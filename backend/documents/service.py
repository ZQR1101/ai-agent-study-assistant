"""Document lifecycle service: create with dedup, friendly IDs, storage."""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from backend.config import get_config
from backend.documents.audit import record_audit
from backend.documents.models import Document
from backend.playbooks.registry import get_playbook


def sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def next_friendly_id(session: Session, prefix: str) -> str:
    """Next human-friendly reference for a prefix, e.g. CG-0007."""

    rows = (
        session.query(Document.friendly_id)
        .filter(Document.friendly_id.like(f"{prefix}-%"))
        .all()
    )
    max_seq = 0
    for (value,) in rows:
        try:
            max_seq = max(max_seq, int(value.rsplit("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return f"{prefix}-{max_seq + 1:04d}"


def source_dir(document_id: str) -> Path:
    """Directory holding the original uploaded file for a document.

    Reads ``PLATFORM_DOCS_DIR`` at call time (tests isolate via this env var);
    falls back to ``<project_root>/data/review_documents``.
    """

    import os

    base = (os.getenv("PLATFORM_DOCS_DIR") or "").strip()
    root = Path(base) if base else get_config().project_root / "data" / "review_documents"
    return root / document_id


def create_document(
    session: Session,
    *,
    playbook_id: str,
    filename: str,
    content: bytes,
    actor: str,
) -> tuple[Document, bool]:
    """Create a document record with content-hash dedup.

    Returns (document, created). When the same content was uploaded before,
    returns the existing record with created=False.
    """

    playbook = get_playbook(playbook_id)  # raises KeyError for unknown ids
    from backend.platform_db import seed_playbook_rules

    seed_playbook_rules(session, playbook_id)  # lazy seeding for new playbooks
    content_hash = sha256_hex(content)
    existing = (
        session.query(Document).filter(Document.source_hash == content_hash).first()
    )
    if existing is not None:
        record_audit(
            session,
            correlation_id="documents",
            event="document.duplicate_rejected",
            actor=actor,
            payload={"hash": content_hash, "existing": existing.friendly_id, "filename": filename},
        )
        session.commit()
        return existing, False

    document = Document(
        id=str(uuid.uuid4()),
        friendly_id=next_friendly_id(session, playbook.friendly_id_prefix),
        playbook_id=playbook.id,
        title=Path(filename).stem or filename,
        source_filename=Path(filename).name,
        source_hash=content_hash,
        status="uploaded",
        created_by=actor,
    )
    session.add(document)
    session.flush()

    directory = source_dir(document.id)
    directory.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix.lower() or ".bin"
    (directory / f"source{suffix}").write_bytes(content)

    record_audit(
        session,
        correlation_id=document.friendly_id,
        event="document.created",
        actor=actor,
        payload={
            "document_id": document.id,
            "playbook": playbook.id,
            "filename": document.source_filename,
            "sha256": content_hash,
        },
    )
    session.commit()
    return document, True


def get_document(session: Session, document_id_or_friendly: str) -> Document | None:
    query = session.query(Document)
    document = query.filter(Document.id == document_id_or_friendly).first()
    if document is None:
        document = (
            session.query(Document)
            .filter(Document.friendly_id == document_id_or_friendly)
            .first()
        )
    return document
