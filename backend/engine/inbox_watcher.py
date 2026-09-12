"""Inbox watcher: watched-folder auto-intake, the zero-touch entry.

A local folder replaces the email trigger from the reference architecture:
any supported document dropped into ``INBOX_DIR`` is ingested, deduplicated
by content hash, and queued for review. Files still being written (size
unstable across the settle interval) are left for the next poll. Malformed
intakes land in a dead-letter state instead of disappearing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from backend.documents.audit import record_audit
from backend.documents.models import Document
from backend.documents.service import create_document
from backend.engine.orchestrator import ReviewConflict, run_review
from backend.platform_db import platform_session

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".docx"}


@dataclass
class WatcherStats:
    ingested: int = 0
    duplicates: int = 0
    dead_lettered: int = 0
    pending: int = 0

    def merge(self, other: "WatcherStats") -> None:
        self.ingested += other.ingested
        self.duplicates += other.duplicates
        self.dead_lettered += other.dead_lettered
        self.pending += other.pending


# Size snapshots persist across polls so file-settling detection works
# call-to-call; entries are dropped once the file leaves the inbox.
_SEEN_SIZES: dict[str, int] = {}


def inbox_dir() -> Path:
    """Watched folder; read at call time so tests can isolate via env."""

    import os

    from backend.config import get_config

    raw = (os.getenv("INBOX_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return get_config().project_root / "data" / "inbox"


def archive_dir() -> Path:
    import os

    from backend.config import get_config

    raw = (os.getenv("INBOX_ARCHIVE_DIR") or "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return get_config().project_root / "data" / "inbox_archive"


def dead_letter_dir() -> Path:
    return archive_dir() / "dead_letter"


def _is_settled(path: Path, interval: float) -> bool:
    """True when the file is safe to ingest.

    ``interval <= 0`` means no settling check — ingest anything non-empty.
    Otherwise a file must be seen twice with a stable non-zero size.
    """

    try:
        size_now = path.stat().st_size
    except OSError:
        return False
    if size_now <= 0:
        return False
    if interval <= 0:
        return True
    previous = _SEEN_SIZES.get(str(path))
    _SEEN_SIZES[str(path)] = size_now
    return previous is not None and previous == size_now


def _forget(path: Path) -> None:
    _SEEN_SIZES.pop(str(path), None)


def _archive(path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    counter = 1
    while target.exists():
        target = target_dir / f"{path.stem}_{counter}{path.suffix}"
        counter += 1
    path.replace(target)
    return target


def poll_once(
    *,
    playbook_id: str,
    actor: str = "inbox",
    settle_interval: float = 0.0,
    custom_llm=None,
    stats: WatcherStats | None = None,
) -> WatcherStats:
    """One ingestion pass over the inbox. Returns per-pass counters."""

    stats = stats or WatcherStats()
    watched = inbox_dir()
    if not watched.exists():
        return stats

    for path in sorted(watched.iterdir()):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        if not _is_settled(path, settle_interval):
            stats.pending += 1
            continue

        session: Session = platform_session()
        try:
            try:
                content = path.read_bytes()
            except OSError as exc:
                record_audit(
                    session,
                    correlation_id="inbox",
                    event="inbox.read_failed",
                    actor=actor,
                    payload={"file": path.name, "error": str(exc)},
                )
                session.commit()
                _forget(_archive(path, dead_letter_dir()))
                stats.dead_lettered += 1
                continue

            try:
                document, created = create_document(
                    session,
                    playbook_id=playbook_id,
                    filename=path.name,
                    content=content,
                    actor=actor,
                )
            except Exception:
                logger.exception("inbox intake failed for %s", path.name)
                session.rollback()
                record_audit(
                    session,
                    correlation_id="inbox",
                    event="inbox.intake_failed",
                    actor=actor,
                    payload={"file": path.name},
                )
                session.commit()
                _forget(_archive(path, dead_letter_dir()))
                stats.dead_lettered += 1
                continue

            if not created:
                record_audit(
                    session,
                    correlation_id=document.friendly_id,
                    event="inbox.duplicate_ignored",
                    actor=actor,
                    payload={"file": path.name},
                )
                session.commit()
                _forget(_archive(path, archive_dir()))
                stats.duplicates += 1
                continue

            friendly = document.friendly_id
            document_id = document.id
            session.commit()
        finally:
            session.close()

        # Review synchronously; failures move the file to dead-letter.
        try:
            run_review(document_id, trigger="inbox", custom_llm=custom_llm)
            _forget(_archive(path, archive_dir()))
            stats.ingested += 1
        except ReviewConflict as exc:
            logger.warning("inbox review conflict for %s: %s", friendly, exc)
            stats.ingested += 1  # record exists; nothing more to do this pass
        except Exception:
            logger.exception("inbox review failed for %s", friendly)
            session: Session = platform_session()
            try:
                document = session.get(Document, document_id)
                if document is not None and document.status != "failed":
                    document.status = "failed"
                    document.status_reason = "收件处理失败，请重试"
                    record_audit(
                        session,
                        correlation_id=friendly,
                        event="review.failed",
                        actor=actor,
                        payload={"trigger": "inbox"},
                    )
                    session.commit()
            finally:
                session.close()
            _forget(_archive(path, dead_letter_dir()))
            stats.dead_lettered += 1
    return stats


def run_watcher(
    *,
    playbook_id: str,
    interval_seconds: float = 10.0,
    settle_interval: float = 2.0,
    once: bool = False,
    custom_llm=None,
):
    """Poll the inbox forever (or once). Blocks the caller."""

    stats = WatcherStats()
    while True:
        try:
            poll_once(
                playbook_id=playbook_id,
                settle_interval=settle_interval,
                custom_llm=custom_llm,
                stats=stats,
            )
        except Exception:
            logger.exception("inbox poll crashed; continuing")
        if once:
            return stats
        time.sleep(interval_seconds)
