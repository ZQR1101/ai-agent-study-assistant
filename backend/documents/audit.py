"""Append-only audit trail helpers.

Convention: audit entries are only ever inserted or read — never updated or
deleted. Every state-changing platform action should append one entry whose
``correlation_id`` ties the event to a document's friendly ID (or ``auth`` for
account events).
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from backend.documents.models import AuditEntry


def record_audit(
    session: Session,
    *,
    correlation_id: str,
    event: str,
    actor: str = "system",
    payload: dict[str, Any] | None = None,
) -> AuditEntry:
    entry = AuditEntry(
        id=str(uuid.uuid4()),
        correlation_id=correlation_id,
        event=event,
        actor=actor,
        payload=payload,
    )
    session.add(entry)
    return entry


def list_audit(session: Session, *, correlation_id: str | None = None, limit: int = 200) -> list[AuditEntry]:
    query = session.query(AuditEntry)
    if correlation_id:
        query = query.filter(AuditEntry.correlation_id == correlation_id)
    return query.order_by(AuditEntry.created_at.desc()).limit(limit).all()
