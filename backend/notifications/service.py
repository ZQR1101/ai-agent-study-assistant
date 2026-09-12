"""Notification service: create, list with read state, mark read.

Notifications are broadcast to the whole team (small-team model); read state
is per user. Email dispatch hooks attach in M2 (mailer) without changing this
contract.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.orm import Session

from backend.notifications.models import Notification, NotificationRead


def create_notification(
    session: Session,
    *,
    type: str,
    title: str,
    body: str = "",
    payload: dict[str, Any] | None = None,
    correlation_id: str | None = None,
) -> Notification:
    notification = Notification(
        id=str(uuid.uuid4()),
        type=type,
        title=title,
        body=body,
        payload=payload,
        correlation_id=correlation_id,
    )
    session.add(notification)
    return notification


def list_notifications(session: Session, user_id: str, *, limit: int = 50) -> list[dict]:
    rows = (
        session.query(Notification)
        .order_by(Notification.created_at.desc())
        .limit(limit)
        .all()
    )
    read_ids = {
        row.notification_id
        for row in session.query(NotificationRead).filter(NotificationRead.user_id == user_id).all()
    }
    return [
        {
            "id": n.id,
            "type": n.type,
            "title": n.title,
            "body": n.body,
            "payload": n.payload,
            "correlation_id": n.correlation_id,
            "created_at": n.created_at.isoformat() if n.created_at else None,
            "read": n.id in read_ids,
        }
        for n in rows
    ]


def unread_count(session: Session, user_id: str) -> int:
    read_ids = {
        row.notification_id
        for row in session.query(NotificationRead).filter(NotificationRead.user_id == user_id).all()
    }
    total = session.query(Notification).count()
    return max(0, total - len(read_ids))


def mark_read(session: Session, user_id: str, notification_id: str) -> bool:
    exists = session.get(Notification, notification_id)
    if exists is None:
        return False
    already = (
        session.query(NotificationRead)
        .filter(
            NotificationRead.notification_id == notification_id,
            NotificationRead.user_id == user_id,
        )
        .first()
    )
    if already is None:
        session.add(NotificationRead(notification_id=notification_id, user_id=user_id))
        session.commit()
    return True


def mark_all_read(session: Session, user_id: str) -> int:
    read_ids = {
        row.notification_id
        for row in session.query(NotificationRead).filter(NotificationRead.user_id == user_id).all()
    }
    unread = (
        session.query(Notification)
        .filter(Notification.id.not_in(read_ids) if read_ids else Notification.id.is_not(None))
        .all()
    )
    for notification in unread:
        session.add(NotificationRead(notification_id=notification.id, user_id=user_id))
    session.commit()
    return len(unread)
