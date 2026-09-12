"""Notification API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.auth.dependencies import require_user
from backend.auth.models import User
from backend.notifications import service
from backend.platform_db import platform_session

router = APIRouter(prefix="/notifications", tags=["Notifications"])


@router.get("")
def list_notifications(user: User = Depends(require_user)) -> dict:
    session: Session = platform_session()
    try:
        items = service.list_notifications(session, user.id)
        return {"notifications": items, "unread": service.unread_count(session, user.id)}
    finally:
        session.close()


@router.post("/read-all")
def read_all(user: User = Depends(require_user)) -> dict:
    session: Session = platform_session()
    try:
        marked = service.mark_all_read(session, user.id)
        return {"marked": marked}
    finally:
        session.close()


@router.post("/{notification_id}/read")
def mark_read(notification_id: str, user: User = Depends(require_user)) -> dict:
    session: Session = platform_session()
    try:
        if not service.mark_read(session, user.id, notification_id):
            raise HTTPException(status_code=404, detail="通知不存在")
        return {"ok": True}
    finally:
        session.close()
