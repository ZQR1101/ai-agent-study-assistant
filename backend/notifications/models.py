"""Notification models: broadcast notifications with per-user read state."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from backend.documents.models import utcnow
from backend.platform_db import PlatformBase

NOTIFICATION_TYPES = ("scored", "finalized", "failed", "rule_suggested", "system")


class Notification(PlatformBase):
    __tablename__ = "notifications"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    type: Mapped[str] = mapped_column(String(30), index=True)
    title: Mapped[str] = mapped_column(String(200))
    body: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(40), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class NotificationRead(PlatformBase):
    __tablename__ = "notification_reads"
    __table_args__ = (
        UniqueConstraint("notification_id", "user_id", name="uq_notification_read"),
    )

    notification_id: Mapped[str] = mapped_column(
        String(36), primary_key=True
    )
    user_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    read_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    created_seq: Mapped[int] = mapped_column(Integer, default=0)  # reserved
