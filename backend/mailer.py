"""Email dispatch for notifications (standard library smtplib only).

When ``EMAIL_ENABLED`` is on, every notification created through
:func:`backend.notifications.service.create_notification_and_email` is also
mailed to ``NOTIFY_EMAILS``. Finalized notifications attach the exported Word
report. SMTP failures are audited and swallowed — mail must never break the
review pipeline.
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

from sqlalchemy.orm import Session

from backend.documents.audit import record_audit

logger = logging.getLogger(__name__)

APP_BASE_URL = "http://127.0.0.1:8000"


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def email_enabled() -> bool:
    return _env("EMAIL_ENABLED").lower() in ("true", "1", "yes", "on") and bool(
        _env("EMAIL_HOST")
    ) and bool(_env("NOTIFY_EMAILS"))


def _recipients() -> list[str]:
    return [addr.strip() for addr in _env("NOTIFY_EMAILS").split(",") if addr.strip()]


def _app_base_url() -> str:
    return _env("APP_BASE_URL", APP_BASE_URL)


def build_message(notification, *, document=None, docx_bytes: bytes | None = None) -> EmailMessage:
    """Build the email for a notification row/instance."""

    message = EmailMessage()
    message["Subject"] = f"[Rulebook] {notification.title}"
    message["From"] = _env("EMAIL_FROM", _env("EMAIL_USER"))
    base = _app_base_url()
    link = ""
    if notification.correlation_id and notification.correlation_id.startswith(("CG-", "DI-", "TY-")):
        link = f"{base}/#/documents/{notification.payload.get('document_id')}" if notification.payload else base
    body_lines = [notification.body or notification.title, ""]
    if link:
        body_lines.append(f"查看详情：{link}")
    message.set_content("\n".join(body_lines))

    if notification.type == "finalized" and docx_bytes:
        message.add_attachment(
            docx_bytes,
            maintype="application",
            subtype="vnd.openxmlformats-officedocument.wordprocessingml.document",
            filename=f"{notification.correlation_id}_report.docx",
        )
    return message


def send_notification_email(session: Session, notification) -> bool:
    """Send one notification email. Returns True when sent; never raises.

    Transaction boundary: this function NEVER commits or rolls back the
    caller's session — a mail failure must not discard pipeline state. Audit
    entries are written on a dedicated session.
    """

    if not email_enabled():
        return False
    correlation = notification.correlation_id or "notifications"

    def _audit(event: str, payload: dict) -> None:
        # Attach to the CALLER's session: a separate session would deadlock on
        # SQLite's single-writer lock while the pipeline transaction is open.
        # The row persists when the caller commits; never commit here.
        record_audit(session, correlation_id=correlation, event=event, payload=payload)

    try:
        docx_bytes = None
        if notification.type == "finalized" and notification.payload:
            from backend.documents.models import Document
            from backend.engine.export import default_docx_key, render_deliverable
            from backend.playbooks import get_playbook

            document = session.get(Document, notification.payload.get("document_id"))
            if document is not None:
                playbook = get_playbook(document.playbook_id)
                docx_bytes = render_deliverable(
                    session, document, default_docx_key(playbook)
                )

        message = build_message(notification, docx_bytes=docx_bytes)
        host = _env("EMAIL_HOST")
        port = int(_env("EMAIL_PORT", "465" if _env("EMAIL_USE_SSL", "true") in ("true", "1") else "587"))
        use_ssl = _env("EMAIL_USE_SSL", "true").lower() in ("true", "1", "yes", "on")
        user = _env("EMAIL_USER")
        password = _env("EMAIL_PASSWORD")

        if use_ssl:
            with smtplib.SMTP_SSL(host, port) as server:
                if user and password:
                    server.login(user, password)
                server.send_message(message, from_addr=message["From"], to_addrs=_recipients())
        else:
            with smtplib.SMTP(host, port) as server:
                if _env("EMAIL_USE_TLS", "true").lower() in ("true", "1", "yes", "on"):
                    server.starttls()
                if user and password:
                    server.login(user, password)
                server.send_message(message, from_addr=message["From"], to_addrs=_recipients())

        _audit("notification.email_sent", {"type": notification.type, "recipients": _recipients()})
        return True
    except Exception as exc:
        logger.exception("通知邮件发送失败")
        _audit("notification.email_failed", {"type": notification.type, "error": str(exc)[:300]})
        return False
