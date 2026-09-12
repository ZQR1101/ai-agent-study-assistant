"""IMAP mail intake: the zero-touch entry from the reference architecture.

A watched inbox replaces the watched folder: unseen mails with supported
attachments are ingested, deduplicated by content hash, routed to a playbook
by subject tag (e.g. ``[CG]`` → contract-compliance) and reviewed. Protocol
access is abstracted behind :class:`MailSource` so tests inject a
:class:`FakeMailSource` — the production path never runs in tests.
"""

from __future__ import annotations

import email
import email.utils
import hashlib
import imaplib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from backend.documents.audit import record_audit
from backend.documents.models import Document
from backend.engine.orchestrator import ReviewConflict, run_review
from backend.platform_db import PlatformBase, platform_session
from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from backend.documents.models import utcnow

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".md", ".txt"}
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024


class MailState(PlatformBase):
    """Per-message processing state so unseen mails are never reprocessed."""

    __tablename__ = "mail_state"

    message_id: Mapped[str] = mapped_column(String(255), primary_key=True)
    status: Mapped[str] = mapped_column(String(20), default="processed")  # processed | failed
    detail: Mapped[str | None] = mapped_column(String(255), nullable=True)
    processed_at: Mapped[object] = mapped_column(DateTime(timezone=True), default=utcnow)


@dataclass
class MailAttachment:
    filename: str
    content: bytes


@dataclass
class MailMessage:
    message_id: str
    subject: str
    attachments: list[MailAttachment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    _imap_number: str | None = None  # internal: for IMAP \Seen marking


class MailSource:
    """Protocol for mail backends. Production: ImapMailSource. Tests: FakeMailSource."""

    def fetch_unseen(self) -> list[MailMessage]:  # pragma: no cover - interface
        raise NotImplementedError

    def mark_seen(self, message: MailMessage) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class ImapMailSource(MailSource):
    def __init__(self, *, host: str, port: int, user: str, password: str,
                 folder: str = "INBOX", use_ssl: bool = True):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.folder = folder
        self.use_ssl = use_ssl
        self._imap: imaplib.IMAP4 | imaplib.IMAP4_SSL | None = None

    def _connect(self):
        if self._imap is not None:
            return
        if self.use_ssl:
            self._imap = imaplib.IMAP4_SSL(self.host, self.port)
        else:
            self._imap = imaplib.IMAP4(self.host, self.port)
        self._imap.login(self.user, self.password)
        status, _ = self._imap.select(self.folder)
        if status != "OK":
            raise RuntimeError(f"无法打开邮箱文件夹: {self.folder}")

    def fetch_unseen(self) -> list[MailMessage]:
        self._connect()
        assert self._imap is not None
        status, data = self._imap.search(None, "UNSEEN")
        if status != "OK":
            return []
        messages: list[MailMessage] = []
        for number in (data[0] or b"").split():
            status, payload = self._imap.fetch(number, "(RFC822)")
            if status != "OK" or not payload or payload[0] is None:
                continue
            raw = payload[0][1]
            message = self._parse_mail(number.decode("ascii"), raw)
            if message is not None:
                messages.append(message)
        return messages

    def _parse_mail(self, number: str, raw: bytes) -> MailMessage | None:
        msg = email.message_from_bytes(raw)
        message_id = (msg.get("Message-ID") or "").strip() or (
            "sha256:" + hashlib.sha256(raw).hexdigest()[:32]
        )
        subject = str(msg.get("Subject") or "").strip()
        message = MailMessage(message_id=message_id, subject=subject, _imap_number=number)

        for part in msg.walk():
            filename = part.get_filename()
            if not filename:
                continue
            decoded_name = str(email.utils.collapse_rfc2231_value(filename) or filename)
            content = part.get_payload(decode=True) or b""
            suffix = Path(decoded_name).suffix.lower()
            if suffix not in SUPPORTED_SUFFIXES:
                message.warnings.append(f"忽略不支持的附件: {decoded_name}")
                continue
            if not content:
                message.warnings.append(f"忽略空附件: {decoded_name}")
                continue
            if len(content) > MAX_ATTACHMENT_BYTES:
                message.warnings.append(f"忽略超大附件(>{MAX_ATTACHMENT_BYTES}B): {decoded_name}")
                continue
            message.attachments.append(
                MailAttachment(filename=decoded_name, content=content)
            )
        return message

    def mark_seen(self, message: MailMessage) -> None:
        if self._imap is None or not message._imap_number:
            return
        try:
            self._imap.store(message._imap_number, "+FLAGS", "\\Seen")
        except Exception:
            logger.exception("标记已读失败: %s", message.message_id)

    def logout(self) -> None:
        if self._imap is not None:
            try:
                self._imap.logout()
            except Exception:
                pass
            self._imap = None


class FakeMailSource(MailSource):
    """Scripted source for tests."""

    def __init__(self, messages: list[MailMessage]):
        self.messages = list(messages)
        self.seen: list[str] = []

    def fetch_unseen(self) -> list[MailMessage]:
        return list(self.messages)

    def mark_seen(self, message: MailMessage) -> None:
        self.seen.append(message.message_id)
        self.messages = [m for m in self.messages if m.message_id != message.message_id]


def get_mail_source() -> MailSource | None:
    """Build the production IMAP source from env, or None when disabled/misconfigured."""

    import os

    if os.getenv("MAIL_ENABLED", "false").lower() not in ("true", "1", "yes", "on"):
        return None
    host = (os.getenv("MAIL_HOST") or "").strip()
    user = (os.getenv("MAIL_USER") or "").strip()
    password = (os.getenv("MAIL_PASSWORD") or "").strip()
    if not host or not user or not password:
        return None
    return ImapMailSource(
        host=host,
        port=int(os.getenv("MAIL_PORT", "993")),
        user=user,
        password=password,
        folder=os.getenv("MAIL_FOLDER", "INBOX"),
        use_ssl=os.getenv("MAIL_USE_SSL", "true").lower() in ("true", "1", "yes", "on"),
    )


def parse_subject_routes() -> dict[str, str]:
    import os

    raw = os.getenv("MAIL_SUBJECT_ROUTES", "CG=contract-compliance,DI=delivery-intake")
    routes: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" in pair:
            tag, playbook = pair.split("=", 1)
            routes[tag.strip().upper()] = playbook.strip()
    return routes


_TAG_RE = re.compile(r"\[([A-Za-z]+)\]")


def route_playbook(subject: str, default_playbook: str) -> str:
    from backend.playbooks import get_playbook

    for tag in _TAG_RE.findall(subject):
        playbook = parse_subject_routes().get(tag.upper())
        if playbook:
            try:
                get_playbook(playbook)
                return playbook
            except KeyError:
                continue
    return default_playbook


def _dead_letter_dir() -> Path:
    import os

    from backend.config import get_config

    raw = (os.getenv("MAIL_DEAD_LETTER_DIR") or "").strip()
    if raw:
        return Path(raw)
    return get_config().project_root / "data" / "mail_dead_letter"


def _already_processed(session: Session, message_id: str) -> bool:
    row = session.get(MailState, message_id)
    return row is not None


def _record_state(session: Session, message_id: str, status: str, detail: str | None = None) -> None:
    session.merge(MailState(message_id=message_id, status=status, detail=detail))
    session.commit()


def process_mailbox(
    *,
    default_playbook: str,
    source: MailSource | None = None,
    custom_llm=None,
    actor: str = "mail",
    stats: dict | None = None,
) -> dict:
    """One ingestion pass over the IMAP inbox. Returns per-pass counters."""

    stats = stats or {"reviewed": 0, "duplicates": 0, "skipped": 0, "failed": 0, "attachments": 0}
    source = source or get_mail_source()
    if source is None:
        return stats

    for mail in source.fetch_unseen():
        session: Session = platform_session()
        try:
            if _already_processed(session, mail.message_id):
                source.mark_seen(mail)
                continue

            from backend.playbooks import get_playbook

            playbook_id = route_playbook(mail.subject, default_playbook)
            try:
                get_playbook(playbook_id)
            except KeyError:
                playbook_id = default_playbook

            if not mail.attachments:
                record_audit(
                    session, correlation_id="mail", event="mail.no_attachment",
                    actor=actor, payload={"subject": mail.subject, "warnings": mail.warnings},
                )
                session.commit()
                _record_state(session, mail.message_id, "processed", "no_attachment")
                source.mark_seen(mail)
                stats["skipped"] += 1
                continue

            mail_failed = False
            for attachment in mail.attachments:
                try:
                    document, created = create_document_record(
                        session, playbook_id=playbook_id, filename=attachment.filename,
                        content=attachment.content, actor=actor,
                    )
                except Exception:
                    logger.exception("邮件附件入库失败: %s", attachment.filename)
                    session.rollback()
                    record_audit(
                        session, correlation_id="mail", event="mail.intake_failed",
                        actor=actor, payload={"file": attachment.filename},
                    )
                    session.commit()
                    _write_dead_letter(attachment)
                    stats["failed"] += 1
                    mail_failed = True
                    continue

                if not created:
                    record_audit(
                        session, correlation_id=document.friendly_id,
                        event="mail.duplicate_ignored", actor=actor,
                        payload={"file": attachment.filename},
                    )
                    session.commit()
                    stats["duplicates"] += 1
                    continue

                document_id = document.id
                friendly = document.friendly_id
                session.commit()
                try:
                    run_review(document_id, trigger="mail", custom_llm=custom_llm)
                    stats["reviewed"] += 1
                    stats["attachments"] += 1
                except ReviewConflict:
                    stats["reviewed"] += 1
                except Exception:
                    logger.exception("邮件附件评审失败: %s", friendly)
                    _write_dead_letter(attachment)
                    stats["failed"] += 1
                    mail_failed = True

            _record_state(
                session, mail.message_id,
                "failed" if mail_failed else "processed",
                f"attachments={len(mail.attachments)}",
            )
            source.mark_seen(mail)
        finally:
            session.close()
    return stats


def create_document_record(session: Session, *, playbook_id: str, filename: str, content: bytes, actor: str):
    from backend.documents.service import create_document

    return create_document(session, playbook_id=playbook_id, filename=filename, content=content, actor=actor)


def _write_dead_letter(attachment: MailAttachment) -> None:
    directory = _dead_letter_dir()
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w.\-一-龥]+", "_", attachment.filename) or "attachment.bin"
    target = directory / safe_name
    counter = 1
    while target.exists():
        target = directory / f"{target.stem}_{counter}{target.suffix}"
        counter += 1
    target.write_bytes(attachment.content)


def run_mail_loop(*, default_playbook: str, interval_seconds: float = 60.0,
                  once: bool = False, custom_llm=None) -> dict:
    stats: dict = {"reviewed": 0, "duplicates": 0, "skipped": 0, "failed": 0, "attachments": 0}
    while True:
        try:
            process_mailbox(default_playbook=default_playbook, custom_llm=custom_llm, stats=stats)
        except Exception:
            logger.exception("mail poll crashed; continuing")
        if once:
            return stats
        time.sleep(interval_seconds)
