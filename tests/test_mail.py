"""Mail intake tests: ingest, dedup, routing, dead-letter with FakeMailSource."""

from __future__ import annotations

import pytest

from backend.engine.mail_inbox import (
    FakeMailSource,
    MailAttachment,
    MailMessage,
    process_mailbox,
    route_playbook,
)
from test_engine_pipeline import (  # noqa: F401
    SAMPLE_CONTRACT,
    FakeLLM,
    _upload,
    admin_headers,
    client,
    fake_llm,
    platform_env,
)


def _mail(message_id: str, subject: str = "[CG] 供应商合同", attachments: list[MailAttachment] | None = None):
    return MailMessage(
        message_id=message_id,
        subject=subject,
        attachments=attachments
        if attachments is not None
        else [MailAttachment(filename="合同.txt", content=SAMPLE_CONTRACT.encode("utf-8"))],
    )


@pytest.fixture()
def mail_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MAIL_DEAD_LETTER_DIR", str(tmp_path / "mail_dead_letter"))
    return tmp_path / "mail_dead_letter"


def wait_terminal(client, headers, document_id, timeout_s=90):
    """Poll until the document leaves parsing/scoring (hybrid retrieval loads
    the real embedding model on first use, so scoring takes a while)."""

    import time

    deadline = time.time() + timeout_s
    detail = client.get(f"/documents/{document_id}", headers=headers).json()
    while time.time() < deadline and detail["document"]["status"] in ("parsing", "scoring"):
        time.sleep(1)
        detail = client.get(f"/documents/{document_id}", headers=headers).json()
    return detail


class TestMailIntake:
    def test_attachment_ingested_and_reviewed(self, client, admin_headers, fake_llm, mail_env):
        source = FakeMailSource([_mail("<m1@t>")])
        stats = process_mailbox(default_playbook="contract-compliance", source=source)
        assert stats["reviewed"] == 1
        assert source.seen == ["<m1@t>"]
        documents = client.get("/documents", headers=admin_headers).json()["documents"]
        assert documents[0]["status"] == "awaiting_review"
        assert documents[0]["source_filename"] == "合同.txt"

    def test_message_id_prevents_reprocessing(self, client, admin_headers, fake_llm, mail_env):
        source = FakeMailSource([_mail("<m2@t>")])
        process_mailbox(default_playbook="contract-compliance", source=source)
        # second pass: fetch_unseen returns empty (mail marked seen) — nothing happens
        stats2 = process_mailbox(default_playbook="contract-compliance", source=FakeMailSource([]))
        assert stats2["reviewed"] == 0
        # same message reappearing unseen (e.g. flag reset) is skipped by mail_state
        stats3 = process_mailbox(
            default_playbook="contract-compliance", source=FakeMailSource([_mail("<m2@t>")])
        )
        assert stats3["reviewed"] == 0
        documents = client.get("/documents", headers=admin_headers).json()["documents"]
        assert len(documents) == 1

    def test_subject_tag_routes_playbook(self, client, admin_headers, fake_llm, mail_env):
        source = FakeMailSource([_mail("<m3@t>", subject="[DI] 客户SOW")])
        process_mailbox(default_playbook="contract-compliance", source=source)
        documents = client.get("/documents", headers=admin_headers).json()["documents"]
        assert documents[0]["playbook_id"] == "delivery-intake"

    def test_no_attachment_skipped_and_audited(self, client, admin_headers, fake_llm, mail_env):
        source = FakeMailSource([_mail("<m4@t>", attachments=[])])
        stats = process_mailbox(default_playbook="contract-compliance", source=source)
        assert stats["skipped"] == 1
        recent = client.get("/documents/audit/recent", headers=admin_headers).json()["audit"]
        assert any(entry["event"] == "mail.no_attachment" for entry in recent)

    def test_duplicate_content_counted_not_reviewed(self, client, admin_headers, fake_llm, mail_env):
        process_mailbox(
            default_playbook="contract-compliance",
            source=FakeMailSource([_mail("<m5a@t>"), _mail("<m5b@t>", subject="[CG] 同附件另一封")]),
        )
        documents = client.get("/documents", headers=admin_headers).json()["documents"]
        assert len(documents) == 1

    def test_unsupported_attachment_dead_lettered(self, client, admin_headers, fake_llm, mail_env):
        source = FakeMailSource(
            [_mail("<m6@t>", attachments=[MailAttachment(filename="virus.exe", content=b"MZ...")])]
        )
        stats = process_mailbox(default_playbook="contract-compliance", source=source)
        assert stats["failed"] == 1
        assert (mail_env / "virus.exe").exists()

    def test_route_playbook_fallback(self):
        assert route_playbook("[XX] 未知标签", "contract-compliance") == "contract-compliance"
        assert route_playbook("无标签", "contract-compliance") == "contract-compliance"


class TestMailer:
    def test_scored_notification_email_sent(self, client, admin_headers, fake_llm, monkeypatch):
        sent = []

        class FakeSMTP:
            def __init__(self, host, port):
                sent.append({"host": host, "port": port, "messages": []})
                FakeSMTP.last = self

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def login(self, user, password):
                sent[-1]["login"] = (user, password)

            def send_message(self, message, from_addr, to_addrs):
                sent[-1]["messages"].append((message["Subject"], to_addrs, message.get_content()))

        monkeypatch.setattr("backend.mailer.smtplib.SMTP_SSL", FakeSMTP)
        monkeypatch.setenv("EMAIL_ENABLED", "true")
        monkeypatch.setenv("EMAIL_HOST", "smtp.test")
        monkeypatch.setenv("EMAIL_PORT", "465")
        monkeypatch.setenv("EMAIL_USER", "bot@corp.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "secret")
        monkeypatch.setenv("NOTIFY_EMAILS", "legal@corp.com,lead@corp.com")

        _upload(client, admin_headers, "合同.txt", SAMPLE_CONTRACT.encode("utf-8"))
        assert sent, "启用邮件后应发送通知"
        subject, recipients, _body = sent[-1]["messages"][0]
        assert "评分完成" in subject
        assert recipients == ["legal@corp.com", "lead@corp.com"]

    def test_disabled_email_never_sends(self, client, admin_headers, fake_llm, monkeypatch):
        sent = []

        class FakeSMTP:
            def __init__(self, *a, **k):
                sent.append(1)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def send_message(self, *a, **k):
                pass

        monkeypatch.setattr("backend.mailer.smtplib.SMTP_SSL", FakeSMTP)
        monkeypatch.setattr("backend.mailer.smtplib.SMTP", FakeSMTP)
        monkeypatch.setenv("EMAIL_ENABLED", "false")
        monkeypatch.setenv("NOTIFY_EMAILS", "x@y.com")
        _upload(client, admin_headers, "n.txt", SAMPLE_CONTRACT.encode("utf-8"))
        assert not sent

    def test_smtp_failure_audited_not_raised(self, client, admin_headers, fake_llm, monkeypatch):
        class ExplodingSMTP:
            def __init__(self, *a, **k):
                raise RuntimeError("smtp down")

        monkeypatch.setattr("backend.mailer.smtplib.SMTP_SSL", ExplodingSMTP)
        monkeypatch.setenv("EMAIL_ENABLED", "true")
        monkeypatch.setenv("EMAIL_HOST", "smtp.test")
        monkeypatch.setenv("NOTIFY_EMAILS", "x@y.com")

        upload = _upload(client, admin_headers, "合同.txt", SAMPLE_CONTRACT.encode("utf-8"))
        # pipeline unaffected (poll through hybrid-retrieval scoring time)
        detail = wait_terminal(client, admin_headers, upload.json()["document_id"])
        assert detail["document"]["status"] == "awaiting_review"
        recent = client.get("/documents/audit/recent", headers=admin_headers).json()["audit"]
        assert any(entry["event"] == "notification.email_failed" for entry in recent)

    def test_finalized_email_carries_docx_attachment(self, client, admin_headers, fake_llm, monkeypatch):
        attachments_seen = []

        class FakeSMTP:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def login(self, *a):
                pass

            def send_message(self, message, from_addr, to_addrs):
                for part in message.walk():
                    fn = part.get_filename()
                    if fn:
                        attachments_seen.append(fn)

        monkeypatch.setattr("backend.mailer.smtplib.SMTP_SSL", FakeSMTP)
        monkeypatch.setenv("EMAIL_ENABLED", "true")
        monkeypatch.setenv("EMAIL_HOST", "smtp.test")
        monkeypatch.setenv("EMAIL_USER", "bot@corp.com")
        monkeypatch.setenv("EMAIL_PASSWORD", "secret")
        monkeypatch.setenv("NOTIFY_EMAILS", "x@y.com")

        upload = _upload(client, admin_headers, "合同.txt", SAMPLE_CONTRACT.encode("utf-8"))
        document_id = upload.json()["document_id"]
        detail = wait_terminal(client, admin_headers, document_id)
        for verdict in detail["verdicts"]:
            if verdict["review_state"] == "awaiting_review":
                client.patch(
                    f"/documents/{document_id}/verdicts/{verdict['id']}",
                    json={"decision": "approve"},
                    headers=admin_headers,
                )
        client.post(f"/documents/{document_id}/finalize", headers=admin_headers)
        assert any(fn.endswith(".docx") for fn in attachments_seen)
