"""Notification center tests: generation from pipeline events, read state."""

from __future__ import annotations

import pytest

from test_engine_pipeline import (  # noqa: F401
    SAMPLE_CONTRACT,
    _upload,
    admin_headers,
    client,
    fake_llm,
    platform_env,
)


class TestNotificationGeneration:
    def test_scored_notification_created_after_review(self, client, admin_headers, fake_llm):
        upload = _upload(client, admin_headers, "合同.txt", SAMPLE_CONTRACT.encode("utf-8"))
        document_id = upload.json()["document_id"]
        notifications = client.get("/notifications", headers=admin_headers).json()
        scored = [n for n in notifications["notifications"] if n["type"] == "scored"]
        assert scored, "评分完成应产生通知"
        target = next(n for n in scored if n["correlation_id"] == upload.json()["friendly_id"])
        assert "待签字" in target["body"]
        assert target["payload"]["document_id"] == document_id
        assert target["read"] is False

    def test_failed_notification_created_on_parse_failure(self, client, admin_headers, fake_llm):
        upload = client.post(
            "/documents/upload",
            files={"file": ("bad.txt", b"", "text/plain")},
            data={"playbook_id": "contract-compliance"},
            headers=admin_headers,
        )
        assert upload.status_code == 400  # empty content rejected before pipeline

    def test_finalized_notification_created(self, client, admin_headers, fake_llm):
        upload = _upload(client, admin_headers, "g.txt", SAMPLE_CONTRACT.encode("utf-8"))
        document_id = upload.json()["document_id"]
        detail = client.get(f"/documents/{document_id}", headers=admin_headers).json()
        for verdict in detail["verdicts"]:
            if verdict["review_state"] == "awaiting_review":
                client.patch(
                    f"/documents/{document_id}/verdicts/{verdict['id']}",
                    json={"decision": "approve"},
                    headers=admin_headers,
                )
        client.post(f"/documents/{document_id}/finalize", headers=admin_headers)
        notifications = client.get("/notifications", headers=admin_headers).json()
        assert any(
            n["type"] == "finalized" and n["correlation_id"] == upload.json()["friendly_id"]
            for n in notifications["notifications"]
        )


class TestReadState:
    def test_unread_count_and_mark_read(self, client, admin_headers, fake_llm):
        before = client.get("/notifications", headers=admin_headers).json()["unread"]
        upload = _upload(client, admin_headers, "h.txt", SAMPLE_CONTRACT.encode("utf-8"))
        after = client.get("/notifications", headers=admin_headers).json()
        assert after["unread"] == before + 1

        target = next(
            n for n in after["notifications"]
            if n["correlation_id"] == upload.json()["friendly_id"]
        )
        client.post(f"/notifications/{target['id']}/read", headers=admin_headers)
        result = client.get("/notifications", headers=admin_headers).json()
        assert result["unread"] == before
        marked = next(n for n in result["notifications"] if n["id"] == target["id"])
        assert marked["read"] is True

    def test_mark_all_read(self, client, admin_headers, fake_llm):
        for i in range(2):
            _upload(client, admin_headers, f"m{i}.txt", (SAMPLE_CONTRACT + str(i)).encode("utf-8"))
        unread_before = client.get("/notifications", headers=admin_headers).json()["unread"]
        assert unread_before >= 2
        marked = client.post("/notifications/read-all", headers=admin_headers).json()["marked"]
        assert marked == unread_before
        assert client.get("/notifications", headers=admin_headers).json()["unread"] == 0

    def test_requires_auth(self, client):
        assert client.get("/notifications").status_code == 401

    def test_read_unknown_notification_404(self, client, admin_headers):
        response = client.post(
            "/notifications/nonexistent/read", headers=admin_headers
        )
        assert response.status_code == 404
