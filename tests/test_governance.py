"""Governance tests: signature gate, review state machine, finalization, export."""

from __future__ import annotations

import pytest

from test_engine_pipeline import (  # noqa: F401  (fixtures shared via import)
    SAMPLE_CONTRACT,
    FakeLLM,
    _upload,
    admin_headers,
    client,
    fake_llm,
    platform_env,
)


def _reviewed_document(client, admin_headers, monkeypatch) -> dict:
    """Upload + force an amber verdict so the signature gate engages."""

    from backend.engine import scoring

    scripted = FakeLLM(
        script={
            "付款账期": {
                "rating": "amber",
                "rationale": "账期60日在黄区边缘",
                "citations": [{"quote": "SAMPLE QUOTE"}],
                "gap_reason": "",
            },
        },
        default={
            "rating": "green",
            "rationale": "条款内容满足规则要求",
            "citations": [{"quote": "SAMPLE QUOTE"}],
            "gap_reason": "",
        },
    )
    monkeypatch.setattr(scoring, "_default_scoring_llm", lambda: scripted)
    upload = _upload(client, admin_headers, "合同.txt", SAMPLE_CONTRACT.encode("utf-8"))
    document_id = upload.json()["document_id"]
    return client.get(f"/documents/{document_id}", headers=admin_headers).json()


def _find_verdict(detail: dict, rule_name: str) -> dict:
    for verdict in detail["verdicts"]:
        if verdict["rule_name"] == rule_name:
            return verdict
    raise AssertionError(f"verdict not found: {rule_name}")


class TestSignatureGate:
    def test_finalize_blocked_while_awaiting_review(self, client, admin_headers, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]
        response = client.post(f"/documents/{document_id}/finalize", headers=admin_headers)
        assert response.status_code == 409
        assert "签字" in response.json()["detail"]

    def test_export_blocked_before_finalize(self, client, admin_headers, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]
        response = client.get(
            f"/documents/{document_id}/export?format=docx", headers=admin_headers
        )
        assert response.status_code == 409

    def test_approve_then_finalize_then_export(self, client, admin_headers, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]
        verdict = _find_verdict(detail, "付款账期")

        decision = client.patch(
            f"/documents/{document_id}/verdicts/{verdict['id']}",
            json={"decision": "approve", "expert_note": "账期可接受"},
            headers=admin_headers,
        )
        assert decision.status_code == 200, decision.text
        assert decision.json()["pending_review"] == 0

        finalize = client.post(f"/documents/{document_id}/finalize", headers=admin_headers)
        assert finalize.status_code == 200
        assert finalize.json()["status"] == "finalized"

        docx = client.get(f"/documents/{document_id}/export?format=docx", headers=admin_headers)
        assert docx.status_code == 200
        assert docx.content[:2] == b"PK"  # docx zip magic
        assert "compliance_report" in docx.headers["Content-Disposition"]

        xlsx = client.get(f"/documents/{document_id}/export?format=xlsx", headers=admin_headers)
        assert xlsx.status_code == 200
        assert xlsx.content[:2] == b"PK"

    def test_illegal_transition_rejected(self, client, admin_headers, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]
        verdict = _find_verdict(detail, "付款账期")

        # approve without note is fine, then approving again is illegal
        first = client.patch(
            f"/documents/{document_id}/verdicts/{verdict['id']}",
            json={"decision": "approve"},
            headers=admin_headers,
        )
        assert first.status_code == 200
        again = client.patch(
            f"/documents/{document_id}/verdicts/{verdict['id']}",
            json={"decision": "approve"},
            headers=admin_headers,
        )
        assert again.status_code == 422

    def test_reject_requires_note(self, client, admin_headers, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]
        verdict = _find_verdict(detail, "付款账期")
        response = client.patch(
            f"/documents/{document_id}/verdicts/{verdict['id']}",
            json={"decision": "reject"},
            headers=admin_headers,
        )
        assert response.status_code == 422
        with_note = client.patch(
            f"/documents/{document_id}/verdicts/{verdict['id']}",
            json={"decision": "reject", "expert_note": "账期超标准，驳回重谈"},
            headers=admin_headers,
        )
        assert with_note.status_code == 200
        assert with_note.json()["verdict"]["review_state"] == "rejected"

    def test_edit_changes_rating_and_scorecard(self, client, admin_headers, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]
        verdict = _find_verdict(detail, "付款账期")
        response = client.patch(
            f"/documents/{document_id}/verdicts/{verdict['id']}",
            json={
                "decision": "edit",
                "new_rating": "red",
                "rationale": "改判：账期不符合60天标准",
                "expert_note": "合同实际账期90天",
            },
            headers=admin_headers,
        )
        assert response.status_code == 200
        after = client.get(f"/documents/{document_id}", headers=admin_headers).json()
        edited = _find_verdict(after, "付款账期")
        assert edited["rating"] == "red"
        assert edited["review_state"] == "edited"
        assert after["document"]["scorecard"]["counts"]["red"] >= 1

    def test_review_events_recorded(self, client, admin_headers, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]
        verdict = _find_verdict(detail, "付款账期")
        client.patch(
            f"/documents/{document_id}/verdicts/{verdict['id']}",
            json={"decision": "approve", "expert_note": "ok"},
            headers=admin_headers,
        )
        events = client.get(
            f"/documents/{document_id}/review-events", headers=admin_headers
        ).json()["events"]
        assert any(event["action"] == "approve" for event in events)
        assert all(event["actor"] for event in events)


class TestDocumentQA:
    def test_ask_answers_with_clause_refs(self, client, admin_headers, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]
        monkeypatch.setattr(
            "backend.engine.scoring._llm_invoke",
            lambda *a, **k: "根据 [条款 5]，可用性为99.95%。",
        )
        response = client.post(
            f"/documents/{document_id}/ask",
            json={"question": "可用性承诺是多少？"},
            headers=admin_headers,
        )
        assert response.status_code == 200
        body = response.json()
        assert "条款 5" in body["answer"]
        assert body["citations"]

    def test_ask_without_llm_returns_502(self, client, admin_headers, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]

        def broken(*args, **kwargs):
            raise RuntimeError("no key")

        monkeypatch.setattr("backend.engine.scoring._llm_invoke", broken)
        response = client.post(
            f"/documents/{document_id}/ask",
            json={"question": "违约金怎么算？"},
            headers=admin_headers,
        )
        assert response.status_code == 502


class TestQueue:
    def test_queue_lists_awaiting_red_first(self, client, admin_headers, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        queue = client.get("/documents/review/queue", headers=admin_headers).json()["items"]
        assert queue
        assert queue[0]["friendly_id"] == detail["document"]["friendly_id"]
        ratings = [item["rating"] for item in queue]
        assert "amber" in ratings or "red" in ratings

    def test_queue_requires_reviewer_role(self, client, admin_headers, platform_env):
        # create an expert and verify access
        password_file = platform_env["db_path"].parent / "bootstrap_admin_password.txt"
        password = ""
        for line in password_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("password:"):
                password = line.removeprefix("password:").strip()
        admin_token = client.post(
            "/auth/login", json={"username": "admin", "password": password}
        ).json()["token"]
        created = client.post(
            "/auth/users",
            json={"username": "expert9", "password": "expert-pass-123", "role": "expert"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert created.status_code == 201
        expert_token = client.post(
            "/auth/login", json={"username": "expert9", "password": "expert-pass-123"}
        ).json()["token"]
        response = client.get(
            "/documents/review/queue", headers={"Authorization": f"Bearer {expert_token}"}
        )
        assert response.status_code == 200


class TestIdempotency:
    def test_double_finalize_is_idempotent(self, client, admin_headers, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]
        verdict = _find_verdict(detail, "付款账期")
        client.patch(
            f"/documents/{document_id}/verdicts/{verdict['id']}",
            json={"decision": "approve"},
            headers=admin_headers,
        )
        first = client.post(f"/documents/{document_id}/finalize", headers=admin_headers)
        second = client.post(f"/documents/{document_id}/finalize", headers=admin_headers)
        assert first.status_code == 200
        assert second.status_code == 200
        assert second.json()["status"] == "finalized"

    def test_decisions_blocked_after_finalize(self, client, admin_headers, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]
        verdict = _find_verdict(detail, "付款账期")
        client.patch(
            f"/documents/{document_id}/verdicts/{verdict['id']}",
            json={"decision": "approve"},
            headers=admin_headers,
        )
        client.post(f"/documents/{document_id}/finalize", headers=admin_headers)
        late = client.patch(
            f"/documents/{document_id}/verdicts/{verdict['id']}",
            json={"decision": "edit", "new_rating": "green"},
            headers=admin_headers,
        )
        assert late.status_code == 409
