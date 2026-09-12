"""Rule suggestion agent tests: draft → accept → live rulebook, dismiss, auth."""

from __future__ import annotations

import json

import pytest

from test_engine_pipeline import (  # noqa: F401
    SAMPLE_CONTRACT,
    FakeLLM,
    _upload,
    admin_headers,
    client,
    fake_llm,
    platform_env,
)


class SuggestionLLM(FakeLLM):
    """Scripted suggestion drafting."""

    def invoke(self, prompt: str):
        class Response:
            def __init__(self, content):
                self.content = content

        if "规则手册治理代理" in prompt:
            return Response(
                json.dumps(
                    {
                        "suggestions": [
                            {
                                "dimension": "商业",
                                "name": "禁止无限期自动续约",
                                "guidance": "存在无限期自动续约条款为红；续约需双方书面确认",
                                "weight": 2,
                                "rationale": "样例合同第X条允许自动续约，现有手册无对应规则",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )
        return super().invoke(prompt)


def _reviewed_document(client, admin_headers, monkeypatch):
    from test_engine_pipeline import SAMPLE_CONTRACT as CONTRACT

    upload = _upload(client, admin_headers, "合同.txt", CONTRACT.encode("utf-8"))
    document_id = upload.json()["document_id"]
    return client.get(f"/documents/{document_id}", headers=admin_headers).json()


class TestSuggestionFlow:
    def test_generate_proposes_and_dedupes(self, client, admin_headers, fake_llm, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]
        monkeypatch.setattr(
            "backend.engine.suggestions._llm_invoke",
            lambda custom_llm, prompt: SuggestionLLM().invoke(prompt).content,
        )

        first = client.post(f"/rule-suggestions/generate/{document_id}", headers=admin_headers)
        assert first.status_code == 202, first.text
        assert first.json()["created"] == 1

        second = client.post(f"/rule-suggestions/generate/{document_id}", headers=admin_headers)
        assert second.json()["created"] == 0  # 同名 proposed 建议去重

        listed = client.get(
            "/rule-suggestions?playbook_id=contract-compliance&status=proposed",
            headers=admin_headers,
        ).json()["suggestions"]
        assert len(listed) == 1
        assert listed[0]["name"] == "禁止无限期自动续约"
        assert listed[0]["status"] == "proposed"

    def test_accept_puts_rule_into_live_rulebook(self, client, admin_headers, fake_llm, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]
        monkeypatch.setattr(
            "backend.engine.suggestions._llm_invoke",
            lambda custom_llm, prompt: SuggestionLLM().invoke(prompt).content,
        )
        client.post(f"/rule-suggestions/generate/{document_id}", headers=admin_headers)
        listed = client.get(
            "/rule-suggestions?playbook_id=contract-compliance&status=proposed",
            headers=admin_headers,
        ).json()["suggestions"]

        accepted = client.post(
            f"/rule-suggestions/{listed[0]['id']}/accept", headers=admin_headers
        )
        assert accepted.status_code == 200, accepted.text

        rules = client.get(
            "/rules?playbook_id=contract-compliance", headers=admin_headers
        ).json()["rules"]
        assert any(r["name"] == "禁止无限期自动续约" and r["active"] for r in rules)
        assert rules[-1]["name"] == "禁止无限期自动续约"  # 追加到手册末尾

        double = client.post(
            f"/rule-suggestions/{listed[0]['id']}/accept", headers=admin_headers
        )
        assert double.status_code == 422  # 已接受的建议不可再次接受

    def test_dismiss_flow(self, client, admin_headers, fake_llm, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]
        monkeypatch.setattr(
            "backend.engine.suggestions._llm_invoke",
            lambda custom_llm, prompt: SuggestionLLM().invoke(prompt).content,
        )
        client.post(f"/rule-suggestions/generate/{document_id}", headers=admin_headers)
        listed = client.get(
            "/rule-suggestions?playbook_id=contract-compliance&status=proposed",
            headers=admin_headers,
        ).json()["suggestions"]
        dismissed = client.post(
            f"/rule-suggestions/{listed[0]['id']}/dismiss", headers=admin_headers
        )
        assert dismissed.json()["status"] == "dismissed"
        still = client.get(
            "/rules?playbook_id=contract-compliance", headers=admin_headers
        ).json()["rules"]
        assert all(r["name"] != "禁止无限期自动续约" for r in still)

    def test_generate_requires_reviewed_document(self, client, admin_headers, fake_llm, monkeypatch):
        import io

        monkeypatch.setattr(
            "backend.engine.suggestions._llm_invoke",
            lambda custom_llm, prompt: SuggestionLLM().invoke(prompt).content,
        )
        # 上传即失败（空文件）——没有可分析的已完成文档
        bad = client.post(
            "/documents/upload",
            files={"file": ("x.txt", b"", "text/plain")},
            data={"playbook_id": "contract-compliance"},
            headers=admin_headers,
        )
        assert bad.status_code == 400
        response = client.post(
            f"/rule-suggestions/generate/does-not-exist", headers=admin_headers
        )
        assert response.status_code == 404


class TestSuggestionAuth:
    def test_generate_requires_admin(self, client, admin_headers, platform_env, monkeypatch):
        password_file = platform_env["db_path"].parent / "bootstrap_admin_password.txt"
        password = ""
        for line in password_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("password:"):
                password = line.removeprefix("password:").strip()
        admin_token = client.post(
            "/auth/login", json={"username": "admin", "password": password}
        ).json()["token"]
        client.post(
            "/auth/users",
            json={"username": "expert7", "password": "expert-pass-123", "role": "expert"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        expert_token = client.post(
            "/auth/login", json={"username": "expert7", "password": "expert-pass-123"}
        ).json()["token"]
        response = client.post(
            "/rule-suggestions/generate/whatever",
            headers={"Authorization": f"Bearer {expert_token}"},
        )
        assert response.status_code == 403

    def test_list_requires_auth(self, client):
        assert client.get("/rule-suggestions").status_code == 401


class TestSuggestionNotification:
    def test_suggestion_creates_notification(self, client, admin_headers, fake_llm, monkeypatch):
        detail = _reviewed_document(client, admin_headers, monkeypatch)
        document_id = detail["document"]["id"]
        monkeypatch.setattr(
            "backend.engine.suggestions._llm_invoke",
            lambda custom_llm, prompt: SuggestionLLM().invoke(prompt).content,
        )
        client.post(f"/rule-suggestions/generate/{document_id}", headers=admin_headers)
        notifications = client.get("/notifications", headers=admin_headers).json()["notifications"]
        assert any(n["type"] == "rule_suggested" for n in notifications)
