"""Platform foundation tests: auth, seed data, playbook registry.

These tests run against a temp platform DB and never touch the LLM or RAG.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def platform_env(tmp_path, monkeypatch):
    """Isolated platform DB per test."""
    db_path = tmp_path / "platform" / "rulebook.db"
    monkeypatch.setenv("PLATFORM_DB_PATH", str(db_path))
    monkeypatch.setenv("AUTH_SECRET", "test-secret-for-platform-tests-0123456789abcdef")
    from backend.platform_db import init_platform_db, reset_platform_db

    reset_platform_db()
    init_platform_db()
    yield {"db_path": db_path}
    reset_platform_db()


@pytest.fixture()
def client(platform_env):
    from backend.server import app

    with TestClient(app) as test_client:
        yield test_client


def _read_bootstrap_password(platform_env) -> str:
    pwd_file = platform_env["db_path"].parent / "bootstrap_admin_password.txt"
    for line in pwd_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("password:"):
            return line.removeprefix("password:").strip()
    raise AssertionError("bootstrap password file missing")


class TestAuthFlow:
    def test_login_with_bootstrap_admin(self, client, platform_env):
        password = _read_bootstrap_password(platform_env)
        response = client.post("/auth/login", json={"username": "admin", "password": password})
        assert response.status_code == 200
        body = response.json()
        assert body["user"]["role"] == "admin"
        assert "token" in body
        assert "rulebook_session" in response.cookies

    def test_login_rejects_wrong_password(self, client):
        response = client.post("/auth/login", json={"username": "admin", "password": "wrong"})
        assert response.status_code == 401

    def test_me_requires_token(self, client):
        assert client.get("/auth/me").json() == {"user": None}

    def test_me_with_token(self, client, platform_env):
        password = _read_bootstrap_password(platform_env)
        login = client.post("/auth/login", json={"username": "admin", "password": password})
        token = login.json()["token"]
        response = client.get(
            "/auth/me", headers={"Authorization": f"Bearer {token}"}
        )
        assert response.status_code == 200
        assert response.json()["user"]["username"] == "admin"

    def test_admin_only_endpoint_forbidden_for_expert(self, client, platform_env):
        password = _read_bootstrap_password(platform_env)
        admin_token = client.post(
            "/auth/login", json={"username": "admin", "password": password}
        ).json()["token"]
        created = client.post(
            "/auth/users",
            json={"username": "expert1", "password": "expert-pass-123", "role": "expert"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert created.status_code == 201

        expert_token = client.post(
            "/auth/login", json={"username": "expert1", "password": "expert-pass-123"}
        ).json()["token"]

        forbidden = client.get(
            "/auth/users", headers={"Authorization": f"Bearer {expert_token}"}
        )
        assert forbidden.status_code == 403

        allowed = client.get(
            "/auth/users", headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert allowed.status_code == 200

    def test_duplicate_username_rejected(self, client, platform_env):
        password = _read_bootstrap_password(platform_env)
        admin_token = client.post(
            "/auth/login", json={"username": "admin", "password": password}
        ).json()["token"]
        payload = {"username": "dup", "password": "password-123"}
        assert client.post("/auth/users", json=payload, headers={"Authorization": f"Bearer {admin_token}"}).status_code == 201
        assert client.post("/auth/users", json=payload, headers={"Authorization": f"Bearer {admin_token}"}).status_code == 409


class TestSeedData:
    def test_both_playbooks_seeded(self, platform_env):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.documents.models import Rule

        engine = create_engine(f"sqlite:///{platform_env['db_path']}")
        session = sessionmaker(bind=engine)()
        try:
            rules = session.query(Rule).all()
            by_playbook = {}
            for rule in rules:
                by_playbook.setdefault(rule.playbook_id, []).append(rule)
            assert set(by_playbook) == {"contract-compliance", "delivery-intake", "dpa-review"}
            assert len(by_playbook["contract-compliance"]) == 15
            assert len(by_playbook["delivery-intake"]) == 12
            assert len(by_playbook["dpa-review"]) == 11
        finally:
            session.close()
            engine.dispose()

    def test_seeding_is_idempotent(self, platform_env):
        from backend.platform_db import init_platform_db
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from backend.documents.models import Rule

        init_platform_db()  # second run must not duplicate
        engine = create_engine(f"sqlite:///{platform_env['db_path']}")
        session = sessionmaker(bind=engine)()
        try:
            assert session.query(Rule).count() == 38
        finally:
            session.close()
            engine.dispose()


class TestPlaybookRegistry:
    def test_specs_validate(self):
        from backend.playbooks import list_playbooks

        for spec in list_playbooks():
            spec.validate()

    def test_register_duplicate_rejected(self):
        from backend.playbooks import get_playbook, register_playbook

        spec = get_playbook("contract-compliance")
        with pytest.raises(ValueError):
            register_playbook(spec)

    def test_register_toy_third_playbook(self):
        from backend.playbooks import list_playbook_ids, register_playbook
        from backend.playbooks.base import PlaybookSpec

        spec = PlaybookSpec(
            id="toy",
            name="玩具剧本",
            description="用于测试",
            friendly_id_prefix="TY",
            dimensions=("甲", "乙"),
            rule_seeds=(
                {"dimension": "甲", "name": "规则一", "guidance": "g"},
                {"dimension": "乙", "name": "规则二", "guidance": "g"},
            ),
        )
        register_playbook(spec)
        try:
            assert "toy" in list_playbook_ids()
        finally:
            from backend.playbooks import registry

            registry._PLAYBOOKS.pop("toy", None)

    def test_invalid_dimension_rejected(self):
        from backend.playbooks.base import PlaybookSpec

        with pytest.raises(ValueError):
            PlaybookSpec(
                id="bad",
                name="bad",
                description="",
                friendly_id_prefix="BD",
                dimensions=("甲",),
                rule_seeds=({"dimension": "乙", "name": "r", "guidance": "g"},),
            ).validate()
