"""Rules CRUD API tests."""

from __future__ import annotations

import pytest

from test_engine_pipeline import (  # noqa: F401
    admin_headers,
    client,
    fake_llm,
    platform_env,
)


def test_list_seeded_rules(client, admin_headers):
    response = client.get(
        "/rules?playbook_id=contract-compliance", headers=admin_headers
    )
    rules = response.json()["rules"]
    assert len(rules) == 15
    assert rules[0]["name"] == "付款账期"
    assert rules[0]["active"] is True


def test_create_and_update_rule(client, admin_headers):
    created = client.post(
        "/rules",
        json={
            "playbook_id": "contract-compliance",
            "dimension": "商业",
            "name": "禁止无限期自动续约",
            "guidance": "存在无限期自动续约条款为红；续约需双方确认",
        },
        headers=admin_headers,
    )
    assert created.status_code == 201, created.text
    rule = created.json()["rule"]

    updated = client.patch(
        f"/rules/{rule['id']}",
        json={"active": False, "weight": 2},
        headers=admin_headers,
    )
    assert updated.status_code == 200
    assert updated.json()["rule"]["active"] is False
    assert updated.json()["rule"]["weight"] == 2

    # the updated rulebook feeds the next document immediately
    listed = client.get(
        "/rules?playbook_id=contract-compliance", headers=admin_headers
    ).json()["rules"]
    assert any(r["name"] == "禁止无限期自动续约" and r["active"] is False for r in listed)


def test_expert_cannot_edit_rulebook(client, admin_headers, platform_env):
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
        json={"username": "expert5", "password": "expert-pass-123", "role": "expert"},
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    expert_token = client.post(
        "/auth/login", json={"username": "expert5", "password": "expert-pass-123"}
    ).json()["token"]
    denied = client.post(
        "/rules",
        json={
            "playbook_id": "contract-compliance",
            "dimension": "商业",
            "name": "x",
            "guidance": "x",
        },
        headers={"Authorization": f"Bearer {expert_token}"},
    )
    assert denied.status_code == 403


def test_unknown_playbook_rejected(client, admin_headers):
    response = client.post(
        "/rules",
        json={"playbook_id": "nope", "dimension": "x", "name": "x", "guidance": "x"},
        headers=admin_headers,
    )
    assert response.status_code == 404
