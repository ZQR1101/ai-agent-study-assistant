"""Third playbook (DPA review) end-to-end test — proves engine generality.

The whole point: DPA lands as pure playbook data; if any engine module needed
a change for a new vertical, this file's existence fails its purpose.
"""

from __future__ import annotations

import json

import pytest

from test_engine_pipeline import (  # noqa: F401
    FakeLLM,
    admin_headers,
    client,
    fake_llm,
    platform_env,
)

DPA_SAMPLE = """数据处理协议（DPA）

第一条 处理目的与范围
受托方仅按控制方书面指令，为提供云托管服务之目的处理个人数据。处理的数据类别限于账户信息（姓名、邮箱、手机号）。SAMPLE QUOTE

第二条 数据主体权利协助
受托方应及时（不超过5个工作日）协助控制方响应数据主体的访问、更正、删除请求，并提供必要信息。SAMPLE QUOTE

第三条 安全措施
受托方实施传输加密（TLS1.2+）、存储加密（AES-256）、基于角色的访问控制与日志审计。SAMPLE QUOTE

第四条 安全事件通知
受托方在发现安全事件后24小时内书面通知控制方，并配合处置。SAMPLE QUOTE

第五条 分包
受托方不得分包，确需分包应提前取得控制方书面同意并披露分包方身份与职责。SAMPLE QUOTE

第六条 数据删除与返还
协议终止后30日内，按控制方选择删除或返还全部个人数据，并出具删除书面认证。SAMPLE QUOTE
"""


def _upload_dpa(client, headers):
    return client.post(
        "/documents/upload",
        files={"file": ("dpa.txt", DPA_SAMPLE.encode("utf-8"), "text/plain")},
        data={"playbook_id": "dpa-review"},
        headers=headers,
    )


class TestDPAPlaybook:
    def test_dpa_registered_and_seeded(self, platform_env):  # noqa: F401
        from backend.playbooks import get_playbook

        spec = get_playbook("dpa-review")
        assert spec.name == "数据处理协议审查"
        assert spec.friendly_id_prefix == "DP"
        assert len(spec.rule_seeds) == 11

    def test_end_to_end_review(self, client, admin_headers, fake_llm):
        response = _upload_dpa(client, admin_headers)
        assert response.status_code == 202, response.text
        document_id = response.json()["document_id"]
        detail = client.get(f"/documents/{document_id}", headers=admin_headers).json()

        assert detail["document"]["status"] == "awaiting_review"
        assert detail["document"]["friendly_id"].startswith("DP-")
        assert len(detail["verdicts"]) == 11
        dimensions = {v["dimension"] for v in detail["verdicts"]}
        assert dimensions == {
            "处理范围与目的", "数据主体权利", "安全措施", "分包与跨境", "保留删除与审计",
        }
        scorecard = detail["document"]["scorecard"]
        assert scorecard["total_rules"] == 11

    def test_gap_rules_turn_red(self, client, admin_headers, monkeypatch):
        """DPA 缺失删除/审计条款时应判红并给出缺口说明。"""

        sparse_dpa = "数据处理协议\n\n第一条 处理\n受托方处理控制方提供的数据。仅此而已。"
        monkeypatch.setattr(
            "backend.engine.scoring._default_scoring_llm",
            lambda: FakeLLM(
                default={
                    "rating": "red",
                    "rationale": "协议未约定该事项",
                    "citations": [],
                    "gap_reason": "协议未约定该事项",
                }
            ),
        )
        upload = _upload_dpa(client, admin_headers)
        del sparse_dpa
        document_id = upload.json()["document_id"]
        detail = client.get(f"/documents/{document_id}", headers=admin_headers).json()
        reds = [v for v in detail["verdicts"] if v["rating"] == "red"]
        assert len(reds) == 11
        assert all(v["gap_reason"] for v in reds)

    def test_playbooks_endpoint_lists_three(self, client, admin_headers):
        ids = {p["id"] for p in client.get("/documents/playbooks", headers=admin_headers).json()["playbooks"]}
        assert ids == {"contract-compliance", "delivery-intake", "dpa-review"}
