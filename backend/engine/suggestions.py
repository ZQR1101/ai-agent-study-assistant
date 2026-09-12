"""Rule suggestion agent — the first generative position in the engine.

Reads one reviewed document (its red/gap verdicts and clause excerpts) and
drafts rulebook additions the handbook does not yet cover. Output is still a
constrained JSON contract (rule fields only); drafts land in
``rule_suggestions`` as ``proposed`` and reach the live rulebook exclusively
through admin acceptance — the generative agent never writes rules directly.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from backend.documents.audit import record_audit
from backend.documents.models import Document, RuleSuggestion, Verdict
from backend.notifications.service import create_notification_and_email

logger = logging.getLogger(__name__)

MAX_SUGGESTIONS = 3
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _llm_invoke(custom_llm, prompt: str) -> str:
    llm = custom_llm
    if llm is None:
        from backend.llm_service import llm as default_llm

        llm = default_llm
    response = llm.invoke(prompt)
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


def _build_prompt(existing_rules: list[str], verdicts: list[Verdict]) -> str:
    gap_lines = []
    for verdict in verdicts:
        tag = "缺口" if verdict.gap_reason else "红灯"
        gap_lines.append(
            f"- [{verdict.rating.upper()}·{tag}] {verdict.rule_id}：{verdict.rationale}"
            + (f"（缺口：{verdict.gap_reason}）" if verdict.gap_reason else "")
        )
    existing_text = "\n".join(f"- {name}" for name in existing_rules) or "（空）"
    return f"""你是规则手册治理代理。当前规则手册包含以下规则：
{existing_text}

以下是某份文档的评审结论（红灯与缺口）：
{chr(10).join(gap_lines) or "（无）"}

请归纳：现有规则手册**未覆盖**、但这份文档暴露出的新检查点。要求：
1. 只提出手册中不存在的规则（不要重述或改写现有规则）；
2. 每条建议必须由上面的评审结论支撑，写明立论依据；
3. 最多 {MAX_SUGGESTIONS} 条；没有值得新增的就返回空数组。

只输出一个 JSON 对象：
{{"suggestions": [{{"dimension": "维度名", "name": "规则名", "guidance": "判定标准（红黄绿分级，引用检查点）", "weight": 1, "rationale": "立论依据"}}]}}"""


def _parse_suggestions(content: str) -> list[dict]:
    match = _JSON_RE.search(content or "")
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    suggestions = data.get("suggestions")
    if not isinstance(suggestions, list):
        return []
    cleaned = []
    for item in suggestions[:MAX_SUGGESTIONS]:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        guidance = str(item.get("guidance") or "").strip()
        dimension = str(item.get("dimension") or "").strip()
        if not name or not guidance or not dimension:
            continue
        cleaned.append(
            {
                "dimension": dimension[:50],
                "name": name[:200],
                "guidance": guidance,
                "weight": max(1, min(5, int(item.get("weight") or 1))),
                "rationale": str(item.get("rationale") or "").strip(),
            }
        )
    return cleaned


def suggest_rules_for_document(
    session: Session,
    document: Document,
    *,
    actor: str,
    custom_llm=None,
) -> dict:
    """Generate and store rule suggestions for one reviewed document."""

    from backend.documents.models import Rule

    existing_names = [
        rule.name for rule in session.query(Rule).filter(Rule.playbook_id == document.playbook_id).all()
    ]
    verdicts = (
        session.query(Verdict)
        .filter(
            Verdict.document_id == document.id,
            Verdict.rating.in_(("red", "amber")),
        )
        .all()
    )

    prompt = _build_prompt(existing_names, verdicts)
    raw = _llm_invoke(custom_llm, prompt)
    drafts = _parse_suggestions(raw)

    existing_lower = {name.lower() for name in existing_names}
    created = []
    for draft in drafts:
        if draft["name"].lower() in existing_lower:
            continue
        duplicate = (
            session.query(RuleSuggestion)
            .filter(
                RuleSuggestion.playbook_id == document.playbook_id,
                RuleSuggestion.name == draft["name"],
                RuleSuggestion.status == "proposed",
            )
            .first()
        )
        if duplicate is not None:
            continue
        suggestion = RuleSuggestion(
            playbook_id=document.playbook_id,
            dimension=draft["dimension"],
            name=draft["name"],
            guidance=draft["guidance"],
            weight=draft["weight"],
            rationale=draft["rationale"],
            source_document_id=document.id,
            status="proposed",
        )
        session.add(suggestion)
        created.append(suggestion)

    session.flush()
    record_audit(
        session,
        correlation_id=document.friendly_id,
        event="rule.suggested",
        actor=actor,
        payload={"count": len(created), "names": [s.name for s in created]},
    )
    if created:
        create_notification_and_email(
            session,
            type="rule_suggested",
            title=f"{len(created)} 条规则建议待确认",
            body=f"{document.friendly_id} 评审后，代理起草了新规则：" + "、".join(s.name for s in created),
            payload={"document_id": document.id, "playbook_id": document.playbook_id},
            correlation_id=document.friendly_id,
        )
    session.commit()
    return {"created": len(created), "suggestions": [s.id for s in created], "drafted": len(drafts)}


def accept_suggestion(session: Session, suggestion: RuleSuggestion, *, actor: str):
    from backend.documents.models import Rule

    if suggestion.status != "proposed":
        raise ValueError(f"建议当前状态不可接受: {suggestion.status}")

    max_ordinal = (
        session.query(Rule)
        .filter(Rule.playbook_id == suggestion.playbook_id)
        .order_by(Rule.ordinal.desc())
        .first()
    )
    rule = Rule(
        playbook_id=suggestion.playbook_id,
        dimension=suggestion.dimension,
        name=suggestion.name,
        guidance=suggestion.guidance,
        weight=suggestion.weight,
        active=True,
        ordinal=(max_ordinal.ordinal + 1) if max_ordinal else 1,
    )
    session.add(rule)
    suggestion.status = "accepted"
    suggestion.decided_by = actor
    suggestion.decided_at = datetime.now(timezone.utc)
    record_audit(
        session,
        correlation_id="rulebook",
        event="rule.suggestion_accepted",
        actor=actor,
        payload={"suggestion_id": suggestion.id, "rule": suggestion.name},
    )
    session.commit()
    return rule
