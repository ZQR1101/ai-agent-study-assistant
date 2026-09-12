"""Rulebook CRUD endpoints. Rules are data: admins edit them here and the next
document scores against the updated rulebook — no code change, no redeploy."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.dependencies import require_admin, require_user
from backend.auth.models import User
from backend.documents.audit import record_audit
from backend.documents.models import Rule
from backend.playbooks import get_playbook
from backend.platform_db import platform_session

router = APIRouter(prefix="/rules", tags=["Rulebook"])


class RuleCreateRequest(BaseModel):
    playbook_id: str
    dimension: str = Field(min_length=1, max_length=50)
    name: str = Field(min_length=1, max_length=200)
    guidance: str = Field(min_length=1, max_length=4000)
    weight: int = Field(default=1, ge=1, le=5)


class RuleUpdateRequest(BaseModel):
    dimension: str | None = Field(default=None, min_length=1, max_length=50)
    name: str | None = Field(default=None, min_length=1, max_length=200)
    guidance: str | None = Field(default=None, min_length=1, max_length=4000)
    weight: int | None = Field(default=None, ge=1, le=5)
    active: bool | None = None


def _payload(rule: Rule) -> dict:
    return {
        "id": rule.id,
        "playbook_id": rule.playbook_id,
        "dimension": rule.dimension,
        "name": rule.name,
        "guidance": rule.guidance,
        "weight": rule.weight,
        "active": rule.active,
        "ordinal": rule.ordinal,
    }


@router.get("")
def list_rules(playbook_id: str, _: User = Depends(require_user)) -> dict:
    session: Session = platform_session()
    try:
        rules = (
            session.query(Rule)
            .filter(Rule.playbook_id == playbook_id)
            .order_by(Rule.ordinal, Rule.created_at)
            .all()
        )
        return {"rules": [_payload(r) for r in rules]}
    finally:
        session.close()


@router.post("", status_code=201)
def create_rule(body: RuleCreateRequest, admin: User = Depends(require_admin)) -> dict:
    try:
        get_playbook(body.playbook_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"未知剧本: {body.playbook_id}")
    session: Session = platform_session()
    try:
        max_ordinal = (
            session.query(Rule)
            .filter(Rule.playbook_id == body.playbook_id)
            .order_by(Rule.ordinal.desc())
            .first()
        )
        rule = Rule(
            id=str(uuid.uuid4()),
            playbook_id=body.playbook_id,
            dimension=body.dimension,
            name=body.name,
            guidance=body.guidance,
            weight=body.weight,
            active=True,
            ordinal=(max_ordinal.ordinal + 1) if max_ordinal else 1,
        )
        session.add(rule)
        record_audit(
            session,
            correlation_id="rulebook",
            event="rule.created",
            actor=admin.username,
            payload={"playbook": body.playbook_id, "rule": body.name},
        )
        session.commit()
        return {"rule": _payload(rule)}
    finally:
        session.close()


@router.patch("/{rule_id}")
def update_rule(
    rule_id: str, body: RuleUpdateRequest, admin: User = Depends(require_admin)
) -> dict:
    session: Session = platform_session()
    try:
        rule = session.get(Rule, rule_id)
        if rule is None:
            raise HTTPException(status_code=404, detail="规则不存在")
        changes = body.model_dump(exclude_none=True)
        if "weight" in changes and changes["weight"] == rule.weight:
            changes.pop("weight")
        if not changes:
            raise HTTPException(status_code=422, detail="没有需要更新的字段")
        for field, value in changes.items():
            setattr(rule, field, value)
        record_audit(
            session,
            correlation_id="rulebook",
            event="rule.updated",
            actor=admin.username,
            payload={"rule_id": rule.id, "name": rule.name, "changes": changes},
        )
        session.commit()
        return {"rule": _payload(rule)}
    finally:
        session.close()
