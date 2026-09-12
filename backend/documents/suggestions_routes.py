"""Rule suggestion API: propose (admin triggers generation), review, accept/dismiss."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.auth.dependencies import require_admin, require_user
from backend.auth.models import User
from backend.documents.audit import record_audit
from backend.documents.models import Document, RuleSuggestion
from backend.documents.service import get_document
from backend.engine.suggestions import suggest_rules_for_document
from backend.platform_db import platform_session

router = APIRouter(prefix="/rule-suggestions", tags=["Rule Suggestions"])


def _payload(suggestion: RuleSuggestion) -> dict:
    return {
        "id": suggestion.id,
        "playbook_id": suggestion.playbook_id,
        "dimension": suggestion.dimension,
        "name": suggestion.name,
        "guidance": suggestion.guidance,
        "weight": suggestion.weight,
        "rationale": suggestion.rationale,
        "source_document_id": suggestion.source_document_id,
        "status": suggestion.status,
        "decided_by": suggestion.decided_by,
        "decided_at": suggestion.decided_at.isoformat() if suggestion.decided_at else None,
        "created_at": suggestion.created_at.isoformat() if suggestion.created_at else None,
    }


class GenerateRequest(BaseModel):
    pass


@router.post("/generate/{document_id}", status_code=202)
def generate(document_id: str, _: GenerateRequest | None = None, admin: User = Depends(require_admin)) -> dict:
    session: Session = platform_session()
    try:
        document = get_document(session, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        if document.status not in ("awaiting_review", "finalized"):
            raise HTTPException(status_code=409, detail="文档尚未完成评审，无法分析规则缺口")
        from backend.engine.suggestions import suggest_rules_for_document

        result = suggest_rules_for_document(session, document=document, actor=admin.username)
        return {"created": result["created"], "suggestion_ids": result["suggestions"]}
    finally:
        session.close()


@router.get("")
def list_suggestions(
    playbook_id: str | None = None,
    status: str | None = None,
    _: User = Depends(require_user),
) -> dict:
    session: Session = platform_session()
    try:
        query = session.query(RuleSuggestion).order_by(RuleSuggestion.created_at.desc())
        if playbook_id:
            query = query.filter(RuleSuggestion.playbook_id == playbook_id)
        if status:
            query = query.filter(RuleSuggestion.status == status)
        return {"suggestions": [_payload(s) for s in query.limit(200).all()]}
    finally:
        session.close()


@router.post("/{suggestion_id}/accept")
def accept(suggestion_id: str, admin: User = Depends(require_admin)) -> dict:
    session: Session = platform_session()
    try:
        suggestion = session.get(RuleSuggestion, suggestion_id)
        if suggestion is None:
            raise HTTPException(status_code=404, detail="建议不存在")
        from backend.engine.suggestions import accept_suggestion

        try:
            rule = accept_suggestion(session, suggestion, actor=admin.username)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {"suggestion_id": suggestion.id, "rule_id": rule.id, "rule_name": rule.name}
    finally:
        session.close()


@router.post("/{suggestion_id}/dismiss")
def dismiss(suggestion_id: str, admin: User = Depends(require_admin)) -> dict:
    session: Session = platform_session()
    try:
        suggestion = session.get(RuleSuggestion, suggestion_id)
        if suggestion is None:
            raise HTTPException(status_code=404, detail="建议不存在")
        if suggestion.status != "proposed":
            raise HTTPException(status_code=422, detail="仅待确认的建议可驳回")
        suggestion.status = "dismissed"
        suggestion.decided_by = admin.username
        suggestion.decided_at = datetime.now(timezone.utc)
        record_audit(
            session,
            correlation_id="rulebook",
            event="rule.suggestion_dismissed",
            actor=admin.username,
            payload={"suggestion_id": suggestion.id, "rule": suggestion.name},
        )
        session.commit()
        return {"suggestion_id": suggestion.id, "status": suggestion.status}
    finally:
        session.close()
