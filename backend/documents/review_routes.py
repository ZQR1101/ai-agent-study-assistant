"""Review & governance endpoints: queue, decisions, finalization, QA, export.

Split from routes.py to keep each module focused; shares the /documents
prefix and helper functions with the main router.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy import case
from sqlalchemy.orm import Session

from backend.auth.dependencies import require_reviewer, require_user
from backend.auth.models import User
from backend.documents.audit import record_audit
from backend.documents.models import Clause, Document, ReviewEvent, Rule, Verdict
from backend.documents.review import (
    IllegalTransition,
    apply_decision,
    finalize_document,
    pending_review_count,
)
from backend.documents.routes import get_document
from backend.engine.export import render_deliverable
from backend.platform_db import platform_session

router = APIRouter(prefix="/documents", tags=["Document Review"])


class VerdictDecisionRequest(BaseModel):
    decision: str
    expert_note: str | None = Field(default=None, max_length=2000)
    new_rating: str | None = None
    rationale: str | None = Field(default=None, max_length=4000)


@router.get("/review/queue")
def review_queue(user: User = Depends(require_reviewer)) -> dict:
    """Cross-document approval queue, red ratings first."""

    session: Session = platform_session()
    try:
        rows = (
            session.query(Verdict, Document, Rule)
            .join(Document, Verdict.document_id == Document.id)
            .outerjoin(Rule, Verdict.rule_id == Rule.id)
            .filter(
                Verdict.review_state == "awaiting_review",
                Document.status == "awaiting_review",
            )
            .order_by(
                case(
                    (Verdict.rating == "red", 0),
                    (Verdict.rating == "amber", 1),
                    else_=2,
                ),
                Document.friendly_id,
            )
            .limit(500)
            .all()
        )
        return {
            "items": [
                {
                    "verdict_id": verdict.id,
                    "document_id": document.id,
                    "friendly_id": document.friendly_id,
                    "document_title": document.title,
                    "playbook_id": document.playbook_id,
                    "rule_name": rule.name if rule else verdict.rule_id,
                    "dimension": rule.dimension if rule else "",
                    "rating": verdict.rating,
                    "rationale": verdict.rationale,
                    "citations": verdict.citations,
                    "gap_reason": verdict.gap_reason,
                    "weight": rule.weight if rule else 1,
                }
                for verdict, document, rule in rows
            ]
        }
    finally:
        session.close()


@router.patch("/{document_id}/verdicts/{verdict_id}")
def decide_verdict(
    document_id: str,
    verdict_id: str,
    body: VerdictDecisionRequest,
    user: User = Depends(require_reviewer),
) -> dict:
    session: Session = platform_session()
    try:
        document = get_document(session, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        if document.status != "awaiting_review":
            raise HTTPException(status_code=409, detail=f"文档当前状态不可评审: {document.status}")
        verdict = session.get(Verdict, verdict_id)
        if verdict is None or verdict.document_id != document.id:
            raise HTTPException(status_code=404, detail="判定不存在")
        try:
            result = apply_decision(
                session,
                document=document,
                verdict=verdict,
                decision=body.decision,
                actor=user.username,
                expert_note=body.expert_note,
                new_rating=body.new_rating,
                rationale=body.rationale,
            )
        except IllegalTransition as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        return {
            "verdict": {
                "id": result.verdict.id,
                "rating": result.verdict.rating,
                "review_state": result.verdict.review_state,
                "expert_note": result.verdict.expert_note,
                "reviewed_by": result.verdict.reviewed_by,
            },
            "pending_review": pending_review_count(session, document),
        }
    finally:
        session.close()


@router.post("/{document_id}/finalize")
def finalize(document_id: str, user: User = Depends(require_reviewer)) -> dict:
    session: Session = platform_session()
    try:
        document = get_document(session, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        try:
            finalize_document(session, document=document, actor=user.username)
        except IllegalTransition as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return {"document_id": document.id, "status": document.status}
    finally:
        session.close()


class QuestionRequest(BaseModel):
    question: str = Field(min_length=2, max_length=500)


@router.post("/{document_id}/ask")
def ask_document(
    document_id: str,
    body: QuestionRequest,
    user: User = Depends(require_user),
) -> dict:
    """Q&A over one document; answers cite clause numbers, refuse when uncovered."""

    session: Session = platform_session()
    try:
        document = get_document(session, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        clauses = (
            session.query(Clause)
            .filter(Clause.document_id == document.id)
            .order_by(Clause.ordinal)
            .all()
        )
        from backend.engine.parsing import ParsedClause
        from backend.engine.qa import answer_question
        from backend.engine.retrieval import select_clauses

        parsed_clauses = [
            ParsedClause(ordinal=c.ordinal, heading=c.heading, text=c.text) for c in clauses
        ]
        selected = select_clauses(parsed_clauses, body.question, char_budget=8000)
        try:
            answer = answer_question(body.question, selected)
        except Exception:
            raise HTTPException(status_code=502, detail="问答模型调用失败")
        record_audit(
            session,
            correlation_id=document.friendly_id,
            event="document.qa",
            actor=user.username,
            payload={"question": body.question},
        )
        session.commit()
        return {
            "answer": answer["answer"],
            "citations": [
                {"ordinal": c.ordinal, "heading": c.heading, "quote": c.text[:200]}
                for c in selected
            ],
        }
    finally:
        session.close()


@router.get("/{document_id}/review-events")
def list_review_events(document_id: str, _: User = Depends(require_user)) -> dict:
    session: Session = platform_session()
    try:
        document = get_document(session, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        events = (
            session.query(ReviewEvent)
            .filter(ReviewEvent.document_id == document.id)
            .order_by(ReviewEvent.created_at.desc())
            .limit(200)
            .all()
        )
        return {
            "events": [
                {
                    "id": event.id,
                    "actor": event.actor,
                    "action": event.action,
                    "detail": event.detail,
                    "created_at": event.created_at.isoformat() if event.created_at else None,
                }
                for event in events
            ]
        }
    finally:
        session.close()


@router.get("/{document_id}/export")
def export_document(
    document_id: str,
    format: str = Query(..., pattern="^(docx|xlsx)$"),
    user: User = Depends(require_reviewer),
) -> Response:
    """Deliverable download. Finalization gate: only signed-off documents export."""

    session: Session = platform_session()
    try:
        document = get_document(session, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        if document.status != "finalized":
            raise HTTPException(
                status_code=409,
                detail="文档尚未定稿：红/黄判定需专家签字后才能导出",
            )
        from backend.playbooks import get_playbook

        playbook = get_playbook(document.playbook_id)
        if format == "xlsx":
            key = "scorecard_xlsx"
        elif "handover_docx" in playbook.deliverables:
            key = "handover_docx"
        else:
            key = "compliance_report_docx"
        try:
            content = render_deliverable(session, document, key)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        filename = f"{document.friendly_id}_{key}"
        media = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            if format == "xlsx"
            else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        return Response(
            content=content,
            media_type=media,
            headers={"Content-Disposition": f'attachment; filename="{filename}.{format}"'},
        )
    finally:
        session.close()
