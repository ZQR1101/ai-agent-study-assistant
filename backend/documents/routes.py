"""Documents API: upload, list, detail, processing, audit."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from pydantic import BaseModel, Field
from sqlalchemy import case
from sqlalchemy.orm import Session, joinedload

from backend.auth.dependencies import require_reviewer, require_user
from backend.auth.models import User
from backend.documents.audit import list_audit, record_audit
from backend.documents.models import (
    Clause,
    Document,
    EngineRun,
    ReviewEvent,
    Rule,
    Verdict,
)
from backend.documents.review import (
    IllegalTransition,
    apply_decision,
    finalize_document,
    pending_review_count,
)
from backend.documents.service import create_document, get_document
from backend.engine.export import render_deliverable
from backend.engine.orchestrator import ReviewConflict, run_review
from backend.playbooks import list_playbooks, playbook_summary
from backend.platform_db import platform_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

UPLOAD_MAX_BYTES = 20 * 1024 * 1024
ALLOWED_SUFFIXES = {".pdf", ".md", ".txt", ".docx"}
ALLOWED_CONTENT_TYPES = {
    ".pdf": {"application/pdf"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
    ".txt": {"text/plain", "application/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/octet-stream",
    },
}


def _set_status(document: Document, status: str, reason: str | None = None) -> None:
    document.status = status
    document.status_reason = reason


@router.get("/playbooks")
def list_playbook_summaries(_: User = Depends(require_user)) -> dict:
    return {"playbooks": [playbook_summary(spec) for spec in list_playbooks()]}


def _validate_upload(filename: str, content_type: str | None, content: bytes) -> str:
    if not filename or filename != Path(filename).name or len(filename) > 128:
        raise HTTPException(status_code=400, detail="无效的文件名")
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"不支持的文件格式: {suffix}")
    allowed = ALLOWED_CONTENT_TYPES[suffix]
    if content_type and content_type.split(";", 1)[0].strip().lower() not in allowed:
        raise HTTPException(status_code=415, detail=f"不允许的 Content-Type: {content_type}")
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="文件内容为空")
    if len(content) > UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail="文件超过 20MB 上限")
    return suffix


def _require_playbook(playbook_id: str) -> None:
    from fastapi import HTTPException as _HTTPException

    from backend.playbooks import get_playbook

    try:
        get_playbook(playbook_id)
    except KeyError:
        raise _HTTPException(status_code=404, detail=f"未知剧本: {playbook_id}")


@router.post("/upload", status_code=202)
def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    playbook_id: str = Form(...),
    user: User = Depends(require_reviewer),
) -> dict:
    content = file.file.read()
    _validate_upload(file.filename or "", file.content_type, content)
    _require_playbook(playbook_id)

    session: Session = platform_session()
    try:
        document, created = create_document(
            session,
            playbook_id=playbook_id,
            filename=file.filename or "",
            content=content,
            actor=user.username,
        )
        if not created:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "相同内容的文档已存在（内容哈希去重）",
                    "document_id": document.id,
                    "friendly_id": document.friendly_id,
                },
            )
        friendly_id = document.friendly_id
        doc_id = document.id
        status = document.status
    finally:
        session.close()

    background_tasks.add_task(_safe_run_review, doc_id, trigger="upload", actor=user.username)
    return {
        "document_id": doc_id,
        "friendly_id": friendly_id,
        "status": status,
        "processing": True,
    }


def _safe_run_review(document_id: str, *, trigger: str, actor: str) -> None:
    """Background wrapper: never let the pipeline crash the worker silently."""

    try:
        run_review(document_id, trigger=trigger)
    except ReviewConflict as exc:
        logger.info("review skipped for %s: %s", document_id, exc)
    except Exception:
        logger.exception("background review failed for %s", document_id)


@router.get("/audit/recent")
def recent_audit(limit: int = 100, _: User = Depends(require_user)) -> dict:
    session: Session = platform_session()
    try:
        entries = list_audit(session, limit=min(max(limit, 1), 500))
        return {
            "audit": [
                {
                    "id": entry.id,
                    "correlation_id": entry.correlation_id,
                    "event": entry.event,
                    "actor": entry.actor,
                    "payload": entry.payload,
                    "created_at": entry.created_at.isoformat() if entry.created_at else None,
                }
                for entry in entries
            ]
        }
    finally:
        session.close()


class ProcessRequest(BaseModel):
    force: bool = False


@router.post("/{document_id}/process", status_code=202)
def process_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    body: ProcessRequest | None = None,
    user: User = Depends(require_reviewer),
) -> dict:
    del body  # accepted for API stability
    session: Session = platform_session()
    try:
        document = get_document(session, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        if document.status in ("parsing", "scoring"):
            raise HTTPException(status_code=409, detail="文档正在处理中")
        if document.status in ("awaiting_review", "finalized") :
            raise HTTPException(status_code=409, detail="文档已有评审结果")
        record_audit(
            session,
            correlation_id=document.friendly_id,
            event="document.process_requested",
            actor=user.username,
        )
        session.commit()
        doc_id = document.id
    finally:
        session.close()

    background_tasks.add_task(_safe_run_review, doc_id, trigger="manual", actor=user.username)
    return {"document_id": doc_id, "processing": True}


def _verdict_payload(verdict: Verdict, rule: Rule) -> dict:
    return {
        "id": verdict.id,
        "rule_id": verdict.rule_id,
        "dimension": rule.dimension,
        "rule_name": rule.name,
        "rule_guidance": rule.guidance,
        "weight": rule.weight,
        "rating": verdict.rating,
        "rationale": verdict.rationale,
        "citations": verdict.citations,
        "gap_reason": verdict.gap_reason,
        "review_state": verdict.review_state,
        "expert_note": verdict.expert_note,
        "reviewed_by": verdict.reviewed_by,
        "reviewed_at": verdict.reviewed_at.isoformat() if verdict.reviewed_at else None,
    }


def _document_payload(document: Document) -> dict:
    return {
        "id": document.id,
        "friendly_id": document.friendly_id,
        "playbook_id": document.playbook_id,
        "title": document.title,
        "source_filename": document.source_filename,
        "status": document.status,
        "status_reason": document.status_reason,
        "scorecard": document.scorecard,
        "created_by": document.created_by,
        "created_at": document.created_at.isoformat() if document.created_at else None,
        "updated_at": document.updated_at.isoformat() if document.updated_at else None,
    }


@router.get("")
def list_documents(user: User = Depends(require_user), status: str | None = None) -> dict:
    session: Session = platform_session()
    try:
        query = session.query(Document).order_by(Document.created_at.desc())
        if status:
            query = query.filter(Document.status == status)
        documents = query.limit(200).all()
        return {"documents": [_document_payload(d) for d in documents]}
    finally:
        session.close()


@router.get("/{document_id}")
def get_document_detail(document_id: str, _: User = Depends(require_user)) -> dict:
    session: Session = platform_session()
    try:
        document = get_document(session, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        rules = {rule.id: rule for rule in session.query(Rule).filter(Rule.playbook_id == document.playbook_id).all()}
        verdicts = (
            session.query(Verdict)
            .options(joinedload(Verdict.rule))
            .filter(Verdict.document_id == document.id)
            .all()
        )
        clauses = session.query(Clause).filter(Clause.document_id == document.id).order_by(Clause.ordinal).all()
        return {
            "document": _document_payload(document),
            "verdicts": [
                _verdict_payload(v, rules.get(v.rule_id) or v.rule)
                for v in verdicts
            ],
            "clauses": [
                {"id": c.id, "ordinal": c.ordinal, "heading": c.heading, "text": c.text}
                for c in clauses
            ],
        }
    finally:
        session.close()


@router.get("/{document_id}/runs")
def list_document_runs(document_id: str, _: User = Depends(require_user)) -> dict:
    session: Session = platform_session()
    try:
        document = get_document(session, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        runs = (
            session.query(EngineRun)
            .filter(EngineRun.document_id == document.id)
            .order_by(EngineRun.started_at.desc())
            .all()
        )
        return {
            "runs": [
                {
                    "id": run.id,
                    "status": run.status,
                    "trigger": run.trigger,
                    "stages": run.stages,
                    "total_latency_ms": run.total_latency_ms,
                    "error": run.error,
                    "started_at": run.started_at.isoformat() if run.started_at else None,
                    "finished_at": run.finished_at.isoformat() if run.finished_at else None,
                }
                for run in runs
            ]
        }
    finally:
        session.close()


@router.get("/{document_id}/audit")
def get_document_audit(document_id: str, _: User = Depends(require_user)) -> dict:
    session: Session = platform_session()
    try:
        document = get_document(session, document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="文档不存在")
        entries = list_audit(session, correlation_id=document.friendly_id)
        return {
            "audit": [
                {
                    "id": entry.id,
                    "event": entry.event,
                    "actor": entry.actor,
                    "payload": entry.payload,
                    "created_at": entry.created_at.isoformat() if entry.created_at else None,
                }
                for entry in entries
            ]
        }
    finally:
        session.close()
