"""Persistence helpers for LLM-as-Judge evaluation results."""

from __future__ import annotations

import json

from sqlalchemy.orm import Session

from backend.db_models import JudgeEvaluation


def _evaluation_to_dict(record: JudgeEvaluation) -> dict:
    try:
        deductions = json.loads(record.deductions_json or "[]")
    except json.JSONDecodeError:
        deductions = []

    return {
        "id": record.id,
        "run_id": record.run_id,
        "session_id": record.session_id,
        "question": record.question,
        "answer": record.answer,
        "judge_model": record.judge_model,
        "accuracy": record.accuracy,
        "completeness": record.completeness,
        "citation_quality": record.citation_quality,
        "overall_score": record.overall_score,
        "verdict": record.verdict,
        "deductions": deductions if isinstance(deductions, list) else [],
        "feedback": record.feedback,
        "raw_output": record.raw_output,
        "judge_feedback": record.judge_feedback,
        "judge_feedback_reason": record.judge_feedback_reason,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


def save_judge_result(
    db: Session,
    *,
    session_id: str | None,
    question: str,
    answer: str,
    evaluation: dict,
    run_id: str | None = None,
) -> dict:
    """Save one judge evaluation and return its serialized representation."""
    record = JudgeEvaluation(
        run_id=run_id,
        session_id=session_id,
        question=question,
        answer=answer,
        judge_model=evaluation.get("judge_model"),
        accuracy=float(evaluation.get("accuracy", 0)),
        completeness=float(evaluation.get("completeness", 0)),
        citation_quality=(
            None
            if evaluation.get("citation_quality") is None
            else float(evaluation.get("citation_quality", 0))
        ),
        overall_score=float(evaluation.get("overall_score", 0)),
        verdict=evaluation.get("verdict"),
        deductions_json=json.dumps(evaluation.get("deductions") or [], ensure_ascii=False),
        feedback=evaluation.get("feedback"),
        raw_output=evaluation.get("raw_output"),
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return _evaluation_to_dict(record)


def list_recent_judge_results(db: Session, limit: int = 20) -> list[dict]:
    """Return recent judge results, newest first."""
    records = (
        db.query(JudgeEvaluation)
        .order_by(JudgeEvaluation.created_at.desc(), JudgeEvaluation.id.desc())
        .limit(limit)
        .all()
    )
    return [_evaluation_to_dict(record) for record in records]


def update_judge_feedback(
    db: Session,
    *,
    result_id: int,
    judge_feedback: str,
    reason: str | None = None,
) -> dict | None:
    """Save user feedback about whether a judge result was reasonable."""
    record = db.get(JudgeEvaluation, result_id)
    if record is None:
        return None

    record.judge_feedback = judge_feedback
    record.judge_feedback_reason = reason
    db.commit()
    db.refresh(record)
    return _evaluation_to_dict(record)


save_judge_evaluation = save_judge_result
list_recent_judge_evaluations = list_recent_judge_results
