"""LLM-as-Judge scoring for assistant answers."""

from __future__ import annotations

import json
import os
from typing import Any

from backend.config import normalize_model


class JudgeEvaluationError(RuntimeError):
    """Raised when the judge model cannot produce a usable score."""


def is_llm_judge_enabled() -> bool:
    """Return True when automatic judge evaluation should run after chat."""
    return os.getenv("ENABLE_LLM_JUDGE", "true").lower() in ("true", "1", "yes")


def is_judge_persistence_enabled() -> bool:
    """Return True when judge results should be persisted to the configured database."""
    value = os.getenv("ENABLE_JUDGE_PERSISTENCE")
    if value is not None:
        return value.lower() in ("true", "1", "yes")
    return bool(os.getenv("DATABASE_URL"))


def _extract_json_object(text: str) -> dict[str, Any] | None:
    content = str(text or "").strip()
    if not content:
        return None

    try:
        parsed = json.loads(content)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    try:
        parsed = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_score(value: Any, fallback: float | None = None) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        if fallback is None:
            raise JudgeEvaluationError("Judge response is missing a numeric score")
        score = fallback
    return max(0.0, min(10.0, score))


def _coerce_optional_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"n/a", "na", "not applicable"}:
        return None
    return _coerce_score(value)


def _json_preview(value: Any, *, limit: int = 4000) -> str:
    if value is None or value == [] or value == {}:
        return "None provided."

    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except TypeError:
        text = str(value)

    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... [truncated]"


def _format_sources(sources: list[dict] | None) -> str:
    if not sources:
        return "No retrieved sources were provided."

    lines = []
    for index, source in enumerate(sources[:6], start=1):
        title = source.get("source") or f"source-{index}"
        snippet = source.get("snippet") or source.get("text") or ""
        score = source.get("score")
        score_text = f" score={score}" if score is not None else ""
        lines.append(f"{index}. {title}{score_text}\n{str(snippet)[:800]}")
    return "\n\n".join(lines)


def compute_verdict(overall_score: float, citation_quality: float | None) -> str:
    """Convert judge scores into a clear pass/fail grade."""
    citation_gate = 10.0 if citation_quality is None else citation_quality
    if overall_score >= 8 and citation_gate >= 7:
        return "PASS"
    if overall_score >= 6:
        return "WEAK_PASS"
    return "FAIL"


def _normalize_deductions(value: Any) -> list[dict]:
    if not isinstance(value, list):
        return []

    deductions: list[dict] = []
    for item in value:
        if isinstance(item, dict):
            metric = str(item.get("metric") or item.get("category") or "Overall").strip()
            reason = str(item.get("reason") or item.get("text") or item.get("description") or "").strip()
            points = item.get("points")
        else:
            metric = "Overall"
            reason = str(item or "").strip()
            points = None

        if not reason:
            continue

        deduction = {"metric": metric or "Overall", "reason": reason}
        try:
            deduction["points"] = round(abs(float(points)), 2)
        except (TypeError, ValueError):
            deduction["points"] = None
        deductions.append(deduction)

    return deductions


def _fallback_deductions(
    *,
    accuracy: float,
    completeness: float,
    citation_quality: float | None,
    overall_score: float,
) -> list[dict]:
    scores = [
        ("准确性", accuracy),
        ("完整性", completeness),
        ("引用质量", citation_quality),
        ("总分", overall_score),
    ]
    deductions = []
    for metric, score in scores:
        if score is None or score >= 9.5:
            continue
        deductions.append(
            {
                "metric": metric,
                "points": round(10 - score, 2),
                "reason": "Judge 给出了低于 10 分的分数，但没有提供具体扣分原因。",
            }
        )
    return deductions


def build_judge_prompt(
    question: str,
    answer: str,
    sources: list[dict] | None = None,
    trace: Any = None,
    runtime_info: dict | None = None,
    tool_calls: list[dict] | None = None,
) -> str:
    """Build the prompt used by the judge model."""
    return f"""
You are a strict evaluator. Do not give 10 unless the answer is excellent and fully supported.
Most normal answers should score between 6 and 8.
Use Simplified Chinese for `feedback` and every `deductions[].reason`.
Use Simplified Chinese metric names in `deductions[].metric`, such as "准确性", "完整性", "引用质量", and "总分".

Evaluate the assistant answer for an AI study assistant. Use the explicit rubric below and explain every deduction.

Accuracy rubric:
- 10 = Completely correct, no obvious errors.
- 8 = Mostly correct, with minor imprecision.
- 6 = Partly wrong or not accurate enough.
- 4 = Key concepts are wrong.
- 2 = Mostly wrong.
- 0 = Completely off-task.

Completeness rubric:
- 10 = Covers all user requirements.
- 8 = Covers most requirements.
- 6 = Missing one important part.
- 4 = Clearly incomplete.
- 2 = Only scattered fragments.
- 0 = Did not complete the task.

Citation Quality rubric:
- 10 = All key conclusions are supported by sources.
- 7 = Most key conclusions are supported by sources.
- 5 = Sources exist but citations are insufficient.
- 3 = RAG was enabled but sources were barely used.
- 0 = Fabricated sources or no source support.
- N/A = The task does not need citations.

Citation Quality must be based on the provided sources and the assistant answer. If RAG/source context exists
but the assistant does not clearly use it for key claims, apply a deduction. If no citation is needed, return null.

Return only valid JSON with this exact shape:
{{
  "accuracy": 0-10,
  "completeness": 0-10,
  "citation_quality": 0-10 or null,
  "overall_score": 0-10,
  "feedback": "one concise Simplified Chinese sentence",
  "deductions": [
    {{"metric": "引用质量", "points": 3, "reason": "已启用 RAG，但回答没有明确引用来源。"}}
  ]
}}

User question:
{question}

Assistant answer:
{answer}

Retrieved sources:
{_format_sources(sources)}

Trace:
{_json_preview(trace)}

Runtime info:
{_json_preview(runtime_info)}

Tool calls:
{_json_preview(tool_calls)}
""".strip()


def judge_answer(
    question: str,
    answer: str,
    *,
    sources: list[dict] | None = None,
    trace: Any = None,
    runtime_info: dict | None = None,
    model: str | None = None,
    judge_llm=None,
) -> dict:
    """Evaluate an assistant answer with an LLM judge."""
    if not str(answer or "").strip():
        raise JudgeEvaluationError("Cannot evaluate an empty answer")

    if judge_llm is None:
        selected_model = normalize_model(os.getenv("JUDGE_MODEL") or model)
        from backend.llm_service import build_llm

        active_llm = build_llm(model=selected_model, temperature=0.0)
    else:
        selected_model = os.getenv("JUDGE_MODEL") or model or "judge"
        active_llm = judge_llm
    tool_calls = []
    if isinstance(runtime_info, dict) and isinstance(runtime_info.get("tool_calls"), list):
        tool_calls = runtime_info.get("tool_calls", [])

    response = active_llm.invoke(
        build_judge_prompt(
            question,
            answer,
            sources=sources,
            trace=trace,
            runtime_info=runtime_info,
            tool_calls=tool_calls,
        )
    )
    raw_output = str(getattr(response, "content", response) or "")
    parsed = _extract_json_object(raw_output)
    if parsed is None:
        raise JudgeEvaluationError("Judge response did not contain valid JSON")

    accuracy = _coerce_score(parsed.get("accuracy"))
    completeness = _coerce_score(parsed.get("completeness"))
    citation_quality = _coerce_optional_score(parsed.get("citation_quality"))
    score_values = [accuracy, completeness]
    if citation_quality is not None:
        score_values.append(citation_quality)
    overall_fallback = round(sum(score_values) / len(score_values), 2)
    overall_score = _coerce_score(parsed.get("overall_score"), fallback=overall_fallback)
    verdict = compute_verdict(overall_score, citation_quality)
    deductions = _normalize_deductions(parsed.get("deductions"))
    if not deductions:
        deductions = _fallback_deductions(
            accuracy=accuracy,
            completeness=completeness,
            citation_quality=citation_quality,
            overall_score=overall_score,
        )

    return {
        "judge_model": selected_model,
        "accuracy": accuracy,
        "completeness": completeness,
        "citation_quality": citation_quality,
        "overall_score": overall_score,
        "verdict": verdict,
        "deductions": deductions,
        "feedback": str(parsed.get("feedback") or "").strip() or None,
        "raw_output": raw_output,
    }
