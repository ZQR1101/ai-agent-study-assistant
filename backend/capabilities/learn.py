from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.history_utils import memory_prompt_block
from backend.llm_service import llm
from backend.rag_service import LEARN_FALLBACK_PREFIX, SIMILARITY_THRESHOLD, with_fallback_prefix
from backend.tool_registry import ToolRegistry


LEARN_CAPABILITY = "learn"


@dataclass
class LearnResult:
    knowledge: str
    summary: str
    quiz: str
    advice: str
    sources: list = field(default_factory=list)
    fallback_used: bool = False
    passed_threshold: bool = False
    highest_score: float | None = None
    threshold: float | None = None
    trace: list[str] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)
    retrieval_info: dict = field(default_factory=dict)

    @property
    def answer(self) -> str:
        return format_learn_answer(self)

    def as_workflow_dict(self) -> dict:
        return {
            "knowledge": self.knowledge,
            "summary": self.summary,
            "quiz": self.quiz,
            "advice": self.advice,
            "sources": self.sources,
            "highest_score": self.highest_score,
            "threshold": self.threshold,
            "passed_threshold": self.passed_threshold,
        }


def format_learn_answer(result: LearnResult | dict) -> str:
    if isinstance(result, LearnResult):
        knowledge = result.knowledge
        summary = result.summary
        quiz = result.quiz
        advice = result.advice
    else:
        knowledge = result.get("knowledge", "")
        summary = result.get("summary", "")
        quiz = result.get("quiz", "")
        advice = result.get("advice", "")

    parts = [f"知识内容：\n{knowledge}"]
    if summary:
        parts.append(f"总结：\n{summary}")
    if quiz:
        parts.append(f"练习题：\n{quiz}")
    if advice:
        parts.append(f"学习建议：\n{advice}")
    return "\n\n".join(parts)


def _default_registry() -> ToolRegistry:
    from backend.tools import TOOL_REGISTRY

    return TOOL_REGISTRY


def _shared_context(
    *,
    history_context: str | None,
    run_id: str | None = None,
    rag_context: str | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    shared: dict[str, Any] = {"history_context": history_context or ""}
    if run_id:
        shared["run_id"] = run_id
    if rag_context:
        shared["rag_context"] = rag_context
    if extra:
        shared.update(extra)
    return shared


def _execute_study(
    registry: ToolRegistry,
    *,
    step_input: str,
    original_input: str,
    operation: str,
    custom_llm,
    top_k: int,
    shared_context: dict,
) -> dict:
    return registry.execute(
        "study",
        step_input=step_input,
        original_input=original_input,
        custom_llm=custom_llm,
        top_k=top_k,
        shared_context=shared_context,
        operation=operation,
        actor=LEARN_CAPABILITY,
    ) or {}


def run_learn(
    topic: str,
    *,
    context: str | None = None,
    custom_llm=None,
    top_k: int = 3,
    score_threshold: float = SIMILARITY_THRESHOLD,
    use_rag: bool = True,
    history_context: str | None = None,
    retrieval_mode: str = "vector",
    reranker_enabled: bool | None = None,
    run_id: str | None = None,
    prefix_on_rag_miss: bool = False,
    registry: ToolRegistry | None = None,
    use_memory: bool = True,
    session_id: str | None = None,
    memory_context: str | None = None,
) -> LearnResult:
    """Run the learn capability: retrieve, explain, summarize, quiz, then advise.

    This owns the turn. It calls `rag_search` and `study` through ToolRegistry
    and does not register itself as a tool.
    """
    active_llm = custom_llm or llm
    tool_registry = registry or _default_registry()
    trace = ["Capability learn：started"]
    plan: list[dict] = []
    sources: list = []
    retrieval_info: dict = {}
    highest_score = None
    threshold = score_threshold
    passed_threshold = False
    fallback_used = False
    rag_context_text = str(context or "").strip()

    if rag_context_text:
        passed_threshold = True
        trace.append("Capability learn：using provided context")
    elif use_rag:
        rag_extra: dict[str, Any] = {
            "retrieval_mode": retrieval_mode,
            "score_threshold": score_threshold,
        }
        if reranker_enabled is not None:
            rag_extra["reranker_enabled"] = reranker_enabled
        plan.append({
            "tool": "rag_search",
            "input": topic,
            "reason": "Retrieve grounding for the lesson.",
        })
        rag_result = tool_registry.execute(
            "rag_search",
            step_input=topic,
            original_input=topic,
            custom_llm=active_llm,
            top_k=top_k,
            shared_context=_shared_context(
                history_context=history_context,
                run_id=run_id,
                extra=rag_extra,
            ),
            generate_answer=False,
            actor=LEARN_CAPABILITY,
        ) or {}
        trace.extend(rag_result.get("trace") or [])
        sources = list(rag_result.get("sources") or [])
        retrieval_info = dict(rag_result.get("retrieval_info") or {})
        highest_score = retrieval_info.get("max_score")
        if retrieval_info.get("threshold") is not None:
            threshold = retrieval_info.get("threshold")
        passed_threshold = bool(
            retrieval_info.get("found")
            or rag_result.get("used_context")
            or rag_result.get("context")
        )
        fallback_used = bool(rag_result.get("fallback_used")) or not passed_threshold
        rag_context_text = rag_result.get("context") or "" if passed_threshold else ""
        if not passed_threshold:
            trace.append("Capability learn：RAG miss, generating without knowledge base")
    else:
        trace.append("Capability learn：RAG skipped")

    explain_result = _execute_study(
        tool_registry,
        step_input=topic,
        original_input=topic,
        operation="explain",
        custom_llm=active_llm,
        top_k=top_k,
        shared_context=_shared_context(
            history_context=history_context,
            run_id=run_id,
            rag_context=rag_context_text or None,
        ),
    )
    plan.append({
        "tool": "study",
        "input": topic,
        "reason": "Explain the topic.",
        "arguments": {"operation": "explain"},
    })
    trace.extend(explain_result.get("trace") or [])
    knowledge = str(explain_result.get("answer") or "")

    followup_shared = _shared_context(history_context=history_context, run_id=run_id)
    followup_input = knowledge or topic
    summary_result = _execute_study(
        tool_registry,
        step_input=followup_input,
        original_input=topic,
        operation="summarize",
        custom_llm=active_llm,
        top_k=top_k,
        shared_context=followup_shared,
    )
    plan.append({
        "tool": "study",
        "input": topic,
        "reason": "Summarize the explanation.",
        "arguments": {"operation": "summarize"},
    })
    trace.extend(summary_result.get("trace") or [])
    summary = str(summary_result.get("answer") or "")

    quiz_result = _execute_study(
        tool_registry,
        step_input=followup_input,
        original_input=topic,
        operation="quiz",
        custom_llm=active_llm,
        top_k=top_k,
        shared_context=followup_shared,
    )
    plan.append({
        "tool": "study",
        "input": topic,
        "reason": "Generate practice questions.",
        "arguments": {"operation": "quiz"},
    })
    trace.extend(quiz_result.get("trace") or [])
    quiz = str(quiz_result.get("answer") or "")

    memory_block = memory_prompt_block(memory_context) if memory_context else ""
    advice_prompt = f"""
请根据下面内容，给出简短的下一步学习建议，不超过 3 条：

最近对话：
{history_context or "无"}
{memory_block}
{summary}
"""
    advice = active_llm.invoke(advice_prompt).content
    trace.append("Capability learn：advice")

    display_knowledge = knowledge
    if prefix_on_rag_miss and use_rag and not passed_threshold and not str(context or "").strip():
        display_knowledge = with_fallback_prefix(knowledge, LEARN_FALLBACK_PREFIX)
        fallback_used = True

    # Record to L1 Memory (lesson_completed event)
    if use_memory and session_id:
        try:
            from backend.memory import get_memory_engine

            engine = get_memory_engine()
            event = engine.record_event(
                session_id=session_id,
                run_id=run_id,
                event_type="lesson_completed",
                data={
                    "topic": topic,
                    "knowledge": knowledge,
                    "summary": summary,
                    "quiz": quiz,
                    "advice": advice,
                    "passed_threshold": passed_threshold,
                    "highest_score": highest_score,
                    "sources": [s.get("source", "") if isinstance(s, dict) else str(s) for s in sources],
                },
            )
            trace.append(f"Memory L1：recorded lesson_completed event {event.event_id}")
            facts = engine.auto_extract_from_lesson(
                event.event_id,
                topic,
                knowledge_score=highest_score,
            )
            trace.append(f"Memory L2：extracted {len(facts)} fact(s)")
            engine.refresh_profile()
            trace.append("Memory L3：profile refreshed")
        except Exception as exc:
            trace.append(f"Memory L1：skip (error: {exc})")

    trace.append("Capability learn：completed")
    return LearnResult(
        knowledge=display_knowledge,
        summary=summary,
        quiz=quiz,
        advice=advice,
        sources=sources,
        fallback_used=fallback_used,
        passed_threshold=passed_threshold,
        highest_score=highest_score,
        threshold=threshold,
        trace=trace,
        plan=plan,
        retrieval_info=retrieval_info,
    )


def learning_workflow(
    topic: str,
    context=None,
    custom_llm=None,
    top_k: int = 3,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    use_rag: bool = True,
    history_context: str | None = None,
) -> dict:
    """Compatibility wrapper for the deprecated `/learn` endpoint."""
    return run_learn(
        topic,
        context=context,
        custom_llm=custom_llm,
        top_k=top_k,
        score_threshold=similarity_threshold,
        use_rag=use_rag,
        history_context=history_context,
    ).as_workflow_dict()
