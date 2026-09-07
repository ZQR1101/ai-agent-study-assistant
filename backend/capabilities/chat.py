from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.llm_service import llm
from backend.rag_service import NO_RAG_ANSWER, RAG_FALLBACK_PREFIX, with_fallback_prefix
from backend.tool_registry import ToolRegistry


CHAT_CAPABILITY = "chat"
CHAT_MODES = frozenset({"chat", "rag", "explain", "summarize", "quiz"})
STUDY_OPERATIONS = frozenset({"explain", "summarize", "quiz"})

_OPERATION_REASONS = {
    "chat": "Answer the question.",
    "explain": "Explain the topic.",
    "summarize": "Summarize the content.",
    "quiz": "Generate practice questions.",
}


@dataclass
class ChatResult:
    answer: str
    sources: list = field(default_factory=list)
    fallback_used: bool = False
    flashcards: list = field(default_factory=list)
    trace: list[str] = field(default_factory=list)
    plan: list[dict] = field(default_factory=list)
    retrieval_info: dict = field(default_factory=dict)
    operation: str = "chat"


def _default_registry() -> ToolRegistry:
    from backend.tools import TOOL_REGISTRY

    return TOOL_REGISTRY


def _shared_context(
    *,
    history_context: str | None,
    run_id: str | None = None,
    rag_context: str | None = None,
    memory_context: str | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    shared: dict[str, Any] = {"history_context": history_context or ""}
    if run_id:
        shared["run_id"] = run_id
    if rag_context:
        shared["rag_context"] = rag_context
    if memory_context:
        shared["memory_context"] = memory_context
    if extra:
        shared.update(extra)
    return shared


def _generation_tool(operation: str) -> str:
    return "study" if operation in STUDY_OPERATIONS else "chat"


def run_chat(
    message: str,
    *,
    mode: str = "chat",
    use_rag: bool = False,
    custom_llm=None,
    top_k: int = 3,
    history_context: str | None = None,
    memory_context: str | None = None,
    retrieval_mode: str = "vector",
    reranker_enabled: bool | None = None,
    run_id: str | None = None,
    registry: ToolRegistry | None = None,
) -> ChatResult:
    """Run the default chat capability: optional retrieve, then one generation tool.

    `explain` / `summarize` / `quiz` / `rag` are mode aliases, not separate capabilities.
    They select a study operation or a retrieve-or-refuse policy. Generation always
    goes through ToolRegistry (`rag_search`, `chat`, `study`).
    """
    active_llm = custom_llm or llm
    tool_registry = registry or _default_registry()
    require_rag = mode == "rag"
    operation = mode if mode in STUDY_OPERATIONS else "chat"
    if require_rag:
        use_rag = True

    trace = [f"Capability chat：started ({mode}/{operation})"]
    plan: list[dict] = []
    sources: list = []
    retrieval_info: dict = {}
    rag_context_text = ""
    passed_threshold = False
    fallback_used = False

    if use_rag:
        rag_extra: dict[str, Any] = {"retrieval_mode": retrieval_mode}
        if reranker_enabled is not None:
            rag_extra["reranker_enabled"] = reranker_enabled
        plan.append({
            "tool": "rag_search",
            "input": message,
            "reason": "Retrieve grounding.",
        })
        rag_result = tool_registry.execute(
            "rag_search",
            step_input=message,
            original_input=message,
            custom_llm=active_llm,
            top_k=top_k,
            shared_context=_shared_context(
                history_context=history_context,
                run_id=run_id,
                memory_context=memory_context,
                extra=rag_extra,
            ),
            generate_answer=False,
            actor=CHAT_CAPABILITY,
        ) or {}
        trace.extend(rag_result.get("trace") or [])
        sources = list(rag_result.get("sources") or [])
        retrieval_info = dict(rag_result.get("retrieval_info") or {})
        passed_threshold = bool(
            retrieval_info.get("found")
            or rag_result.get("used_context")
            or rag_result.get("context")
        )
        fallback_used = bool(rag_result.get("fallback_used")) or not passed_threshold
        rag_context_text = rag_result.get("context") or "" if passed_threshold else ""
        if not passed_threshold:
            trace.append("Capability chat：RAG miss")

    if require_rag and not passed_threshold:
        trace.append("Capability chat：require_rag miss, refusing ungrounded answer")
        trace.append("Capability chat：completed")
        return ChatResult(
            answer=NO_RAG_ANSWER,
            sources=[],
            fallback_used=True,
            trace=trace,
            plan=plan,
            retrieval_info=retrieval_info,
            operation=operation,
        )

    generation_tool = _generation_tool(operation)
    execute_kwargs: dict[str, Any] = {
        "step_input": message,
        "original_input": message,
        "custom_llm": active_llm,
        "top_k": top_k,
        "shared_context": _shared_context(
            history_context=history_context,
            run_id=run_id,
            rag_context=rag_context_text or None,
            memory_context=memory_context,
        ),
        "actor": CHAT_CAPABILITY,
    }
    plan_step: dict[str, Any] = {
        "tool": generation_tool,
        "input": message,
        "reason": _OPERATION_REASONS[operation],
    }
    if generation_tool == "study":
        execute_kwargs["operation"] = operation
        plan_step["arguments"] = {"operation": operation}

    plan.append(plan_step)
    generation = tool_registry.execute(generation_tool, **execute_kwargs) or {}
    trace.extend(generation.get("trace") or [])
    answer = str(generation.get("answer") or "")
    flashcards = list(generation.get("flashcards") or [])
    if generation.get("sources"):
        sources = list(generation["sources"])

    if use_rag and not passed_threshold:
        answer = with_fallback_prefix(answer, RAG_FALLBACK_PREFIX)
        fallback_used = True

    trace.append("Capability chat：completed")
    return ChatResult(
        answer=answer,
        sources=sources,
        fallback_used=fallback_used,
        flashcards=flashcards,
        trace=trace,
        plan=plan,
        retrieval_info=retrieval_info,
        operation=operation,
    )
