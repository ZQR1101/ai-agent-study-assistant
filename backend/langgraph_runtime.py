from __future__ import annotations

from time import perf_counter
from typing import Any, TypedDict

from pydantic import ValidationError

from backend.agent_core import _extract_json_object
from backend.history_utils import format_history, normalize_history
from backend.llm_service import build_llm, normalize_model
from backend.schemas import AgentPlan, ChatRequest
from backend.tools import TOOL_REGISTRY


class LangGraphRuntimeUnavailableError(RuntimeError):
    pass


def _elapsed_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1000))


class LangGraphAgentState(TypedDict, total=False):
    message: str
    model: str
    temperature: float
    top_k: int
    history: list[dict]
    history_context: str
    custom_llm: Any
    planner_mode: str
    planner_fallback: bool
    planner_error: str

    intent: str
    use_rag: bool
    need_chat: bool
    need_explain: bool
    need_summarize: bool
    need_quiz: bool
    need_flashcard: bool

    plan: list[dict]
    sources: list[dict]
    flashcards: list[dict]
    trace: list[str]
    graph_path: list[str]
    tool_calls: list[dict]

    rag_context: str
    step_outputs: list[dict]
    last_output: str
    final_answer: str

    error: str


def _append_trace(state: LangGraphAgentState, item: str) -> list[str]:
    return [*state.get("trace", []), item]


def _append_graph_path(state: LangGraphAgentState, node_name: str) -> list[str]:
    return [*state.get("graph_path", []), node_name]


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _model_to_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump()

    return model.dict()


def _validate_agent_plan(data: dict) -> dict:
    if hasattr(AgentPlan, "model_validate"):
        plan = AgentPlan.model_validate(data)
    else:
        plan = AgentPlan.parse_obj(data)

    return _model_to_dict(plan)


def _has_positive_intent(text: str, keywords: list[str], negative_keywords: list[str]) -> bool:
    if _contains_any(text, negative_keywords):
        return False

    return _contains_any(text, keywords)


def detect_intent(message: str, use_rag_requested: bool = False) -> dict:
    normalized = message.lower()
    rag_keywords = [
        "根据知识库",
        "根据文档",
        "根据资料",
        "基于资料",
        "结合知识库",
        "只用知识库",
        "只根据知识库",
        "从文档中",
        "根据上传内容",
        "根据刚才资料",
        "知识库",
        "文档",
        "资料",
        "knowledge base",
        "rag context",
    ]
    rag_negative_keywords = [
        "不用知识库",
        "不要用知识库",
        "不使用知识库",
        "不根据知识库",
        "不要根据知识库",
    ]
    explain_keywords = [
        "什么是",
        "解释",
        "讲解",
        "用简单话说",
        "通俗解释",
        "帮我理解",
        "这个概念",
        "为什么",
        "区别是什么",
        "帮我复习",
        "what is",
        "explain",
    ]
    summarize_keywords = [
        "总结",
        "概括",
        "提炼",
        "归纳",
        "核心思想",
        "主要内容",
        "简短总结",
        "summarize",
        "summary",
    ]
    summarize_negative_keywords = [
        "不需要总结",
        "不要总结",
        "不用总结",
        "不总结",
    ]
    quiz_keywords = [
        "出题",
        "练习题",
        "测验",
        "quiz",
        "自测题",
        "检查我",
        "题目",
        "选择题",
        "问答题",
        "3 道题",
        "3道题",
        "三道题",
        "quiz me",
        "question",
    ]
    quiz_negative_keywords = [
        "不要出题",
        "不用出题",
        "不出题",
        "不生成练习题",
        "不要生成练习题",
        "不用生成练习题",
    ]
    flashcard_keywords = [
        "记忆卡片",
        "flashcard",
        "卡片",
        "复习卡",
        "背诵卡",
        "做成卡片",
        "生成卡片",
        "帮我复习",
    ]
    flashcard_negative_keywords = [
        "不要卡片",
        "不用卡片",
        "不生成卡片",
        "不要生成卡片",
        "不用生成卡片",
    ]

    explicit_no_rag = _contains_any(normalized, rag_negative_keywords)
    use_rag = (use_rag_requested or _contains_any(normalized, rag_keywords)) and not explicit_no_rag
    need_summarize = _has_positive_intent(
        normalized,
        summarize_keywords,
        summarize_negative_keywords,
    )
    need_quiz = _has_positive_intent(normalized, quiz_keywords, quiz_negative_keywords)
    need_flashcard = _has_positive_intent(
        normalized,
        flashcard_keywords,
        flashcard_negative_keywords,
    )
    need_explain = _contains_any(normalized, explain_keywords)
    if use_rag and not need_summarize and not need_explain:
        need_explain = True
    if not any([need_explain, need_summarize, need_quiz, need_flashcard]):
        need_explain = True

    intent_parts = []
    if use_rag:
        intent_parts.append("rag")
    if need_summarize:
        intent_parts.append("summarize")
    if need_explain:
        intent_parts.append("explain")
    if need_flashcard:
        intent_parts.append("flashcard")
    if need_quiz:
        intent_parts.append("quiz")

    return {
        "intent": "+".join(intent_parts),
        "use_rag": use_rag,
        "need_chat": False,
        "need_explain": need_explain,
        "need_summarize": need_summarize,
        "need_quiz": need_quiz,
        "need_flashcard": need_flashcard,
    }


def _plan_from_intent(message: str, intent: dict) -> list[dict]:
    steps: list[dict] = []

    def add_step(tool: str, reason: str) -> None:
        if any(step["tool"] == tool for step in steps):
            return

        steps.append({
            "tool": tool,
            "input": message,
            "reason": reason,
        })

    if intent.get("use_rag"):
        add_step("rag", "User requested knowledge-base grounded context.")

    if intent.get("need_summarize"):
        add_step("summarize", "User requested a concise summary.")

    if intent.get("need_explain"):
        add_step("explain", "User requested an explanation.")

    if intent.get("need_flashcard"):
        add_step("flashcard", "User requested memory cards.")

    if intent.get("need_quiz"):
        add_step("quiz", "User requested practice questions.")

    if not steps:
        add_step("explain", "Default learning response.")

    return steps


_LANGGRAPH_TOOL_ORDER = ["rag", "summarize", "explain", "chat", "flashcard", "quiz"]


def _tool_descriptions_for_prompt() -> str:
    return "\n".join(
        f"- {tool.name}：{tool.description}"
        for tool in TOOL_REGISTRY.values()
    )


def _normalize_plan_steps(steps: list[dict]) -> list[dict]:
    by_tool: dict[str, dict] = {}

    for step in steps:
        tool = str(step.get("tool", ""))
        if tool not in TOOL_REGISTRY or tool in by_tool:
            continue

        by_tool[tool] = {
            "tool": tool,
            "input": str(step.get("input") or ""),
            "reason": step.get("reason"),
        }

    return [
        by_tool[tool]
        for tool in _LANGGRAPH_TOOL_ORDER
        if tool in by_tool and by_tool[tool]["input"]
    ]


def _intent_from_plan_steps(steps: list[dict]) -> dict:
    tools = [step.get("tool") for step in steps]
    intent_parts = [tool for tool in _LANGGRAPH_TOOL_ORDER if tool in tools]

    return {
        "intent": "+".join(intent_parts) or "explain",
        "use_rag": "rag" in tools,
        "need_chat": "chat" in tools,
        "need_explain": "explain" in tools,
        "need_summarize": "summarize" in tools,
        "need_quiz": "quiz" in tools,
        "need_flashcard": "flashcard" in tools,
    }


def _rule_planner_state(message: str, state: LangGraphAgentState) -> dict:
    intent = detect_intent(message, use_rag_requested=state.get("use_rag", False))
    plan = _plan_from_intent(message, intent)
    return {
        **intent,
        "plan": plan,
        "planner_fallback": False,
        "planner_error": "",
    }


def plan_with_llm(state: LangGraphAgentState) -> dict:
    active_llm = state.get("custom_llm")
    if active_llm is None:
        raise RuntimeError("LLM planner requires a custom_llm")

    message = state.get("message", "")
    history_block = state.get("history_context") or "无"
    planner_prompt = f"""
你是 AI Study Assistant 的 LangGraph Planner。
请把用户请求拆成 1 到 5 个 LangGraph 执行步骤，并只返回 JSON object。

可用工具：
{_tool_descriptions_for_prompt()}

AgentPlan schema：
{{
  "goal": "用户任务目标，非空字符串",
  "steps": [
    {{
      "tool": "chat|rag|explain|summarize|quiz|flashcard",
      "input": "传给工具的输入，非空字符串",
      "reason": "为什么使用这个工具"
    }}
  ],
  "fallback": false
}}

规划规则：
- 如果用户要求根据知识库、文档或资料回答，先使用 rag。
- 如果用户要求解释概念，使用 explain。
- 如果用户要求总结、概括、提炼，使用 summarize。
- 如果用户要求卡片或复习卡，使用 flashcard。
- 如果用户要求出题、练习题或测验，使用 quiz。
- 如果用户没有明确学习工具需求，使用 chat。
- 工具不要重复。
- 推荐执行顺序：rag -> summarize/explain/chat -> flashcard -> quiz。
- 明确否定的任务不要加入，例如“不要出题”就不要加入 quiz。

示例 1：
用户输入：请解释 RAG，并出 3 道练习题
输出：
{{
  "goal": "解释 RAG 并生成练习题",
  "steps": [
    {{"tool": "explain", "input": "RAG", "reason": "用户要求解释概念"}},
    {{"tool": "quiz", "input": "基于 RAG 生成 3 道练习题", "reason": "用户要求出题"}}
  ],
  "fallback": false
}}

示例 2：
用户输入：根据知识库解释 agentic rag，生成记忆卡片，并出 3 道题
输出：
{{
  "goal": "基于知识库解释 agentic rag，生成卡片和练习题",
  "steps": [
    {{"tool": "rag", "input": "agentic rag", "reason": "用户要求根据知识库回答，先检索相关内容"}},
    {{"tool": "explain", "input": "基于知识库解释 agentic rag", "reason": "解释概念"}},
    {{"tool": "flashcard", "input": "基于 agentic rag 生成记忆卡片", "reason": "用户要求生成卡片"}},
    {{"tool": "quiz", "input": "基于 agentic rag 生成 3 道题", "reason": "用户要求出题"}}
  ],
  "fallback": false
}}

最近对话：
{history_block}

当前用户请求：
{message}

最终输出要求：
你必须只输出一个 JSON object。
不要输出 Markdown。
不要输出 ```json。
不要输出任何解释文字。
不要输出 schema 以外的字段。
JSON 必须符合 AgentPlan schema。
"""
    response = active_llm.invoke(planner_prompt)
    data = _extract_json_object(response.content)
    if not data:
        raise ValueError("JSON parse failed")

    try:
        plan = _validate_agent_plan(data)
    except ValidationError as exc:
        message = exc.errors()[0].get("msg") if exc.errors() else str(exc)
        raise ValueError(f"schema validation failed: {message}") from exc

    steps = _normalize_plan_steps(plan.get("steps", []))
    if not steps:
        raise ValueError("planner returned empty steps")

    if any(step.get("tool") not in TOOL_REGISTRY for step in steps):
        raise ValueError("planner returned unknown tool")

    intent = _intent_from_plan_steps(steps)
    return {
        **intent,
        "plan": steps,
        "planner_fallback": False,
        "planner_error": "",
    }


def _first_plan_input(state: LangGraphAgentState, tool_name: str) -> str:
    for step in state.get("plan", []):
        if step.get("tool") == tool_name and step.get("input"):
            return str(step["input"])

    return state.get("message", "")


def _build_shared_context(state: LangGraphAgentState) -> dict:
    return {
        "original_input": state.get("message", ""),
        "history_context": state.get("history_context", ""),
        "rag_context": state.get("rag_context", ""),
        "sources": state.get("sources", []),
        "step_outputs": state.get("step_outputs", []),
        "last_output": state.get("last_output", ""),
    }


def _merge_unique_sources(existing: list[dict], incoming: list[dict]) -> list[dict]:
    merged = [*existing]
    seen = {
        (
            source.get("source"),
            source.get("score"),
            source.get("snippet") or source.get("text"),
        )
        for source in merged
    }

    for source in incoming:
        key = (
            source.get("source"),
            source.get("score"),
            source.get("snippet") or source.get("text"),
        )
        if key not in seen:
            merged.append(source)
            seen.add(key)

    return merged


def _yes_no(value: Any) -> str:
    return "是" if value else "否"


def _tool_trace(node_name: str, tool_name: str, result: dict) -> list[str]:
    trace = [
        f"LangGraph node: {node_name}",
        f"调用工具：{tool_name}",
        f"工具说明：{result.get('tool_description', '')}",
        f"执行成功：{_yes_no(result.get('tool_success'))}",
        f"使用上下文：{_yes_no(result.get('used_context'))}",
    ]

    if result.get("context_sources"):
        trace.append(f"上下文来源：{' + '.join(result['context_sources'])}")
    if "latency_ms" in result:
        trace.append(f"{tool_name}: latency_ms={result['latency_ms']}")
    if result.get("error"):
        trace.append(f"错误：{result['error']}")

    trace.extend(item for item in result.get("trace", []) if item not in trace)
    return trace


def _normalize_tool_result(tool_name: str, result: dict, success: bool, description: str, error: str = "") -> dict:
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "trace": result.get("trace", []),
        "context": result.get("context", ""),
        "flashcards": result.get("flashcards", []),
        "tool_name": tool_name,
        "tool_description": description,
        "tool_success": success,
        "used_context": result.get("used_context", False),
        "context_sources": result.get("context_sources", []),
        "fallback_used": result.get("fallback_used", False),
        "error": error,
    }


def _run_registry_tool_raw(tool_name: str, step_input: str, state: LangGraphAgentState) -> dict:
    tool_spec = TOOL_REGISTRY.get(tool_name)

    if tool_spec is None:
        return _normalize_tool_result(
            tool_name=tool_name,
            result={"answer": f"LangGraph tool not found: {tool_name}"},
            success=False,
            description="",
            error=f"Unknown tool: {tool_name}",
        )

    try:
        raw_result = tool_spec.run(
            step_input=step_input,
            original_input=state.get("message", ""),
            custom_llm=state.get("custom_llm"),
            top_k=state.get("top_k", 3),
            shared_context=_build_shared_context(state),
        )
    except Exception as exc:
        return _normalize_tool_result(
            tool_name=tool_name,
            result={"answer": f"LangGraph tool failed: {tool_name}"},
            success=False,
            description=tool_spec.description,
            error=str(exc),
        )

    return _normalize_tool_result(
        tool_name=tool_spec.name,
        result=raw_result or {},
        success=True,
        description=tool_spec.description,
    )


def run_registry_tool_for_state(
    tool_name: str,
    step_input: str,
    state: LangGraphAgentState,
) -> LangGraphAgentState:
    tool_started_at = perf_counter()
    result = _run_registry_tool_raw(tool_name, step_input, state)
    latency_ms = _elapsed_ms(tool_started_at)
    result["latency_ms"] = latency_ms
    answer = result.get("answer", "")
    incoming_sources = result.get("sources", [])
    incoming_flashcards = result.get("flashcards", [])
    sources = _merge_unique_sources(state.get("sources", []), incoming_sources)
    flashcards = [*state.get("flashcards", []), *incoming_flashcards]
    step_outputs = [
        *state.get("step_outputs", []),
        {
            "tool": tool_name,
            "input": step_input,
            "answer": answer,
            "sources": incoming_sources,
            "flashcards": incoming_flashcards,
            "success": bool(result.get("tool_success")),
            "error": result.get("error", ""),
        },
    ]
    trace = [
        *state.get("trace", []),
        *_tool_trace(tool_name, tool_name, result),
    ]
    tool_call = {
        "node": tool_name,
        "tool": tool_name,
        "description": result.get("tool_description", ""),
        "success": bool(result.get("tool_success")),
        "used_context": bool(result.get("used_context")),
        "context_sources": result.get("context_sources", []),
        "output_length": len(answer),
        "latency_ms": latency_ms,
    }
    if result.get("error"):
        tool_call["error"] = result["error"]

    new_state: LangGraphAgentState = {
        **state,
        "sources": sources,
        "flashcards": flashcards,
        "step_outputs": step_outputs,
        "tool_calls": [*state.get("tool_calls", []), tool_call],
        "last_output": answer,
        "trace": trace,
    }

    if result.get("context"):
        new_state["rag_context"] = result["context"]
    if result.get("error"):
        new_state["error"] = result["error"]

    return new_state


def planner_node(state: LangGraphAgentState) -> LangGraphAgentState:
    message = state.get("message", "")
    planner_mode = state.get("planner_mode") or "rule"
    planner_started_at = perf_counter()
    planner_result = _rule_planner_state(message, state)
    fallback_reason = ""

    if planner_mode == "llm":
        try:
            planner_result = plan_with_llm(state)
        except Exception as exc:  # noqa: BLE001 - fallback must keep the backend available.
            fallback_reason = str(exc)
            planner_result = {
                **_rule_planner_state(message, state),
                "planner_fallback": True,
                "planner_error": fallback_reason,
            }

    plan = planner_result["plan"]
    latency_ms = _elapsed_ms(planner_started_at)
    trace = [
        *state.get("trace", []),
        "planner: start",
        f"planner: mode={planner_mode}",
        f"planner: intent={planner_result['intent']}",
        f"planner: use_rag={planner_result['use_rag']}",
        f"planner: need_explain={planner_result['need_explain']}",
        f"planner: need_summarize={planner_result['need_summarize']}",
        f"planner: need_flashcard={planner_result['need_flashcard']}",
        f"planner: need_quiz={planner_result['need_quiz']}",
        f"planner: fallback={planner_result.get('planner_fallback', False)}",
        f"planner: produced {len(plan)} step(s)",
        f"planner: latency_ms={latency_ms}",
    ]
    if fallback_reason:
        trace.append(f"planner: fallback reason={fallback_reason}")

    planner_call = {
        "node": "planner",
        "tool": "planner",
        "description": f"{planner_mode} planner",
        "success": not bool(planner_result.get("planner_fallback")),
        "used_context": bool(state.get("history_context")),
        "context_sources": ["history"] if state.get("history_context") else [],
        "output_length": len(plan),
        "latency_ms": latency_ms,
    }
    if planner_result.get("planner_error"):
        planner_call["error"] = planner_result["planner_error"]

    return {
        **state,
        **planner_result,
        "planner_mode": planner_mode,
        "plan": plan,
        "trace": trace,
        "tool_calls": [*state.get("tool_calls", []), planner_call],
        "graph_path": _append_graph_path(state, "planner"),
    }


def route_after_planner(state: LangGraphAgentState) -> str:
    if state.get("use_rag"):
        return "rag"

    if state.get("need_summarize"):
        return "summarize"

    if state.get("need_explain"):
        return "explain"

    if state.get("need_chat"):
        return "chat"

    if state.get("need_flashcard"):
        return "flashcard"

    if state.get("need_quiz"):
        return "quiz"

    return "explain"


def rag_node(state: LangGraphAgentState) -> LangGraphAgentState:
    query = _first_plan_input(state, "rag")
    state_with_trace = {
        **state,
        "trace": _append_trace(state, f"rag: input={query}"),
        "graph_path": _append_graph_path(state, "rag"),
    }
    return run_registry_tool_for_state("rag", query, state_with_trace)


def route_after_rag(state: LangGraphAgentState) -> str:
    if state.get("need_summarize"):
        return "summarize"

    if state.get("need_explain"):
        return "explain"

    if state.get("need_chat"):
        return "chat"

    if state.get("need_flashcard"):
        return "flashcard"

    if state.get("need_quiz"):
        return "quiz"

    return "explain"


def explain_node(state: LangGraphAgentState) -> LangGraphAgentState:
    topic = _first_plan_input(state, "explain")
    state_with_trace = {
        **state,
        "trace": _append_trace(state, f"explain: input={topic}"),
        "graph_path": _append_graph_path(state, "explain"),
    }
    return run_registry_tool_for_state("explain", topic, state_with_trace)


def chat_node(state: LangGraphAgentState) -> LangGraphAgentState:
    topic = _first_plan_input(state, "chat")
    state_with_trace = {
        **state,
        "trace": _append_trace(state, f"chat: input={topic}"),
        "graph_path": _append_graph_path(state, "chat"),
    }
    return run_registry_tool_for_state("chat", topic, state_with_trace)


def summarize_node(state: LangGraphAgentState) -> LangGraphAgentState:
    topic = _first_plan_input(state, "summarize")
    state_with_trace = {
        **state,
        "trace": _append_trace(state, f"summarize: input={topic}"),
        "graph_path": _append_graph_path(state, "summarize"),
    }
    return run_registry_tool_for_state("summarize", topic, state_with_trace)


def route_after_main_content(state: LangGraphAgentState) -> str:
    if state.get("graph_path", [])[-1:] == ["summarize"] and state.get("need_explain"):
        return "explain"

    if state.get("need_flashcard"):
        return "flashcard"

    if state.get("need_quiz"):
        return "quiz"

    return "finalizer"


def flashcard_node(state: LangGraphAgentState) -> LangGraphAgentState:
    topic = _first_plan_input(state, "flashcard")
    state_with_trace = {
        **state,
        "trace": _append_trace(state, f"flashcard: input={topic}"),
        "graph_path": _append_graph_path(state, "flashcard"),
    }
    return run_registry_tool_for_state("flashcard", topic, state_with_trace)


def route_after_flashcard(state: LangGraphAgentState) -> str:
    if state.get("need_quiz"):
        return "quiz"

    return "finalizer"


def quiz_node(state: LangGraphAgentState) -> LangGraphAgentState:
    topic = _first_plan_input(state, "quiz")
    state_with_trace = {
        **state,
        "trace": _append_trace(state, f"quiz: input={topic}"),
        "graph_path": _append_graph_path(state, "quiz"),
    }
    return run_registry_tool_for_state("quiz", topic, state_with_trace)


def route_to_finalizer(state: LangGraphAgentState) -> str:
    return "finalizer"


def _latest_step_answer(state: LangGraphAgentState, tool_name: str) -> str:
    for output in reversed(state.get("step_outputs", [])):
        if output.get("tool") == tool_name and output.get("answer"):
            return str(output["answer"])

    return ""


def _failed_steps(state: LangGraphAgentState) -> list[dict]:
    return [output for output in state.get("step_outputs", []) if not output.get("success", True)]


def compose_final_answer(state: LangGraphAgentState) -> str:
    chat_answer = _latest_step_answer(state, "chat")
    explain_answer = _latest_step_answer(state, "explain")
    summarize_answer = _latest_step_answer(state, "summarize")
    quiz_answer = _latest_step_answer(state, "quiz")
    rag_answer = _latest_step_answer(state, "rag")
    flashcard_count = len(state.get("flashcards", []))
    failures = _failed_steps(state)
    sections = []

    if summarize_answer:
        sections.append(f"## 1. 内容总结\n{summarize_answer}")
    elif explain_answer:
        sections.append(f"## 1. 知识讲解\n{explain_answer}")
    elif chat_answer:
        sections.append(f"## 1. 回复\n{chat_answer}")
    elif rag_answer and not flashcard_count and not quiz_answer:
        sections.append(f"## 1. 知识库检索结果\n{rag_answer}")

    if flashcard_count:
        sections.append(
            f"## {len(sections) + 1}. 记忆卡片\n"
            f"已生成 {flashcard_count} 张记忆卡片，请在下方卡片区域查看、翻面或下载。"
        )

    if quiz_answer:
        sections.append(f"## {len(sections) + 1}. 练习题\n{quiz_answer}")

    if failures:
        failed_tools = ", ".join(str(step.get("tool")) for step in failures)
        sections.append(
            f"## {len(sections) + 1}. 执行提示\n"
            f"部分 LangGraph 工具执行失败：{failed_tools}。请查看 trace 获取更多信息。"
        )

    if not sections:
        fallback = state.get("last_output") or "暂时没有生成有效的 LangGraph 结果。"
        sections.append(f"## 1. 执行结果\n{fallback}")

    return "# 学习结果\n\n" + "\n\n".join(sections)


def finalizer_node(state: LangGraphAgentState) -> LangGraphAgentState:
    final_answer = compose_final_answer(state)

    return {
        **state,
        "final_answer": final_answer,
        "graph_path": _append_graph_path(state, "finalizer"),
        "trace": _append_trace(state, "finalizer: composed final answer"),
    }


def build_runtime_info(state: LangGraphAgentState) -> dict:
    graph_path = state.get("graph_path", [])
    return {
        "runtime": "langgraph",
        "graph_path": graph_path,
        "node_count": len(graph_path),
        "tool_calls": state.get("tool_calls", []),
        "finalizer_used": "finalizer" in graph_path,
        "planner_mode": state.get("planner_mode") or "rule",
        "planner_fallback": bool(state.get("planner_fallback", False)),
        "planner_error": state.get("planner_error") or None,
        "error": state.get("error") or None,
    }


def build_langgraph_workflow():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise LangGraphRuntimeUnavailableError(
            "LangGraph is not installed. Install the optional dependency with "
            "`pip install langgraph` or install project requirements."
        ) from exc

    graph_builder = StateGraph(LangGraphAgentState)
    graph_builder.add_node("planner", planner_node)
    graph_builder.add_node("rag", rag_node)
    graph_builder.add_node("chat", chat_node)
    graph_builder.add_node("explain", explain_node)
    graph_builder.add_node("summarize", summarize_node)
    graph_builder.add_node("flashcard", flashcard_node)
    graph_builder.add_node("quiz", quiz_node)
    graph_builder.add_node("finalizer", finalizer_node)
    graph_builder.add_edge(START, "planner")
    graph_builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "rag": "rag",
            "explain": "explain",
            "summarize": "summarize",
            "chat": "chat",
            "flashcard": "flashcard",
            "quiz": "quiz",
        },
    )
    graph_builder.add_conditional_edges(
        "rag",
        route_after_rag,
        {
            "explain": "explain",
            "summarize": "summarize",
            "chat": "chat",
            "flashcard": "flashcard",
            "quiz": "quiz",
        },
    )
    graph_builder.add_conditional_edges(
        "chat",
        route_after_main_content,
        {
            "flashcard": "flashcard",
            "quiz": "quiz",
            "finalizer": "finalizer",
        },
    )
    graph_builder.add_conditional_edges(
        "explain",
        route_after_main_content,
        {
            "flashcard": "flashcard",
            "quiz": "quiz",
            "finalizer": "finalizer",
        },
    )
    graph_builder.add_conditional_edges(
        "summarize",
        route_after_main_content,
        {
            "explain": "explain",
            "flashcard": "flashcard",
            "quiz": "quiz",
            "finalizer": "finalizer",
        },
    )
    graph_builder.add_conditional_edges(
        "flashcard",
        route_after_flashcard,
        {
            "quiz": "quiz",
            "finalizer": "finalizer",
        },
    )
    graph_builder.add_conditional_edges(
        "quiz",
        route_to_finalizer,
        {
            "finalizer": "finalizer",
        },
    )
    graph_builder.add_edge("finalizer", END)
    return graph_builder.compile()


def run_langgraph_workflow(
    message: str,
    *,
    custom_llm=None,
    model: str = "mimo-v2.5",
    temperature: float = 0.7,
    top_k: int = 3,
    history: list[dict] | None = None,
    history_context: str = "",
    use_rag: bool = False,
    planner_mode: str = "rule",
) -> dict:
    graph = build_langgraph_workflow()
    result = graph.invoke({
        "message": message,
        "model": model,
        "temperature": temperature,
        "top_k": top_k,
        "history": history or [],
        "history_context": history_context,
        "custom_llm": custom_llm,
        "planner_mode": planner_mode,
        "planner_fallback": False,
        "planner_error": "",
        "use_rag": use_rag,
        "plan": [],
        "sources": [],
        "flashcards": [],
        "trace": ["langgraph_runtime: start"],
        "graph_path": [],
        "tool_calls": [],
        "rag_context": "",
        "step_outputs": [],
        "last_output": "",
        "final_answer": "",
    })

    return {
        "answer": result.get("final_answer") or compose_final_answer(result),
        "sources": result.get("sources", []),
        "trace": result.get("trace", []),
        "plan": result.get("plan", []),
        "flashcards": result.get("flashcards", []),
        "step_outputs": result.get("step_outputs", []),
        "runtime_info": build_runtime_info(result),
    }


def _group_langgraph_trace(trace: list[str]) -> list[dict]:
    return [{
        "title": "LangGraph Runtime",
        "items": [item for item in trace if item],
    }]


def run_langgraph_chat_request(request: ChatRequest) -> dict:
    selected_model = normalize_model(request.model)
    history_messages = normalize_history(request.history)
    history_context = format_history(history_messages)
    custom_llm = build_llm(model=selected_model, temperature=request.temperature)
    trace = [
        "执行方式：LangGraph",
        "LangGraph workflow enabled",
        f"mode: {request.mode}",
        f"model: {selected_model}",
        f"temperature: {request.temperature}",
        f"use_agent: {request.use_agent}",
        f"use_rag: {request.use_rag}",
        f"use_langgraph: {request.use_langgraph}",
        f"planner_mode: {request.planner_mode}",
        f"top_k: {request.top_k}",
    ]

    try:
        result = run_langgraph_workflow(
            request.message,
            custom_llm=custom_llm,
            model=selected_model,
            temperature=request.temperature,
            top_k=request.top_k,
            history=history_messages,
            history_context=history_context,
            use_rag=request.use_rag,
            planner_mode=request.planner_mode,
        )
    except LangGraphRuntimeUnavailableError as exc:
        trace.append(f"LangGraph unavailable: {exc}")
        return {
            "answer": f"LangGraph workflow is unavailable: {exc}",
            "mode": "langgraph",
            "model": selected_model,
            "sources": [],
            "trace": _group_langgraph_trace(trace),
            "plan": [],
            "flashcards": [],
            "runtime_info": {
                "runtime": "langgraph",
                "graph_path": [],
                "node_count": 0,
                "tool_calls": [],
                "finalizer_used": False,
                "planner_mode": request.planner_mode,
                "planner_fallback": False,
                "planner_error": None,
                "error": str(exc),
            },
        }

    trace.extend(result.get("trace", []))
    return {
        "answer": result.get("answer", ""),
        "mode": "langgraph",
        "model": selected_model,
        "sources": result.get("sources", []),
        "trace": _group_langgraph_trace(trace),
        "plan": result.get("plan", []),
        "flashcards": result.get("flashcards", []),
        "runtime_info": result.get("runtime_info", {}),
    }
