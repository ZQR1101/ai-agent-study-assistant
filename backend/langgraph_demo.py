from typing import Any, Dict, List, TypedDict

from backend.tools import TOOL_REGISTRY


class LangGraphDemoUnavailableError(RuntimeError):
    pass


class LangGraphDemoState(TypedDict, total=False):
    message: str
    intent: str
    use_rag: bool
    need_quiz: bool
    need_flashcard: bool
    plan: List[Dict[str, Any]]
    rag_context: str
    answer_parts: List[str]
    sources: List[Dict[str, Any]]
    flashcards: List[Dict[str, Any]]
    trace: List[str]
    step_outputs: List[Dict[str, Any]]
    last_output: str


def _append_trace(state: LangGraphDemoState, item: str) -> List[str]:
    return [*state.get("trace", []), item]


def _first_plan_input(state: LangGraphDemoState, tool_name: str) -> str:
    for step in state.get("plan", []):
        if step.get("tool") == tool_name and step.get("input"):
            return str(step["input"])

    return state.get("message", "")


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def _build_shared_context(state: LangGraphDemoState) -> dict:
    return {
        "original_input": state.get("message", ""),
        "history_context": "",
        "rag_context": state.get("rag_context", ""),
        "sources": state.get("sources", []),
        "step_outputs": state.get("step_outputs", []),
        "last_output": state.get("last_output", ""),
    }


def run_registry_tool(
    tool_name: str,
    step_input: str,
    state: LangGraphDemoState,
    custom_llm=None,
    top_k: int = 3,
) -> dict:
    tool_spec = TOOL_REGISTRY.get(tool_name)

    if tool_spec is None:
        return {
            "answer": f"LangGraph demo tool not found: {tool_name}",
            "sources": [],
            "trace": [f"error: unknown tool {tool_name}"],
            "context": "",
            "flashcards": [],
            "tool_name": tool_name,
            "tool_description": "",
            "tool_success": False,
            "used_context": False,
            "context_sources": [],
            "error": f"Unknown tool: {tool_name}",
        }

    try:
        result = tool_spec.run(
            step_input=step_input,
            original_input=state.get("message", ""),
            custom_llm=custom_llm,
            top_k=top_k,
            shared_context=_build_shared_context(state),
        )
    except Exception as exc:
        return {
            "answer": f"LangGraph demo tool failed: {tool_name}",
            "sources": [],
            "trace": [],
            "context": "",
            "flashcards": [],
            "tool_name": tool_name,
            "tool_description": tool_spec.description,
            "tool_success": False,
            "used_context": False,
            "context_sources": [],
            "error": str(exc),
        }

    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "trace": result.get("trace", []),
        "context": result.get("context", ""),
        "flashcards": result.get("flashcards", []),
        "tool_name": tool_spec.name,
        "tool_description": tool_spec.description,
        "tool_success": True,
        "used_context": result.get("used_context", False),
        "context_sources": result.get("context_sources", []),
        "error": "",
    }


def _merge_unique_sources(existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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


def _tool_trace(node_name: str, tool_name: str, result: dict) -> List[str]:
    trace = [
        f"{node_name}: LangGraph node: {node_name}",
        f"{node_name}: \u8c03\u7528\u5de5\u5177\uff1a{tool_name}",
        f"{node_name}: \u5de5\u5177\u8bf4\u660e\uff1a{result.get('tool_description', '')}",
        f"{node_name}: \u6267\u884c\u6210\u529f\uff1a{'\u662f' if result.get('tool_success') else '\u5426'}",
        f"{node_name}: \u4f7f\u7528\u4e0a\u4e0b\u6587\uff1a{'\u662f' if result.get('used_context') else '\u5426'}",
        f"{node_name}: call tool={tool_name}",
        f"{node_name}: tool description={result.get('tool_description', '')}",
        f"{node_name}: tool success={'yes' if result.get('tool_success') else 'no'}",
        f"{node_name}: used context={'yes' if result.get('used_context') else 'no'}",
    ]

    if result.get("context_sources"):
        trace.append(f"{node_name}: context sources={'+'.join(result['context_sources'])}")
    if result.get("error"):
        trace.append(f"{node_name}: \u9519\u8bef\uff1a{result['error']}")
        trace.append(f"{node_name}: error={result['error']}")

    trace.extend(result.get("trace", []))
    return trace


def _apply_tool_result(
    state: LangGraphDemoState,
    node_name: str,
    tool_name: str,
    step_input: str,
    result: dict,
    include_answer: bool = True,
) -> LangGraphDemoState:
    answer = result.get("answer", "")
    sources = _merge_unique_sources(state.get("sources", []), result.get("sources", []))
    flashcards = [*state.get("flashcards", []), *result.get("flashcards", [])]
    answer_parts = [*state.get("answer_parts", [])]

    if include_answer and answer:
        answer_parts.append(answer)

    step_outputs = [
        *state.get("step_outputs", []),
        {
            "tool": tool_name,
            "input": step_input,
            "answer": answer,
            "success": bool(result.get("tool_success")),
        },
    ]
    trace = [
        *state.get("trace", []),
        *_tool_trace(node_name, tool_name, result),
    ]
    new_state = {
        **state,
        "answer_parts": answer_parts,
        "sources": sources,
        "flashcards": flashcards,
        "step_outputs": step_outputs,
        "last_output": answer,
        "trace": trace,
    }

    if result.get("context"):
        new_state["rag_context"] = result["context"]

    return new_state


def planner_node(state: LangGraphDemoState) -> LangGraphDemoState:
    message = state.get("message", "")
    trace = _append_trace(state, "planner: start")
    normalized = message.lower()
    use_rag = _contains_any(
        normalized,
        ["知识库", "根据资料", "根据文档", "资料", "文档", "knowledge base", "rag context"],
    )
    need_quiz = _contains_any(
        normalized,
        ["出题", "练习题", "测验", "题", "quiz", "question"],
    )
    need_flashcard = _contains_any(
        normalized,
        ["记忆卡片", "flashcard", "卡片"],
    )
    intent_parts = ["explain"]

    if use_rag:
        intent_parts.insert(0, "rag")
    if need_flashcard:
        intent_parts.append("flashcard")
    if need_quiz:
        intent_parts.append("quiz")

    plan_steps = [
        {
            "tool": tool,
            "input": message,
            "reason": "LangGraph demo rule-based planner",
        }
        for tool in intent_parts
    ]
    intent = "+".join(intent_parts)
    trace.extend([
        f"planner: intent={intent}",
        f"planner: use_rag={use_rag}",
        f"planner: need_flashcard={need_flashcard}",
        f"planner: need_quiz={need_quiz}",
        f"planner: produced {len(plan_steps)} step(s)",
    ])

    return {
        **state,
        "intent": intent,
        "use_rag": use_rag,
        "need_quiz": need_quiz,
        "need_flashcard": need_flashcard,
        "plan": plan_steps,
        "trace": trace,
    }


def route_after_planner(state: LangGraphDemoState) -> str:
    if state.get("use_rag"):
        return "rag"

    return "explain"


def rag_node(state: LangGraphDemoState) -> LangGraphDemoState:
    query = _first_plan_input(state, "rag")
    state_with_trace = {
        **state,
        "trace": _append_trace(state, f"rag: input={query}"),
    }
    result = run_registry_tool("rag", query, state_with_trace)
    return _apply_tool_result(state_with_trace, "rag", "rag", query, result)


def route_after_rag(state: LangGraphDemoState) -> str:
    return "explain"


def explain_node(state: LangGraphDemoState) -> LangGraphDemoState:
    topic = _first_plan_input(state, "explain")
    state_with_trace = {
        **state,
        "trace": _append_trace(state, f"explain: input={topic}"),
    }
    result = run_registry_tool("explain", topic, state_with_trace)
    return _apply_tool_result(state_with_trace, "explain", "explain", topic, result)


def route_after_explain(state: LangGraphDemoState) -> str:
    if state.get("need_flashcard"):
        return "flashcard"

    if state.get("need_quiz"):
        return "quiz"

    return "end"


def flashcard_node(state: LangGraphDemoState) -> LangGraphDemoState:
    topic = _first_plan_input(state, "flashcard")
    state_with_trace = {
        **state,
        "trace": _append_trace(state, f"flashcard: input={topic}"),
    }
    result = run_registry_tool("flashcard", topic, state_with_trace)
    return _apply_tool_result(state_with_trace, "flashcard", "flashcard", topic, result)


def route_after_flashcard(state: LangGraphDemoState) -> str:
    if state.get("need_quiz"):
        return "quiz"

    return "end"


def quiz_node(state: LangGraphDemoState) -> LangGraphDemoState:
    topic = _first_plan_input(state, "quiz")
    state_with_trace = {
        **state,
        "trace": _append_trace(state, f"quiz: input={topic}"),
    }
    result = run_registry_tool("quiz", topic, state_with_trace)
    return _apply_tool_result(state_with_trace, "quiz", "quiz", topic, result)


def build_demo_graph():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:
        raise LangGraphDemoUnavailableError(
            "LangGraph is not installed. Install the optional dependency with "
            "`pip install langgraph` or install project requirements."
        ) from exc

    graph_builder = StateGraph(LangGraphDemoState)
    graph_builder.add_node("planner", planner_node)
    graph_builder.add_node("rag", rag_node)
    graph_builder.add_node("explain", explain_node)
    graph_builder.add_node("flashcard", flashcard_node)
    graph_builder.add_node("quiz", quiz_node)
    graph_builder.add_edge(START, "planner")
    graph_builder.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "rag": "rag",
            "explain": "explain",
        },
    )
    graph_builder.add_conditional_edges(
        "rag",
        route_after_rag,
        {
            "explain": "explain",
        },
    )
    graph_builder.add_conditional_edges(
        "explain",
        route_after_explain,
        {
            "flashcard": "flashcard",
            "quiz": "quiz",
            "end": END,
        },
    )
    graph_builder.add_conditional_edges(
        "flashcard",
        route_after_flashcard,
        {
            "quiz": "quiz",
            "end": END,
        },
    )
    graph_builder.add_edge("quiz", END)
    return graph_builder.compile()


def run_langgraph_demo(message: str) -> dict:
    graph = build_demo_graph()
    result = graph.invoke({
        "message": message,
        "plan": [],
        "rag_context": "",
        "answer_parts": [],
        "sources": [],
        "flashcards": [],
        "step_outputs": [],
        "last_output": "",
        "trace": ["langgraph_demo: start"],
    })

    return {
        "answer": "\n\n".join(result.get("answer_parts", [])),
        "sources": result.get("sources", []),
        "trace": result.get("trace", []),
        "plan": result.get("plan", []),
        "flashcards": result.get("flashcards", []),
    }
