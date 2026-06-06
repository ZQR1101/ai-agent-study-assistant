from typing import Any, Dict, List, TypedDict

from backend.llm_service import explain, generate_questions
from backend.rag_service import get_rag_context


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


def _append_trace(state: LangGraphDemoState, item: str) -> List[str]:
    return [*state.get("trace", []), item]


def _first_plan_input(state: LangGraphDemoState, tool_name: str) -> str:
    for step in state.get("plan", []):
        if step.get("tool") == tool_name and step.get("input"):
            return str(step["input"])

    return state.get("message", "")


def _contains_any(text: str, keywords: List[str]) -> bool:
    return any(keyword in text for keyword in keywords)


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
    trace = _append_trace(state, f"rag: query={query}")
    rag_result = get_rag_context(query)
    found = bool(rag_result.get("found"))
    trace.append(f"rag: found={found}")

    return {
        **state,
        "rag_context": rag_result.get("context", ""),
        "sources": rag_result.get("sources", []),
        "trace": trace,
    }


def route_after_rag(state: LangGraphDemoState) -> str:
    return "explain"


def explain_node(state: LangGraphDemoState) -> LangGraphDemoState:
    topic = _first_plan_input(state, "explain")
    context = state.get("rag_context") or None
    trace = _append_trace(state, f"explain: context={'yes' if context else 'no'}")
    answer = explain(topic, context=context)
    answer_parts = [*state.get("answer_parts", []), answer]

    return {
        **state,
        "answer_parts": answer_parts,
        "trace": trace,
    }


def route_after_explain(state: LangGraphDemoState) -> str:
    if state.get("need_flashcard"):
        return "flashcard"

    if state.get("need_quiz"):
        return "quiz"

    return "end"


def flashcard_node(state: LangGraphDemoState) -> LangGraphDemoState:
    topic = _first_plan_input(state, "flashcard")
    trace = _append_trace(state, "flashcard: start")
    source_text = state.get("rag_context") or topic
    flashcards = [
        {
            "front": f"What is the key idea of {topic}?",
            "back": "Review the explanation and retrieved context, then state the core concept in your own words.",
            "tags": ["langgraph-demo"],
            "difficulty": "medium",
        }
    ]
    answer_parts = [
        *state.get("answer_parts", []),
        f"Flashcards generated: {len(flashcards)}\nSource basis: {source_text[:120]}",
    ]
    trace.append(f"flashcard: generated {len(flashcards)} card(s)")

    return {
        **state,
        "flashcards": flashcards,
        "answer_parts": answer_parts,
        "trace": trace,
    }


def route_after_flashcard(state: LangGraphDemoState) -> str:
    if state.get("need_quiz"):
        return "quiz"

    return "end"


def quiz_node(state: LangGraphDemoState) -> LangGraphDemoState:
    topic = _first_plan_input(state, "quiz")
    context = state.get("rag_context") or None
    trace = _append_trace(state, f"quiz: context={'yes' if context else 'no'}")
    quiz = generate_questions(topic, context=context)
    answer_parts = [*state.get("answer_parts", []), quiz]

    return {
        **state,
        "answer_parts": answer_parts,
        "trace": trace,
    }


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
        "trace": ["langgraph_demo: start"],
    })

    return {
        "answer": "\n\n".join(result.get("answer_parts", [])),
        "sources": result.get("sources", []),
        "trace": result.get("trace", []),
        "plan": result.get("plan", []),
        "flashcards": result.get("flashcards", []),
    }
