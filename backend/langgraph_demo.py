from backend import langgraph_runtime as runtime
from backend.langgraph_runtime import (
    LangGraphAgentState,
    LangGraphRuntimeUnavailableError,
    build_langgraph_workflow,
    chat_node,
    compose_final_answer,
    detect_intent,
    finalizer_node,
    planner_node,
    rag_search_node,
    route_after_chat,
    route_after_planner,
    route_after_rag_search,
    route_after_study,
    route_to_finalizer,
    run_langgraph_workflow,
    run_registry_tool_for_state,
    study_node,
)
from backend.tools import TOOL_REGISTRY


LangGraphDemoState = LangGraphAgentState
LangGraphDemoUnavailableError = LangGraphRuntimeUnavailableError


def _sync_registry_for_compat() -> None:
    runtime.TOOL_REGISTRY = TOOL_REGISTRY


def build_demo_graph():
    _sync_registry_for_compat()
    return build_langgraph_workflow()


def run_registry_tool(
    tool_name: str,
    step_input: str,
    state: LangGraphDemoState,
    custom_llm=None,
    top_k: int = 3,
    operation: str | None = None,
    generate_answer: bool | None = None,
) -> dict:
    _sync_registry_for_compat()
    runtime_state = {
        **state,
        "custom_llm": custom_llm,
        "top_k": top_k,
    }
    return runtime._run_registry_tool_raw(
        tool_name,
        step_input,
        runtime_state,
        operation=operation,
        generate_answer=generate_answer,
    )


def run_langgraph_demo(message: str, custom_llm=None, top_k: int = 3) -> dict:
    _sync_registry_for_compat()
    return run_langgraph_workflow(
        message,
        custom_llm=custom_llm,
        top_k=top_k,
    )
