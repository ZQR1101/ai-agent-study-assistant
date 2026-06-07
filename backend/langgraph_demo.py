from backend import langgraph_runtime as runtime
from backend.langgraph_runtime import (
    LangGraphAgentState,
    LangGraphRuntimeUnavailableError,
    build_langgraph_workflow,
    compose_final_answer,
    detect_intent,
    explain_node,
    finalizer_node,
    flashcard_node,
    planner_node,
    quiz_node,
    rag_node,
    route_after_flashcard,
    route_after_main_content,
    route_after_planner,
    route_after_rag,
    route_to_finalizer,
    run_langgraph_workflow,
    run_registry_tool_for_state,
    summarize_node,
)
from backend.tools import TOOL_REGISTRY


LangGraphDemoState = LangGraphAgentState
LangGraphDemoUnavailableError = LangGraphRuntimeUnavailableError


def _sync_registry_for_compat() -> None:
    runtime.TOOL_REGISTRY = TOOL_REGISTRY


def build_demo_graph():
    _sync_registry_for_compat()
    return build_langgraph_workflow()


def run_registry_tool(tool_name: str, step_input: str, state: LangGraphDemoState, custom_llm=None, top_k: int = 3) -> dict:
    _sync_registry_for_compat()
    runtime_state = {
        **state,
        "custom_llm": custom_llm,
        "top_k": top_k,
    }
    return runtime._run_registry_tool_raw(tool_name, step_input, runtime_state)


def run_langgraph_demo(message: str, custom_llm=None, top_k: int = 3) -> dict:
    _sync_registry_for_compat()
    return run_langgraph_workflow(
        message,
        custom_llm=custom_llm,
        top_k=top_k,
    )
