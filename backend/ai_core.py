from backend.agent_core import agent_router, run_agent
from backend.capabilities.chat import CHAT_MODES, run_chat
from backend.capabilities.learn import learning_workflow, run_learn
from backend.config import get_config, is_image_model
from backend.history_utils import format_history, normalize_history
from backend.image_service import generate_image
from backend.llm_service import (
    attach_usage_to_runtime_info,
    build_llm,
    normalize_model,
    track_llm_usage,
)
from backend.schemas import ChatRequest


def _plan_steps_for_response(plan: dict | None) -> list[dict]:
    if not plan or not isinstance(plan.get("steps"), list):
        return []

    return [
        {
            "tool": str(step.get("tool", "")),
            "input": str(step.get("input", "")),
            "reason": step.get("reason"),
            **({"arguments": step.get("arguments")} if step.get("arguments") else {}),
        }
        for step in plan["steps"]
        if isinstance(step, dict)
    ]


def _append_trace_block(blocks: list[dict], title: str, items: list[str]) -> None:
    filtered_items = [item for item in items if item]

    if filtered_items:
        blocks.append({
            "title": title,
            "items": filtered_items,
        })


def _group_trace_items(trace: list[str]) -> list[dict]:
    request_items = []
    rag_items = []
    route_items = []
    agent_items = []
    result_items = []
    other_items = []

    for item in trace:
        if item == "收到用户请求":
            request_items.append(item)
        elif (
            item.startswith("mode：")
            or item.startswith("model：")
            or item.startswith("temperature：")
            or item.startswith("use_rag：")
            or item.startswith("use_agent：")
            or item.startswith("top_k：")
            or item.startswith("retrieval_mode：")
            or item.startswith("session_id：")
            or item.startswith("memory 注入：")
            or item.startswith("使用 history：")
            or item.startswith("history 消息数：")
            or item.startswith("模型 ")
        ):
            request_items.append(item)
        elif item.startswith("RAG ") or item.startswith("外层 RAG"):
            rag_items.append(item)
        elif item.startswith("最终执行的模式"):
            route_items.append(item)
        elif item.startswith("Agent ") or item.startswith("Capability "):
            agent_items.append(item)
        elif item.startswith("是否启用 fallback"):
            result_items.append(item)
        else:
            other_items.append(item)

    blocks = []
    _append_trace_block(blocks, "请求参数", request_items)
    _append_trace_block(blocks, "RAG 检索", rag_items)
    _append_trace_block(blocks, "路由决策", route_items)
    _append_trace_block(blocks, "Agent 执行", agent_items)
    _append_trace_block(blocks, "执行结果", result_items)
    _append_trace_block(blocks, "其他信息", other_items)
    return blocks


def _resolve_memory_context() -> str | None:
    if not get_config().enable_memory:
        return None
    try:
        from backend.memory import get_memory_engine

        context = get_memory_engine().get_context_for_llm()
    except Exception:
        return None
    return context or None


def run_langgraph_chat_request(request: ChatRequest) -> dict:
    from backend.langgraph_runtime import run_langgraph_chat_request as run_runtime_chat_request

    return run_runtime_chat_request(request)


def run_chat_request(request: ChatRequest) -> dict:
    selected_model = normalize_model(request.model)
    if is_image_model(selected_model):
        trace = [
            "received image generation request",
            f"model: {selected_model}",
            "provider: DashScope Wanx",
        ]
        try:
            result = generate_image(request.message, selected_model)
            image_markdown = "\n".join(f"![generated image]({url})" for url in result["image_urls"])
            image_cards = [
                {
                    "front": "图卡",
                    "back": request.message,
                    "tags": ["image", selected_model],
                    "difficulty": "easy",
                    "card_type": "image",
                    "image_url": url,
                    "image_alt": request.message,
                }
                for url in result["image_urls"]
            ]
            answer = f"已生成图片：\n\n{image_markdown}"
            trace.append(f"task_id: {result['task_id']}")
            trace.append(f"api_key_source: {result['api_key_source']}")
        except RuntimeError as exc:
            answer = str(exc)
            image_cards = []
            trace.append(f"error: {exc}")

        return {
            "answer": answer,
            "mode": "image",
            "model": selected_model,
            "sources": [],
            "trace": _group_trace_items(trace),
            "plan": [],
            "flashcards": image_cards,
            "runtime_info": {},
        }

    if request.mode == "auto" and request.use_langgraph:
        return run_langgraph_chat_request(request)

    use_rag = request.use_rag or request.mode == "rag"
    history_messages = normalize_history(request.history)
    history_context = format_history(history_messages)
    trace = [
        "收到用户请求",
        f"mode：{request.mode}",
        f"model：{selected_model}",
        f"temperature：{request.temperature}",
        f"use_rag：{use_rag}",
        f"use_agent：{request.use_agent}",
        f"top_k：{request.top_k}",
        f"retrieval_mode：{request.retrieval_mode}",
        f"reranker_enabled：{request.reranker_enabled}",
        f"session_id：{request.session_id or '无'}",
        f"run_id：{request.run_id or '无'}",
        f"使用 history：{'是' if history_messages else '否'}",
        f"history 消息数：{len(history_messages)}",
    ]
    if selected_model != request.model:
        trace.append(f"模型 {request.model} 不可用，已回退到 {selected_model}")

    memory_context = _resolve_memory_context()
    trace.append(f"memory 注入：{'是' if memory_context else '否'}")

    custom_llm = track_llm_usage(
        build_llm(model=selected_model, temperature=request.temperature),
        selected_model,
    )
    sources = []
    answer = ""
    executed_mode = request.mode
    fallback_used = False
    plan = []
    flashcards = []
    pending_actions = []
    runtime_info = {}

    if request.mode == "learn":
        executed_mode = "learn"
        trace.append("最终执行的模式：learn")
        if use_rag:
            trace.append(
                f"外层 RAG 检索：跳过，交给 Learn capability 的 rag_search 执行"
                f"（retrieval_mode={request.retrieval_mode}）"
            )
        learn_result = run_learn(
            request.message,
            custom_llm=custom_llm,
            top_k=request.top_k,
            use_rag=use_rag,
            history_context=history_context,
            retrieval_mode=request.retrieval_mode,
            reranker_enabled=request.reranker_enabled,
            run_id=request.run_id,
            prefix_on_rag_miss=True,
            memory_context=memory_context,
            session_id=request.session_id,
        )
        answer = learn_result.answer
        sources = learn_result.sources
        plan = learn_result.plan
        fallback_used = fallback_used or learn_result.fallback_used
        trace.extend(learn_result.trace)
        runtime_info["capability"] = "learn"
        runtime_info.update(learn_result.retrieval_info)

    elif request.mode in CHAT_MODES:
        executed_mode = request.mode
        trace.append(f"最终执行的模式：{request.mode}")
        chat_result = run_chat(
            request.message,
            mode=request.mode,
            use_rag=use_rag,
            custom_llm=custom_llm,
            top_k=request.top_k,
            history_context=history_context,
            retrieval_mode=request.retrieval_mode,
            reranker_enabled=request.reranker_enabled,
            run_id=request.run_id,
            memory_context=memory_context,
        )
        answer = chat_result.answer
        sources = chat_result.sources
        plan = chat_result.plan
        flashcards = chat_result.flashcards
        fallback_used = fallback_used or chat_result.fallback_used
        trace.extend(chat_result.trace)
        runtime_info["capability"] = "chat"
        runtime_info["operation"] = chat_result.operation
        runtime_info.update(chat_result.retrieval_info)

    else:
        executed_mode = "agent"
        trace.append("最终执行的模式：agent")
        if use_rag:
            trace.append(
                f"外层 RAG 检索：跳过，交给 Agent rag tool 执行"
                f"（retrieval_mode={request.retrieval_mode}）"
            )
        agent_result = run_agent(
            request.message,
            custom_llm=custom_llm,
            prefer_rag=use_rag,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            reranker_enabled=request.reranker_enabled,
            history_context=history_context,
            memory_context=memory_context,
            run_id=request.run_id,
            session_id=request.session_id,
        )
        answer = agent_result["answer"]
        sources = agent_result.get("sources", [])
        plan = _plan_steps_for_response(agent_result.get("plan"))
        flashcards = agent_result.get("flashcards", [])
        runtime_info = agent_result.get("runtime_info", {})
        pending_actions = agent_result.get("pending_actions", [])
        fallback_used = fallback_used or agent_result.get("fallback_used", False)
        trace.extend(agent_result["trace"])

    trace.append(f"是否启用 fallback：{'是' if fallback_used else '否'}")

    return {
        "answer": answer,
        "mode": executed_mode,
        "model": selected_model,
        "sources": sources,
        "trace": _group_trace_items(trace),
        "plan": plan,
        "flashcards": flashcards,
        "runtime_info": attach_usage_to_runtime_info(runtime_info, custom_llm),
        "pending_actions": pending_actions,
    }
