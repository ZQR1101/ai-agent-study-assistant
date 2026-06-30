from backend.agent_core import agent_router, run_agent
from backend.config import is_image_model
from backend.history_utils import format_history, normalize_history
from backend.image_service import generate_image
from backend.llm_service import (
    attach_usage_to_runtime_info,
    build_llm,
    chat,
    explain,
    generate_questions,
    llm,
    normalize_model,
    summarize,
    track_llm_usage,
)
from backend.rag_service import (
    LEARN_FALLBACK_PREFIX,
    NO_RAG_ANSWER,
    RAG_FALLBACK_PREFIX,
    SIMILARITY_THRESHOLD,
    append_rag_trace,
    get_rag_context,
    rag_answer,
    rag_answer_with_sources,
    rag_context_for_trace,
    with_fallback_prefix,
)
from backend.schemas import ChatRequest


def learning_workflow(
    topic: str,
    context=None,
    custom_llm=None,
    top_k: int = 3,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    use_rag: bool = True,
    history_context: str | None = None,
) -> dict:
    active_llm = custom_llm or llm
    sources = []
    highest_score = None
    threshold = similarity_threshold
    passed_threshold = False

    if context:
        knowledge = explain(
            topic,
            context=context,
            custom_llm=active_llm,
            history_context=history_context,
        )
        passed_threshold = True
    elif use_rag:
        rag_context = get_rag_context(topic, top_k=top_k, score_threshold=similarity_threshold)
        sources = rag_context["sources"]
        highest_score = rag_context["max_score"]
        threshold = rag_context["threshold"]
        passed_threshold = rag_context["found"]

        if rag_context["found"]:
            knowledge = explain(
                topic,
                context=rag_context["context"],
                custom_llm=active_llm,
                history_context=history_context,
            )
        else:
            knowledge = explain(topic, custom_llm=active_llm, history_context=history_context)
    else:
        knowledge = explain(topic, custom_llm=active_llm, history_context=history_context)

    summary = summarize(knowledge, custom_llm=active_llm, history_context=history_context)
    quiz = generate_questions(knowledge, custom_llm=active_llm, history_context=history_context)

    advice_prompt = f"""
请根据下面内容，给出简短的下一步学习建议，不超过 3 条：

最近对话：
{history_context or "无"}

{summary}
"""
    advice = active_llm.invoke(advice_prompt).content

    return {
        "knowledge": knowledge,
        "summary": summary,
        "quiz": quiz,
        "advice": advice,
        "sources": sources,
        "highest_score": highest_score,
        "threshold": threshold,
        "passed_threshold": passed_threshold,
    }


def _format_learning_result(result: dict) -> str:
    parts = [f"知识内容：\n{result.get('knowledge', '')}"]

    if result.get("summary"):
        parts.append(f"总结：\n{result['summary']}")

    if result.get("quiz"):
        parts.append(f"练习题：\n{result['quiz']}")

    if result.get("advice"):
        parts.append(f"学习建议：\n{result['advice']}")

    return "\n\n".join(parts)


def _plan_steps_for_response(plan: dict | None) -> list[dict]:
    if not plan or not isinstance(plan.get("steps"), list):
        return []

    return [
        {
            "tool": str(step.get("tool", "")),
            "input": str(step.get("input", "")),
            "reason": step.get("reason"),
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
            or item.startswith("使用 history：")
            or item.startswith("history 消息数：")
            or item.startswith("模型 ")
        ):
            request_items.append(item)
        elif item.startswith("RAG ") or item.startswith("外层 RAG"):
            rag_items.append(item)
        elif item.startswith("最终执行的模式"):
            route_items.append(item)
        elif item.startswith("Agent "):
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
    agent_handles_rag = request.mode == "auto" and request.use_agent
    history_messages = normalize_history(request.history)
    history_context = format_history(history_messages)
    rag_question = (
        f"历史对话：\n{history_context}\n\n当前问题：{request.message}"
        if history_context
        else request.message
    )
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
        f"使用 history：{'是' if history_messages else '否'}",
        f"history 消息数：{len(history_messages)}",
    ]
    if selected_model != request.model:
        trace.append(f"模型 {request.model} 不可用，已回退到 {selected_model}")

    custom_llm = track_llm_usage(
        build_llm(model=selected_model, temperature=request.temperature),
        selected_model,
    )
    sources = []
    answer = ""
    executed_mode = request.mode
    fallback_used = False
    rag_context = None
    plan = []
    flashcards = []
    runtime_info = {}

    if use_rag and not agent_handles_rag:
        rag_context = get_rag_context(
            rag_question,
            request.top_k,
            retrieval_mode=request.retrieval_mode,
            reranker_enabled=request.reranker_enabled,
        )
        trace.append(f"RAG query 使用 history：{'是' if history_context else '否'}")
        append_rag_trace(trace, rag_context_for_trace(rag_context, request.top_k))
        runtime_info.update({
            "retrieval_mode": rag_context.get("retrieval_mode", request.retrieval_mode),
            "candidate_k": rag_context.get("candidate_k"),
            "vector_candidates": rag_context.get("vector_candidates", 0),
            "bm25_candidates": rag_context.get("bm25_candidates", 0),
            "hybrid_used": rag_context.get("hybrid_used", False),
            "reranker_enabled": rag_context.get("reranker_enabled", False),
            "reranker_used": rag_context.get("reranker_used", False),
            "reranker_model": rag_context.get("reranker_model"),
            "reranker_top_n": rag_context.get("reranker_top_n"),
            "reranker_error": rag_context.get("reranker_error"),
        })
    elif use_rag and agent_handles_rag:
        trace.append(f"外层 RAG 检索：跳过，交给 Agent rag tool 执行（retrieval_mode={request.retrieval_mode}）")

    if request.mode == "rag":
        executed_mode = "rag"
        trace.append("最终执行的模式：rag")

        if rag_context and rag_context["found"]:
            answer = chat(
                request.message,
                context=rag_context["context"],
                custom_llm=custom_llm,
                history_context=history_context,
            )
            sources = rag_context["sources"]
        else:
            answer = NO_RAG_ANSWER

    elif request.mode == "chat":
        executed_mode = "chat"
        trace.append("最终执行的模式：chat")

        if rag_context and rag_context["found"]:
            answer = chat(
                request.message,
                context=rag_context["context"],
                custom_llm=custom_llm,
                history_context=history_context,
            )
            sources = rag_context["sources"]
        else:
            answer = chat(request.message, custom_llm=custom_llm, history_context=history_context)
            if use_rag and rag_context and not rag_context["found"]:
                fallback_used = True
                answer = with_fallback_prefix(answer, RAG_FALLBACK_PREFIX)

    elif request.mode == "explain":
        executed_mode = "explain"
        trace.append("最终执行的模式：explain")

        if rag_context and rag_context["found"]:
            answer = explain(
                request.message,
                context=rag_context["context"],
                custom_llm=custom_llm,
                history_context=history_context,
            )
            sources = rag_context["sources"]
        else:
            answer = explain(request.message, custom_llm=custom_llm, history_context=history_context)
            if use_rag and rag_context and not rag_context["found"]:
                fallback_used = True
                answer = with_fallback_prefix(answer, RAG_FALLBACK_PREFIX)

    elif request.mode == "summarize":
        executed_mode = "summarize"
        trace.append("最终执行的模式：summarize")

        if rag_context and rag_context["found"]:
            answer = summarize(
                request.message,
                context=rag_context["context"],
                custom_llm=custom_llm,
                history_context=history_context,
            )
            sources = rag_context["sources"]
        else:
            answer = summarize(request.message, custom_llm=custom_llm, history_context=history_context)
            if use_rag and rag_context and not rag_context["found"]:
                fallback_used = True
                answer = with_fallback_prefix(answer, RAG_FALLBACK_PREFIX)

    elif request.mode == "quiz":
        executed_mode = "quiz"
        trace.append("最终执行的模式：quiz")

        if rag_context and rag_context["found"]:
            answer = generate_questions(
                request.message,
                context=rag_context["context"],
                custom_llm=custom_llm,
                history_context=history_context,
            )
            sources = rag_context["sources"]
        else:
            answer = generate_questions(
                request.message,
                custom_llm=custom_llm,
                history_context=history_context,
            )
            if use_rag and rag_context and not rag_context["found"]:
                fallback_used = True
                answer = with_fallback_prefix(answer, RAG_FALLBACK_PREFIX)

    elif request.mode == "learn":
        executed_mode = "learn"
        trace.append("最终执行的模式：learn")

        if rag_context and rag_context["found"]:
            result = learning_workflow(
                request.message,
                context=rag_context["context"],
                custom_llm=custom_llm,
                use_rag=False,
                history_context=history_context,
            )
            sources = rag_context["sources"]
        else:
            result = learning_workflow(
                request.message,
                custom_llm=custom_llm,
                use_rag=False,
                history_context=history_context,
            )
            if use_rag and rag_context and not rag_context["found"]:
                fallback_used = True
                result["knowledge"] = with_fallback_prefix(
                    result["knowledge"],
                    LEARN_FALLBACK_PREFIX,
                )

        answer = _format_learning_result(result)

    elif request.mode == "auto" or request.use_agent:
        executed_mode = "agent"
        trace.append("最终执行的模式：agent")
        agent_result = run_agent(
            request.message,
            custom_llm=custom_llm,
            prefer_rag=use_rag,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            reranker_enabled=request.reranker_enabled,
            history_context=history_context,
        )
        answer = agent_result["answer"]
        sources = agent_result.get("sources", [])
        plan = _plan_steps_for_response(agent_result.get("plan"))
        flashcards = agent_result.get("flashcards", [])
        runtime_info = agent_result.get("runtime_info", {})
        fallback_used = fallback_used or agent_result.get("fallback_used", False)
        trace.extend(agent_result["trace"])

    else:
        executed_mode = "agent"
        trace.append("最终执行的模式：agent")
        agent_result = run_agent(
            request.message,
            custom_llm=custom_llm,
            prefer_rag=use_rag,
            top_k=request.top_k,
            retrieval_mode=request.retrieval_mode,
            reranker_enabled=request.reranker_enabled,
            history_context=history_context,
        )
        answer = agent_result["answer"]
        sources = agent_result.get("sources", [])
        plan = _plan_steps_for_response(agent_result.get("plan"))
        flashcards = agent_result.get("flashcards", [])
        runtime_info = agent_result.get("runtime_info", {})
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
    }
