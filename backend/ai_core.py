import json
import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from backend.rag_store import SIMILARITY_THRESHOLD, search_relevant_chunks
from backend.schemas import ChatRequest

load_dotenv()

DEFAULT_MODEL = "mimo-v2.5"
DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"
SUPPORTED_MODELS = {DEFAULT_MODEL}
SOURCE_SNIPPET_LENGTH = 400

NO_RAG_ANSWER = "知识库中没有找到与该问题相关的内容。你可以上传相关资料，或切换到普通聊天模式。"
RAG_FALLBACK_PREFIX = "知识库中没有找到相关内容，以下内容未使用知识库，仅基于模型通用知识生成。"
LEARN_FALLBACK_PREFIX = "知识库中没有找到相关内容，以下学习内容未使用知识库，仅基于模型通用知识生成。"


def normalize_model(model: str | None) -> str:
    if model in SUPPORTED_MODELS:
        return model
    return DEFAULT_MODEL


def build_llm(model: str = DEFAULT_MODEL, temperature: float = 0.7, max_tokens: int = 2000):
    api_key = (
        os.getenv("MY_MIMO_API_KEY")
        or os.getenv("MIMO_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )
    selected_model = normalize_model(model)

    return ChatOpenAI(
        api_key=api_key,
        base_url=os.getenv("MIMO_BASE_URL", DEFAULT_BASE_URL),
        model=selected_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )


llm = build_llm()


def _format_score(score) -> str:
    if score is None:
        return "无"
    return f"{score:.4f}"


def _truncate_text(text: str, max_length: int = SOURCE_SNIPPET_LENGTH) -> str:
    clean_text = " ".join(str(text or "").split())
    if len(clean_text) <= max_length:
        return clean_text
    return clean_text[:max_length].rstrip() + "..."


def _context_prompt(task: str, text: str, context: str) -> str:
    return f"""
请优先根据下面的知识库内容完成任务。
如果知识库内容不足以回答，再说明不足之处。

知识库内容：
{context}

任务：
{task}

用户输入：
{text}
"""


def get_rag_context(
    question: str,
    top_k: int = 3,
    score_threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    search_result = search_relevant_chunks(
        question,
        top_k=top_k,
        similarity_threshold=score_threshold,
        include_metadata=True,
    )
    chunks = search_result["chunks"]
    max_score = search_result["highest_score"]

    if not chunks or max_score is None or max_score < score_threshold:
        return {
            "found": False,
            "context": "",
            "sources": [],
            "max_score": max_score,
            "threshold": score_threshold,
        }

    context_parts = []
    source_chunks = []

    for chunk in chunks:
        context_parts.append(
            f"来源文件：{chunk['source']}\n"
            f"相似度：{chunk['score']:.4f}\n"
            f"内容：\n{chunk['text']}"
        )
        source_chunks.append({
            "source": chunk["source"],
            "score": float(chunk["score"]),
            "snippet": _truncate_text(chunk["text"]),
            "text": _truncate_text(chunk["text"]),
        })

    return {
        "found": True,
        "context": "\n\n---\n\n".join(context_parts),
        "sources": source_chunks,
        "max_score": max_score,
        "threshold": score_threshold,
    }


def chat(text: str, context=None, custom_llm=None) -> str:
    active_llm = custom_llm or llm
    prompt = text

    if context:
        prompt = _context_prompt("回答用户问题", text, context)

    response = active_llm.invoke(prompt)
    return response.content


def explain(text: str, context=None, custom_llm=None) -> str:
    active_llm = custom_llm or llm

    if context:
        prompt = _context_prompt("请用简单易懂的中文解释用户输入", text, context)
    else:
        prompt = f"请用简单易懂的中文解释：\n{text}"

    response = active_llm.invoke(prompt)
    return response.content


def summarize(text: str, context=None, custom_llm=None) -> str:
    active_llm = custom_llm or llm

    if context:
        prompt = _context_prompt("请总结与用户输入相关的知识库内容", text, context)
    else:
        prompt = f"请总结以下内容。如果内容很短或像一个主题，请先解释它的含义，再做简短总结：\n{text}"

    response = active_llm.invoke(prompt)
    return response.content


def generate_questions(text: str, context=None, custom_llm=None) -> str:
    active_llm = custom_llm or llm

    if context:
        prompt = _context_prompt("请基于知识库内容出 3 道练习题，并给出答案", text, context)
    else:
        prompt = f"请根据以下主题或知识点出 3 道练习题，并给出答案：\n{text}"

    response = active_llm.invoke(prompt)
    return response.content


def rag_answer_with_sources(
    question: str,
    custom_llm=None,
    top_k: int = 3,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> dict:
    rag_context = get_rag_context(
        question,
        top_k=top_k,
        score_threshold=similarity_threshold,
    )

    if not rag_context["found"]:
        return {
            "answer": NO_RAG_ANSWER,
            "sources": [],
            "highest_score": rag_context["max_score"],
            "threshold": rag_context["threshold"],
            "passed_threshold": False,
        }

    answer = chat(question, context=rag_context["context"], custom_llm=custom_llm)

    return {
        "answer": answer,
        "sources": rag_context["sources"],
        "highest_score": rag_context["max_score"],
        "threshold": rag_context["threshold"],
        "passed_threshold": True,
    }


def rag_answer(
    question: str,
    custom_llm=None,
    top_k: int = 3,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
) -> str:
    result = rag_answer_with_sources(
        question,
        custom_llm=custom_llm,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
    )
    source_text = "\n".join([
        f"- {source.get('source')} ({_format_score(source.get('score'))})"
        for source in result["sources"]
    ])

    return f"""
{result["answer"]}

---

参考来源：
{source_text}
"""


def _extract_json_object(text: str) -> dict | None:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None

    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _fallback_agent_plan(user_input: str, reason: str = "planner json parse failed") -> dict:
    lowered = user_input.lower()

    if any(word in lowered for word in ["quiz", "题", "练习", "测试"]):
        tool = "quiz"
    elif any(word in lowered for word in ["summary", "summarize", "总结", "摘要"]):
        tool = "summarize"
    elif any(word in lowered for word in ["rag", "知识库", "根据我的资料", "根据文档"]):
        tool = "rag"
    elif any(word in lowered for word in ["解释", "什么是", "what is", "why"]):
        tool = "explain"
    else:
        tool = "chat"

    return {
        "goal": "根据用户请求选择最合适的学习助手能力。",
        "fallback": True,
        "fallback_reason": reason,
        "steps": [
            {
                "tool": tool,
                "input": user_input,
                "reason": "使用本地规则生成的 fallback 单步计划。",
            }
        ],
    }


def plan_agent_steps(user_input: str, custom_llm=None) -> dict:
    active_llm = custom_llm or llm
    planner_prompt = f"""
你是 AI 学习助手的 Planner。
请把用户请求拆成 1 到 3 个执行步骤，并只返回 JSON，不要返回 Markdown。

可用工具：
- chat：普通回答
- explain：解释概念
- summarize：总结内容
- quiz：生成练习题
- rag：查询本地知识库

JSON 格式：
{{
  "goal": "用户目标",
  "steps": [
    {{
      "tool": "chat|explain|summarize|quiz|rag",
      "input": "传给工具的输入",
      "reason": "为什么使用这个工具"
    }}
  ]
}}

用户请求：
{user_input}
"""
    response = active_llm.invoke(planner_prompt)
    plan = _extract_json_object(response.content)

    if not plan or not isinstance(plan.get("steps"), list) or not plan["steps"]:
        return _fallback_agent_plan(user_input)

    allowed_tools = {"chat", "explain", "summarize", "quiz", "rag"}
    normalized_steps = []

    for step in plan["steps"][:3]:
        if not isinstance(step, dict):
            continue

        tool = str(step.get("tool", "")).strip().lower()
        if tool not in allowed_tools:
            continue

        normalized_steps.append({
            "tool": tool,
            "input": str(step.get("input") or user_input),
            "reason": str(step.get("reason") or "planner selected this tool"),
        })

    if not normalized_steps:
        return _fallback_agent_plan(user_input, reason="planner returned no valid tools")

    return {
        "goal": str(plan.get("goal") or "完成用户请求"),
        "fallback": False,
        "steps": normalized_steps,
    }


def _execute_agent_tool(tool: str, tool_input: str, custom_llm=None, top_k: int = 3) -> dict:
    active_llm = custom_llm or llm

    if tool == "chat":
        return {
            "answer": chat(tool_input, custom_llm=active_llm),
            "sources": [],
            "trace": [],
            "fallback_used": False,
        }

    if tool == "explain":
        return {
            "answer": explain(tool_input, custom_llm=active_llm),
            "sources": [],
            "trace": [],
            "fallback_used": False,
        }

    if tool == "summarize":
        return {
            "answer": summarize(tool_input, custom_llm=active_llm),
            "sources": [],
            "trace": [],
            "fallback_used": False,
        }

    if tool == "quiz":
        return {
            "answer": generate_questions(tool_input, custom_llm=active_llm),
            "sources": [],
            "trace": [],
            "fallback_used": False,
        }

    if tool == "rag":
        rag_context = get_rag_context(tool_input, top_k=top_k)
        rag_sources = rag_context.get("sources", [])
        trace = [
            f"RAG query：{tool_input}",
            f"RAG max_score：{_format_score(rag_context.get('max_score'))}",
            f"RAG threshold：{_format_score(rag_context.get('threshold'))}",
            f"RAG 是否命中：{'是' if rag_context.get('found') else '否'}",
            f"RAG sources：{_source_names(rag_sources)}",
        ]

        if rag_context.get("found"):
            return {
                "answer": chat(tool_input, context=rag_context["context"], custom_llm=active_llm),
                "sources": rag_sources,
                "trace": trace,
                "fallback_used": False,
            }

        trace.append("Agent RAG 未命中，未使用知识库来源")
        answer = _with_fallback_prefix(
            chat(tool_input, custom_llm=active_llm),
            RAG_FALLBACK_PREFIX,
        )
        return {
            "answer": answer,
            "sources": [],
            "trace": trace,
            "fallback_used": True,
        }

    return {
        "answer": chat(tool_input, custom_llm=active_llm),
        "sources": [],
        "trace": [],
        "fallback_used": False,
    }


def run_agent(user_input: str, custom_llm=None, prefer_rag: bool = False, top_k: int = 3) -> dict:
    active_llm = custom_llm or llm
    trace = ["Agent Planner：开始分析用户请求"]
    plan = plan_agent_steps(user_input, custom_llm=active_llm)

    trace.append(f"Agent Planner goal：{plan.get('goal')}")
    trace.append(f"Agent Planner fallback：{'是' if plan.get('fallback') else '否'}")
    if plan.get("fallback_reason"):
        trace.append(f"Agent Planner fallback reason：{plan['fallback_reason']}")

    if prefer_rag and not any(step.get("tool") == "rag" for step in plan["steps"]):
        plan["steps"].insert(0, {
            "tool": "rag",
            "input": user_input,
            "reason": "use_rag=true，优先尝试知识库检索",
        })
        plan["steps"] = plan["steps"][:3]
        trace.append("Agent Planner：use_rag=true，已插入 rag step")

    previous_result = ""
    step_outputs = []
    all_sources = []
    fallback_used = False

    for index, step in enumerate(plan["steps"], start=1):
        tool = step["tool"]
        tool_input = step.get("input") or user_input
        if "{previous_result}" in tool_input:
            tool_input = tool_input.replace("{previous_result}", previous_result)

        trace.append(f"Agent Step {index} tool={tool}")
        trace.append(f"Agent Step {index} plan：tool={tool}, reason={step.get('reason')}")
        result = _execute_agent_tool(tool, tool_input, custom_llm=active_llm, top_k=top_k)
        result_answer = result.get("answer", "")
        result_sources = result.get("sources", [])
        previous_result = result_answer
        all_sources.extend(result_sources)
        fallback_used = fallback_used or result.get("fallback_used", False)
        trace.extend(result.get("trace", []))
        step_outputs.append(f"步骤 {index}（{tool}）：\n{result_answer}")
        trace.append(f"Agent Step {index} done：输出长度 {len(result_answer)}")

    answer = "\n\n".join(step_outputs) if step_outputs else chat(user_input, custom_llm=active_llm)

    return {
        "answer": answer,
        "trace": trace,
        "plan": plan,
        "sources": all_sources,
        "fallback_used": fallback_used,
    }


def agent_router(user_input: str, custom_llm=None) -> str:
    return run_agent(user_input, custom_llm=custom_llm)["answer"]


def learning_workflow(
    topic: str,
    context=None,
    custom_llm=None,
    top_k: int = 3,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    use_rag: bool = True,
) -> dict:
    active_llm = custom_llm or llm
    sources = []
    highest_score = None
    threshold = similarity_threshold
    passed_threshold = False

    if context:
        knowledge = explain(topic, context=context, custom_llm=active_llm)
        passed_threshold = True
    elif use_rag:
        rag_context = get_rag_context(topic, top_k=top_k, score_threshold=similarity_threshold)
        sources = rag_context["sources"]
        highest_score = rag_context["max_score"]
        threshold = rag_context["threshold"]
        passed_threshold = rag_context["found"]

        if rag_context["found"]:
            knowledge = explain(topic, context=rag_context["context"], custom_llm=active_llm)
        else:
            knowledge = explain(topic, custom_llm=active_llm)
    else:
        knowledge = explain(topic, custom_llm=active_llm)

    summary = summarize(knowledge, custom_llm=active_llm)
    quiz = generate_questions(knowledge, custom_llm=active_llm)

    advice_prompt = f"""
请根据下面内容，给出简短的下一步学习建议，不超过 3 条：

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


def _source_names(sources: list[dict]) -> list[str]:
    return sorted(set(source.get("source", "") for source in sources if source.get("source")))


def _append_rag_trace(trace: list[str], rag_context: dict | None) -> None:
    if not rag_context:
        return

    trace.append(f"RAG top_k：{rag_context.get('top_k')}")
    trace.append(f"RAG max_score：{_format_score(rag_context.get('max_score'))}")
    trace.append(f"RAG 阈值：{_format_score(rag_context.get('threshold'))}")
    trace.append(f"RAG 是否通过阈值：{'是' if rag_context.get('found') else '否'}")
    trace.append(f"RAG sources：{_source_names(rag_context.get('sources', []))}")


def _rag_context_for_trace(rag_context: dict, top_k: int) -> dict:
    return {
        **rag_context,
        "top_k": top_k,
    }


def _with_fallback_prefix(answer: str, prefix: str) -> str:
    return f"{prefix}\n\n{answer}"


def run_chat_request(request: ChatRequest) -> dict:
    use_rag = request.use_rag or request.mode == "rag"
    agent_handles_rag = request.mode == "auto" and request.use_agent
    selected_model = normalize_model(request.model)
    trace = [
        "收到用户请求",
        f"mode：{request.mode}",
        f"model：{selected_model}",
        f"temperature：{request.temperature}",
        f"use_rag：{use_rag}",
    ]
    if selected_model != request.model:
        trace.append(f"模型 {request.model} 不可用，已回退到 {selected_model}")

    custom_llm = build_llm(model=selected_model, temperature=request.temperature)
    sources = []
    answer = ""
    executed_mode = request.mode
    fallback_used = False
    rag_context = None

    if use_rag and not agent_handles_rag:
        rag_context = get_rag_context(request.message, request.top_k)
        _append_rag_trace(trace, _rag_context_for_trace(rag_context, request.top_k))
    elif use_rag and agent_handles_rag:
        trace.append("外层 RAG 检索：跳过，交给 Agent rag tool 执行")

    if request.mode == "rag":
        executed_mode = "rag"
        trace.append("最终执行的模式：rag")

        if rag_context and rag_context["found"]:
            answer = chat(request.message, context=rag_context["context"], custom_llm=custom_llm)
            sources = rag_context["sources"]
        else:
            answer = NO_RAG_ANSWER

    elif request.mode == "chat":
        executed_mode = "chat"
        trace.append("最终执行的模式：chat")

        if rag_context and rag_context["found"]:
            answer = chat(request.message, context=rag_context["context"], custom_llm=custom_llm)
            sources = rag_context["sources"]
        else:
            answer = chat(request.message, custom_llm=custom_llm)
            if use_rag and rag_context and not rag_context["found"]:
                fallback_used = True
                answer = _with_fallback_prefix(answer, RAG_FALLBACK_PREFIX)

    elif request.mode == "explain":
        executed_mode = "explain"
        trace.append("最终执行的模式：explain")

        if rag_context and rag_context["found"]:
            answer = explain(request.message, context=rag_context["context"], custom_llm=custom_llm)
            sources = rag_context["sources"]
        else:
            answer = explain(request.message, custom_llm=custom_llm)
            if use_rag and rag_context and not rag_context["found"]:
                fallback_used = True
                answer = _with_fallback_prefix(answer, RAG_FALLBACK_PREFIX)

    elif request.mode == "summarize":
        executed_mode = "summarize"
        trace.append("最终执行的模式：summarize")

        if rag_context and rag_context["found"]:
            answer = summarize(request.message, context=rag_context["context"], custom_llm=custom_llm)
            sources = rag_context["sources"]
        else:
            answer = summarize(request.message, custom_llm=custom_llm)
            if use_rag and rag_context and not rag_context["found"]:
                fallback_used = True
                answer = _with_fallback_prefix(answer, RAG_FALLBACK_PREFIX)

    elif request.mode == "quiz":
        executed_mode = "quiz"
        trace.append("最终执行的模式：quiz")

        if rag_context and rag_context["found"]:
            answer = generate_questions(
                request.message,
                context=rag_context["context"],
                custom_llm=custom_llm,
            )
            sources = rag_context["sources"]
        else:
            answer = generate_questions(request.message, custom_llm=custom_llm)
            if use_rag and rag_context and not rag_context["found"]:
                fallback_used = True
                answer = _with_fallback_prefix(answer, RAG_FALLBACK_PREFIX)

    elif request.mode == "learn":
        executed_mode = "learn"
        trace.append("最终执行的模式：learn")

        if rag_context and rag_context["found"]:
            result = learning_workflow(
                request.message,
                context=rag_context["context"],
                custom_llm=custom_llm,
                use_rag=False,
            )
            sources = rag_context["sources"]
        else:
            result = learning_workflow(
                request.message,
                custom_llm=custom_llm,
                use_rag=False,
            )
            if use_rag and rag_context and not rag_context["found"]:
                fallback_used = True
                result["knowledge"] = _with_fallback_prefix(
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
        )
        answer = agent_result["answer"]
        sources = agent_result.get("sources", [])
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
        )
        answer = agent_result["answer"]
        sources = agent_result.get("sources", [])
        fallback_used = fallback_used or agent_result.get("fallback_used", False)
        trace.extend(agent_result["trace"])

    trace.append(f"是否启用 fallback：{'是' if fallback_used else '否'}")

    return {
        "answer": answer,
        "mode": executed_mode,
        "model": selected_model,
        "sources": sources,
        "trace": trace,
    }
