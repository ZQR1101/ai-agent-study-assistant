import json
import os
import re

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from backend.rag_store import SIMILARITY_THRESHOLD, search_relevant_chunks
from backend.schemas import AgentPlan, ChatRequest, FlashcardPayload
from backend.tools import ToolSpec

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
    expanded_query = search_result.get("expanded_query", question)
    raw_count = search_result.get("raw_count", 0)
    valid_count = search_result.get("valid_count", len(chunks))
    discarded_invalid_count = search_result.get("discarded_invalid_count", 0)

    if not chunks or max_score is None or max_score < score_threshold:
        return {
            "found": False,
            "context": "",
            "sources": [],
            "max_score": max_score,
            "threshold": score_threshold,
            "expanded_query": expanded_query,
            "raw_count": raw_count,
            "valid_count": valid_count,
            "discarded_invalid_count": discarded_invalid_count,
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
        "expanded_query": expanded_query,
        "raw_count": raw_count,
        "valid_count": valid_count,
        "discarded_invalid_count": discarded_invalid_count,
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


def _validate_flashcard_payload(data: dict) -> dict:
    if hasattr(FlashcardPayload, "model_validate"):
        payload = FlashcardPayload.model_validate(data)
    else:
        payload = FlashcardPayload.parse_obj(data)

    return _model_to_dict(payload)


def _tool_descriptions_for_prompt() -> str:
    return "\n".join(
        f"- {tool.name}：{tool.description}"
        for tool in TOOL_REGISTRY.values()
    )


def _tool_names_for_prompt() -> str:
    return "|".join(TOOL_REGISTRY.keys())


def _fallback_agent_plan(user_input: str, reason: str = "planner json parse failed") -> dict:
    lowered = user_input.lower()

    if any(word in lowered for word in ["flashcard", "卡片", "记忆卡", "抽认卡"]):
        tool = "flashcard"
    elif any(word in lowered for word in ["quiz", "题", "练习", "测试"]):
        tool = "quiz"
    elif any(word in lowered for word in ["summary", "summarize", "总结", "摘要"]):
        tool = "summarize"
    elif any(word in lowered for word in ["rag", "知识库", "根据我的资料", "根据文档"]):
        tool = "rag"
    elif any(word in lowered for word in ["解释", "什么是", "what is", "why"]):
        tool = "explain"
    else:
        tool = "chat"

    if tool not in TOOL_REGISTRY:
        tool = "chat"

    fallback_plan = {
        "goal": "根据用户请求选择最合适的学习助手能力。",
        "fallback": True,
        "steps": [
            {
                "tool": tool,
                "input": user_input,
                "reason": "使用本地规则生成的 fallback 单步计划。",
            }
        ],
    }
    validated_plan = _validate_agent_plan(fallback_plan)
    validated_plan["fallback_reason"] = reason
    validated_plan["planner_json_parse"] = "失败"
    validated_plan["planner_schema_validate"] = "失败"
    return validated_plan


def plan_agent_steps(user_input: str, custom_llm=None) -> dict:
    active_llm = custom_llm or llm
    planner_prompt = f"""
你是 AI 学习助手的 Planner。
请把用户请求拆成 1 到 4 个执行步骤，并只返回 JSON object。

可用工具：
{_tool_descriptions_for_prompt()}

AgentPlan schema：
{{
  "goal": "用户任务目标，非空字符串",
  "steps": [
    {{
      "tool": "{_tool_names_for_prompt()}",
      "input": "传给工具的输入，非空字符串",
      "reason": "为什么使用这个工具"
    }}
  ],
  "fallback": false
}}

工具名只能是：{', '.join(TOOL_REGISTRY.keys())}。

示例 1：
用户输入：什么是 RAG
输出：
{{
  "goal": "解释 RAG 的概念",
  "steps": [
    {{
      "tool": "explain",
      "input": "RAG",
      "reason": "用户询问概念定义，使用 explain 工具解释"
    }}
  ],
  "fallback": false
}}

示例 2：
用户输入：请解释 RAG，并出 3 道练习题
输出：
{{
  "goal": "解释 RAG 并生成练习题",
  "steps": [
    {{
      "tool": "explain",
      "input": "RAG",
      "reason": "先解释概念"
    }},
    {{
      "tool": "quiz",
      "input": "基于 RAG 生成 3 道练习题",
      "reason": "用户要求出题"
    }}
  ],
  "fallback": false
}}

示例 3：
用户输入：根据知识库解释 agentic rag，并出 3 道练习题
输出：
{{
  "goal": "基于知识库解释 agentic rag 并生成练习题",
  "steps": [
    {{
      "tool": "rag",
      "input": "agentic rag",
      "reason": "用户要求根据知识库回答，先检索相关内容"
    }},
    {{
      "tool": "explain",
      "input": "基于知识库内容解释 agentic rag",
      "reason": "解释检索到的概念"
    }},
    {{
      "tool": "quiz",
      "input": "基于 agentic rag 生成 3 道练习题",
      "reason": "用户要求生成练习题"
    }}
  ],
  "fallback": false
}}

示例 4：
用户输入：根据知识库解释 agentic rag，生成记忆卡片，并出 3 道练习题
输出：
{{
  "goal": "基于知识库解释 agentic rag，生成记忆卡片并出题",
  "steps": [
    {{
      "tool": "rag",
      "input": "agentic rag",
      "reason": "用户要求根据知识库回答，先检索相关内容"
    }},
    {{
      "tool": "explain",
      "input": "基于知识库内容解释 agentic rag",
      "reason": "先帮助用户理解概念"
    }},
    {{
      "tool": "flashcard",
      "input": "基于 agentic rag 生成 3-5 张复习记忆卡片",
      "reason": "用户要求生成记忆卡片用于复习"
    }},
    {{
      "tool": "quiz",
      "input": "基于 agentic rag 生成 3 道练习题",
      "reason": "用户要求生成练习题"
    }}
  ],
  "fallback": false
}}

当前用户请求：
{user_input}

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
        return _fallback_agent_plan(user_input, reason="planner json parse failed")

    try:
        plan = _validate_agent_plan(data)
    except ValidationError as exc:
        fallback_plan = _fallback_agent_plan(
            user_input,
            reason=f"planner schema validate failed: {exc.errors()[0].get('msg')}",
        )
        fallback_plan["planner_json_parse"] = "成功"
        return fallback_plan

    if not plan.get("steps"):
        fallback_plan = _fallback_agent_plan(user_input, reason="planner returned empty steps")
        fallback_plan["planner_json_parse"] = "成功"
        return fallback_plan

    if any(step.get("tool") not in TOOL_REGISTRY for step in plan["steps"]):
        fallback_plan = _fallback_agent_plan(user_input, reason="planner returned unknown tool")
        fallback_plan["planner_json_parse"] = "成功"
        return fallback_plan

    plan["steps"] = plan["steps"][:4]
    plan["fallback"] = False
    plan["planner_json_parse"] = "成功"
    plan["planner_schema_validate"] = "成功"
    return plan


def _is_valid_agent_context(value: str | None) -> bool:
    text = str(value or "").strip()

    if not text:
        return False

    invalid_markers = [
        NO_RAG_ANSWER,
        "知识库中没有找到相关内容",
        "知识库中没有找到与该问题相关的内容",
    ]
    return not any(marker in text for marker in invalid_markers)


def _build_agent_tool_context(shared_context: dict | None) -> tuple[str | None, list[str]]:
    if not shared_context:
        return None, []

    context_parts = []
    context_sources = []
    rag_context = shared_context.get("rag_context", "")
    last_output = shared_context.get("last_output", "")

    if _is_valid_agent_context(rag_context):
        context_parts.append(f"【知识库内容】\n{rag_context}")
        context_sources.append("rag_context")

    if _is_valid_agent_context(last_output):
        context_parts.append(f"【上一步输出】\n{last_output}")
        context_sources.append("previous_step_output")

    if not context_parts:
        return None, []

    return "\n\n".join(context_parts), context_sources


def _base_tool_result(
    answer: str,
    sources: list | None = None,
    trace: list[str] | None = None,
    context: str = "",
    fallback_used: bool = False,
    used_context: bool = False,
    context_sources: list[str] | None = None,
    flashcards: list | None = None,
) -> dict:
    return {
        "answer": answer,
        "sources": sources or [],
        "trace": trace or [],
        "context": context,
        "fallback_used": fallback_used,
        "used_context": used_context,
        "context_sources": context_sources or [],
        "flashcards": flashcards or [],
    }


def _format_flashcards_markdown(cards: list[dict]) -> str:
    parts = ["## 记忆卡片"]

    for index, card in enumerate(cards, start=1):
        tags = " / ".join(card.get("tags") or [])
        parts.append(
            f"### 卡片 {index}\n"
            f"**正面：** {card.get('front', '')}\n"
            f"**背面：** {card.get('back', '')}\n"
            f"**标签：** {tags}\n"
            f"**难度：** {card.get('difficulty', 'medium')}"
        )

    return "\n\n".join(parts)


def _run_chat_tool(
    step_input: str,
    original_input: str = "",
    custom_llm=None,
    top_k: int = 3,
    shared_context: dict | None = None,
) -> dict:
    active_llm = custom_llm or llm
    tool_context, context_sources = _build_agent_tool_context(shared_context)
    answer = chat(step_input, context=tool_context, custom_llm=active_llm)
    return _base_tool_result(
        answer=answer,
        used_context=bool(tool_context),
        context_sources=context_sources,
    )


def _run_explain_tool(
    step_input: str,
    original_input: str = "",
    custom_llm=None,
    top_k: int = 3,
    shared_context: dict | None = None,
) -> dict:
    active_llm = custom_llm or llm
    tool_context, context_sources = _build_agent_tool_context(shared_context)
    answer = explain(step_input, context=tool_context, custom_llm=active_llm)
    return _base_tool_result(
        answer=answer,
        used_context=bool(tool_context),
        context_sources=context_sources,
    )


def _run_summarize_tool(
    step_input: str,
    original_input: str = "",
    custom_llm=None,
    top_k: int = 3,
    shared_context: dict | None = None,
) -> dict:
    active_llm = custom_llm or llm
    tool_context, context_sources = _build_agent_tool_context(shared_context)
    answer = summarize(step_input, context=tool_context, custom_llm=active_llm)
    return _base_tool_result(
        answer=answer,
        used_context=bool(tool_context),
        context_sources=context_sources,
    )


def _run_quiz_tool(
    step_input: str,
    original_input: str = "",
    custom_llm=None,
    top_k: int = 3,
    shared_context: dict | None = None,
) -> dict:
    active_llm = custom_llm or llm
    tool_context, context_sources = _build_agent_tool_context(shared_context)
    answer = generate_questions(step_input, context=tool_context, custom_llm=active_llm)
    return _base_tool_result(
        answer=answer,
        used_context=bool(tool_context),
        context_sources=context_sources,
    )


def _run_flashcard_tool(
    step_input: str,
    original_input: str = "",
    custom_llm=None,
    top_k: int = 3,
    shared_context: dict | None = None,
) -> dict:
    active_llm = custom_llm or llm
    tool_context, context_sources = _build_agent_tool_context(shared_context)
    source_text = tool_context or step_input or original_input
    trace = []

    prompt = f"""
你是 AI 学习助手的 flashcard 工具。
请根据给定学习内容生成适合学生复习的记忆卡片，并且只输出 JSON object。

FlashcardPayload schema：
{{
  "cards": [
    {{
      "front": "卡片正面问题，非空字符串",
      "back": "卡片背面答案，非空字符串",
      "tags": ["标签1", "标签2"],
      "difficulty": "easy|medium|hard"
    }}
  ]
}}

示例：
{{
  "cards": [
    {{
      "front": "什么是 RAG？",
      "back": "RAG 是检索增强生成，即先检索相关知识，再让模型基于知识回答。",
      "tags": ["RAG", "基础概念"],
      "difficulty": "easy"
    }},
    {{
      "front": "Agentic RAG 和传统 RAG 的区别是什么？",
      "back": "传统 RAG 通常是固定的检索-生成流程，而 Agentic RAG 可以自主规划、调用工具、评估结果并迭代优化。",
      "tags": ["RAG", "Agent"],
      "difficulty": "medium"
    }}
  ]
}}

学习内容：
{source_text}

用户要求：
{step_input}

输出要求：
- 默认生成 3-5 张卡片。
- 每张卡片必须包含 front、back、tags、difficulty。
- difficulty 只能是 easy、medium、hard。
- 如果学习内容来自知识库，请尽量贴合知识库内容，不要编造。
- 不要输出 Markdown。
- 不要输出 ```json。
- 不要输出任何解释文字。
- 不要输出 schema 以外的字段。
- 你必须只输出一个 JSON object。
"""

    raw_response = active_llm.invoke(prompt).content
    data = _extract_json_object(raw_response)
    flashcards = []

    if data:
        try:
            payload = _validate_flashcard_payload(data)
            flashcards = payload.get("cards", [])[:5]
            trace.append("Flashcard JSON parse：成功")
            trace.append("Flashcard schema validate：成功")
        except ValidationError as exc:
            trace.append("Flashcard JSON parse：成功")
            trace.append(f"Flashcard schema validate：失败：{exc.errors()[0].get('msg')}")
    else:
        trace.append("Flashcard JSON parse：失败")
        trace.append("Flashcard schema validate：失败")

    if flashcards:
        answer = f"已生成 {len(flashcards)} 张记忆卡片，请在下方卡片区域查看、复制或下载。"
    else:
        fallback_prompt = f"""
请基于以下学习内容生成适合学生复习的记忆卡片。

【学习内容】
{source_text}

【用户要求】
{step_input}

输出要求：
- 默认生成 3-5 张卡片。
- 每张卡片必须包含：正面、背面、标签、难度。
- 难度使用 easy / medium / hard。
- 使用清晰 Markdown。

格式：
## 记忆卡片

### 卡片 1
**正面：** ...
**背面：** ...
**标签：** ...
**难度：** medium
"""
        answer = active_llm.invoke(fallback_prompt).content
        trace.append("Flashcard fallback：已返回 Markdown answer，flashcards=[]")

    return _base_tool_result(
        answer=answer,
        trace=trace,
        used_context=bool(tool_context),
        context_sources=context_sources,
        flashcards=flashcards,
    )


def _run_rag_tool(
    step_input: str,
    original_input: str = "",
    custom_llm=None,
    top_k: int = 3,
    shared_context: dict | None = None,
) -> dict:
    active_llm = custom_llm or llm
    rag_context = get_rag_context(step_input, top_k=top_k)
    rag_sources = rag_context.get("sources", [])
    trace = [
        f"RAG query：{step_input}",
        f"RAG expanded_query：{rag_context.get('expanded_query')}",
        f"RAG max_score：{_format_score(rag_context.get('max_score'))}",
        f"RAG threshold：{_format_score(rag_context.get('threshold'))}",
        f"RAG 原始候选数：{rag_context.get('raw_count')}",
        f"RAG 有效候选数：{rag_context.get('valid_count')}",
        f"RAG 丢弃无效 chunk 数：{rag_context.get('discarded_invalid_count')}",
        f"RAG 是否命中：{'是' if rag_context.get('found') else '否'}",
        f"RAG sources：{_source_names(rag_sources)}",
    ]

    if rag_context.get("found"):
        answer = chat(step_input, context=rag_context["context"], custom_llm=active_llm)
        return _base_tool_result(
            answer=answer,
            sources=rag_sources,
            context=rag_context["context"],
            trace=trace,
        )

    trace.append("Agent RAG 未命中，未使用知识库来源")
    answer = _with_fallback_prefix(
        chat(step_input, custom_llm=active_llm),
        RAG_FALLBACK_PREFIX,
    )
    return _base_tool_result(
        answer=answer,
        trace=trace,
        fallback_used=True,
    )


TOOL_REGISTRY = {
    "chat": ToolSpec(
        name="chat",
        description="普通聊天或通用问答",
        run=_run_chat_tool,
    ),
    "rag": ToolSpec(
        name="rag",
        description="从本地知识库检索相关内容并回答",
        run=_run_rag_tool,
    ),
    "explain": ToolSpec(
        name="explain",
        description="用简单中文解释概念",
        run=_run_explain_tool,
    ),
    "summarize": ToolSpec(
        name="summarize",
        description="总结输入内容",
        run=_run_summarize_tool,
    ),
    "quiz": ToolSpec(
        name="quiz",
        description="根据内容生成练习题",
        run=_run_quiz_tool,
    ),
    "flashcard": ToolSpec(
        name="flashcard",
        description="根据知识点生成适合复习的记忆卡片，包括正面问题、背面答案、标签和难度",
        run=_run_flashcard_tool,
    ),
}


def _execute_agent_tool(
    tool: str,
    tool_input: str,
    custom_llm=None,
    top_k: int = 3,
    shared_context: dict | None = None,
) -> dict:
    tool_spec = TOOL_REGISTRY.get(tool)

    if tool_spec is None:
        fallback_tool = TOOL_REGISTRY["chat"]
        result = fallback_tool.run(
            step_input=tool_input,
            original_input=(shared_context or {}).get("original_input", ""),
            custom_llm=custom_llm,
            top_k=top_k,
            shared_context=shared_context,
        )
        result["trace"].append(f"Agent unknown tool：{tool}，已回退到 chat")
        result["tool_name"] = "chat"
        result["tool_description"] = fallback_tool.description
        result["tool_success"] = True
        return result

    result = tool_spec.run(
        step_input=tool_input,
        original_input=(shared_context or {}).get("original_input", ""),
        custom_llm=custom_llm,
        top_k=top_k,
        shared_context=shared_context,
    )
    result["tool_name"] = tool_spec.name
    result["tool_description"] = tool_spec.description
    result["tool_success"] = True
    return result


def run_agent(user_input: str, custom_llm=None, prefer_rag: bool = False, top_k: int = 3) -> dict:
    active_llm = custom_llm or llm
    trace = ["Agent Planner：开始分析用户请求"]
    plan = plan_agent_steps(user_input, custom_llm=active_llm)

    trace.append(f"Agent Planner goal：{plan.get('goal')}")
    trace.append(f"Agent Planner JSON parse：{plan.get('planner_json_parse', '成功')}")
    trace.append(f"Agent Planner schema validate：{plan.get('planner_schema_validate', '成功')}")
    trace.append(f"Agent Planner fallback：{'是' if plan.get('fallback') else '否'}")
    if plan.get("fallback_reason"):
        trace.append(f"Agent Planner fallback reason：{plan['fallback_reason']}")

    if prefer_rag and not any(step.get("tool") == "rag" for step in plan["steps"]):
        plan["steps"].insert(0, {
            "tool": "rag",
            "input": user_input,
            "reason": "use_rag=true，优先尝试知识库检索",
        })
        plan["steps"] = plan["steps"][:4]
        trace.append("Agent Planner：use_rag=true，已插入 rag step")

    previous_result = ""
    step_outputs = []
    all_sources = []
    all_flashcards = []
    fallback_used = False
    shared_context = {
        "original_input": user_input,
        "rag_context": "",
        "sources": [],
        "step_outputs": [],
        "last_output": "",
    }

    for index, step in enumerate(plan["steps"], start=1):
        tool = step["tool"]
        tool_input = step.get("input") or user_input
        if "{previous_result}" in tool_input:
            tool_input = tool_input.replace("{previous_result}", previous_result)

        trace.append(f"Agent Step {index} tool={tool}")
        trace.append(f"Agent Step {index} plan：tool={tool}, reason={step.get('reason')}")
        result = _execute_agent_tool(
            tool,
            tool_input,
            custom_llm=active_llm,
            top_k=top_k,
            shared_context=shared_context,
        )
        result_answer = result.get("answer", "")
        result_sources = result.get("sources", [])
        result_flashcards = result.get("flashcards", [])
        previous_result = result_answer
        all_sources.extend(result_sources)
        all_flashcards.extend(result_flashcards)
        fallback_used = fallback_used or result.get("fallback_used", False)
        trace.append(f"Agent Step {index} 工具说明：{result.get('tool_description', '')}")
        trace.append(f"Agent Step {index} 工具执行成功：{'是' if result.get('tool_success') else '否'}")
        trace.extend(result.get("trace", []))
        trace.append(f"Agent Step {index} 使用上下文：{'是' if result.get('used_context') else '否'}")
        if result.get("context_sources"):
            trace.append(f"Agent Step {index} 上下文来源：{' + '.join(result['context_sources'])}")

        if tool == "rag" and _is_valid_agent_context(result.get("context")):
            shared_context["rag_context"] = result["context"]
            shared_context["sources"] = result_sources

        shared_context["step_outputs"].append({
            "tool": tool,
            "input": tool_input,
            "answer": result_answer,
        })
        shared_context["last_output"] = result_answer
        step_outputs.append(f"步骤 {index}（{tool}）：\n{result_answer}")
        trace.append(f"Agent Step {index} done：输出长度 {len(result_answer)}")

    answer = "\n\n".join(step_outputs) if step_outputs else chat(user_input, custom_llm=active_llm)

    return {
        "answer": answer,
        "trace": trace,
        "plan": plan,
        "sources": all_sources,
        "flashcards": all_flashcards,
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
    trace.append(f"RAG expanded_query：{rag_context.get('expanded_query')}")
    trace.append(f"RAG max_score：{_format_score(rag_context.get('max_score'))}")
    trace.append(f"RAG 阈值：{_format_score(rag_context.get('threshold'))}")
    trace.append(f"RAG 原始候选数：{rag_context.get('raw_count')}")
    trace.append(f"RAG 有效候选数：{rag_context.get('valid_count')}")
    trace.append(f"RAG 丢弃无效 chunk 数：{rag_context.get('discarded_invalid_count')}")
    trace.append(f"RAG 是否通过阈值：{'是' if rag_context.get('found') else '否'}")
    trace.append(f"RAG sources：{_source_names(rag_context.get('sources', []))}")


def _rag_context_for_trace(rag_context: dict, top_k: int) -> dict:
    return {
        **rag_context,
        "top_k": top_k,
    }


def _with_fallback_prefix(answer: str, prefix: str) -> str:
    return f"{prefix}\n\n{answer}"


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
        f"use_agent：{request.use_agent}",
        f"top_k：{request.top_k}",
    ]
    if selected_model != request.model:
        trace.append(f"模型 {request.model} 不可用，已回退到 {selected_model}")

    custom_llm = build_llm(model=selected_model, temperature=request.temperature)
    sources = []
    answer = ""
    executed_mode = request.mode
    fallback_used = False
    rag_context = None
    plan = []
    flashcards = []

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
        plan = _plan_steps_for_response(agent_result.get("plan"))
        flashcards = agent_result.get("flashcards", [])
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
        plan = _plan_steps_for_response(agent_result.get("plan"))
        flashcards = agent_result.get("flashcards", [])
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
    }
