import json
import re

from pydantic import ValidationError

from backend.llm_service import chat, explain, generate_questions, llm, summarize
from backend.rag_service import (
    NO_RAG_ANSWER,
    RAG_FALLBACK_PREFIX,
    format_score,
    get_rag_context,
    source_names,
    with_fallback_prefix,
)
from backend.schemas import FlashcardPayload
from backend.tool_actions import (
    delete_knowledge_file,
    delete_run,
    delete_saved_item,
    rebuild_rag_index_tool,
    reset_rag_index,
    reset_saved_items,
    save_flashcards,
    save_note,
    save_quiz,
)
from backend.tool_registry import ToolCategory, ToolRegistry, ToolSpec


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


def _validate_flashcard_payload(data: dict) -> dict:
    if hasattr(FlashcardPayload, "model_validate"):
        payload = FlashcardPayload.model_validate(data)
    else:
        payload = FlashcardPayload.parse_obj(data)

    return _model_to_dict(payload)


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


def build_agent_tool_context(shared_context: dict | None) -> tuple[str | None, list[str]]:
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
    retrieval_info: dict | None = None,
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
        "retrieval_info": retrieval_info or {},
    }


def _run_chat_tool(
    step_input: str,
    original_input: str = "",
    custom_llm=None,
    top_k: int = 3,
    shared_context: dict | None = None,
) -> dict:
    active_llm = custom_llm or llm
    tool_context, context_sources = build_agent_tool_context(shared_context)
    history_context = (shared_context or {}).get("history_context", "")
    answer = chat(
        step_input,
        context=tool_context,
        custom_llm=active_llm,
        history_context=history_context,
    )
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
    tool_context, context_sources = build_agent_tool_context(shared_context)
    history_context = (shared_context or {}).get("history_context", "")
    answer = explain(
        step_input,
        context=tool_context,
        custom_llm=active_llm,
        history_context=history_context,
    )
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
    tool_context, context_sources = build_agent_tool_context(shared_context)
    history_context = (shared_context or {}).get("history_context", "")
    answer = summarize(
        step_input,
        context=tool_context,
        custom_llm=active_llm,
        history_context=history_context,
    )
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
    tool_context, context_sources = build_agent_tool_context(shared_context)
    history_context = (shared_context or {}).get("history_context", "")
    answer = generate_questions(
        step_input,
        context=tool_context,
        custom_llm=active_llm,
        history_context=history_context,
    )
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
    tool_context, context_sources = build_agent_tool_context(shared_context)
    history_context = (shared_context or {}).get("history_context", "")
    source_text = tool_context or history_context or step_input or original_input
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
    generate_answer: bool = True,
) -> dict:
    active_llm = custom_llm or llm
    history_context = (shared_context or {}).get("history_context", "")
    retrieval_mode = (shared_context or {}).get("retrieval_mode", "vector")
    rag_query = step_input
    rag_kwargs = {}
    if shared_context is not None and "reranker_enabled" in shared_context:
        rag_kwargs["reranker_enabled"] = bool(shared_context["reranker_enabled"])
    rag_context = get_rag_context(
        rag_query,
        top_k=top_k,
        retrieval_mode=retrieval_mode,
        **rag_kwargs,
    )
    rag_sources = rag_context.get("sources", [])
    retrieval_info = {
        "retrieval_mode": rag_context.get("retrieval_mode", retrieval_mode),
        "candidate_k": rag_context.get("candidate_k"),
        "vector_candidates": rag_context.get("vector_candidates", 0),
        "bm25_candidates": rag_context.get("bm25_candidates", 0),
        "hybrid_used": rag_context.get("hybrid_used", False),
        "reranker_enabled": rag_context.get("reranker_enabled", False),
        "reranker_used": rag_context.get("reranker_used", False),
        "reranker_model": rag_context.get("reranker_model"),
        "reranker_top_n": rag_context.get("reranker_top_n"),
        "reranker_error": rag_context.get("reranker_error"),
    }
    trace = [
        f"RAG query：{step_input}",
        "RAG query 使用 history：否",
        f"RAG answer 使用 history：{'是' if history_context else '否'}",
        f"RAG retrieval_mode：{retrieval_info['retrieval_mode']}",
        f"RAG candidate_k：{retrieval_info['candidate_k']}",
        f"RAG expanded_query：{rag_context.get('expanded_query')}",
        f"RAG max_score：{format_score(rag_context.get('max_score'))}",
        f"RAG threshold：{format_score(rag_context.get('threshold'))}",
        f"RAG vector_candidates：{retrieval_info['vector_candidates']}",
        f"RAG bm25_candidates：{retrieval_info['bm25_candidates']}",
        f"RAG hybrid_used：{'是' if retrieval_info['hybrid_used'] else '否'}",
        f"RAG reranker_enabled：{'是' if retrieval_info['reranker_enabled'] else '否'}",
        f"RAG reranker_used：{'是' if retrieval_info['reranker_used'] else '否'}",
        f"RAG 原始候选数：{rag_context.get('raw_count')}",
        f"RAG 有效候选数：{rag_context.get('valid_count')}",
        f"RAG 丢弃无效 chunk 数：{rag_context.get('discarded_invalid_count')}",
        f"RAG 是否命中：{'是' if rag_context.get('found') else '否'}",
        f"RAG sources：{source_names(rag_sources)}",
    ]

    if rag_context.get("error"):
        trace.append(f"RAG error：{rag_context.get('error')}")

    if rag_context.get("found"):
        if not generate_answer:
            return _base_tool_result(
                answer="",
                sources=rag_sources,
                context=rag_context["context"],
                trace=[*trace, "RAG answer generation：skipped"],
                used_context=True,
                context_sources=source_names(rag_sources),
                retrieval_info=retrieval_info,
            )

        answer = chat(
            step_input,
            context=rag_context["context"],
            custom_llm=active_llm,
            history_context=history_context,
        )
        return _base_tool_result(
            answer=answer,
            sources=rag_sources,
            context=rag_context["context"],
            trace=trace,
            used_context=True,
            context_sources=source_names(rag_sources),
            retrieval_info=retrieval_info,
        )

    trace.append("Agent RAG 未命中，未使用知识库来源")
    answer = with_fallback_prefix(
        chat(step_input, custom_llm=active_llm, history_context=history_context),
        RAG_FALLBACK_PREFIX,
    )
    return _base_tool_result(
        answer=answer,
        trace=trace,
        fallback_used=True,
        retrieval_info=retrieval_info,
    )


_LEGACY_TOOL_REGISTRY = {
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


# Public registry v2. Legacy implementations above remain private runners so
# existing workflows can migrate without changing generation behavior.


def _detect_study_operations(step_input: str) -> list[str]:
    text = str(step_input or "").lower()
    markers = {
        "summarize": ("summarize", "summary", "总结", "摘要", "概括"),
        "explain": ("explain", "what is", "why", "解释", "讲解", "什么是"),
        "flashcard": ("flashcard", "card", "卡片", "记忆卡", "抽认卡"),
        "quiz": ("quiz", "question", "test", "测验", "练习题", "出题"),
    }
    operations = [
        operation
        for operation, keywords in markers.items()
        if any(keyword in text for keyword in keywords)
    ]
    return operations or ["explain"]


def _run_study_tool(
    step_input: str,
    original_input: str = "",
    custom_llm=None,
    top_k: int = 3,
    shared_context: dict | None = None,
    operation: str | list[str] | None = None,
) -> dict:
    """Run one or more learning-content operations through one public tool."""
    requested = [operation] if isinstance(operation, str) else list(operation or [])
    requested = [
        item
        for item in requested
        if item in {"explain", "summarize", "quiz", "flashcard"}
    ]
    operations = requested or _detect_study_operations(step_input)
    runners = {
        "explain": _run_explain_tool,
        "summarize": _run_summarize_tool,
        "quiz": _run_quiz_tool,
        "flashcard": _run_flashcard_tool,
    }
    answers = []
    trace = [f"study operations: {', '.join(operations)}"]
    flashcards = []
    used_context = False
    context_sources = []
    for item in operations:
        result = runners[item](
            step_input=step_input,
            original_input=original_input,
            custom_llm=custom_llm,
            top_k=top_k,
            shared_context=shared_context,
        )
        if result.get("answer"):
            answers.append(result["answer"])
        trace.extend(result.get("trace", []))
        flashcards.extend(result.get("flashcards", []))
        used_context = used_context or bool(result.get("used_context"))
        for source in result.get("context_sources", []):
            if source not in context_sources:
                context_sources.append(source)
    return _base_tool_result(
        answer="\n\n".join(answers),
        trace=trace,
        used_context=used_context,
        context_sources=context_sources,
        flashcards=flashcards,
    )


TOOL_REGISTRY = ToolRegistry(
    [
        ToolSpec("chat", "General conversation and question answering.", _run_chat_tool),
        ToolSpec(
            "rag_search",
            "Search the local knowledge base and answer from retrieved evidence.",
            _run_rag_tool,
        ),
        ToolSpec(
            "study",
            "Explain, summarize, create quizzes, or create flashcards in one tool.",
            _run_study_tool,
        ),
        ToolSpec("save_note", "Save a study note.", save_note, ToolCategory.WRITE),
        ToolSpec(
            "save_flashcards",
            "Save generated flashcards.",
            save_flashcards,
            ToolCategory.WRITE,
        ),
        ToolSpec("save_quiz", "Save a quiz or question set.", save_quiz, ToolCategory.WRITE),
        ToolSpec(
            "delete_saved_item",
            "Delete one saved note, flashcard set, or quiz.",
            delete_saved_item,
            ToolCategory.DANGEROUS,
            True,
            False,
        ),
        ToolSpec(
            "delete_run",
            "Delete one persisted execution Run.",
            delete_run,
            ToolCategory.DANGEROUS,
            True,
            False,
        ),
        ToolSpec(
            "delete_knowledge_file",
            "Delete a file from the local knowledge base.",
            delete_knowledge_file,
            ToolCategory.DANGEROUS,
            True,
            False,
        ),
        ToolSpec(
            "reset_saved_items",
            "Delete all saved study data, optionally in one collection.",
            reset_saved_items,
            ToolCategory.DANGEROUS,
            True,
            False,
        ),
        ToolSpec(
            "reset_rag_index",
            "Remove the persisted and in-memory RAG index.",
            reset_rag_index,
            ToolCategory.DANGEROUS,
            True,
            False,
        ),
        ToolSpec(
            "rebuild_rag_index",
            "Rebuild and replace the RAG index from knowledge files.",
            rebuild_rag_index_tool,
            ToolCategory.DANGEROUS,
            True,
            False,
        ),
    ],
)
