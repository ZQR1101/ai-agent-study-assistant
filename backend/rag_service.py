import re
from time import perf_counter

from backend.config import QUERY_REWRITE_MODES, get_config
from backend.history_utils import truncate_text
from backend.llm_service import chat
from backend.rag_store import SIMILARITY_THRESHOLD, search_relevant_chunks

SOURCE_SNIPPET_LENGTH = 400
QUERY_FUSION_K = 60

_QUERY_REWRITE_FOLLOW_UP_PATTERN = re.compile(
    r"(?:它|这个|那个|上述|前面|刚才|其中|这两者|他们|她们|这些|那些|"
    r"该(?:方法|模型|接口|算法|功能|配置|文档|章节|项目))"
)
_QUERY_REWRITE_STRONG_ANCHOR_PATTERN = re.compile(
    r"/[A-Za-z0-9_./-]+|[A-Za-z][A-Za-z0-9_.-]*|\d+(?:\.\d+)*"
)
_QUERY_REWRITE_EXACT_VALUE_PATTERN = re.compile(
    r"(?:[/\\][A-Za-z0-9_.\-/\\]+|"
    r"(?<![A-Za-z0-9_])v?\d+(?:\.\d+)+(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])[A-Z]{2,}-?\d{2,}(?![A-Za-z0-9_])|"
    r"(?<![A-Za-z0-9_])[A-Za-z0-9_-]+\."
    r"(?:py|js|jsx|ts|tsx|java|go|rs|md|txt|json|ya?ml|toml|ini|pdf)"
    r"(?![A-Za-z0-9_]))",
    re.IGNORECASE,
)

NO_RAG_ANSWER = "知识库中没有找到与该问题相关的内容。你可以上传相关资料，或切换到普通聊天模式。"
RAG_FALLBACK_PREFIX = "知识库中没有找到相关内容，以下内容未使用知识库，仅基于模型通用知识生成。"
LEARN_FALLBACK_PREFIX = "知识库中没有找到相关内容，以下学习内容未使用知识库，仅基于模型通用知识生成。"

QUERY_REWRITE_PROMPT = """你是 RAG 检索查询改写器。请把用户问题改写成更适合知识库检索的一行查询。

要求：
- 去掉寒暄、口语化表达和“帮我/请/讲讲/介绍一下”等操作性噪声。
- 如果用户用了“它/这个/上面/前面”等指代，并且历史对话能明确指向对象，请补全指代。
- 保留专有名词、中英文缩写、数字、文件名、接口名和路径。
- 不要回答问题，不要解释，只输出改写后的检索查询。

历史对话：
{history_context}

用户问题：
{question}

检索查询："""


def format_score(score) -> str:
    if score is None:
        return "无"
    return f"{score:.4f}"


def truncate_source_text(text: str, max_length: int = SOURCE_SNIPPET_LENGTH) -> str:
    return truncate_text(text, max_length)


def _response_text(response) -> str:
    return str(getattr(response, "content", response) or "")


def _clean_rewritten_query(text: str) -> str:
    cleaned = " ".join(str(text or "").split()).strip()
    cleaned = cleaned.strip("`'\"“”‘’")
    cleaned = cleaned.removeprefix("检索查询：").removeprefix("检索查询:")
    cleaned = cleaned.removeprefix("改写后的检索查询：").removeprefix("改写后的检索查询:")
    cleaned = cleaned.strip("`'\"“”‘’ ")
    return cleaned[:300].strip()


def should_rewrite_query(question: str, history_context: str | None) -> bool:
    query = " ".join(str(question or "").split()).strip()
    if not query or not str(history_context or "").strip():
        return False
    if len(query) > 30 or not _QUERY_REWRITE_FOLLOW_UP_PATTERN.search(query):
        return False
    if _QUERY_REWRITE_EXACT_VALUE_PATTERN.search(query):
        return False

    anchors = {
        value.casefold()
        for value in _QUERY_REWRITE_STRONG_ANCHOR_PATTERN.findall(query)
        if value.strip()
    }
    return len(anchors) <= 1


def _query_rewrite_decision(
    question: str,
    history_context: str | None,
    mode: str,
) -> dict:
    normalized_mode = str(mode or "off").strip().lower()
    if normalized_mode not in QUERY_REWRITE_MODES:
        normalized_mode = "off"
    if normalized_mode == "off":
        return {"mode": normalized_mode, "enabled": False, "reason": "mode_off"}
    if normalized_mode == "always":
        return {"mode": normalized_mode, "enabled": True, "reason": "mode_always"}
    if not str(history_context or "").strip():
        return {"mode": normalized_mode, "enabled": False, "reason": "missing_history"}
    if should_rewrite_query(question, history_context):
        return {"mode": normalized_mode, "enabled": True, "reason": "context_follow_up"}
    return {"mode": normalized_mode, "enabled": False, "reason": "query_self_contained"}


def _query_result_key(chunk: dict, position: int):
    if chunk.get("chunk_id"):
        return chunk["chunk_id"]
    source_text_key = (chunk.get("source"), chunk.get("text"))
    if any(source_text_key):
        return source_text_key
    return ("position", position)


def _merge_search_errors(*errors) -> str | None:
    messages = list(dict.fromkeys(str(error) for error in errors if error))
    return "; ".join(messages) if messages else None


def _fuse_query_search_results(
    original_result: dict,
    rewritten_result: dict,
    top_k: int,
) -> dict:
    fused = {}
    for query_name, search_result in (
        ("original", original_result),
        ("rewritten", rewritten_result),
    ):
        for rank, chunk in enumerate(search_result.get("chunks", []), start=1):
            key = _query_result_key(chunk, rank)
            if key not in fused:
                fused[key] = {
                    **chunk,
                    "query_fusion_score": 0.0,
                }
            fused[key]["query_fusion_score"] += 1.0 / (QUERY_FUSION_K + rank)
            fused[key][f"{query_name}_query_rank"] = rank

    all_chunks = list(fused.values())
    all_chunks.sort(key=lambda item: item["query_fusion_score"], reverse=True)
    chunks = all_chunks[:top_k]
    passed_threshold = bool(
        original_result.get("passed_threshold")
        or rewritten_result.get("passed_threshold")
    )
    if not passed_threshold:
        chunks = []

    merged = {
        **original_result,
        "chunks": chunks,
        "highest_score": max(
            (float(item.get("score", 0.0)) for item in chunks),
            default=None,
        ),
        "passed_threshold": passed_threshold and bool(chunks),
        "raw_count": len(all_chunks),
        "valid_count": len(chunks),
        "discarded_invalid_count": int(original_result.get("discarded_invalid_count", 0))
        + int(rewritten_result.get("discarded_invalid_count", 0)),
        "error": _merge_search_errors(
            original_result.get("error"),
            rewritten_result.get("error"),
        ),
        "candidate_k": max(
            int(original_result.get("candidate_k") or 0),
            int(rewritten_result.get("candidate_k") or 0),
        ) or None,
        "vector_candidates": int(original_result.get("vector_candidates", 0))
        + int(rewritten_result.get("vector_candidates", 0)),
        "bm25_candidates": int(original_result.get("bm25_candidates", 0))
        + int(rewritten_result.get("bm25_candidates", 0)),
        "reranker_used": bool(
            original_result.get("reranker_used")
            or rewritten_result.get("reranker_used")
        ),
        "reranker_error": _merge_search_errors(
            original_result.get("reranker_error"),
            rewritten_result.get("reranker_error"),
        ),
        "original_expanded_query": original_result.get("expanded_query"),
        "rewrite_expanded_query": rewritten_result.get("expanded_query"),
        "query_fusion_used": True,
    }
    return merged


def rewrite_query_for_retrieval(
    question: str,
    custom_llm=None,
    history_context: str | None = None,
) -> dict:
    started_at = perf_counter()

    def result(query: str, used: bool, error: str | None) -> dict:
        return {
            "query": query,
            "used": used,
            "error": error,
            "latency_ms": round((perf_counter() - started_at) * 1000, 3),
        }

    original_query = " ".join(str(question or "").split()).strip()
    if not original_query or custom_llm is None:
        return result(original_query, False, None)

    prompt = QUERY_REWRITE_PROMPT.format(
        history_context=(history_context or "无").strip() or "无",
        question=original_query,
    )
    try:
        rewritten = _clean_rewritten_query(_response_text(custom_llm.invoke(prompt)))
    except Exception as exc:
        return result(original_query, False, str(exc))

    if not rewritten:
        return result(original_query, False, "empty rewritten query")

    return result(rewritten, rewritten != original_query, None)


def get_rag_context(
    question: str,
    top_k: int = 3,
    score_threshold: float = SIMILARITY_THRESHOLD,
    retrieval_mode: str = "vector",
    candidate_k: int | None = None,
    reranker_enabled: bool = False,
    reranker_top_n: int | None = None,
    query_rewrite_llm=None,
    history_context: str | None = None,
    query_rewrite_mode: str | None = None,
) -> dict:
    configured_mode = query_rewrite_mode or get_config().query_rewrite_mode
    rewrite_decision = _query_rewrite_decision(
        question,
        history_context,
        configured_mode,
    )
    query_rewrite = rewrite_query_for_retrieval(
        question,
        custom_llm=query_rewrite_llm if rewrite_decision["enabled"] else None,
        history_context=history_context,
    )
    retrieval_query = query_rewrite["query"]
    search_kwargs = {
        "top_k": top_k,
        "similarity_threshold": score_threshold,
        "include_metadata": True,
        "retrieval_mode": retrieval_mode,
        "candidate_k": candidate_k,
        "reranker_enabled": reranker_enabled,
        "reranker_top_n": reranker_top_n,
    }
    original_search_result = search_relevant_chunks(
        question,
        **search_kwargs,
    )
    if query_rewrite["used"]:
        rewritten_search_result = search_relevant_chunks(
            retrieval_query,
            **search_kwargs,
        )
        search_result = _fuse_query_search_results(
            original_search_result,
            rewritten_search_result,
            top_k,
        )
    else:
        search_result = {
            **original_search_result,
            "query_fusion_used": False,
        }
    chunks = search_result["chunks"]
    max_score = search_result["highest_score"]
    expanded_query = search_result.get("expanded_query", question)
    raw_count = search_result.get("raw_count", 0)
    valid_count = search_result.get("valid_count", len(chunks))
    discarded_invalid_count = search_result.get("discarded_invalid_count", 0)
    error = search_result.get("error")
    passed_threshold = search_result.get("passed_threshold", bool(chunks))
    result_threshold = search_result.get("threshold", score_threshold)
    reranker_info = {
        "reranker_enabled": search_result.get("reranker_enabled", False),
        "reranker_used": search_result.get("reranker_used", False),
        "reranker_model": search_result.get("reranker_model"),
        "reranker_top_n": search_result.get("reranker_top_n"),
        "reranker_error": search_result.get("reranker_error"),
    }
    rewrite_reason = rewrite_decision["reason"]
    if rewrite_decision["enabled"] and query_rewrite_llm is None:
        rewrite_reason = "llm_unavailable"
    elif query_rewrite["error"]:
        rewrite_reason = "rewrite_error"
    elif query_rewrite["used"]:
        rewrite_reason = "rewritten"
    elif rewrite_decision["enabled"]:
        rewrite_reason = "rewrite_unchanged"
    rewrite_info = {
        "original_query": question,
        "retrieval_query": retrieval_query,
        "retrieval_queries": (
            [question, retrieval_query] if query_rewrite["used"] else [question]
        ),
        "query_rewrite_mode": rewrite_decision["mode"],
        "query_rewrite_attempted": bool(
            rewrite_decision["enabled"] and query_rewrite_llm is not None
        ),
        "query_rewrite_used": query_rewrite["used"],
        "query_rewrite_error": query_rewrite["error"],
        "query_rewrite_reason": rewrite_reason,
        "query_rewrite_latency_ms": query_rewrite["latency_ms"],
        "query_fusion_used": search_result.get("query_fusion_used", False),
    }

    if not chunks or not passed_threshold:
        return {
            "found": False,
            "context": "",
            "sources": [],
            "retrieved_chunks": [],
            "max_score": max_score,
            "threshold": result_threshold,
            "expanded_query": expanded_query,
            "raw_count": raw_count,
            "valid_count": valid_count,
            "discarded_invalid_count": discarded_invalid_count,
            "error": error,
            **rewrite_info,
            "retrieval_mode": search_result.get("retrieval_mode", retrieval_mode),
            "candidate_k": search_result.get("candidate_k"),
            "vector_candidates": search_result.get("vector_candidates", 0),
            "bm25_candidates": search_result.get("bm25_candidates", 0),
            "hybrid_used": search_result.get("hybrid_used", False),
            **reranker_info,
        }

    context_parts = []
    source_chunks = []

    for chunk in chunks:
        retrieval = chunk.get("retrieval", retrieval_mode)
        context_parts.append(
            f"来源文件：{chunk['source']}\n"
            f"文档：{chunk.get('document_title') or chunk.get('document') or chunk['source']}\n"
            f"章节：{chunk.get('section') or '无'}\n"
            f"标题：{chunk.get('title') or '无'}\n"
            f"检索方式：{retrieval}\n"
            f"得分：{chunk['score']:.4f}\n"
            f"内容：\n{chunk['text']}"
        )
        source_payload = {
            "source": chunk["source"],
            "score": float(chunk["score"]),
            "snippet": truncate_source_text(chunk["text"]),
            "text": truncate_source_text(chunk["text"]),
            "chunk_id": chunk.get("chunk_id"),
            "retrieval": retrieval,
            "document": chunk.get("document"),
            "document_title": chunk.get("document_title"),
            "title": chunk.get("title"),
            "section": chunk.get("section"),
            "headings": chunk.get("headings", []),
        }
        for key in (
            "vector_score",
            "bm25_score",
            "vector_rank",
            "bm25_rank",
            "rerank_score",
            "rerank_rank",
            "reranker_used",
            "query_fusion_score",
            "original_query_rank",
            "rewritten_query_rank",
        ):
            if chunk.get(key) is not None:
                source_payload[key] = chunk[key]
        source_chunks.append(source_payload)

    return {
        "found": True,
        "context": "\n\n---\n\n".join(context_parts),
        "sources": source_chunks,
        "retrieved_chunks": chunks,
        "max_score": max_score,
        "threshold": result_threshold,
        "expanded_query": expanded_query,
        "raw_count": raw_count,
        "valid_count": valid_count,
        "discarded_invalid_count": discarded_invalid_count,
        "error": error,
        **rewrite_info,
        "retrieval_mode": search_result.get("retrieval_mode", retrieval_mode),
        "candidate_k": search_result.get("candidate_k"),
        "vector_candidates": search_result.get("vector_candidates", 0),
        "bm25_candidates": search_result.get("bm25_candidates", 0),
        "hybrid_used": search_result.get("hybrid_used", False),
        **reranker_info,
    }


def rag_answer_with_sources(
    question: str,
    custom_llm=None,
    top_k: int = 3,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    retrieval_mode: str = "vector",
    reranker_enabled: bool = False,
    reranker_top_n: int | None = None,
    history_context: str | None = None,
) -> dict:
    rag_context = get_rag_context(
        question,
        top_k=top_k,
        score_threshold=similarity_threshold,
        retrieval_mode=retrieval_mode,
        reranker_enabled=reranker_enabled,
        reranker_top_n=reranker_top_n,
        query_rewrite_llm=custom_llm,
        history_context=history_context,
    )

    if not rag_context["found"]:
        return {
            "answer": NO_RAG_ANSWER,
            "sources": [],
            "highest_score": rag_context["max_score"],
            "threshold": rag_context["threshold"],
            "passed_threshold": False,
            "retrieval_mode": rag_context.get("retrieval_mode", retrieval_mode),
            "reranker_enabled": rag_context.get("reranker_enabled", False),
            "reranker_used": rag_context.get("reranker_used", False),
            "reranker_error": rag_context.get("reranker_error"),
        }

    answer = chat(question, context=rag_context["context"], custom_llm=custom_llm)

    return {
        "answer": answer,
        "sources": rag_context["sources"],
        "highest_score": rag_context["max_score"],
        "threshold": rag_context["threshold"],
        "passed_threshold": True,
        "retrieval_mode": rag_context.get("retrieval_mode", retrieval_mode),
        "reranker_enabled": rag_context.get("reranker_enabled", False),
        "reranker_used": rag_context.get("reranker_used", False),
        "reranker_error": rag_context.get("reranker_error"),
    }


def rag_answer(
    question: str,
    custom_llm=None,
    top_k: int = 3,
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    retrieval_mode: str = "vector",
    reranker_enabled: bool = False,
    reranker_top_n: int | None = None,
    history_context: str | None = None,
) -> str:
    result = rag_answer_with_sources(
        question,
        custom_llm=custom_llm,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        retrieval_mode=retrieval_mode,
        reranker_enabled=reranker_enabled,
        reranker_top_n=reranker_top_n,
        history_context=history_context,
    )
    source_text = "\n".join([
        f"- {source.get('source')} ({format_score(source.get('score'))})"
        for source in result["sources"]
    ])

    return f"""
{result["answer"]}

---

参考来源：
{source_text}
"""


def source_names(sources: list[dict]) -> list[str]:
    return sorted(set(source.get("source", "") for source in sources if source.get("source")))


def append_rag_trace(trace: list[str], rag_context: dict | None) -> None:
    if not rag_context:
        return

    trace.append(f"RAG top_k：{rag_context.get('top_k')}")
    trace.append(f"RAG retrieval_mode：{rag_context.get('retrieval_mode', 'vector')}")
    trace.append(f"RAG candidate_k：{rag_context.get('candidate_k')}")
    trace.append(f"RAG original_query：{rag_context.get('original_query')}")
    trace.append(f"RAG retrieval_query：{rag_context.get('retrieval_query')}")
    trace.append(f"RAG expanded_query：{rag_context.get('expanded_query')}")
    trace.append(f"RAG query_rewrite_used：{'是' if rag_context.get('query_rewrite_used') else '否'}")
    trace.append(f"RAG query_rewrite_mode：{rag_context.get('query_rewrite_mode', 'off')}")
    trace.append(f"RAG query_rewrite_reason：{rag_context.get('query_rewrite_reason')}")
    trace.append(f"RAG query_rewrite_latency_ms：{rag_context.get('query_rewrite_latency_ms', 0)}")
    trace.append(f"RAG query_fusion_used：{'是' if rag_context.get('query_fusion_used') else '否'}")
    if rag_context.get("query_rewrite_error"):
        trace.append(f"RAG query_rewrite_error：{rag_context.get('query_rewrite_error')}")
    trace.append(f"RAG max_score：{format_score(rag_context.get('max_score'))}")
    trace.append(f"RAG 阈值：{format_score(rag_context.get('threshold'))}")
    trace.append(f"RAG vector_candidates：{rag_context.get('vector_candidates', 0)}")
    trace.append(f"RAG bm25_candidates：{rag_context.get('bm25_candidates', 0)}")
    trace.append(f"RAG hybrid_used：{'是' if rag_context.get('hybrid_used') else '否'}")
    trace.append(f"RAG reranker_enabled：{'是' if rag_context.get('reranker_enabled') else '否'}")
    trace.append(f"RAG reranker_used：{'是' if rag_context.get('reranker_used') else '否'}")
    trace.append(f"RAG reranker_model：{rag_context.get('reranker_model')}")
    trace.append(f"RAG reranker_top_n：{rag_context.get('reranker_top_n')}")
    if rag_context.get("reranker_error"):
        trace.append(f"RAG reranker_error：{rag_context.get('reranker_error')}")
    trace.append(f"RAG 原始候选数：{rag_context.get('raw_count')}")
    trace.append(f"RAG 有效候选数：{rag_context.get('valid_count')}")
    trace.append(f"RAG 丢弃无效 chunk 数：{rag_context.get('discarded_invalid_count')}")
    trace.append(f"RAG 是否通过阈值：{'是' if rag_context.get('found') else '否'}")
    trace.append(f"RAG sources：{source_names(rag_context.get('sources', []))}")


def rag_context_for_trace(rag_context: dict, top_k: int) -> dict:
    return {
        **rag_context,
        "top_k": top_k,
    }


def with_fallback_prefix(answer: str, prefix: str) -> str:
    return f"{prefix}\n\n{answer}"
