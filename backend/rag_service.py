from backend.history_utils import truncate_text
from backend.llm_service import chat
from backend.rag_store import SIMILARITY_THRESHOLD, search_relevant_chunks

SOURCE_SNIPPET_LENGTH = 400

NO_RAG_ANSWER = "知识库中没有找到与该问题相关的内容。你可以上传相关资料，或切换到普通聊天模式。"
RAG_FALLBACK_PREFIX = "知识库中没有找到相关内容，以下内容未使用知识库，仅基于模型通用知识生成。"
LEARN_FALLBACK_PREFIX = "知识库中没有找到相关内容，以下学习内容未使用知识库，仅基于模型通用知识生成。"


def format_score(score) -> str:
    if score is None:
        return "无"
    return f"{score:.4f}"


def truncate_source_text(text: str, max_length: int = SOURCE_SNIPPET_LENGTH) -> str:
    return truncate_text(text, max_length)


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
            "snippet": truncate_source_text(chunk["text"]),
            "text": truncate_source_text(chunk["text"]),
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
    trace.append(f"RAG expanded_query：{rag_context.get('expanded_query')}")
    trace.append(f"RAG max_score：{format_score(rag_context.get('max_score'))}")
    trace.append(f"RAG 阈值：{format_score(rag_context.get('threshold'))}")
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
