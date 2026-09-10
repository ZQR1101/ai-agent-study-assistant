"""Shared query-rewrite search wrapper used by offline evaluation harnesses.

Wraps production retrieval with the query rewrite + two-query fusion flow so
evaluate_rag_retrieval and benchmark_rag_batch measure the same pipeline.
"""
from __future__ import annotations

from typing import Any, Callable


def build_query_rewrite_search(
    mode: str,
) -> Callable[..., Any] | None:
    if mode == "off":
        return None

    from backend.config import get_config
    from backend.llm_service import build_llm
    from backend.rag_service import (
        _fuse_query_search_results,
        _query_rewrite_decision,
        _rerank_search_result,
        rewrite_query_for_retrieval,
        search_with_conditional_rewrite,
    )
    from backend.rag_store import search_relevant_chunks
    from backend.reranker import is_reranker_enabled

    config = get_config()
    rewrite_llm = build_llm(model=config.model, temperature=0.0)
    rewrite_cache: dict[tuple[str, str], dict] = {}
    rewrite_stats = {"api_call_count": 0}

    def cached_rewrite(question: str, history_context: str | None = None) -> dict:
        cache_key = (question, history_context or "")
        if cache_key in rewrite_cache:
            return rewrite_cache[cache_key]

        decision = _query_rewrite_decision(question, history_context, mode)
        attempted = bool(decision["enabled"])
        if attempted:
            rewrite_stats["api_call_count"] += 1
        rewrite = rewrite_query_for_retrieval(
            question,
            custom_llm=rewrite_llm if attempted else None,
            history_context=history_context,
        )
        reason = decision["reason"]
        if rewrite["error"]:
            reason = "rewrite_error"
        elif rewrite["used"]:
            reason = "rewritten"
        elif attempted:
            reason = "rewrite_unchanged"
        cached = {
            "decision": decision,
            "rewrite": rewrite,
            "attempted": attempted,
            "reason": reason,
        }
        rewrite_cache[cache_key] = cached
        return cached

    def search(
        question: str,
        top_k: int = 3,
        similarity_threshold: float = 0.55,
        include_metadata: bool = False,
        retrieval_mode: str = "vector",
        candidate_k: int | None = None,
        reranker_enabled: bool = False,
        reranker_top_n: int | None = None,
        history_context: str | None = None,
        **_kwargs,
    ):
        cache_key = (question, history_context or "")
        rewrite_cache_hit = cache_key in rewrite_cache
        base_search_kwargs = {
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "include_metadata": True,
            "retrieval_mode": retrieval_mode,
            "candidate_k": candidate_k,
            "reranker_enabled": reranker_enabled,
            "reranker_top_n": reranker_top_n,
        }
        if mode == "conditional":
            search_result, rewrite_delta = search_with_conditional_rewrite(
                question,
                history_context=history_context,
                query_rewrite_llm=rewrite_llm,
                top_k=top_k,
                search_kwargs=base_search_kwargs,
                # cached_rewrite wraps the raw rewrite payload with decision
                # stats; the adaptive helper expects the payload itself.
                rewrite_fn=lambda q, h: cached_rewrite(q, h)["rewrite"],
            )
            metadata = {
                **search_result,
                **rewrite_delta,
                "query_rewrite_mode": mode,
            }
            metadata.update({
                "original_query": question,
                "query_rewrite_latency_included": bool(
                    rewrite_delta["query_rewrite_attempted"] and not rewrite_cache_hit
                ),
            })
            return metadata if include_metadata else metadata["chunks"]

        cached = cached_rewrite(question, history_context)
        rewrite = cached["rewrite"]
        rerank_after_fusion = rewrite["used"] and is_reranker_enabled(reranker_enabled)
        search_kwargs = {**base_search_kwargs, "apply_reranker": not rerank_after_fusion}
        original_result = search_relevant_chunks(
            question,
            **search_kwargs,
        )
        if rewrite["used"]:
            rewritten_result = search_relevant_chunks(
                rewrite["query"],
                **search_kwargs,
            )
            fusion_pool_size = top_k
            if rerank_after_fusion:
                fusion_pool_size = (
                    int(original_result.get("reranker_top_n") or 0) or top_k
                )
            metadata = _fuse_query_search_results(
                original_result,
                rewritten_result,
                fusion_pool_size,
            )
            if rerank_after_fusion:
                metadata = _rerank_search_result(metadata, question, top_k)
        else:
            metadata = {
                **original_result,
                "query_fusion_used": False,
            }

        metadata.update({
            "original_query": question,
            "retrieval_query": rewrite["query"],
            "retrieval_queries": (
                [question, rewrite["query"]] if rewrite["used"] else [question]
            ),
            "query_rewrite_mode": cached["decision"]["mode"],
            "query_rewrite_attempted": cached["attempted"],
            "query_rewrite_used": rewrite["used"],
            "query_rewrite_error": rewrite["error"],
            "query_rewrite_reason": cached["reason"],
            "query_rewrite_latency_ms": rewrite["latency_ms"],
            "query_rewrite_latency_included": bool(
                cached["attempted"] and not rewrite_cache_hit
            ),
        })
        chunks = metadata.get("chunks", [])
        return metadata if include_metadata else chunks

    search.rewrite_cache = rewrite_cache
    search.rewrite_stats = rewrite_stats
    return search
