from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CASES_PATH = Path("eval_cases/rag_retrieval_cases.json")
DEFAULT_JSON_OUTPUT = Path("outputs/rag_retrieval_eval.json")
DEFAULT_MARKDOWN_OUTPUT = Path("outputs/rag_retrieval_eval.md")
BASE_MODES = ("vector", "bm25", "hybrid")
RERANKER_MODE = "hybrid_reranker"
SUPPORTED_MODES = (*BASE_MODES, RERANKER_MODE)
SNIPPET_LENGTH = 320


def configure_offline_embedding() -> None:
    # A missing local model becomes a vector-mode failure instead of a download.
    os.environ.setdefault("EMBEDDING_MODEL_LOCAL_ONLY", "true")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def project_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def parse_modes(value: str) -> list[str]:
    modes = [item.strip().lower() for item in str(value).split(",") if item.strip()]
    invalid = [mode for mode in modes if mode not in SUPPORTED_MODES]
    if not modes or invalid:
        detail = ", ".join(invalid) if invalid else "empty mode list"
        raise argparse.ArgumentTypeError(f"invalid retrieval modes: {detail}")
    return list(dict.fromkeys(modes))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate local RAG retrieval modes with a fixed offline case set.",
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES_PATH))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--modes", type=parse_modes, default=list(BASE_MODES))
    parser.add_argument("--output", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN_OUTPUT))
    parser.add_argument("--with-answer", action="store_true")
    parser.add_argument("--with-judge", action="store_true")
    parser.add_argument("--with-reranker", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.with_reranker and RERANKER_MODE not in args.modes:
        args.modes.append(RERANKER_MODE)
    return args


def _string_list(value: Any, field: str, case_id: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"case {case_id}: {field} must be a list of non-empty strings")
    return [item.strip() for item in value]


def load_cases(path: str | Path) -> list[dict]:
    case_path = project_path(path)
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("evaluation cases must be a non-empty JSON list")

    cases = []
    seen_ids = set()
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"case {index}: expected an object")
        case_id = str(raw_case.get("id") or "").strip()
        question = str(raw_case.get("question") or "").strip()
        if not case_id or not question:
            raise ValueError(f"case {index}: id and question are required")
        if case_id in seen_ids:
            raise ValueError(f"duplicate case id: {case_id}")
        seen_ids.add(case_id)
        expected_sources = _string_list(
            raw_case.get("expected_sources", []), "expected_sources", case_id
        )
        is_negative = bool(raw_case.get("is_negative", False))
        cases.append({
            "id": case_id,
            "question": question,
            "expected_keywords": _string_list(raw_case.get("expected_keywords", []), "expected_keywords", case_id),
            "expected_sources": expected_sources,
            "case_type": str(raw_case.get("case_type") or "fact_lookup").strip(),
            "batch": str(raw_case.get("batch") or "unspecified").strip(),
            "is_negative": is_negative,
            "requires_ocr": bool(raw_case.get("requires_ocr", False)),
            "notes": str(raw_case.get("notes") or "").strip(),
        })
    return cases


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _truncate(value: Any, limit: int = SNIPPET_LENGTH) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _normalize_source(value: str) -> str:
    return str(value or "").replace("\\", "/").strip().casefold()


def _source_matches(expected: str, actual: str) -> bool:
    expected_key = _normalize_source(expected)
    actual_key = _normalize_source(actual)
    if not expected_key or not actual_key:
        return False
    return (
        expected_key == actual_key
        or actual_key.endswith("/" + expected_key)
        or Path(actual_key).name == Path(expected_key).name
    )


def compute_ranking_metrics(
    chunks: list[dict],
    expected_sources: list[str],
    expected_keywords: list[str],
) -> dict:
    best_source_rank = None
    best_keyword_rank = None

    for rank, chunk in enumerate(chunks, start=1):
        source = str(chunk.get("source") or "")
        if best_source_rank is None and any(
            _source_matches(expected, source) for expected in expected_sources
        ):
            best_source_rank = rank

        keyword_text = " ".join(
            str(chunk.get(key) or "") for key in ("source", "text", "snippet")
        ).casefold()
        if best_keyword_rank is None and any(
            keyword.casefold() in keyword_text for keyword in expected_keywords
        ):
            best_keyword_rank = rank

        if best_source_rank is not None and best_keyword_rank is not None:
            break

    return {
        "top1_source_hit": int(best_source_rank is not None and best_source_rank <= 1),
        "top3_source_hit": int(best_source_rank is not None and best_source_rank <= 3),
        "top5_source_hit": int(best_source_rank is not None and best_source_rank <= 5),
        "top_k_source_hit": int(best_source_rank is not None),
        "mrr": 1.0 / best_source_rank if best_source_rank is not None else 0.0,
        "best_expected_source_rank": best_source_rank,
        "best_expected_keyword_rank": best_keyword_rank,
    }


def serialize_chunk(chunk: dict, rank: int) -> dict:
    text = chunk.get("text") or chunk.get("snippet") or ""
    payload = {
        "rank": rank,
        "source": str(chunk.get("source") or "unknown"),
        "score": _float_or_none(chunk.get("score")),
        "retrieval": str(chunk.get("retrieval") or "unknown"),
        "snippet": _truncate(text),
    }
    for key in (
        "vector_score",
        "bm25_score",
        "vector_rank",
        "bm25_rank",
        "rerank_score",
        "rerank_rank",
        "reranker_used",
        "chunk_id",
    ):
        value = chunk.get(key)
        if value is not None:
            payload[key] = _float_or_none(value) if key.endswith("score") else value
    return payload


def score_retrieval(case: dict, mode: str, top_k: int, chunks: list[dict]) -> dict:
    ranked_chunks = chunks[:top_k]
    serialized = [serialize_chunk(chunk, rank) for rank, chunk in enumerate(ranked_chunks, start=1)]
    haystack = "\n".join(
        f"{chunk['source']} {chunk['snippet']}"
        for chunk in serialized
    ).casefold()
    actual_sources = [chunk["source"] for chunk in serialized]
    matched_keywords = [
        keyword
        for keyword in case["expected_keywords"]
        if keyword.casefold() in haystack
    ]
    matched_sources = [
        expected
        for expected in case["expected_sources"]
        if any(_source_matches(expected, actual) for actual in actual_sources)
    ]
    unique_sources = list(dict.fromkeys(actual_sources))
    keyword_hits = len(matched_keywords)
    source_hits = len(matched_sources)
    ranking_metrics = compute_ranking_metrics(
        ranked_chunks,
        case["expected_sources"],
        case["expected_keywords"],
    )
    return {
        "success": True,
        "error": None,
        "warning": None,
        "retrieval_mode": mode,
        "top_k": top_k,
        "sources": unique_sources,
        "scores": [chunk["score"] for chunk in serialized],
        "snippets": [chunk["snippet"] for chunk in serialized],
        "chunks": serialized,
        "matched_expected_keywords": matched_keywords,
        "matched_expected_sources": matched_sources,
        "keyword_hit_count": keyword_hits,
        "source_hit_count": source_hits,
        "retrieval_score": keyword_hits + source_hits,
        "ranking_metrics": ranking_metrics,
        "reranker_enabled": mode == RERANKER_MODE,
        "reranker_used": any(bool(chunk.get("reranker_used")) for chunk in serialized),
        "is_negative": bool(case.get("is_negative", False)),
        "fallback_success": bool(case.get("is_negative", False) and not serialized),
        "source_pollution": bool(case.get("is_negative", False) and serialized),
    }


def failed_result(mode: str, top_k: int, error: Exception | str) -> dict:
    return {
        "success": False,
        "error": str(error),
        "warning": None,
        "retrieval_mode": mode,
        "top_k": top_k,
        "sources": [],
        "scores": [],
        "snippets": [],
        "chunks": [],
        "matched_expected_keywords": [],
        "matched_expected_sources": [],
        "keyword_hit_count": 0,
        "source_hit_count": 0,
        "retrieval_score": 0,
        "ranking_metrics": {
            "top1_source_hit": 0,
            "top3_source_hit": 0,
            "top5_source_hit": 0,
            "top_k_source_hit": 0,
            "mrr": 0.0,
            "best_expected_source_rank": None,
            "best_expected_keyword_rank": None,
        },
        "reranker_enabled": mode == RERANKER_MODE,
        "reranker_used": False,
        "is_negative": False,
        "fallback_success": False,
        "source_pollution": False,
        "latency_ms": 0.0,
    }


def run_retrieval(
    case: dict,
    mode: str,
    top_k: int,
    search_fn: Callable[..., Any] | None = None,
) -> dict:
    started_at = perf_counter()
    try:
        if search_fn is None:
            configure_offline_embedding()
            from backend.rag_store import search_relevant_chunks

            search_fn = search_relevant_chunks
        retrieval_mode = "hybrid" if mode == RERANKER_MODE else mode
        raw_result = search_fn(
            case["question"],
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            reranker_enabled=mode == RERANKER_MODE,
            include_metadata=True,
        )
        metadata = raw_result if isinstance(raw_result, dict) else {"chunks": raw_result}
        chunks = metadata.get("chunks", [])
        if not isinstance(chunks, list):
            raise TypeError("retrieval result chunks must be a list")
        result = score_retrieval(case, mode, top_k, chunks)
        retrieval_error = metadata.get("error")
        if retrieval_error and not chunks and mode == "vector":
            result = failed_result(mode, top_k, retrieval_error)
            result["is_negative"] = bool(case.get("is_negative", False))
            result["latency_ms"] = round((perf_counter() - started_at) * 1000, 3)
            return result
        if retrieval_error:
            result["warning"] = str(retrieval_error)
        for key in (
            "expanded_query",
            "candidate_k",
            "vector_candidates",
            "bm25_candidates",
            "hybrid_used",
            "threshold",
            "highest_score",
            "passed_threshold",
            "reranker_enabled",
            "reranker_used",
            "reranker_model",
            "reranker_top_n",
            "reranker_error",
        ):
            if key in metadata:
                result[key] = metadata.get(key)
        result["latency_ms"] = round((perf_counter() - started_at) * 1000, 3)
        return result
    except Exception as error:
        result = failed_result(mode, top_k, error)
        result["is_negative"] = bool(case.get("is_negative", False))
        result["latency_ms"] = round((perf_counter() - started_at) * 1000, 3)
        return result


def _run_answer(question: str, mode: str, top_k: int, answer_fn: Callable[..., dict] | None) -> dict:
    try:
        if answer_fn is None:
            from backend.rag_service import rag_answer_with_sources

            answer_fn = rag_answer_with_sources
        retrieval_mode = "hybrid" if mode == RERANKER_MODE else mode
        result = answer_fn(
            question,
            top_k=top_k,
            retrieval_mode=retrieval_mode,
            reranker_enabled=mode == RERANKER_MODE,
        )
        return {
            "success": True,
            "error": None,
            "text": str(result.get("answer") or ""),
            "sources": result.get("sources") or [],
            "passed_threshold": result.get("passed_threshold"),
        }
    except Exception as error:
        return {"success": False, "error": str(error), "text": "", "sources": []}


def _run_judge(
    case: dict,
    retrieval: dict,
    answer: dict,
    judge_fn: Callable[..., dict] | None,
    judge_enabled: bool,
) -> dict:
    if not judge_enabled:
        return {"success": False, "error": "ENABLE_LLM_JUDGE is disabled"}
    if not answer.get("success") or not answer.get("text"):
        return {"success": False, "error": answer.get("error") or "answer generation failed"}
    try:
        if judge_fn is None:
            from backend.judge_service import judge_answer

            judge_fn = judge_answer
        sources = answer.get("sources") or retrieval.get("chunks") or []
        evaluation = judge_fn(case["question"], answer["text"], sources=sources)
        return {"success": True, "error": None, "evaluation": evaluation}
    except Exception as error:
        return {"success": False, "error": str(error)}


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) * percentile + 0.999999)) - 1))
    return ordered[index]


def build_mode_summary(cases: list[dict], modes: list[str]) -> dict:
    summary = {}
    case_count = len(cases)
    for mode in modes:
        results = [case["results"][mode] for case in cases]
        total_keyword_hits = sum(result["keyword_hit_count"] for result in results)
        total_source_hits = sum(result["source_hit_count"] for result in results)
        total_score = sum(result["retrieval_score"] for result in results)
        positive_results = [
            case["results"][mode]
            for case in cases
            if not bool(case.get("is_negative", False))
        ]
        negative_results = [
            case["results"][mode]
            for case in cases
            if bool(case.get("is_negative", False))
        ]
        ranking_metrics = [result.get("ranking_metrics") or {} for result in positive_results]
        source_ranks = [
            metrics.get("best_expected_source_rank")
            for metrics in ranking_metrics
            if metrics.get("best_expected_source_rank") is not None
        ]
        keyword_ranks = [
            metrics.get("best_expected_keyword_rank")
            for metrics in ranking_metrics
            if metrics.get("best_expected_keyword_rank") is not None
        ]
        positive_count = len(positive_results)
        negative_count = len(negative_results)
        latencies = [float(result.get("latency_ms", 0.0)) for result in results]
        top1_hits = sum(int(metrics.get("top1_source_hit", 0)) for metrics in ranking_metrics)
        top3_hits = sum(int(metrics.get("top3_source_hit", 0)) for metrics in ranking_metrics)
        top5_hits = sum(int(metrics.get("top5_source_hit", 0)) for metrics in ranking_metrics)
        top_k_hits = sum(int(metrics.get("top_k_source_hit", 0)) for metrics in ranking_metrics)
        fallback_successes = sum(bool(result.get("fallback_success")) for result in negative_results)
        pollution_cases = sum(bool(result.get("source_pollution")) for result in negative_results)
        summary[mode] = {
            "total_keyword_hits": total_keyword_hits,
            "total_source_hits": total_source_hits,
            "average_retrieval_score": round(total_score / case_count, 3) if case_count else 0.0,
            "successful_cases": sum(bool(result.get("success")) for result in results),
            "failed_cases": sum(not bool(result.get("success")) for result in results),
            "reranker_used_cases": sum(bool(result.get("reranker_used")) for result in results),
            "positive_case_count": positive_count,
            "negative_case_count": negative_count,
            "total_top1_source_hits": top1_hits,
            "total_top3_source_hits": top3_hits,
            "total_top5_source_hits": top5_hits,
            "total_top_k_source_hits": top_k_hits,
            "top1_source_hit_rate": round(top1_hits / positive_count, 4) if positive_count else 0.0,
            "top3_source_hit_rate": round(top3_hits / positive_count, 4) if positive_count else 0.0,
            "top5_source_hit_rate": round(top5_hits / positive_count, 4) if positive_count else 0.0,
            "top_k_source_hit_rate": round(top_k_hits / positive_count, 4) if positive_count else 0.0,
            "average_mrr": round(
                sum(float(metrics.get("mrr", 0.0)) for metrics in ranking_metrics) / positive_count,
                3,
            ) if positive_count else 0.0,
            "average_best_expected_source_rank": round(
                sum(source_ranks) / len(source_ranks), 3
            ) if source_ranks else None,
            "average_best_expected_keyword_rank": round(
                sum(keyword_ranks) / len(keyword_ranks), 3
            ) if keyword_ranks else None,
            "average_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
            "p95_latency_ms": round(_percentile(latencies, 0.95), 3),
            "fallback_success_count": fallback_successes,
            "fallback_success_rate": round(fallback_successes / negative_count, 4) if negative_count else None,
            "source_pollution_count": pollution_cases,
            "source_pollution_rate": round(pollution_cases / negative_count, 4) if negative_count else None,
        }
    return summary


def build_hybrid_reranker_diagnostics(cases: list[dict]) -> list[dict]:
    diagnostics = []
    for case in cases:
        if bool(case.get("is_negative", False)):
            continue
        results = case.get("results", {})
        if "hybrid" not in results or RERANKER_MODE not in results:
            continue

        hybrid = results["hybrid"]
        reranked = results[RERANKER_MODE]
        hybrid_rank = (hybrid.get("ranking_metrics") or {}).get("best_expected_source_rank")
        reranked_rank = (reranked.get("ranking_metrics") or {}).get("best_expected_source_rank")

        if hybrid_rank is None and reranked_rank is None:
            verdict = "same"
        elif hybrid_rank is None:
            verdict = "improved"
        elif reranked_rank is None:
            verdict = "worse"
        elif reranked_rank < hybrid_rank:
            verdict = "improved"
        elif reranked_rank > hybrid_rank:
            verdict = "worse"
        else:
            verdict = "same"

        diagnostics.append({
            "case_id": case.get("id"),
            "hybrid_best_source_rank": hybrid_rank,
            "hybrid_reranker_best_source_rank": reranked_rank,
            "rank_delta": (
                hybrid_rank - reranked_rank
                if hybrid_rank is not None and reranked_rank is not None
                else None
            ),
            "hybrid_source_hit_count": hybrid.get("source_hit_count", 0),
            "hybrid_reranker_source_hit_count": reranked.get("source_hit_count", 0),
            "verdict": verdict,
        })
    return diagnostics


def evaluate_cases(
    cases: list[dict],
    modes: list[str],
    top_k: int,
    *,
    with_answer: bool = False,
    with_judge: bool = False,
    search_fn: Callable[..., Any] | None = None,
    answer_fn: Callable[..., dict] | None = None,
    judge_fn: Callable[..., dict] | None = None,
    judge_enabled: bool | None = None,
) -> dict:
    if judge_enabled is None and with_judge:
        from backend.judge_service import is_llm_judge_enabled

        judge_enabled = is_llm_judge_enabled()
    effective_answer = with_answer or with_judge
    evaluated_cases = []
    for case in cases:
        case_result = {**case, "results": {}}
        for mode in modes:
            retrieval = run_retrieval(case, mode, top_k, search_fn=search_fn)
            if effective_answer:
                retrieval["answer"] = _run_answer(case["question"], mode, top_k, answer_fn)
            if with_judge:
                retrieval["judge"] = _run_judge(
                    case,
                    retrieval,
                    retrieval["answer"],
                    judge_fn,
                    bool(judge_enabled),
                )
            case_result["results"][mode] = retrieval
        evaluated_cases.append(case_result)

    report = {
        "summary": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "case_count": len(evaluated_cases),
            "positive_case_count": sum(not bool(case.get("is_negative", False)) for case in evaluated_cases),
            "negative_case_count": sum(bool(case.get("is_negative", False)) for case in evaluated_cases),
            "modes": modes,
            "top_k": top_k,
            "with_answer": effective_answer,
            "with_judge": with_judge,
            "with_reranker": RERANKER_MODE in modes,
        },
        "mode_summary": build_mode_summary(evaluated_cases, modes),
        "cases": evaluated_cases,
    }
    if "hybrid" in modes and RERANKER_MODE in modes:
        report["hybrid_reranker_diagnostics"] = build_hybrid_reranker_diagnostics(
            evaluated_cases
        )
    return report


def _md_escape(value: Any) -> str:
    return str(value if value is not None else "-").replace("|", "\\|").replace("\n", " ")


def _md_metric(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# RAG Retrieval Evaluation",
        "",
        "## Summary",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Positive cases: {summary.get('positive_case_count', summary['case_count'])}",
        f"- Negative cases: {summary.get('negative_case_count', 0)}",
        f"- Top K: {summary['top_k']}",
        f"- Modes: {', '.join(summary['modes'])}",
        f"- With answer: {summary['with_answer']}",
        f"- With judge: {summary['with_judge']}",
        f"- With reranker: {summary.get('with_reranker', False)}",
        "",
        "| Mode | Top-1 | Top-3 | Top-K | Avg MRR | Avg Latency ms | P95 ms | Fallback Success | Source Pollution | Success | Failed | Reranker Used |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in summary["modes"]:
        item = report["mode_summary"][mode]
        lines.append(
            f"| {mode} | {_md_metric(item['top1_source_hit_rate'])} | "
            f"{_md_metric(item['top3_source_hit_rate'])} | {_md_metric(item['top_k_source_hit_rate'])} | "
            f"{_md_metric(item['average_mrr'])} | {_md_metric(item['average_latency_ms'])} | "
            f"{_md_metric(item['p95_latency_ms'])} | {_md_metric(item['fallback_success_rate'])} | "
            f"{_md_metric(item['source_pollution_rate'])} | "
            f"{item['successful_cases']} | {item['failed_cases']} | "
            f"{item.get('reranker_used_cases', 0)} |"
        )

    diagnostics = report.get("hybrid_reranker_diagnostics")
    if diagnostics is None and "hybrid" in summary["modes"] and RERANKER_MODE in summary["modes"]:
        diagnostics = build_hybrid_reranker_diagnostics(report["cases"])
    if diagnostics is not None:
        lines.extend([
            "",
            "## Hybrid vs Hybrid Reranker Diagnostics",
            "",
            "Positive rank delta means the expected source moved closer to rank 1.",
            "",
            "| Case ID | Hybrid Best Source Rank | Hybrid Reranker Best Source Rank | Rank Delta | Hybrid Source Hits | Hybrid Reranker Source Hits | Verdict |",
            "|---|---:|---:|---:|---:|---:|---|",
        ])
        for item in diagnostics:
            lines.append(
                f"| {_md_escape(item['case_id'])} | {_md_metric(item['hybrid_best_source_rank'])} | "
                f"{_md_metric(item['hybrid_reranker_best_source_rank'])} | "
                f"{_md_metric(item['rank_delta'])} | {item['hybrid_source_hit_count']} | "
                f"{item['hybrid_reranker_source_hit_count']} | {item['verdict']} |"
            )

    lines.extend(["", "## Case Details", ""])
    for case in report["cases"]:
        lines.extend([
            f"### {case['id']} - {case['question']}",
            "",
            f"**Expected keywords:** {', '.join(case['expected_keywords']) or '(none)'}",
            "",
            f"**Expected sources:** {', '.join(case['expected_sources']) or '(none)'}",
            "",
            f"**Case type:** {case.get('case_type', 'fact_lookup')}",
            "",
            f"**Negative sample:** {case.get('is_negative', False)}",
            "",
        ])
        if case.get("notes"):
            lines.extend([f"**Notes:** {case['notes']}", ""])
        for mode in summary["modes"]:
            result = case["results"][mode]
            ranking = result.get("ranking_metrics") or {}
            lines.extend([
                f"#### {mode.title()}",
                "",
                f"- Status: {'OK' if result['success'] else 'FAILED'}",
                f"- Keyword hits: {result['keyword_hit_count']} ({', '.join(result['matched_expected_keywords']) or 'none'})",
                f"- Source hits: {result['source_hit_count']} ({', '.join(result['matched_expected_sources']) or 'none'})",
                f"- Retrieval score: {result['retrieval_score']}",
                f"- Best expected source rank: {_md_metric(ranking.get('best_expected_source_rank'))}",
                f"- Best expected keyword rank: {_md_metric(ranking.get('best_expected_keyword_rank'))}",
                f"- MRR: {_md_metric(ranking.get('mrr', 0.0))}",
                f"- Top1/Top3/Top5 source hit: {ranking.get('top1_source_hit', 0)}/{ranking.get('top3_source_hit', 0)}/{ranking.get('top5_source_hit', 0)}",
                f"- Latency ms: {_md_metric(result.get('latency_ms'))}",
                f"- Fallback success: {result.get('fallback_success', False)}",
                f"- Source pollution: {result.get('source_pollution', False)}",
                f"- Reranker enabled: {result.get('reranker_enabled', False)}",
                f"- Reranker used: {result.get('reranker_used', False)}",
            ])
            if result.get("reranker_error"):
                lines.append(f"- Reranker error: `{_md_escape(result['reranker_error'])}`")
            if result.get("error"):
                lines.append(f"- Error: `{_md_escape(result['error'])}`")
            if result.get("warning"):
                lines.append(f"- Warning: `{_md_escape(result['warning'])}`")
            lines.extend([
                "",
                "| Rank | Source | Score | Retrieval | Vector Score | BM25 Score | Rerank Score | Rerank Rank |",
                "|---:|---|---:|---|---:|---:|---:|---:|",
            ])
            if result["chunks"]:
                for chunk in result["chunks"]:
                    lines.append(
                        f"| {chunk['rank']} | {_md_escape(chunk['source'])} | {_md_escape(chunk['score'])} | "
                        f"{_md_escape(chunk['retrieval'])} | {_md_escape(chunk.get('vector_score'))} | "
                        f"{_md_escape(chunk.get('bm25_score'))} | {_md_escape(chunk.get('rerank_score'))} | "
                        f"{_md_escape(chunk.get('rerank_rank'))} |"
                    )
                lines.append("")
                for chunk in result["chunks"]:
                    lines.extend([
                        f"**Snippet {chunk['rank']}:** {_md_escape(chunk['snippet'])}",
                        "",
                    ])
            else:
                lines.extend(["| - | No chunks | - | - | - | - | - | - |", ""])
            answer = result.get("answer")
            if answer:
                lines.extend([
                    f"**Answer status:** {'OK' if answer.get('success') else 'FAILED'}",
                    "",
                    _truncate(answer.get("text") or answer.get("error"), 1000),
                    "",
                ])
            judge = result.get("judge")
            if judge:
                evaluation = judge.get("evaluation") or {}
                lines.extend([
                    f"**Judge status:** {'OK' if judge.get('success') else 'FAILED'}",
                    f"**Judge score:** {evaluation.get('overall_score', '-')}",
                    f"**Judge error:** {judge.get('error') or '-'}",
                    "",
                ])
    return "\n".join(lines).rstrip() + "\n"


def write_json_report(report: dict, path: str | Path) -> Path:
    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_markdown_report(report: dict, path: str | Path) -> Path:
    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_markdown(report), encoding="utf-8")
    return output_path


def main() -> int:
    args = parse_args()
    try:
        cases_path = project_path(args.cases)
        cases = load_cases(cases_path)
        print(f"Evaluating {len(cases)} cases across modes: {', '.join(args.modes)}")
        report = evaluate_cases(
            cases,
            args.modes,
            args.top_k,
            with_answer=args.with_answer,
            with_judge=args.with_judge,
        )
        report["summary"]["cases_file"] = str(cases_path)
        json_path = write_json_report(report, args.output)
        markdown_path = write_markdown_report(report, args.markdown)
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {markdown_path}")
        for mode, item in report["mode_summary"].items():
            print(
                f"{mode}: keyword_hits={item['total_keyword_hits']} "
                f"source_hits={item['total_source_hits']} "
                f"top1={item['top1_source_hit_rate']:.3f} "
                f"top3={item['top3_source_hit_rate']:.3f} "
                f"mrr={item['average_mrr']:.3f} "
                f"avg_latency_ms={item['average_latency_ms']:.3f} "
                f"p95_latency_ms={item['p95_latency_ms']:.3f} "
                f"failed={item['failed_cases']}"
            )
        return 0
    except Exception as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
