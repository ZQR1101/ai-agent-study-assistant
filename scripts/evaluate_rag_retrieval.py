from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CASES_PATH = Path("eval_cases/rag_retrieval_cases.json")
DEFAULT_JSON_OUTPUT = Path("outputs/rag_retrieval_eval.json")
DEFAULT_MARKDOWN_OUTPUT = Path("outputs/rag_retrieval_eval.md")
SUPPORTED_MODES = ("vector", "bm25", "hybrid")
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
    parser.add_argument("--modes", type=parse_modes, default=list(SUPPORTED_MODES))
    parser.add_argument("--output", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN_OUTPUT))
    parser.add_argument("--with-answer", action="store_true")
    parser.add_argument("--with-judge", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
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
        cases.append({
            "id": case_id,
            "question": question,
            "expected_keywords": _string_list(raw_case.get("expected_keywords", []), "expected_keywords", case_id),
            "expected_sources": _string_list(raw_case.get("expected_sources", []), "expected_sources", case_id),
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


def serialize_chunk(chunk: dict, rank: int) -> dict:
    text = chunk.get("text") or chunk.get("snippet") or ""
    payload = {
        "rank": rank,
        "source": str(chunk.get("source") or "unknown"),
        "score": _float_or_none(chunk.get("score")),
        "retrieval": str(chunk.get("retrieval") or "unknown"),
        "snippet": _truncate(text),
    }
    for key in ("vector_score", "bm25_score", "vector_rank", "bm25_rank", "chunk_id"):
        value = chunk.get(key)
        if value is not None:
            payload[key] = _float_or_none(value) if key.endswith("score") else value
    return payload


def score_retrieval(case: dict, mode: str, top_k: int, chunks: list[dict]) -> dict:
    serialized = [serialize_chunk(chunk, rank) for rank, chunk in enumerate(chunks[:top_k], start=1)]
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
    }


def run_retrieval(
    case: dict,
    mode: str,
    top_k: int,
    search_fn: Callable[..., Any] | None = None,
) -> dict:
    try:
        if search_fn is None:
            configure_offline_embedding()
            from backend.rag_store import search_relevant_chunks

            search_fn = search_relevant_chunks
        raw_result = search_fn(
            case["question"],
            top_k=top_k,
            retrieval_mode=mode,
            include_metadata=True,
        )
        metadata = raw_result if isinstance(raw_result, dict) else {"chunks": raw_result}
        chunks = metadata.get("chunks", [])
        if not isinstance(chunks, list):
            raise TypeError("retrieval result chunks must be a list")
        result = score_retrieval(case, mode, top_k, chunks)
        retrieval_error = metadata.get("error")
        if retrieval_error and not chunks and mode == "vector":
            return failed_result(mode, top_k, retrieval_error)
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
        ):
            if key in metadata:
                result[key] = metadata.get(key)
        return result
    except Exception as error:
        return failed_result(mode, top_k, error)


def _run_answer(question: str, mode: str, top_k: int, answer_fn: Callable[..., dict] | None) -> dict:
    try:
        if answer_fn is None:
            from backend.rag_service import rag_answer_with_sources

            answer_fn = rag_answer_with_sources
        result = answer_fn(question, top_k=top_k, retrieval_mode=mode)
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


def build_mode_summary(cases: list[dict], modes: list[str]) -> dict:
    summary = {}
    case_count = len(cases)
    for mode in modes:
        results = [case["results"][mode] for case in cases]
        total_keyword_hits = sum(result["keyword_hit_count"] for result in results)
        total_source_hits = sum(result["source_hit_count"] for result in results)
        total_score = sum(result["retrieval_score"] for result in results)
        summary[mode] = {
            "total_keyword_hits": total_keyword_hits,
            "total_source_hits": total_source_hits,
            "average_retrieval_score": round(total_score / case_count, 3) if case_count else 0.0,
            "successful_cases": sum(bool(result.get("success")) for result in results),
            "failed_cases": sum(not bool(result.get("success")) for result in results),
        }
    return summary


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

    return {
        "summary": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "case_count": len(evaluated_cases),
            "modes": modes,
            "top_k": top_k,
            "with_answer": effective_answer,
            "with_judge": with_judge,
        },
        "mode_summary": build_mode_summary(evaluated_cases, modes),
        "cases": evaluated_cases,
    }


def _md_escape(value: Any) -> str:
    return str(value if value is not None else "-").replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    lines = [
        "# RAG Retrieval Evaluation",
        "",
        "## Summary",
        "",
        f"- Cases: {summary['case_count']}",
        f"- Top K: {summary['top_k']}",
        f"- Modes: {', '.join(summary['modes'])}",
        f"- With answer: {summary['with_answer']}",
        f"- With judge: {summary['with_judge']}",
        "",
        "| Mode | Keyword Hits | Source Hits | Avg Score | Success | Failed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mode in summary["modes"]:
        item = report["mode_summary"][mode]
        lines.append(
            f"| {mode} | {item['total_keyword_hits']} | {item['total_source_hits']} | "
            f"{item['average_retrieval_score']:.3f} | {item['successful_cases']} | {item['failed_cases']} |"
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
        ])
        if case.get("notes"):
            lines.extend([f"**Notes:** {case['notes']}", ""])
        for mode in summary["modes"]:
            result = case["results"][mode]
            lines.extend([
                f"#### {mode.title()}",
                "",
                f"- Status: {'OK' if result['success'] else 'FAILED'}",
                f"- Keyword hits: {result['keyword_hit_count']} ({', '.join(result['matched_expected_keywords']) or 'none'})",
                f"- Source hits: {result['source_hit_count']} ({', '.join(result['matched_expected_sources']) or 'none'})",
                f"- Retrieval score: {result['retrieval_score']}",
            ])
            if result.get("error"):
                lines.append(f"- Error: `{_md_escape(result['error'])}`")
            if result.get("warning"):
                lines.append(f"- Warning: `{_md_escape(result['warning'])}`")
            lines.extend([
                "",
                "| Rank | Source | Score | Retrieval | Vector Score | BM25 Score |",
                "|---:|---|---:|---|---:|---:|",
            ])
            if result["chunks"]:
                for chunk in result["chunks"]:
                    lines.append(
                        f"| {chunk['rank']} | {_md_escape(chunk['source'])} | {_md_escape(chunk['score'])} | "
                        f"{_md_escape(chunk['retrieval'])} | {_md_escape(chunk.get('vector_score'))} | "
                        f"{_md_escape(chunk.get('bm25_score'))} |"
                    )
                lines.append("")
                for chunk in result["chunks"]:
                    lines.extend([
                        f"**Snippet {chunk['rank']}:** {_md_escape(chunk['snippet'])}",
                        "",
                    ])
            else:
                lines.extend(["| - | No chunks | - | - | - | - |", ""])
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
                f"avg_score={item['average_retrieval_score']:.3f} "
                f"failed={item['failed_cases']}"
            )
        return 0
    except Exception as error:
        print(f"Evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
