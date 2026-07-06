from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_rag_retrieval import (  # noqa: E402
    BASE_MODES,
    RERANKER_MODE,
    configure_offline_embedding,
    evaluate_cases,
    load_cases,
    run_retrieval,
    write_json_report,
    write_markdown_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild the local RAG index and run one reproducible corpus batch benchmark.",
    )
    parser.add_argument("--version", required=True, help="Batch label such as v1, v2, or v3")
    parser.add_argument("--cases", required=True, nargs="+")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/rag_corpus_benchmark")
    parser.add_argument("--without-reranker", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    return args


def collect_parse_metrics(documents: list[dict]) -> dict:
    methods = Counter(str(document.get("parse_method") or "unknown") for document in documents)
    extensions = Counter(Path(str(document.get("source") or "")).suffix.lower() or "<none>" for document in documents)
    failures = [
        document
        for document in documents
        if not str(document.get("text") or "").strip()
        or str(document.get("parse_method") or "") == "failed"
    ]
    return {
        "document_count": len(documents),
        "document_count_by_extension": dict(sorted(extensions.items())),
        "parse_method_counts": dict(sorted(methods.items())),
        "parse_failure_count": len(failures),
        "parse_failure_sources": [str(document.get("source") or "unknown") for document in failures],
        "ocr_trigger_count": sum(bool(document.get("need_ocr")) for document in documents),
        "ocr_used_count": sum(bool(document.get("ocr_used")) for document in documents),
        "corrupted_pdf_count": sum(bool(document.get("corrupted_pdf")) for document in documents),
        "safe_fallback_count": sum(bool(document.get("safe_fallback")) for document in documents),
        "documents": [
            {
                "source": str(document.get("source") or "unknown"),
                "parse_method": str(document.get("parse_method") or "unknown"),
                "text_char_count": int(document.get("text_char_count") or 0),
                "need_ocr": bool(document.get("need_ocr")),
                "ocr_used": bool(document.get("ocr_used")),
                "corrupted_pdf": bool(document.get("corrupted_pdf")),
                "safe_fallback": bool(document.get("safe_fallback")),
                "warnings": list(document.get("warnings") or []),
            }
            for document in documents
        ],
    }


def rebuild_and_measure(version: str) -> dict:
    from backend import rag_store
    from backend.config import get_config

    config = get_config()
    documents = rag_store.load_documents()
    metrics = collect_parse_metrics(documents)
    started_at = perf_counter()
    rag_store.rebuild_rag_index(documents=documents)
    build_ms = (perf_counter() - started_at) * 1000
    status = rag_store.get_rag_index_status()
    qs = getattr(rag_store, "last_build_quality_stats", {})
    quality_summary = {
        "chunks_kept": qs.get("kept", 0),
        "chunks_low_quality": qs.get("low_quality", 0),
        "chunks_dropped": qs.get("dropped", 0),
        "dropped_count": qs.get("dropped", 0),
        "low_quality_count": qs.get("low_quality", 0),
    }
    metrics.update({
        "version": version,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "index_build_ms": round(build_ms, 3),
        "chunk_count": len(rag_store.chunks),
        "index_ready": bool(status.get("ready")),
        "index_error": status.get("error"),
        "embedding_model": config.embedding_model,
        "embedding_model_local_only": config.embedding_model_local_only,
        "ocr_enabled": config.enable_ocr,
        "ocr_engine": config.ocr_engine,
        "reranker_configured": config.enable_reranker and bool(config.reranker_model),
        "reranker_model": config.reranker_model or None,
        "quality_filter": quality_summary,
    })
    return metrics


def warm_up(cases: list[dict], modes: list[str], top_k: int) -> dict[str, float]:
    case = next((item for item in cases if not item.get("is_negative")), cases[0])
    timings = {}
    for mode in modes:
        started_at = perf_counter()
        run_retrieval(case, mode, top_k)
        timings[mode] = round((perf_counter() - started_at) * 1000, 3)
    return timings


def main() -> int:
    args = parse_args()
    configure_offline_embedding()
    version = args.version.strip().lower()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    seen_case_ids = set()
    for case_path in args.cases:
        for case in load_cases(case_path):
            if case["id"] in seen_case_ids:
                raise ValueError(f"duplicate case id across files: {case['id']}")
            seen_case_ids.add(case["id"])
            cases.append(case)
    modes = list(BASE_MODES)
    if not args.without_reranker:
        modes.append(RERANKER_MODE)

    index_metrics = rebuild_and_measure(version)
    index_path = output_dir / f"{version}_index.json"
    index_path.write_text(json.dumps(index_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.build_only:
        print(f"Index metrics: {index_path}")
        print(
            f"documents={index_metrics['document_count']} chunks={index_metrics['chunk_count']} "
            f"build_ms={index_metrics['index_build_ms']:.3f} "
            f"parse_failures={index_metrics['parse_failure_count']} "
            f"ocr_triggered={index_metrics['ocr_trigger_count']} ocr_used={index_metrics['ocr_used_count']}"
        )
        return 0
    warmup_latency_ms = warm_up(cases, modes, args.top_k)
    report = evaluate_cases(cases, modes, args.top_k)
    report["summary"].update({
        "version": version,
        "cases_files": [str(Path(path)) for path in args.cases],
        "warmup_latency_ms": warmup_latency_ms,
        "index_metrics": index_metrics,
    })

    json_path = write_json_report(report, output_dir / f"{version}_retrieval.json")
    markdown_path = write_markdown_report(report, output_dir / f"{version}_retrieval.md")

    print(f"Index metrics: {index_path}")
    print(f"Retrieval JSON: {json_path}")
    print(f"Retrieval Markdown: {markdown_path}")
    print(
        f"documents={index_metrics['document_count']} chunks={index_metrics['chunk_count']} "
        f"build_ms={index_metrics['index_build_ms']:.3f} "
        f"parse_failures={index_metrics['parse_failure_count']} "
        f"ocr_triggered={index_metrics['ocr_trigger_count']} ocr_used={index_metrics['ocr_used_count']}"
    )
    for mode, item in report["mode_summary"].items():
        print(
            f"{mode}: top1={item['top1_source_hit_rate']:.3f} "
            f"top3={item['top3_source_hit_rate']:.3f} mrr={item['average_mrr']:.3f} "
            f"avg_ms={item['average_latency_ms']:.3f} p95_ms={item['p95_latency_ms']:.3f} "
            f"fallback={item['fallback_success_rate']} pollution={item['source_pollution_rate']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
