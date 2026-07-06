from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "rag_corpus_benchmark"
REPORT_PATH = PROJECT_ROOT / "reports" / "RAG_V1_V2_V3_BENCHMARK.md"
METRICS_PATH = PROJECT_ROOT / "reports" / "RAG_V1_V2_V3_METRICS.json"
MODES = ("vector", "bm25", "hybrid", "hybrid_reranker")
MODE_LABELS = {
    "vector": "Vector",
    "bm25": "BM25",
    "hybrid": "Hybrid",
    "hybrid_reranker": "Hybrid+Reranker",
}


def _load(name: str) -> dict:
    return json.loads((OUTPUT_DIR / name).read_text(encoding="utf-8"))


def _pct(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.1f}%"


def _num(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _ocr_subset(report: dict, mode: str) -> dict:
    cases = [
        case
        for case in report["cases"]
        if case.get("requires_ocr") and not case.get("is_negative")
    ]
    metrics = [case["results"][mode]["ranking_metrics"] for case in cases]
    count = len(metrics)
    return {
        "count": count,
        "top1": sum(item.get("top1_source_hit", 0) for item in metrics) / count if count else 0.0,
        "top3": sum(item.get("top3_source_hit", 0) for item in metrics) / count if count else 0.0,
        "top_k": sum(item.get("top_k_source_hit", 0) for item in metrics) / count if count else 0.0,
        "mrr": sum(float(item.get("mrr", 0.0)) for item in metrics) / count if count else 0.0,
    }


def _case_rank(report: dict, case_id: str, mode: str):
    case = next(case for case in report["cases"] if case["id"] == case_id)
    return case["results"][mode]["ranking_metrics"].get("best_expected_source_rank")


def render_report() -> str:
    retrieval = {
        "V1": _load("v1_retrieval.json"),
        "V2": _load("v2_retrieval.json"),
        "V3": _load("v3_ocr_on_retrieval.json"),
    }
    indexes = {
        "V1": _load("v1_index.json"),
        "V2": _load("v2_index.json"),
        "V3": _load("v3_ocr_on_index.json"),
    }
    ocr_off_report = _load("v3_ocr_off_retrieval.json")
    ocr_off_index = _load("v3_ocr_off_index.json")
    v3 = retrieval["V3"]
    v3_index = indexes["V3"]

    lines = [
        "# RAG V1/V2/V3 Local Benchmark",
        "",
        f"- generated_at: `{datetime.now(timezone.utc).isoformat()}`",
        "- environment: local Windows process, one warm-up query per mode, one measured retrieval per case",
        "- generation: no LLM answer or LLM-as-Judge calls; retrieval only",
        "- Top-K: 5",
        "- reports_source: `outputs/rag_corpus_benchmark/*.json`",
        "",
        "## Corpus and Index Comparison",
        "",
        "| Batch | Documents | Chunks | Index build ms | Parse failures | OCR triggered | OCR used | Cases (positive/negative) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for batch in ("V1", "V2", "V3"):
        index = indexes[batch]
        summary = retrieval[batch]["summary"]
        lines.append(
            f"| {batch} | {index['document_count']} | {index['chunk_count']} | "
            f"{index['index_build_ms']:.3f} | {index['parse_failure_count']} | "
            f"{index['ocr_trigger_count']} | {index['ocr_used_count']} | "
            f"{summary['case_count']} ({summary['positive_case_count']}/{summary['negative_case_count']}) |"
        )
    lines.extend([
        "",
        "V1 contains local project documents. V2 is cumulative V1 plus five curated official-document summaries. "
        "V3 is cumulative V2 plus four original research PDFs, two OCR fixtures derived from real paper pages, and a provenance manifest.",
        "",
        "`index_build_ms` is the measured `rebuild_rag_index` interval from each batch run. V3 reused the already parsed document objects to avoid a second OCR pass, so build-time trend across V2 and V3 should not be presented as a strict end-to-end speedup comparison.",
        "",
        "## Retrieval Comparison",
        "",
        "| Batch | Mode | Top-1 | Top-3 | MRR | Avg ms | P95 ms | Fallback success | Source pollution |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for batch in ("V1", "V2", "V3"):
        for mode in MODES:
            item = retrieval[batch]["mode_summary"][mode]
            lines.append(
                f"| {batch} | {MODE_LABELS[mode]} | {_pct(item['top1_source_hit_rate'])} | "
                f"{_pct(item['top3_source_hit_rate'])} | {_num(item['average_mrr'])} | "
                f"{_num(item['average_latency_ms'])} | {_num(item['p95_latency_ms'])} | "
                f"{_pct(item['fallback_success_rate'])} | {_pct(item['source_pollution_rate'])} |"
            )

    lines.extend([
        "",
        "Top-1, Top-3 and MRR use positive cases only. Fallback success and source pollution use negative cases only. "
        "A negative case is a successful fallback only when retrieval returns no sources; returning any source counts as pollution.",
        "",
        "## V3 PDF Parse Results",
        "",
        "| Source | Parse method | Characters | Need OCR | OCR used |",
        "|---|---|---:|---:|---:|",
    ])
    paper_documents = [
        item for item in v3_index["documents"] if item["source"].lower().endswith(".pdf") and item["source"].startswith("paper_")
    ]
    for item in paper_documents:
        lines.append(
            f"| `{item['source']}` | {item['parse_method']} | {item['text_char_count']} | "
            f"{item['need_ocr']} | {item['ocr_used']} |"
        )

    lines.extend([
        "",
        "## OCR On vs Off",
        "",
        "| Setting | Chunks | Parse failures | OCR triggered | OCR used |",
        "|---|---:|---:|---:|---:|",
        f"| OCR on | {v3_index['chunk_count']} | {v3_index['parse_failure_count']} | {v3_index['ocr_trigger_count']} | {v3_index['ocr_used_count']} |",
        f"| OCR off | {ocr_off_index['chunk_count']} | {ocr_off_index['parse_failure_count']} | {ocr_off_index['ocr_trigger_count']} | {ocr_off_index['ocr_used_count']} |",
        "",
        "### OCR-specific positive cases (n=2)",
        "",
        "| Mode | OCR on Top-1 | OCR off Top-1 | OCR on Top-3 | OCR off Top-3 | OCR on Top-K | OCR off Top-K | OCR on MRR | OCR off MRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in MODES:
        on = _ocr_subset(v3, mode)
        off = _ocr_subset(ocr_off_report, mode)
        lines.append(
            f"| {MODE_LABELS[mode]} | {_pct(on['top1'])} | {_pct(off['top1'])} | "
            f"{_pct(on['top3'])} | {_pct(off['top3'])} | {_pct(on['top_k'])} | "
            f"{_pct(off['top_k'])} | {_num(on['mrr'])} | {_num(off['mrr'])} |"
        )

    lines.extend([
        "",
        "### OCR case ranks",
        "",
        "| Case | Mode | OCR on rank | OCR off rank |",
        "|---|---|---:|---:|",
    ])
    for case_id in ("v3_ocr_scanned_marker", "v3_ocr_mixed_marker"):
        for mode in MODES:
            on_rank = _case_rank(v3, case_id, mode)
            off_rank = _case_rank(ocr_off_report, case_id, mode)
            lines.append(
                f"| `{case_id}` | {MODE_LABELS[mode]} | {on_rank if on_rank is not None else 'miss'} | "
                f"{off_rank if off_rank is not None else 'miss'} |"
            )

    hybrid = v3["mode_summary"]["hybrid"]
    reranker = v3["mode_summary"]["hybrid_reranker"]
    bm25 = v3["mode_summary"]["bm25"]
    vector = v3["mode_summary"]["vector"]
    latency_ratio = reranker["average_latency_ms"] / hybrid["average_latency_ms"]
    lines.extend([
        "",
        "## Conclusions",
        "",
        f"1. On the final 40 positive cases, Hybrid+Reranker achieved Top-1 {_pct(reranker['top1_source_hit_rate'])}, "
        f"Top-3 {_pct(reranker['top3_source_hit_rate'])}, and MRR {_num(reranker['average_mrr'])}. "
        f"Compared with Hybrid, MRR changed by {reranker['average_mrr'] - hybrid['average_mrr']:+.3f}, while average latency was {latency_ratio:.1f}x higher.",
        f"2. BM25 remained a strong baseline at Top-1 {_pct(bm25['top1_source_hit_rate'])}, Top-3 {_pct(bm25['top3_source_hit_rate'])}, "
        f"and MRR {_num(bm25['average_mrr'])}; on this corpus it outperformed non-reranked Hybrid MRR {_num(hybrid['average_mrr'])}.",
        f"3. OCR on added {v3_index['chunk_count'] - ocr_off_index['chunk_count']} chunks and reduced parse failures from "
        f"{ocr_off_index['parse_failure_count']} to {v3_index['parse_failure_count']}. It made the pure scanned fixture retrievable by BM25 at rank 1, "
        "Hybrid at rank 3, and Hybrid+Reranker at rank 5. Vector still missed it, and every mode missed the mixed-PDF marker in Top-5 because OCR joined the marker into one token.",
        f"4. Negative handling is not uniform: Vector fallback success was {_pct(vector['fallback_success_rate'])} with "
        f"source pollution {_pct(vector['source_pollution_rate'])}, while BM25, Hybrid and Hybrid+Reranker returned sources for every final negative case.",
        "",
        "## Cautious Resume STAR Bullets",
        "",
        f"- Expanded a local AI study assistant knowledge base in three traceable stages from {indexes['V1']['document_count']} to {v3_index['document_count']} documents and from {indexes['V1']['chunk_count']} to {v3_index['chunk_count']} indexed chunks, combining project notes, curated official documentation, and four arXiv papers.",
        f"- Built a 55-case offline retrieval benchmark ({v3['summary']['positive_case_count']} positive, {v3['summary']['negative_case_count']} negative) covering fact lookup, concept explanation, acronyms, mixed Chinese/English queries, OCR, and out-of-knowledge questions.",
        f"- Compared Vector, BM25, Hybrid RRF, and CrossEncoder reranking; on the final local run, reranking improved Hybrid MRR from {_num(hybrid['average_mrr'])} to {_num(reranker['average_mrr'])} and Top-1 from {_pct(hybrid['top1_source_hit_rate'])} to {_pct(reranker['top1_source_hit_rate'])}.",
        f"- Quantified the reranking tradeoff instead of reporting accuracy alone: average retrieval latency increased from {_num(hybrid['average_latency_ms'])} ms for Hybrid to {_num(reranker['average_latency_ms'])} ms for Hybrid+Reranker on the same machine and case set.",
        f"- Added native, scanned, and mixed PDF parsing checks; OCR recovered {v3_index['chunk_count'] - ocr_off_index['chunk_count']} additional chunks and changed a pure scanned-PDF source from a miss to rank 1 with BM25.",
        f"- Added negative-query evaluation and found a concrete reliability gap: Vector rejected {_pct(vector['fallback_success_rate'])} of final negatives, while lexical and hybrid modes had {_pct(bm25['source_pollution_rate'])} source pollution under the current no-threshold behavior.",
        "- Preserved source URLs, collection timestamps, SHA-256 hashes, per-document parse methods, and generated JSON metrics so benchmark claims can be reproduced and audited.",
        "",
        "## Results Too Small or Fragile for a Resume Claim",
        "",
        "- Do not claim a general OCR accuracy percentage: the OCR-specific positive subset contains only 2 cases and two derived fixtures.",
        "- Do not claim production latency or throughput: each query was measured once on one local Windows machine, without concurrency or repeated confidence intervals.",
        "- Do not generalize the ranking winner beyond this corpus: the final positive set has 40 cases and only four research-paper topics.",
        "- Do not present the V1/V2/V3 index-build times as a strict scaling curve: the V3 runner reused parsed documents to avoid a duplicate OCR pass.",
        "- Do not claim answer correctness from this run: it evaluates source ranking only and does not call an LLM answer generator, human annotator, or LLM-as-Judge.",
        "- Treat the 15 negative cases as a diagnostic signal, not a calibrated production out-of-domain distribution.",
        "",
    ])
    return "\n".join(lines)


def compact_metrics() -> dict:
    batches = {
        "v1": (_load("v1_index.json"), _load("v1_retrieval.json")),
        "v2": (_load("v2_index.json"), _load("v2_retrieval.json")),
        "v3_ocr_on": (_load("v3_ocr_on_index.json"), _load("v3_ocr_on_retrieval.json")),
        "v3_ocr_off": (_load("v3_ocr_off_index.json"), _load("v3_ocr_off_retrieval.json")),
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_k": 5,
        "batches": {},
    }
    for name, (index, retrieval) in batches.items():
        payload["batches"][name] = {
            "index": {
                key: index.get(key)
                for key in (
                    "document_count",
                    "document_count_by_extension",
                    "chunk_count",
                    "index_build_ms",
                    "parse_failure_count",
                    "parse_failure_sources",
                    "ocr_trigger_count",
                    "ocr_used_count",
                    "parse_method_counts",
                    "embedding_model",
                    "reranker_model",
                )
            },
            "cases": {
                key: retrieval["summary"].get(key)
                for key in ("case_count", "positive_case_count", "negative_case_count")
            },
            "mode_summary": retrieval["mode_summary"],
        }
        reranker_model = payload["batches"][name]["index"].get("reranker_model")
        if reranker_model:
            payload["batches"][name]["index"]["reranker_model"] = Path(reranker_model).name

    on_report = batches["v3_ocr_on"][1]
    off_report = batches["v3_ocr_off"][1]
    payload["ocr_specific_case_count"] = sum(
        case.get("requires_ocr") and not case.get("is_negative")
        for case in on_report["cases"]
    )
    payload["ocr_specific_mode_summary"] = {
        mode: {
            "ocr_on": _ocr_subset(on_report, mode),
            "ocr_off": _ocr_subset(off_report, mode),
        }
        for mode in MODES
    }
    payload["ocr_case_ranks"] = {
        case_id: {
            mode: {
                "ocr_on": _case_rank(on_report, case_id, mode),
                "ocr_off": _case_rank(off_report, case_id, mode),
            }
            for mode in MODES
        }
        for case_id in ("v3_ocr_scanned_marker", "v3_ocr_mixed_marker")
    }
    return payload


def main() -> int:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(render_report(), encoding="utf-8")
    METRICS_PATH.write_text(
        json.dumps(compact_metrics(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Benchmark report: {REPORT_PATH}")
    print(f"Compact metrics: {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
