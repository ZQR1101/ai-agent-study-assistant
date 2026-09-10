"""Offline parameter sweep for RAG chunking and hybrid fusion settings.

For every (chunk_size, overlap) combination an in-memory index is built
(nothing is written to rag_index/), the production retrieval path is evaluated
against the offline case set, and RRF weights plus the similarity threshold are
swept on top. Results land in a JSON + Markdown comparison report so tuning is
driven by measured metrics instead of intuition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate_rag_retrieval import (  # noqa: E402
    _md_metric,
    configure_offline_embedding,
    evaluate_cases,
    load_cases,
    project_path,
    write_json_report,
    write_markdown_report,
)

DEFAULT_CASES = Path("eval_cases/rag_retrieval_cases.json")
DEFAULT_JSON_OUTPUT = Path("outputs/rag_param_sweep.json")
DEFAULT_MARKDOWN_OUTPUT = Path("outputs/rag_param_sweep.md")
SWEEP_MODES = ("vector", "bm25", "hybrid")
METRIC_FIELDS = (
    "top1_source_hit_rate",
    "top3_source_hit_rate",
    "top_k_source_hit_rate",
    "average_mrr",
    "source_pollution_rate",
    "fallback_success_rate",
    "average_latency_ms",
    "p95_latency_ms",
    "failed_cases",
)
BEST_CONFIG_COUNT = 5


def parse_int_list(value: str) -> list[int]:
    items: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            number = int(part)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid integer: {part}") from error
        if number < 0:
            raise argparse.ArgumentTypeError(f"invalid non-negative integer: {part}")
        items.append(number)
    if not items:
        raise argparse.ArgumentTypeError("empty integer list")
    return list(dict.fromkeys(items))


def parse_float_list(value: str) -> list[float]:
    items: list[float] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            number = float(part)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"invalid number: {part}") from error
        if number <= 0:
            raise argparse.ArgumentTypeError(f"invalid positive number: {part}")
        items.append(round(number, 6))
    if not items:
        raise argparse.ArgumentTypeError("empty number list")
    return list(dict.fromkeys(items))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep chunking and hybrid fusion parameters offline and compare metrics.",
    )
    parser.add_argument("--cases", default=str(DEFAULT_CASES))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-sizes", type=parse_int_list, default=[300, 500, 800])
    parser.add_argument("--overlaps", type=parse_int_list, default=[0, 100])
    parser.add_argument("--vector-weights", type=parse_float_list, default=[1.0])
    parser.add_argument("--bm25-weights", type=parse_float_list, default=[0.6, 1.0, 1.15, 1.5])
    parser.add_argument(
        "--similarity-thresholds",
        type=parse_float_list,
        default=[0.45, 0.5, 0.55, 0.6],
    )
    parser.add_argument("--output", default=str(DEFAULT_JSON_OUTPUT))
    parser.add_argument("--markdown", default=str(DEFAULT_MARKDOWN_OUTPUT))
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    return args


def build_chunk_grid(chunk_sizes: list[int], overlaps: list[int]) -> tuple[list[dict], list[dict]]:
    combos: list[dict] = []
    skipped: list[dict] = []
    for chunk_size in chunk_sizes:
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive: {chunk_size}")
        for overlap in overlaps:
            combo = {"chunk_size": chunk_size, "overlap": overlap}
            if overlap >= chunk_size:
                skipped.append({**combo, "reason": "overlap must be smaller than chunk_size"})
            else:
                combos.append(combo)
    if not combos:
        raise ValueError("no valid chunk_size/overlap combination")
    return combos, skipped


def build_fusion_grid(
    vector_weights: list[float],
    bm25_weights: list[float],
    thresholds: list[float],
) -> list[dict]:
    return [
        {
            "vector_weight": vector_weight,
            "bm25_weight": bm25_weight,
            "similarity_threshold": threshold,
        }
        for vector_weight in vector_weights
        for bm25_weight in bm25_weights
        for threshold in thresholds
    ]


def config_label(chunk_cfg: dict, fusion_cfg: dict | None = None) -> str:
    label = f"cs{chunk_cfg['chunk_size']}-ov{chunk_cfg['overlap']}"
    if fusion_cfg:
        label += (
            f"-w{fusion_cfg['vector_weight']:g}/{fusion_cfg['bm25_weight']:g}"
            f"-t{fusion_cfg['similarity_threshold']:g}"
        )
    return label


def build_in_memory_index(
    documents: list[dict],
    chunk_size: int,
    overlap: int,
) -> tuple[list[dict], Any, dict]:
    from backend import rag_store

    started_at = perf_counter()
    chunks, quality_stats = rag_store.build_chunks(
        documents=documents,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    index = None
    if chunks:
        model = rag_store.get_embedding_model()
        faiss = rag_store._get_faiss()
        np = rag_store._get_numpy()
        embeddings = model.encode(
            [rag_store._chunk_embedding_text(chunk) for chunk in chunks]
        )
        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

    stats = {
        "chunk_count": len(chunks),
        "chunks_kept": quality_stats.get("kept", 0),
        "chunks_low_quality": quality_stats.get("low_quality", 0),
        "chunks_dropped": quality_stats.get("dropped", 0),
        "build_ms": round((perf_counter() - started_at) * 1000, 3),
    }
    return chunks, index, stats


def install_search_context(chunks: list[dict], index: Any) -> Callable[[], None]:
    from backend import rag_store

    snapshot = (rag_store.chunks, rag_store.index, rag_store.rag_index_error)
    rag_store.chunks = chunks
    rag_store.index = index
    rag_store.rag_index_error = None
    rag_store._reset_bm25_index()

    def restore() -> None:
        rag_store.chunks, rag_store.index, rag_store.rag_index_error = snapshot
        rag_store._reset_bm25_index()

    return restore


def make_search_fn(
    similarity_threshold: float,
    vector_weight: float,
    bm25_weight: float,
) -> Callable[..., Any]:
    from backend import rag_store

    def search_fn(question: str, **kwargs):
        original = (rag_store.HYBRID_VECTOR_WEIGHT, rag_store.HYBRID_BM25_WEIGHT)
        rag_store.HYBRID_VECTOR_WEIGHT = vector_weight
        rag_store.HYBRID_BM25_WEIGHT = bm25_weight
        try:
            return rag_store.search_relevant_chunks(
                question,
                similarity_threshold=similarity_threshold,
                **kwargs,
            )
        finally:
            rag_store.HYBRID_VECTOR_WEIGHT, rag_store.HYBRID_BM25_WEIGHT = original

    return search_fn


def metrics_row(config: dict, mode: str, mode_summary: dict) -> dict:
    chunk_part = {"chunk_size": config["chunk_size"], "overlap": config["overlap"]}
    fusion_part = config if "vector_weight" in config else None
    row = {
        **config,
        "mode": mode,
        "label": config_label(chunk_part, fusion_part),
    }
    for field in METRIC_FIELDS:
        row[field] = mode_summary.get(field)
    return row


def rank_fusion_rows(rows: list[dict]) -> list[dict]:
    def sort_key(row: dict):
        return (
            -(row.get("top3_source_hit_rate") or 0.0),
            -(row.get("average_mrr") or 0.0),
            row.get("source_pollution_rate")
            if row.get("source_pollution_rate") is not None
            else 1.0,
            row.get("average_latency_ms") or 0.0,
        )

    return sorted(rows, key=sort_key)


def current_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def capture_baseline_fusion() -> dict:
    from backend import rag_store

    return {
        "vector_weight": float(rag_store.HYBRID_VECTOR_WEIGHT),
        "bm25_weight": float(rag_store.HYBRID_BM25_WEIGHT),
        "similarity_threshold": float(rag_store.SIMILARITY_THRESHOLD),
    }


def run_sweep(
    cases: list[dict],
    chunk_grid: list[dict],
    skipped_chunk_combos: list[dict],
    fusion_grid: list[dict],
    top_k: int,
    cases_path: Path,
) -> dict:
    from backend import rag_store

    documents = rag_store.load_documents()
    baseline_fusion = capture_baseline_fusion()
    started_at = perf_counter()
    chunk_configs: list[dict] = []
    mode_rows: list[dict] = []
    fusion_rows: list[dict] = []

    for chunk_cfg in chunk_grid:
        label = config_label(chunk_cfg)
        print(f"[sweep] building {label} ...")
        chunks, index, stats = build_in_memory_index(
            documents,
            chunk_cfg["chunk_size"],
            chunk_cfg["overlap"],
        )
        chunk_configs.append({**chunk_cfg, **stats})
        if index is None:
            print(f"[sweep] {label}: no chunks produced, skipping evaluation")
            continue
        restore = install_search_context(chunks, index)
        try:
            baseline_fn = make_search_fn(
                baseline_fusion["similarity_threshold"],
                baseline_fusion["vector_weight"],
                baseline_fusion["bm25_weight"],
            )
            baseline_report = evaluate_cases(
                cases,
                list(SWEEP_MODES),
                top_k,
                search_fn=baseline_fn,
            )
            for mode in SWEEP_MODES:
                mode_rows.append(
                    metrics_row(
                        chunk_cfg,
                        mode,
                        baseline_report["mode_summary"][mode],
                    )
                )
            print(
                f"[sweep] {label}: baseline "
                + " ".join(
                    f"{mode} top3={baseline_report['mode_summary'][mode]['top3_source_hit_rate']:.3f}"
                    for mode in SWEEP_MODES
                )
            )

            for fusion_cfg in fusion_grid:
                fusion_label = config_label(chunk_cfg, fusion_cfg)
                search_fn = make_search_fn(
                    fusion_cfg["similarity_threshold"],
                    fusion_cfg["vector_weight"],
                    fusion_cfg["bm25_weight"],
                )
                try:
                    fusion_report = evaluate_cases(
                        cases,
                        ["hybrid"],
                        top_k,
                        search_fn=search_fn,
                    )
                    fusion_rows.append(
                        metrics_row(
                            {**chunk_cfg, **fusion_cfg},
                            "hybrid",
                            fusion_report["mode_summary"]["hybrid"],
                        )
                    )
                    summary = fusion_report["mode_summary"]["hybrid"]
                    print(
                        f"[sweep] {fusion_label}: top3={summary['top3_source_hit_rate']:.3f} "
                        f"mrr={summary['average_mrr']:.3f} "
                        f"pollution={summary['source_pollution_rate']}"
                    )
                except Exception as error:
                    print(f"[sweep] {fusion_label}: FAILED ({error})")
                    fusion_rows.append({
                        **chunk_cfg,
                        **fusion_cfg,
                        "mode": "hybrid",
                        "label": fusion_label,
                        "error": str(error),
                    })
        finally:
            restore()

    model_name_or_path, _ = rag_store.get_embedding_model_settings()
    return {
        "summary": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": current_git_commit(),
            "cases_file": str(cases_path),
            "cases_sha256": file_sha256(cases_path),
            "case_count": len(cases),
            "top_k": top_k,
            "embedding_model": model_name_or_path,
            "document_count": len(documents),
            "baseline_fusion": baseline_fusion,
            "chunk_grid": chunk_grid,
            "skipped_chunk_combos": skipped_chunk_combos,
            "fusion_grid": fusion_grid,
            "runtime_s": round(perf_counter() - started_at, 3),
        },
        "chunk_configs": chunk_configs,
        "mode_rows": mode_rows,
        "fusion_rows": fusion_rows,
        "best_configs": rank_fusion_rows(
            [row for row in fusion_rows if "error" not in row]
        )[:BEST_CONFIG_COUNT],
    }


def _md_escape(value: Any) -> str:
    return str(value if value is not None else "-").replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict) -> str:
    summary = report["summary"]
    chunk_stats = {
        config_label(config): config for config in report["chunk_configs"]
    }
    lines = [
        "# RAG Parameter Sweep",
        "",
        "## Reproducibility",
        "",
        f"- Generated at: {summary['generated_at']}",
        f"- Git commit: {summary.get('git_commit') or 'unknown'}",
        f"- Cases: {summary['cases_file']} (sha256 `{summary.get('cases_sha256') or 'unknown'}`)",
        f"- Case count: {summary['case_count']} (top_k={summary['top_k']})",
        f"- Documents: {summary['document_count']}",
        f"- Embedding model: {summary['embedding_model']}",
        f"- Baseline fusion: "
        f"w={summary['baseline_fusion']['vector_weight']:g}/{summary['baseline_fusion']['bm25_weight']:g}, "
        f"threshold={summary['baseline_fusion']['similarity_threshold']:g}",
        f"- Runtime: {summary['runtime_s']}s",
        "",
    ]
    if summary.get("skipped_chunk_combos"):
        lines.append(
            f"- Skipped chunk combos: "
            f"{', '.join(config_label(item) for item in summary['skipped_chunk_combos'])}"
        )
        lines.append("")

    lines.extend([
        "## Chunk Sweep (baseline fusion params)",
        "",
        "| Config | Chunks | Build ms | Mode | Top-1 | Top-3 | Top-K | MRR | Pollution | Fallback | Avg ms | P95 ms |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["mode_rows"]:
        config = chunk_stats.get(row["label"], {})
        lines.append(
            f"| {row['label']} | {config.get('chunk_count', '-')} | "
            f"{_md_metric(config.get('build_ms'))} | {row['mode']} | "
            f"{_md_metric(row.get('top1_source_hit_rate'))} | "
            f"{_md_metric(row.get('top3_source_hit_rate'))} | "
            f"{_md_metric(row.get('top_k_source_hit_rate'))} | "
            f"{_md_metric(row.get('average_mrr'))} | "
            f"{_md_metric(row.get('source_pollution_rate'))} | "
            f"{_md_metric(row.get('fallback_success_rate'))} | "
            f"{_md_metric(row.get('average_latency_ms'))} | "
            f"{_md_metric(row.get('p95_latency_ms'))} |"
        )

    lines.extend([
        "",
        "## Fusion Sweep (hybrid)",
        "",
        "| Config | Vec W | BM25 W | Threshold | Top-1 | Top-3 | Top-K | MRR | Pollution | Fallback | Avg ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in report["fusion_rows"]:
        if "error" in row:
            lines.append(
                f"| {row['label']} | {row.get('vector_weight')} | {row.get('bm25_weight')} | "
                f"{row.get('similarity_threshold')} | - | - | - | - | - | - | "
                f"FAILED: {_md_escape(row['error'])} |"
            )
            continue
        lines.append(
            f"| {row['label']} | {row['vector_weight']:g} | {row['bm25_weight']:g} | "
            f"{row['similarity_threshold']:g} | {_md_metric(row.get('top1_source_hit_rate'))} | "
            f"{_md_metric(row.get('top3_source_hit_rate'))} | "
            f"{_md_metric(row.get('top_k_source_hit_rate'))} | "
            f"{_md_metric(row.get('average_mrr'))} | "
            f"{_md_metric(row.get('source_pollution_rate'))} | "
            f"{_md_metric(row.get('fallback_success_rate'))} | "
            f"{_md_metric(row.get('average_latency_ms'))} |"
        )

    lines.extend([
        "",
        "## Best Fusion Configs",
        "",
        "Ranked by top-3 source hit rate, then MRR, then lower source pollution.",
        "",
        "| Rank | Config | Top-1 | Top-3 | Top-K | MRR | Pollution | Avg ms |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for rank, row in enumerate(report["best_configs"], start=1):
        lines.append(
            f"| {rank} | {row['label']} | {_md_metric(row.get('top1_source_hit_rate'))} | "
            f"{_md_metric(row.get('top3_source_hit_rate'))} | "
            f"{_md_metric(row.get('top_k_source_hit_rate'))} | "
            f"{_md_metric(row.get('average_mrr'))} | "
            f"{_md_metric(row.get('source_pollution_rate'))} | "
            f"{_md_metric(row.get('average_latency_ms'))} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    args = parse_args()
    configure_offline_embedding()
    try:
        cases_path = project_path(args.cases)
        cases = load_cases(cases_path)
        chunk_grid, skipped = build_chunk_grid(args.chunk_sizes, args.overlaps)
        fusion_grid = build_fusion_grid(
            args.vector_weights,
            args.bm25_weights,
            args.similarity_thresholds,
        )
        total_combos = len(chunk_grid) * (len(SWEEP_MODES) + len(fusion_grid))
        print(
            f"Sweeping {len(chunk_grid)} chunk configs x {len(fusion_grid)} fusion configs "
            f"({total_combos} evaluations, {len(cases)} cases each)"
        )
        report = run_sweep(
            cases,
            chunk_grid,
            skipped,
            fusion_grid,
            args.top_k,
            cases_path,
        )
        json_path = write_json_report(report, args.output)
        output_path = project_path(args.markdown)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(render_markdown(report), encoding="utf-8")
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {output_path}")
        for rank, row in enumerate(report["best_configs"], start=1):
            print(
                f"best #{rank}: {row['label']} top3={row.get('top3_source_hit_rate')} "
                f"mrr={row.get('average_mrr')} pollution={row.get('source_pollution_rate')}"
            )
        return 0
    except Exception as error:
        print(f"Parameter sweep failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
