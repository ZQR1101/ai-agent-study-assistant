import argparse
import unittest
from unittest.mock import patch

import numpy as np

from backend import rag_store
from scripts.rag_param_sweep import (
    build_chunk_grid,
    build_fusion_grid,
    build_in_memory_index,
    config_label,
    install_search_context,
    make_search_fn,
    metrics_row,
    parse_float_list,
    parse_int_list,
    rank_fusion_rows,
    render_markdown,
)


class SweepGridTests(unittest.TestCase):
    def test_parse_int_list_dedupes_and_validates(self):
        self.assertEqual(parse_int_list("500, 300,500"), [500, 300])
        self.assertEqual(parse_int_list("0"), [0])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_int_list("-1")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_int_list("")

    def test_parse_float_list_validates(self):
        self.assertEqual(parse_float_list("1.15, 1.0,1.15"), [1.15, 1.0])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_float_list("-0.5")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_float_list("abc")

    def test_build_chunk_grid_skips_overlap_larger_than_chunk(self):
        combos, skipped = build_chunk_grid([500], [100, 500])

        self.assertEqual(combos, [{"chunk_size": 500, "overlap": 100}])
        self.assertEqual(
            skipped,
            [{
                "chunk_size": 500,
                "overlap": 500,
                "reason": "overlap must be smaller than chunk_size",
            }],
        )

    def test_build_chunk_grid_rejects_fully_invalid_grid(self):
        with self.assertRaises(ValueError):
            build_chunk_grid([100], [100])

    def test_build_fusion_grid_is_cross_product(self):
        grid = build_fusion_grid([1.0], [0.6, 1.0], [0.5, 0.55])

        self.assertEqual(len(grid), 4)
        self.assertEqual(
            grid[0],
            {"vector_weight": 1.0, "bm25_weight": 0.6, "similarity_threshold": 0.5},
        )

    def test_config_label_formats_params(self):
        self.assertEqual(
            config_label({"chunk_size": 500, "overlap": 100}),
            "cs500-ov100",
        )
        self.assertEqual(
            config_label(
                {"chunk_size": 500, "overlap": 100},
                {
                    "vector_weight": 1.0,
                    "bm25_weight": 1.15,
                    "similarity_threshold": 0.55,
                },
            ),
            "cs500-ov100-w1/1.15-t0.55",
        )


class WeightedSearchFnTests(unittest.TestCase):
    def test_search_fn_overrides_weights_and_threshold_then_restores(self):
        observed = {}

        def fake_search(question, **kwargs):
            observed["weights"] = (
                rag_store.HYBRID_VECTOR_WEIGHT,
                rag_store.HYBRID_BM25_WEIGHT,
            )
            observed["kwargs"] = kwargs
            return {"chunks": []}

        with (
            patch.object(rag_store, "HYBRID_VECTOR_WEIGHT", 1.0),
            patch.object(rag_store, "HYBRID_BM25_WEIGHT", 1.15),
            patch.object(rag_store, "search_relevant_chunks", side_effect=fake_search),
        ):
            search_fn = make_search_fn(0.5, 0.8, 1.4)
            result = search_fn("question", top_k=3, retrieval_mode="hybrid")

        self.assertEqual(result, {"chunks": []})
        self.assertEqual(observed["weights"], (0.8, 1.4))
        self.assertEqual(observed["kwargs"]["similarity_threshold"], 0.5)
        self.assertEqual(observed["kwargs"]["top_k"], 3)
        self.assertEqual(rag_store.HYBRID_VECTOR_WEIGHT, 1.0)
        self.assertEqual(rag_store.HYBRID_BM25_WEIGHT, 1.15)


class SearchContextTests(unittest.TestCase):
    def test_install_search_context_swaps_and_restores(self):
        original_chunks = rag_store.chunks
        original_index = rag_store.index
        new_chunks = [{"source": "a.md", "chunk_index": 0}]

        restore = install_search_context(new_chunks, "fake-index")
        try:
            self.assertIs(rag_store.chunks, new_chunks)
            self.assertEqual(rag_store.index, "fake-index")
            self.assertIsNone(rag_store.rag_index_error)
        finally:
            restore()

        self.assertIs(rag_store.chunks, original_chunks)
        self.assertIs(rag_store.index, original_index)


class FakeIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.added = None

    def add(self, vectors):
        self.added = vectors


class FakeFaiss:
    @staticmethod
    def normalize_L2(vectors):
        pass

    @staticmethod
    def IndexFlatIP(dimension):
        return FakeIndex(dimension)


class FakeModel:
    def encode(self, texts):
        return [[float(len(text)), 1.0] for text in texts]


class InMemoryIndexTests(unittest.TestCase):
    def test_build_in_memory_index_uses_given_chunk_params(self):
        captured = {}

        def fake_build_chunks(documents=None, chunk_size=500, overlap=100):
            captured["chunk_size"] = chunk_size
            captured["overlap"] = overlap
            return (
                [{"source": "a.md", "text": "abc"}],
                {"kept": 1, "low_quality": 0, "dropped": 0},
            )

        with (
            patch.object(rag_store, "build_chunks", side_effect=fake_build_chunks),
            patch.object(rag_store, "get_embedding_model", return_value=FakeModel()),
            patch.object(rag_store, "_get_faiss", return_value=FakeFaiss),
            patch.object(rag_store, "_get_numpy", return_value=np),
        ):
            chunks, index, stats = build_in_memory_index([{"source": "a.md"}], 300, 50)

        self.assertEqual(captured, {"chunk_size": 300, "overlap": 50})
        self.assertEqual(len(chunks), 1)
        self.assertEqual(index.dimension, 2)
        self.assertEqual(len(index.added), 1)
        self.assertEqual(
            stats,
            {
                "chunk_count": 1,
                "chunks_kept": 1,
                "chunks_low_quality": 0,
                "chunks_dropped": 0,
                "build_ms": stats["build_ms"],
            },
        )
        self.assertGreaterEqual(stats["build_ms"], 0)

    def test_build_in_memory_index_skips_embedding_for_empty_chunks(self):
        with patch.object(
            rag_store,
            "build_chunks",
            return_value=([], {"kept": 0, "low_quality": 0, "dropped": 0}),
        ):
            chunks, index, stats = build_in_memory_index([], 500, 100)

        self.assertEqual(chunks, [])
        self.assertIsNone(index)
        self.assertEqual(stats["chunk_count"], 0)


class MetricsRowTests(unittest.TestCase):
    def test_metrics_row_selects_metric_fields_only(self):
        row = metrics_row(
            {"chunk_size": 500, "overlap": 0},
            "vector",
            {"top1_source_hit_rate": 0.5, "top3_source_hit_rate": 0.9, "unrelated": "x"},
        )

        self.assertEqual(row["label"], "cs500-ov0")
        self.assertEqual(row["mode"], "vector")
        self.assertEqual(row["top1_source_hit_rate"], 0.5)
        self.assertNotIn("unrelated", row)

    def test_metrics_row_labels_fusion_configs(self):
        row = metrics_row(
            {
                "chunk_size": 500,
                "overlap": 100,
                "vector_weight": 1.0,
                "bm25_weight": 1.5,
                "similarity_threshold": 0.6,
            },
            "hybrid",
            {},
        )

        self.assertEqual(row["label"], "cs500-ov100-w1/1.5-t0.6")

    def test_rank_fusion_rows_orders_by_top3_mrr_pollution(self):
        rows = [
            {"label": "a", "top3_source_hit_rate": 0.8, "average_mrr": 0.5, "source_pollution_rate": 0.2},
            {"label": "b", "top3_source_hit_rate": 0.9, "average_mrr": 0.4, "source_pollution_rate": 0.5},
            {"label": "c", "top3_source_hit_rate": 0.8, "average_mrr": 0.5, "source_pollution_rate": 0.1},
        ]

        self.assertEqual(
            [row["label"] for row in rank_fusion_rows(rows)],
            ["b", "c", "a"],
        )


class RenderMarkdownTests(unittest.TestCase):
    @staticmethod
    def _fusion_row(**overrides) -> dict:
        row = {
            "label": "cs500-ov100-w1/1.15-t0.55",
            "vector_weight": 1.0,
            "bm25_weight": 1.15,
            "similarity_threshold": 0.55,
            "top1_source_hit_rate": 0.5,
            "top3_source_hit_rate": 0.9,
            "top_k_source_hit_rate": 1.0,
            "average_mrr": 0.6,
            "source_pollution_rate": 0.0,
            "fallback_success_rate": None,
            "average_latency_ms": 10.0,
        }
        row.update(overrides)
        return row

    def _report(self, fusion_rows: list[dict]) -> dict:
        return {
            "summary": {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "git_commit": "abc1234",
                "cases_file": "eval_cases/rag_retrieval_cases.json",
                "cases_sha256": "deadbeef",
                "case_count": 2,
                "top_k": 5,
                "document_count": 3,
                "embedding_model": "local/model",
                "baseline_fusion": {
                    "vector_weight": 1.0,
                    "bm25_weight": 1.15,
                    "similarity_threshold": 0.55,
                },
                "runtime_s": 12.5,
                "skipped_chunk_combos": [],
            },
            "chunk_configs": [
                {"chunk_size": 500, "overlap": 100, "chunk_count": 42, "build_ms": 123.4}
            ],
            "mode_rows": [
                {
                    "label": "cs500-ov100",
                    "mode": "hybrid",
                    "top1_source_hit_rate": 0.5,
                    "top3_source_hit_rate": 0.9,
                    "top_k_source_hit_rate": 1.0,
                    "average_mrr": 0.6,
                    "source_pollution_rate": 0.0,
                    "fallback_success_rate": None,
                    "average_latency_ms": 10.0,
                    "p95_latency_ms": 20.0,
                }
            ],
            "fusion_rows": fusion_rows,
            "best_configs": [self._fusion_row()],
        }

    def test_render_markdown_includes_tables_and_best_configs(self):
        markdown = render_markdown(self._report([self._fusion_row()]))

        self.assertIn("## Chunk Sweep", markdown)
        self.assertIn("## Fusion Sweep", markdown)
        self.assertIn("## Best Fusion Configs", markdown)
        self.assertIn("cs500-ov100-w1/1.15-t0.55", markdown)
        self.assertIn("abc1234", markdown)
        self.assertIn("0.900", markdown)
        self.assertIn("deadbeef", markdown)

    def test_render_markdown_keeps_failed_rows_aligned(self):
        markdown = render_markdown(
            self._report([self._fusion_row(error="embedding model missing")])
        )

        failed_line = next(
            line for line in markdown.splitlines() if "FAILED" in line
        )
        self.assertEqual(failed_line.count("|"), 12)
        self.assertIn("embedding model missing", failed_line)


if __name__ == "__main__":
    unittest.main()
