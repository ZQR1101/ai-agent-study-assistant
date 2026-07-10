import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from scripts.evaluate_rag_retrieval import (
    build_hybrid_reranker_diagnostics,
    build_mode_summary,
    compute_ranking_metrics,
    evaluate_cases,
    load_cases,
    parse_args,
    render_markdown,
    score_retrieval,
    write_json_report,
    write_markdown_report,
)
from scripts.benchmark_rag_batch import build_query_rewrite_search


def sample_case(case_id: str = "case-1") -> dict:
    return {
        "id": case_id,
        "question": "What is Alpha?",
        "expected_keywords": ["Alpha", "database"],
        "expected_sources": ["alpha.md", "database.md"],
        "notes": "offline test case",
    }


def sample_chunks() -> list[dict]:
    return [
        {
            "source": "alpha.md",
            "text": "Alpha is a database retrieval concept.",
            "score": 0.9,
            "retrieval": "vector",
            "vector_score": 0.9,
        }
    ]


class RagRetrievalEvaluationTests(unittest.TestCase):
    def test_benchmark_reuses_one_rewrite_across_retrieval_modes(self):
        class FakeRewriteLLM:
            def __init__(self):
                self.calls = 0

            def invoke(self, _prompt):
                self.calls += 1
                return SimpleNamespace(content="Alpha database")

        fake_llm = FakeRewriteLLM()
        search_result = {
            "chunks": sample_chunks(),
            "highest_score": 0.9,
            "threshold": 0.55,
            "passed_threshold": True,
            "expanded_query": "Alpha",
            "raw_count": 1,
            "valid_count": 1,
            "discarded_invalid_count": 0,
            "error": None,
            "retrieval_mode": "hybrid",
            "candidate_k": 5,
            "vector_candidates": 1,
            "bm25_candidates": 1,
            "hybrid_used": True,
            "reranker_enabled": False,
            "reranker_used": False,
            "reranker_model": None,
            "reranker_top_n": None,
            "reranker_error": None,
        }

        with (
            patch("backend.llm_service.build_llm", return_value=fake_llm),
            patch("backend.rag_store.search_relevant_chunks", return_value=search_result),
        ):
            search = build_query_rewrite_search("always")
            first = search("What is Alpha?", retrieval_mode="vector", include_metadata=True)
            second = search("What is Alpha?", retrieval_mode="hybrid", include_metadata=True)

        self.assertEqual(fake_llm.calls, 1)
        self.assertEqual(search.rewrite_stats["api_call_count"], 1)
        self.assertEqual(len(search.rewrite_cache), 1)
        self.assertTrue(first["query_rewrite_latency_included"])
        self.assertFalse(second["query_rewrite_latency_included"])


    def test_load_cases_reads_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.json"
            path.write_text(json.dumps([sample_case()]), encoding="utf-8")

            cases = load_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["id"], "case-1")
        self.assertEqual(cases[0]["expected_sources"], ["alpha.md", "database.md"])

    def test_load_cases_preserves_batch_negative_and_ocr_metadata(self):
        case = {
            **sample_case(),
            "case_type": "ocr_fact",
            "batch": "v3",
            "is_negative": True,
            "requires_ocr": True,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.json"
            path.write_text(json.dumps([case]), encoding="utf-8")
            loaded = load_cases(path)[0]

        self.assertEqual(loaded["case_type"], "ocr_fact")
        self.assertEqual(loaded["batch"], "v3")
        self.assertTrue(loaded["is_negative"])
        self.assertTrue(loaded["requires_ocr"])

    def test_expected_keyword_hits_are_counted(self):
        result = score_retrieval(sample_case(), "vector", 5, sample_chunks())

        self.assertEqual(result["keyword_hit_count"], 2)
        self.assertEqual(result["matched_expected_keywords"], ["Alpha", "database"])

    def test_expected_source_hits_are_counted(self):
        result = score_retrieval(sample_case(), "vector", 5, sample_chunks())

        self.assertEqual(result["source_hit_count"], 1)
        self.assertEqual(result["matched_expected_sources"], ["alpha.md"])
        self.assertEqual(result["retrieval_score"], 3)

    def test_ranking_metrics_track_source_and_keyword_positions(self):
        chunks = [
            {
                "source": "other.md",
                "text": "Alpha appears before the expected source.",
            },
            {
                "source": "docs/alpha.md",
                "text": "Relevant source content.",
            },
            {"source": "third.md", "text": "Unrelated content."},
        ]

        metrics = compute_ranking_metrics(
            chunks,
            expected_sources=["alpha.md"],
            expected_keywords=["Alpha"],
        )

        self.assertEqual(metrics["top1_source_hit"], 0)
        self.assertEqual(metrics["top3_source_hit"], 1)
        self.assertEqual(metrics["top5_source_hit"], 1)
        self.assertEqual(metrics["best_expected_source_rank"], 2)
        self.assertEqual(metrics["best_expected_keyword_rank"], 1)
        self.assertEqual(metrics["mrr"], 0.5)

    def test_ranking_metrics_detect_top1_source_hit(self):
        metrics = compute_ranking_metrics(
            [{"source": "alpha.md", "text": "Relevant content."}],
            expected_sources=["alpha.md"],
            expected_keywords=[],
        )

        self.assertEqual(metrics["top1_source_hit"], 1)
        self.assertEqual(metrics["top3_source_hit"], 1)
        self.assertEqual(metrics["mrr"], 1.0)

    def test_ranking_metrics_return_zero_mrr_without_source_hit(self):
        metrics = compute_ranking_metrics(
            [{"source": "other.md", "text": "Alpha keyword only."}],
            expected_sources=["alpha.md"],
            expected_keywords=["Alpha"],
        )

        self.assertIsNone(metrics["best_expected_source_rank"])
        self.assertEqual(metrics["best_expected_keyword_rank"], 1)
        self.assertEqual(metrics["mrr"], 0.0)

    def test_mode_summary_aggregates_results(self):
        cases = [
            {
                "results": {
                    "bm25": {
                        "success": True,
                        "keyword_hit_count": 2,
                        "source_hit_count": 1,
                        "retrieval_score": 3,
                    }
                }
            },
            {
                "results": {
                    "bm25": {
                        "success": False,
                        "keyword_hit_count": 0,
                        "source_hit_count": 0,
                        "retrieval_score": 0,
                    }
                }
            },
        ]

        summary = build_mode_summary(cases, ["bm25"])["bm25"]

        self.assertEqual(summary["total_keyword_hits"], 2)
        self.assertEqual(summary["total_source_hits"], 1)
        self.assertEqual(summary["average_retrieval_score"], 1.5)
        self.assertEqual(summary["failed_cases"], 1)

    def test_mode_summary_aggregates_ranking_metrics(self):
        cases = [
            {
                "results": {
                    "hybrid": {
                        "success": True,
                        "keyword_hit_count": 1,
                        "source_hit_count": 1,
                        "retrieval_score": 2,
                        "ranking_metrics": {
                            "top1_source_hit": 1,
                            "top3_source_hit": 1,
                            "top5_source_hit": 1,
                            "mrr": 1.0,
                            "best_expected_source_rank": 1,
                            "best_expected_keyword_rank": 2,
                        },
                    }
                }
            },
            {
                "results": {
                    "hybrid": {
                        "success": True,
                        "keyword_hit_count": 0,
                        "source_hit_count": 1,
                        "retrieval_score": 1,
                        "ranking_metrics": {
                            "top1_source_hit": 0,
                            "top3_source_hit": 1,
                            "top5_source_hit": 1,
                            "mrr": 0.5,
                            "best_expected_source_rank": 2,
                            "best_expected_keyword_rank": None,
                        },
                    }
                }
            },
            {
                "results": {
                    "hybrid": {
                        "success": True,
                        "keyword_hit_count": 0,
                        "source_hit_count": 0,
                        "retrieval_score": 0,
                        "ranking_metrics": {
                            "top1_source_hit": 0,
                            "top3_source_hit": 0,
                            "top5_source_hit": 0,
                            "mrr": 0.0,
                            "best_expected_source_rank": None,
                            "best_expected_keyword_rank": None,
                        },
                    }
                }
            },
        ]

        summary = build_mode_summary(cases, ["hybrid"])["hybrid"]

        self.assertEqual(summary["total_top1_source_hits"], 1)
        self.assertEqual(summary["total_top3_source_hits"], 2)
        self.assertEqual(summary["total_top5_source_hits"], 2)
        self.assertEqual(summary["average_mrr"], 0.5)
        self.assertEqual(summary["average_best_expected_source_rank"], 1.5)
        self.assertEqual(summary["average_best_expected_keyword_rank"], 2.0)

    def test_markdown_report_can_be_generated(self):
        def search_fn(_question, **_kwargs):
            return {"chunks": sample_chunks(), "error": None}

        report = evaluate_cases([sample_case()], ["vector"], 5, search_fn=search_fn)
        markdown = render_markdown(report)

        self.assertIn("# RAG Retrieval Evaluation", markdown)
        self.assertIn("| vector |", markdown)
        self.assertIn("alpha.md", markdown)
        self.assertIn("Snippet 1", markdown)
        self.assertIn("Top-1", markdown)
        self.assertIn("P95 ms", markdown)
        self.assertIn("Source Pollution", markdown)
        self.assertIn("Avg MRR", markdown)
        self.assertIn("Best expected source rank: 1", markdown)

    def test_negative_summary_tracks_fallback_and_source_pollution(self):
        negative_cases = [
            {
                "id": "fallback",
                "is_negative": True,
                "results": {
                    "hybrid": {
                        "success": True,
                        "keyword_hit_count": 0,
                        "source_hit_count": 0,
                        "retrieval_score": 0,
                        "ranking_metrics": {},
                        "fallback_success": True,
                        "source_pollution": False,
                        "latency_ms": 10.0,
                    }
                },
            },
            {
                "id": "polluted",
                "is_negative": True,
                "results": {
                    "hybrid": {
                        "success": True,
                        "keyword_hit_count": 0,
                        "source_hit_count": 0,
                        "retrieval_score": 0,
                        "ranking_metrics": {},
                        "fallback_success": False,
                        "source_pollution": True,
                        "latency_ms": 30.0,
                    }
                },
            },
        ]

        summary = build_mode_summary(negative_cases, ["hybrid"])["hybrid"]

        self.assertEqual(summary["fallback_success_rate"], 0.5)
        self.assertEqual(summary["source_pollution_rate"], 0.5)
        self.assertEqual(summary["average_latency_ms"], 20.0)
        self.assertEqual(summary["p95_latency_ms"], 30.0)

    def test_query_rewrite_metrics_are_aggregated_and_rendered(self):
        cases = []
        for used, latency in ((True, 120.0), (False, 80.0)):
            cases.append({
                "results": {
                    "hybrid": {
                        "success": True,
                        "keyword_hit_count": 0,
                        "source_hit_count": 0,
                        "retrieval_score": 0,
                        "ranking_metrics": {},
                        "query_rewrite_attempted": True,
                        "query_rewrite_used": used,
                        "query_rewrite_latency_ms": latency,
                        "query_fusion_used": used,
                    }
                }
            })

        mode_summary = build_mode_summary(cases, ["hybrid"])
        summary = mode_summary["hybrid"]
        self.assertEqual(summary["query_rewrite_attempt_count"], 2)
        self.assertEqual(summary["query_rewrite_success_count"], 1)
        self.assertEqual(summary["query_rewrite_success_rate"], 0.5)
        self.assertEqual(summary["query_rewrite_fallback_count"], 1)
        self.assertEqual(summary["average_query_rewrite_latency_ms"], 100.0)

        report = {
            "summary": {
                "case_count": 0,
                "positive_case_count": 0,
                "negative_case_count": 0,
                "modes": ["hybrid"],
                "top_k": 5,
                "with_answer": False,
                "with_judge": False,
                "with_reranker": False,
            },
            "mode_summary": mode_summary,
            "cases": [],
        }
        self.assertIn("## Query Rewrite Diagnostics", render_markdown(report))

    def test_json_report_can_be_written(self):
        report = {
            "summary": {"case_count": 0, "modes": [], "top_k": 5},
            "mode_summary": {},
            "cases": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_json_report(report, Path(tmpdir) / "report.json")
            loaded = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(loaded["summary"]["top_k"], 5)

    def test_markdown_report_can_be_written(self):
        def search_fn(_question, **_kwargs):
            return {"chunks": sample_chunks(), "error": None}

        report = evaluate_cases([sample_case()], ["vector"], 5, search_fn=search_fn)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_markdown_report(report, Path(tmpdir) / "report.md")
            content = path.read_text(encoding="utf-8")

        self.assertIn("## Summary", content)
        self.assertIn("## Case Details", content)

    def test_mode_failure_does_not_stop_evaluation(self):
        def search_fn(_question, *, retrieval_mode, **_kwargs):
            if retrieval_mode == "vector":
                raise RuntimeError("vector unavailable")
            return {"chunks": sample_chunks(), "error": None}

        report = evaluate_cases(
            [sample_case()],
            ["vector", "bm25", "hybrid"],
            5,
            search_fn=search_fn,
        )
        results = report["cases"][0]["results"]

        self.assertFalse(results["vector"]["success"])
        self.assertIn("vector unavailable", results["vector"]["error"])
        self.assertTrue(results["bm25"]["success"])
        self.assertTrue(results["hybrid"]["success"])
        self.assertEqual(report["mode_summary"]["vector"]["failed_cases"], 1)

    def test_with_judge_generates_answer_and_records_evaluation(self):
        def search_fn(_question, **_kwargs):
            return {"chunks": sample_chunks(), "error": None}

        def answer_fn(_question, **_kwargs):
            return {
                "answer": "Alpha answer",
                "sources": sample_chunks(),
                "passed_threshold": True,
            }

        def judge_fn(_question, _answer, **_kwargs):
            return {"overall_score": 9.0, "verdict": "PASS"}

        report = evaluate_cases(
            [sample_case()],
            ["hybrid"],
            5,
            with_judge=True,
            search_fn=search_fn,
            answer_fn=answer_fn,
            judge_fn=judge_fn,
            judge_enabled=True,
        )
        result = report["cases"][0]["results"]["hybrid"]

        self.assertTrue(result["answer"]["success"])
        self.assertTrue(result["judge"]["success"])
        self.assertEqual(result["judge"]["evaluation"]["overall_score"], 9.0)

    def test_with_reranker_adds_hybrid_reranker_mode(self):
        with patch.object(
            sys,
            "argv",
            ["evaluate_rag_retrieval.py", "--with-reranker"],
        ):
            args = parse_args()

        self.assertEqual(args.modes, ["vector", "bm25", "hybrid", "hybrid_reranker"])

    def test_hybrid_reranker_uses_hybrid_search_with_reranker_flag(self):
        calls = []

        def search_fn(_question, **kwargs):
            calls.append(kwargs)
            return {
                "chunks": [
                    {
                        **sample_chunks()[0],
                        "rerank_score": 0.95,
                        "rerank_rank": 1,
                        "reranker_used": True,
                    }
                ],
                "error": None,
                "reranker_enabled": True,
                "reranker_used": True,
            }

        report = evaluate_cases(
            [sample_case()],
            ["hybrid_reranker"],
            5,
            search_fn=search_fn,
        )
        result = report["cases"][0]["results"]["hybrid_reranker"]

        self.assertEqual(calls[0]["retrieval_mode"], "hybrid")
        self.assertTrue(calls[0]["reranker_enabled"])
        self.assertTrue(result["reranker_used"])
        self.assertEqual(result["chunks"][0]["rerank_rank"], 1)

    def test_hybrid_reranker_diagnostics_detect_all_verdicts(self):
        def result(rank, source_hits=1):
            return {
                "success": True,
                "keyword_hit_count": 0,
                "source_hit_count": source_hits,
                "retrieval_score": source_hits,
                "ranking_metrics": {
                    "top1_source_hit": int(rank == 1),
                    "top3_source_hit": int(rank is not None and rank <= 3),
                    "top5_source_hit": int(rank is not None and rank <= 5),
                    "mrr": 1.0 / rank if rank else 0.0,
                    "best_expected_source_rank": rank,
                    "best_expected_keyword_rank": None,
                },
            }

        cases = [
            {
                "id": "improved",
                "results": {
                    "hybrid": result(3),
                    "hybrid_reranker": result(1),
                },
            },
            {
                "id": "same",
                "results": {
                    "hybrid": result(2),
                    "hybrid_reranker": result(2),
                },
            },
            {
                "id": "worse",
                "results": {
                    "hybrid": result(1),
                    "hybrid_reranker": result(None, source_hits=0),
                },
            },
        ]

        diagnostics = build_hybrid_reranker_diagnostics(cases)

        self.assertEqual([item["verdict"] for item in diagnostics], ["improved", "same", "worse"])
        self.assertEqual(diagnostics[0]["rank_delta"], 2)
        self.assertIsNone(diagnostics[2]["rank_delta"])

        report = {
            "summary": {
                "case_count": 3,
                "modes": ["hybrid", "hybrid_reranker"],
                "top_k": 5,
                "with_answer": False,
                "with_judge": False,
                "with_reranker": True,
            },
            "mode_summary": build_mode_summary(cases, ["hybrid", "hybrid_reranker"]),
            "hybrid_reranker_diagnostics": diagnostics,
            "cases": [],
        }
        markdown = render_markdown(report)

        self.assertIn("## Hybrid vs Hybrid Reranker Diagnostics", markdown)
        self.assertIn("| improved |", markdown)
        self.assertIn("| same |", markdown)
        self.assertIn("| worse |", markdown)


if __name__ == "__main__":
    unittest.main()
