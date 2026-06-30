import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.evaluate_rag_retrieval import (
    build_mode_summary,
    evaluate_cases,
    load_cases,
    parse_args,
    render_markdown,
    score_retrieval,
    write_json_report,
    write_markdown_report,
)


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
    def test_load_cases_reads_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cases.json"
            path.write_text(json.dumps([sample_case()]), encoding="utf-8")

            cases = load_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0]["id"], "case-1")
        self.assertEqual(cases[0]["expected_sources"], ["alpha.md", "database.md"])

    def test_expected_keyword_hits_are_counted(self):
        result = score_retrieval(sample_case(), "vector", 5, sample_chunks())

        self.assertEqual(result["keyword_hit_count"], 2)
        self.assertEqual(result["matched_expected_keywords"], ["Alpha", "database"])

    def test_expected_source_hits_are_counted(self):
        result = score_retrieval(sample_case(), "vector", 5, sample_chunks())

        self.assertEqual(result["source_hit_count"], 1)
        self.assertEqual(result["matched_expected_sources"], ["alpha.md"])
        self.assertEqual(result["retrieval_score"], 3)

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

    def test_markdown_report_can_be_generated(self):
        def search_fn(_question, **_kwargs):
            return {"chunks": sample_chunks(), "error": None}

        report = evaluate_cases([sample_case()], ["vector"], 5, search_fn=search_fn)
        markdown = render_markdown(report)

        self.assertIn("# RAG Retrieval Evaluation", markdown)
        self.assertIn("| vector |", markdown)
        self.assertIn("alpha.md", markdown)
        self.assertIn("Snippet 1", markdown)

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


if __name__ == "__main__":
    unittest.main()
