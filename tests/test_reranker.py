import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import rag_store
from backend.ai_core import run_chat_request
from backend.reranker import (
    _reset_reranker_state,
    get_reranker_error,
    get_reranker_model,
    rerank_chunks,
    rerank_chunks_with_metadata,
)
from backend.schemas import ChatRequest
from backend.server import DebugRagRequest, debug_rag_api


def sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "a",
            "source": "a.md",
            "text": "Alpha candidate with enough content for reranker testing.",
            "score": 0.9,
            "retrieval": "hybrid",
        },
        {
            "chunk_id": "b",
            "source": "b.md",
            "text": "Beta candidate with more relevant content for the query.",
            "score": 0.8,
            "retrieval": "hybrid",
        },
    ]


def sample_search_metadata() -> dict:
    return {
        "chunks": sample_chunks(),
        "highest_score": 0.9,
        "threshold": None,
        "passed_threshold": True,
        "expanded_query": "query",
        "raw_count": 2,
        "valid_count": 2,
        "discarded_invalid_count": 0,
        "error": None,
        "retrieval_mode": "hybrid",
        "candidate_k": 10,
        "vector_candidates": 2,
        "bm25_candidates": 2,
        "hybrid_used": True,
    }


class RerankerTests(unittest.TestCase):
    def setUp(self):
        _reset_reranker_state()

    def tearDown(self):
        _reset_reranker_state()

    def test_disabled_config_does_not_load_model(self):
        with (
            patch.dict(
                os.environ,
                {"ENABLE_RERANKER": "false", "RERANKER_MODEL": "unused-model"},
                clear=False,
            ),
            patch("backend.reranker._get_cross_encoder") as mock_cross_encoder,
        ):
            self.assertIsNone(get_reranker_model())

        mock_cross_encoder.assert_not_called()

    def test_model_is_loaded_once_for_repeated_calls(self):
        model = object()
        with (
            patch.dict(
                os.environ,
                {"ENABLE_RERANKER": "true", "RERANKER_MODEL": "mock/model"},
                clear=False,
            ),
            patch("backend.reranker._get_cross_encoder") as mock_get_cross_encoder,
        ):
            mock_get_cross_encoder.return_value.return_value = model
            first = get_reranker_model()
            second = get_reranker_model()

        self.assertIs(first, model)
        self.assertIs(second, model)
        mock_get_cross_encoder.return_value.assert_called_once_with("mock/model")

    def test_failed_load_retries_after_cooldown(self):
        model = object()
        with (
            patch.dict(
                os.environ,
                {"ENABLE_RERANKER": "true", "RERANKER_MODEL": "mock/model"},
                clear=False,
            ),
            patch("backend.reranker._get_cross_encoder") as mock_get_cross_encoder,
        ):
            encoder_class = mock_get_cross_encoder.return_value
            encoder_class.side_effect = [RuntimeError("temporary failure"), model]

            with patch("backend.reranker.monotonic", return_value=100.0):
                self.assertIsNone(get_reranker_model())
            with patch("backend.reranker.monotonic", return_value=110.0):
                self.assertIsNone(get_reranker_model())
            with patch("backend.reranker.monotonic", return_value=131.0):
                self.assertIs(get_reranker_model(), model)

        self.assertEqual(encoder_class.call_count, 2)
        self.assertIsNone(get_reranker_error())

    def test_reset_allows_immediate_retry_after_failure(self):
        model = object()
        with (
            patch.dict(
                os.environ,
                {"ENABLE_RERANKER": "true", "RERANKER_MODEL": "mock/model"},
                clear=False,
            ),
            patch("backend.reranker._get_cross_encoder") as mock_get_cross_encoder,
            patch("backend.reranker.monotonic", return_value=100.0),
        ):
            encoder_class = mock_get_cross_encoder.return_value
            encoder_class.side_effect = [RuntimeError("temporary failure"), model]
            self.assertIsNone(get_reranker_model())

            _reset_reranker_state()

            self.assertIs(get_reranker_model(), model)

        self.assertEqual(encoder_class.call_count, 2)

    def test_disabled_request_keeps_search_results_unchanged(self):
        metadata = sample_search_metadata()
        expected_chunks = [dict(chunk) for chunk in metadata["chunks"]]
        with (
            patch("backend.rag_store._search_hybrid_chunks_with_metadata", return_value=metadata),
            patch("backend.rag_store.rerank_chunks_with_metadata") as mock_rerank,
        ):
            result = rag_store.search_relevant_chunks(
                "query",
                top_k=2,
                retrieval_mode="hybrid",
                reranker_enabled=False,
                include_metadata=True,
            )

        self.assertEqual(result["chunks"], expected_chunks)
        self.assertFalse(result["reranker_enabled"])
        self.assertFalse(result["reranker_used"])
        mock_rerank.assert_not_called()

    def test_missing_model_falls_back_without_interrupting(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_model = Path(tmpdir) / "missing-reranker"
            with patch.dict(
                os.environ,
                {
                    "ENABLE_RERANKER": "true",
                    "RERANKER_MODEL": str(missing_model),
                    "RERANKER_TOP_N": "20",
                },
                clear=False,
            ):
                result = rerank_chunks_with_metadata(
                    "query",
                    sample_chunks(),
                    top_k=1,
                    enabled=True,
                )

        self.assertTrue(result["reranker_enabled"])
        self.assertFalse(result["reranker_used"])
        self.assertEqual(result["chunks"][0]["source"], "a.md")
        self.assertIn("Reranker model path not found", result["reranker_error"])

    def test_enabled_search_reranks_only_top_n_candidates(self):
        metadata = sample_search_metadata()
        reranked_metadata = {
            "chunks": [metadata["chunks"][1]],
            "reranker_enabled": True,
            "reranker_used": True,
            "reranker_model": "mock/model",
            "reranker_top_n": 2,
            "reranker_error": None,
        }
        with (
            patch.dict(
                os.environ,
                {
                    "ENABLE_RERANKER": "true",
                    "RERANKER_MODEL": "mock/model",
                    "RERANKER_TOP_N": "2",
                },
                clear=False,
            ),
            patch(
                "backend.rag_store._search_hybrid_chunks_with_metadata",
                return_value=metadata,
            ) as mock_hybrid,
            patch(
                "backend.rag_store.rerank_chunks_with_metadata",
                return_value=reranked_metadata,
            ) as mock_rerank,
        ):
            result = rag_store.search_relevant_chunks(
                "query",
                top_k=1,
                retrieval_mode="hybrid",
                reranker_enabled=True,
                include_metadata=True,
            )

        self.assertEqual(mock_hybrid.call_args.kwargs["top_k"], 2)
        self.assertEqual(mock_hybrid.call_args.kwargs["candidate_k"], 2)
        self.assertEqual(len(mock_rerank.call_args.args[1]), 2)
        self.assertEqual(result["chunks"][0]["source"], "b.md")

    def test_rerank_chunks_uses_mock_scores_to_reorder(self):
        class FakeCrossEncoder:
            def predict(self, pairs, show_progress_bar=False):
                self.pairs = pairs
                return [0.1, 0.95]

        with (
            patch.dict(
                os.environ,
                {"ENABLE_RERANKER": "true", "RERANKER_MODEL": "mock/model"},
                clear=False,
            ),
            patch("backend.reranker.get_reranker_model", return_value=FakeCrossEncoder()),
        ):
            result = rerank_chunks("Beta query", sample_chunks(), top_k=2)

        self.assertEqual([chunk["source"] for chunk in result], ["b.md", "a.md"])
        self.assertEqual(result[0]["rerank_score"], 0.95)
        self.assertEqual(result[0]["rerank_rank"], 1)
        self.assertTrue(result[0]["reranker_used"])
        self.assertEqual(result[0]["score"], 0.8)

    def test_debug_rag_accepts_reranker_flag(self):
        metadata = {
            **sample_search_metadata(),
            "chunks": [{**sample_chunks()[1], "rerank_score": 0.95, "rerank_rank": 1}],
            "reranker_enabled": True,
            "reranker_used": True,
            "reranker_model": "mock/model",
            "reranker_top_n": 20,
            "reranker_error": None,
        }
        request = DebugRagRequest(
            text="Beta query",
            top_k=1,
            retrieval_mode="hybrid",
            reranker_enabled=True,
        )
        with patch(
            "backend.rag_store.search_relevant_chunks",
            return_value=metadata,
        ) as mock_search:
            response = debug_rag_api(request)

        mock_search.assert_called_once_with(
            "Beta query",
            top_k=1,
            retrieval_mode="hybrid",
            reranker_enabled=True,
            include_metadata=True,
        )
        self.assertTrue(response["reranker_enabled"])
        self.assertTrue(response["reranker_used"])
        self.assertEqual(response["chunks"][0]["rerank_rank"], 1)

    def test_chat_runtime_info_contains_reranker_status(self):
        rag_context = {
            "found": True,
            "context": "reranked context",
            "sources": [],
            "max_score": 0.8,
            "threshold": None,
            "expanded_query": "query",
            "raw_count": 2,
            "valid_count": 1,
            "discarded_invalid_count": 0,
            "error": None,
            "retrieval_mode": "hybrid",
            "candidate_k": 20,
            "vector_candidates": 2,
            "bm25_candidates": 2,
            "hybrid_used": True,
            "reranker_enabled": True,
            "reranker_used": True,
            "reranker_model": "mock/model",
            "reranker_top_n": 20,
            "reranker_error": None,
        }
        request = ChatRequest(
            message="query",
            mode="rag",
            use_rag=True,
            retrieval_mode="hybrid",
            reranker_enabled=True,
        )
        with (
            patch("backend.ai_core.get_rag_context", return_value=rag_context),
            patch("backend.ai_core.build_llm", return_value=object()),
            patch("backend.ai_core.chat", return_value="answer"),
        ):
            response = run_chat_request(request)

        runtime_info = response["runtime_info"]
        self.assertTrue(runtime_info["reranker_enabled"])
        self.assertTrue(runtime_info["reranker_used"])
        self.assertEqual(runtime_info["reranker_model"], "mock/model")
        self.assertEqual(runtime_info["reranker_top_n"], 20)
        self.assertIsNone(runtime_info["reranker_error"])


if __name__ == "__main__":
    unittest.main()
