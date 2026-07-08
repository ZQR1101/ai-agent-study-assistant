import unittest
from unittest.mock import patch

from backend import rag_store
from backend.rag_store import (
    expand_query,
    reciprocal_rank_fusion,
    search_hybrid_chunks,
    search_keyword_chunks,
    search_relevant_chunks,
    tokenize_for_bm25,
)


class HybridSearchTests(unittest.TestCase):
    def setUp(self):
        self.original_chunks = rag_store.chunks
        self.original_index = rag_store.index
        self.original_error = rag_store.rag_index_error
        rag_store.chunks = []
        rag_store.index = None
        rag_store.rag_index_error = None
        rag_store._reset_bm25_index()

    def tearDown(self):
        rag_store.chunks = self.original_chunks
        rag_store.index = self.original_index
        rag_store.rag_index_error = self.original_error
        rag_store._reset_bm25_index()

    def test_tokenizer_preserves_config_api_and_path_tokens(self):
        tokens = tokenize_for_bm25(
            "ENABLE_DB_HISTORY uses planner_mode and /rag/warmup in backend/rag_store.py"
        )

        self.assertIn("ENABLE_DB_HISTORY", tokens)
        self.assertIn("planner_mode", tokens)
        self.assertIn("/rag/warmup", tokens)
        self.assertIn("backend/rag_store.py", tokens)

    def test_tokenizer_filters_single_chinese_characters_and_instruction_noise(self):
        tokens = tokenize_for_bm25("什么是skill，并给我一个skill学习路线")

        self.assertIn("skill", tokens)
        self.assertIn("学习", tokens)
        self.assertIn("路线", tokens)
        self.assertNotIn("什", tokens)
        self.assertNotIn("什么", tokens)
        self.assertNotIn("给我", tokens)

    def test_skill_query_is_cleaned_and_expanded(self):
        expanded = expand_query("什么是skill，并给我一个skill学习路线")

        self.assertNotIn("什么是", expanded)
        self.assertNotIn("给我一个", expanded)
        self.assertIn("Agent Skill", expanded)
        self.assertIn("SKILL.md", expanded)

    def test_bm25_hits_keyword_chunk(self):
        rag_store.chunks = [
            {
                "source": "db.md",
                "text": "ENABLE_DB_HISTORY controls PostgreSQL session persistence and backend chat history storage.",
            },
            {
                "source": "rag.md",
                "text": "RAG warmup loads embedding models and existing vector indexes in a background thread.",
            },
        ]

        results = search_keyword_chunks("ENABLE_DB_HISTORY", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "db.md")
        self.assertEqual(results[0]["retrieval"], "bm25")
        self.assertGreater(results[0]["bm25_score"], 0)

    def test_bm25_matches_chunk_metadata(self):
        rag_store.chunks = [
            {
                "source": "backend_notes.md",
                "document": "backend_notes.md",
                "document_title": "Backend Notes",
                "title": "Billing API",
                "section": "Backend Notes > Billing API",
                "headings": ["Backend Notes", "Billing API"],
                "text": "This paragraph explains request validation, persistence, and response handling.",
            }
        ]

        results = search_keyword_chunks("Billing API", top_k=1)

        self.assertEqual(results[0]["source"], "backend_notes.md")
        self.assertEqual(results[0]["title"], "Billing API")
        self.assertEqual(results[0]["section"], "Backend Notes > Billing API")

    def test_bm25_ranks_agent_skill_document_first_for_compound_query(self):
        rag_store.chunks = [
            {
                "source": "agent_hot_04_agent_skills_portable_workflows.md",
                "text": "Agent Skills are reusable workflow knowledge packages built around SKILL.md instructions.",
            },
            {
                "source": "generic_learning_guide.md",
                "text": "学习路线包括快速看课程、理解概念、制作项目、调试并长期迭代。",
            },
            {
                "source": "prompt_engineering.md",
                "text": "提示工程课程提供学习路线、示例输出、知识检查和项目练习。",
            },
        ]

        result = search_relevant_chunks(
            "什么是skill，并给我一个skill学习路线",
            top_k=3,
            retrieval_mode="bm25",
            include_metadata=True,
        )

        self.assertEqual(
            result["chunks"][0]["source"],
            "agent_hot_04_agent_skills_portable_workflows.md",
        )

    def test_rrf_promotes_chunk_present_in_both_lists(self):
        vector_results = [
            {"chunk_id": "b", "source": "b.md", "text": "vector first", "score": 0.9},
            {"chunk_id": "a", "source": "a.md", "text": "shared chunk", "score": 0.8},
        ]
        bm25_results = [
            {"chunk_id": "a", "source": "a.md", "text": "shared chunk", "score": 6.0},
            {"chunk_id": "c", "source": "c.md", "text": "keyword second", "score": 5.0},
        ]

        fused = reciprocal_rank_fusion([vector_results, bm25_results], top_k=3)

        self.assertEqual(fused[0]["chunk_id"], "a")
        self.assertEqual(fused[0]["retrieval"], "hybrid")
        self.assertEqual(fused[0]["vector_rank"], 2)
        self.assertEqual(fused[0]["bm25_rank"], 1)

    def test_search_relevant_chunks_vector_mode_uses_vector_path(self):
        vector_metadata = {
            "chunks": [{"source": "v.md", "text": "vector", "score": 0.8, "retrieval": "vector"}],
            "highest_score": 0.8,
            "threshold": 0.55,
            "passed_threshold": True,
            "expanded_query": "query",
            "raw_count": 1,
            "valid_count": 1,
            "discarded_invalid_count": 0,
            "error": None,
            "retrieval_mode": "vector",
            "candidate_k": 5,
            "vector_candidates": 1,
            "bm25_candidates": 0,
            "hybrid_used": False,
        }
        with patch("backend.rag_store._search_vector_chunks_with_metadata", return_value=vector_metadata) as mock_vector:
            results = search_relevant_chunks("query", retrieval_mode="vector")

        self.assertEqual(results, vector_metadata["chunks"])
        mock_vector.assert_called_once()

    def test_search_relevant_chunks_bm25_mode(self):
        rag_store.chunks = [
            {
                "source": "api.md",
                "text": "/rag/warmup exposes the background RAG warmup endpoint for loading embedding state.",
            }
        ]

        result = search_relevant_chunks(
            "/rag/warmup",
            retrieval_mode="bm25",
            include_metadata=True,
        )

        self.assertEqual(result["retrieval_mode"], "bm25")
        self.assertTrue(result["passed_threshold"])
        self.assertEqual(result["chunks"][0]["retrieval"], "bm25")

    def test_search_relevant_chunks_hybrid_mode(self):
        vector_result = {
            "chunk_id": "doc:0",
            "source": "doc.md",
            "text": "vector candidate with enough meaningful content for validation",
            "score": 0.8,
            "retrieval": "vector",
            "vector_score": 0.8,
            "vector_rank": 1,
        }
        bm25_result = {
            "chunk_id": "doc:0",
            "source": "doc.md",
            "text": "vector candidate with enough meaningful content for validation",
            "score": 4.2,
            "retrieval": "bm25",
            "bm25_score": 4.2,
            "bm25_rank": 1,
        }
        with (
            patch("backend.rag_store.search_vector_chunks", return_value=[vector_result]),
            patch("backend.rag_store.search_keyword_chunks", return_value=[bm25_result]),
        ):
            result = search_relevant_chunks(
                "query",
                retrieval_mode="hybrid",
                include_metadata=True,
            )

        self.assertEqual(result["retrieval_mode"], "hybrid")
        self.assertTrue(result["hybrid_used"])
        self.assertEqual(result["chunks"][0]["retrieval"], "hybrid")
        self.assertEqual(result["chunks"][0]["vector_rank"], 1)
        self.assertEqual(result["chunks"][0]["bm25_rank"], 1)

    def test_hybrid_prefers_bm25_exact_match_when_lists_do_not_overlap(self):
        vector_result = {
            "chunk_id": "vector:0",
            "source": "generic_learning_guide.md",
            "text": "generic learning path content",
            "score": 0.56,
            "retrieval": "vector",
            "vector_score": 0.56,
            "vector_rank": 1,
        }
        bm25_result = {
            "chunk_id": "bm25:0",
            "source": "agent_skills.md",
            "text": "Agent Skill and SKILL.md workflow knowledge",
            "score": 8.2,
            "retrieval": "bm25",
            "bm25_score": 8.2,
            "bm25_rank": 1,
        }
        with (
            patch("backend.rag_store.search_vector_chunks", return_value=[vector_result]),
            patch("backend.rag_store.search_keyword_chunks", return_value=[bm25_result]),
        ):
            results = search_hybrid_chunks("Agent Skill", top_k=2)

        self.assertEqual(results[0]["source"], "agent_skills.md")

    def test_empty_knowledge_base_does_not_error(self):
        with (
            patch("backend.rag_store._load_chunks_file_only", return_value=False),
            patch("backend.rag_store.build_chunks", return_value=([], {})),
        ):
            results = search_relevant_chunks("anything", retrieval_mode="bm25")

        self.assertEqual(results, [])

    def test_hybrid_falls_back_to_vector_when_bm25_empty(self):
        vector_result = {
            "chunk_id": "vector:0",
            "source": "vector.md",
            "text": "vector only result",
            "score": 0.8,
            "retrieval": "vector",
        }
        with (
            patch("backend.rag_store.search_vector_chunks", return_value=[vector_result]),
            patch("backend.rag_store.search_keyword_chunks", return_value=[]),
        ):
            results = search_hybrid_chunks("query", top_k=1)

        self.assertEqual(results[0]["chunk_id"], "vector:0")
        self.assertEqual(results[0]["retrieval"], "hybrid")

    def test_hybrid_falls_back_to_bm25_when_vector_empty(self):
        bm25_result = {
            "chunk_id": "bm25:0",
            "source": "bm25.md",
            "text": "keyword only result",
            "score": 5.0,
            "retrieval": "bm25",
        }
        with (
            patch("backend.rag_store.search_vector_chunks", return_value=[]),
            patch("backend.rag_store.search_keyword_chunks", return_value=[bm25_result]),
        ):
            results = search_hybrid_chunks("query", top_k=1)

        self.assertEqual(results[0]["chunk_id"], "bm25:0")
        self.assertEqual(results[0]["retrieval"], "hybrid")


if __name__ == "__main__":
    unittest.main()
