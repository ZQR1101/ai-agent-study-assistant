import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend import rag_service


class FakeLLM:
    def __init__(self, text: str):
        self.text = text
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.text)


class RagServiceTests(unittest.TestCase):
    def test_rewrite_query_for_retrieval_uses_llm_and_history(self):
        llm = FakeLLM("Agent Skill 工作流知识包")

        result = rag_service.rewrite_query_for_retrieval(
            "这个怎么做学习路线？",
            custom_llm=llm,
            history_context="用户：Agent Skill 是什么？",
        )

        self.assertEqual(result["query"], "Agent Skill 工作流知识包")
        self.assertTrue(result["used"])
        self.assertIsNone(result["error"])
        self.assertIn("Agent Skill 是什么", llm.prompts[0])

    def test_rewrite_query_for_retrieval_falls_back_without_llm(self):
        result = rag_service.rewrite_query_for_retrieval("帮我讲讲 RAG")

        self.assertEqual(result["query"], "帮我讲讲 RAG")
        self.assertFalse(result["used"])
        self.assertIsNone(result["error"])

    def test_get_rag_context_searches_with_rewritten_query(self):
        llm = FakeLLM("RAG 检索增强生成")
        search_result = {
            "chunks": [],
            "highest_score": None,
            "threshold": 0.55,
            "passed_threshold": False,
            "expanded_query": "RAG 检索增强生成 retrieval augmented generation",
            "raw_count": 0,
            "valid_count": 0,
            "discarded_invalid_count": 0,
            "error": None,
            "retrieval_mode": "hybrid",
            "candidate_k": 10,
            "vector_candidates": 0,
            "bm25_candidates": 0,
            "hybrid_used": True,
            "reranker_enabled": False,
            "reranker_used": False,
            "reranker_model": None,
            "reranker_top_n": None,
            "reranker_error": None,
        }

        with patch("backend.rag_service.search_relevant_chunks", return_value=search_result) as search:
            context = rag_service.get_rag_context(
                "帮我讲讲它",
                retrieval_mode="hybrid",
                query_rewrite_llm=llm,
                history_context="用户：RAG 是什么？",
            )

        self.assertFalse(context["found"])
        self.assertEqual(context["retrieval_query"], "RAG 检索增强生成")
        self.assertTrue(context["query_rewrite_used"])
        self.assertEqual(search.call_args.args[0], "RAG 检索增强生成")

    def test_rag_answer_does_not_enable_query_rewrite_by_default(self):
        llm = FakeLLM("answer")
        rag_context = {
            "found": True,
            "context": "retrieved context",
            "sources": [],
            "max_score": 0.8,
            "threshold": 0.55,
            "retrieval_mode": "hybrid",
            "reranker_enabled": False,
            "reranker_used": False,
            "reranker_error": None,
        }

        with patch("backend.rag_service.get_rag_context", return_value=rag_context) as get_context:
            result = rag_service.rag_answer_with_sources(
                "帮我讲讲 RAG",
                custom_llm=llm,
                retrieval_mode="hybrid",
            )

        self.assertEqual(result["answer"], "answer")
        self.assertNotIn("检索查询", llm.prompts[0])
        self.assertNotIn("query_rewrite_llm", get_context.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
