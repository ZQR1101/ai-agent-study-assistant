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


def make_search_result(chunks=None, *, retrieval_mode="hybrid"):
    chunks = list(chunks or [])
    return {
        "chunks": chunks,
        "highest_score": max((chunk["score"] for chunk in chunks), default=None),
        "threshold": 0.55,
        "passed_threshold": bool(chunks),
        "expanded_query": "expanded query",
        "raw_count": len(chunks),
        "valid_count": len(chunks),
        "discarded_invalid_count": 0,
        "error": None,
        "retrieval_mode": retrieval_mode,
        "candidate_k": 10,
        "vector_candidates": len(chunks),
        "bm25_candidates": 0,
        "hybrid_used": retrieval_mode == "hybrid",
        "reranker_enabled": False,
        "reranker_used": False,
        "reranker_model": None,
        "reranker_top_n": None,
        "reranker_error": None,
    }


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
        self.assertGreaterEqual(result["latency_ms"], 0)

    def test_conditional_rewrite_requires_context_dependent_follow_up(self):
        history = "用户：Kubernetes HPA 是什么？"

        self.assertTrue(rag_service.should_rewrite_query("那它怎么配置？", history))
        self.assertFalse(rag_service.should_rewrite_query("那它怎么配置？", None))
        self.assertFalse(
            rag_service.should_rewrite_query(
                "这个 Kubernetes HPA 最小副本数是多少？",
                history,
            )
        )
        self.assertFalse(
            rag_service.should_rewrite_query("这个在 /api/v1/items 怎么配置？", history)
        )
        self.assertFalse(
            rag_service.should_rewrite_query("这个在 config.yaml 里怎么配置？", history)
        )
        self.assertFalse(
            rag_service.should_rewrite_query("这个在版本v1.2里怎么配置？", history)
        )

    def test_get_rag_context_searches_with_rewritten_query(self):
        llm = FakeLLM("RAG 检索增强生成")
        search_result = make_search_result()

        with patch("backend.rag_service.search_relevant_chunks", return_value=search_result) as search:
            context = rag_service.get_rag_context(
                "帮我讲讲它",
                retrieval_mode="hybrid",
                query_rewrite_llm=llm,
                history_context="用户：RAG 是什么？",
                query_rewrite_mode="always",
            )

        self.assertFalse(context["found"])
        self.assertEqual(context["retrieval_query"], "RAG 检索增强生成")
        self.assertTrue(context["query_rewrite_used"])
        self.assertTrue(context["query_fusion_used"])
        self.assertEqual(
            [call.args[0] for call in search.call_args_list],
            ["帮我讲讲它", "RAG 检索增强生成"],
        )

    def test_conditional_rewrite_keeps_original_and_rewritten_candidates(self):
        llm = FakeLLM("Kubernetes HPA 配置")
        original_chunk = {
            "chunk_id": "original",
            "source": "follow-up.md",
            "text": "原始问题直接命中的内容",
            "score": 0.8,
            "retrieval": "hybrid",
        }
        rewritten_chunk = {
            "chunk_id": "rewritten",
            "source": "kubernetes.md",
            "text": "Kubernetes HPA 配置内容",
            "score": 0.9,
            "retrieval": "hybrid",
        }

        def search_side_effect(query, **_kwargs):
            if query == "那它怎么配置？":
                return make_search_result([original_chunk])
            return make_search_result([rewritten_chunk])

        with patch(
            "backend.rag_service.search_relevant_chunks",
            side_effect=search_side_effect,
        ):
            context = rag_service.get_rag_context(
                "那它怎么配置？",
                top_k=2,
                query_rewrite_llm=llm,
                history_context="用户：Kubernetes HPA 是什么？",
                query_rewrite_mode="conditional",
            )

        self.assertTrue(context["found"])
        self.assertEqual(
            {source["source"] for source in context["sources"]},
            {"follow-up.md", "kubernetes.md"},
        )
        self.assertEqual(context["query_rewrite_reason"], "rewritten")

    def test_off_mode_never_invokes_rewrite_llm(self):
        llm = FakeLLM("不应使用")
        with patch(
            "backend.rag_service.search_relevant_chunks",
            return_value=make_search_result(),
        ) as search:
            context = rag_service.get_rag_context(
                "那它怎么配置？",
                query_rewrite_llm=llm,
                history_context="用户：Kubernetes HPA 是什么？",
                query_rewrite_mode="off",
            )

        self.assertEqual(llm.prompts, [])
        self.assertFalse(context["query_rewrite_attempted"])
        self.assertEqual(context["query_rewrite_reason"], "mode_off")
        search.assert_called_once()

    def test_rag_answer_delegates_rewrite_policy_to_context_builder(self):
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
        self.assertIs(get_context.call_args.kwargs["query_rewrite_llm"], llm)


if __name__ == "__main__":
    unittest.main()
