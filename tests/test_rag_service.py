import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend import rag_service, rag_store
from backend.rag_service import _context_block_overhead


class FakeLLM:
    def __init__(self, text: str):
        self.text = text
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.text)


def make_chunk(chunk_id: str, score: float) -> dict:
    return {
        "chunk_id": chunk_id,
        "source": f"{chunk_id}.md",
        "text": f"{chunk_id} 的内容",
        "score": score,
        "retrieval": "hybrid",
    }


def make_store_chunk(source: str, chunk_index: int) -> dict:
    return {
        "source": source,
        "chunk_index": chunk_index,
        "text": f"{source} #{chunk_index} 内容",
        "document_title": source.replace(".md", ""),
        "title": f"{source} 标题 {chunk_index}",
        "section": f"{source} 章节 {chunk_index}",
        "headings": [],
    }


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

    def test_question_self_contained_classifier(self):
        self.assertTrue(
            rag_service._question_is_self_contained("这个 Kubernetes HPA 最小副本数是多少？")
        )
        self.assertTrue(rag_service._question_is_self_contained("这个在 config.yaml 里怎么配置？"))
        self.assertTrue(rag_service._question_is_self_contained("那它和 v1.2 有什么区别？"))
        self.assertFalse(rag_service._question_is_self_contained("那它怎么配置？"))
        self.assertFalse(rag_service._question_is_self_contained("继续说说"))
        self.assertFalse(rag_service._question_is_self_contained("展开讲讲它的部署方式"))
        self.assertTrue(rag_service._question_is_self_contained(""))

    def test_is_retrieval_insufficient(self):
        self.assertTrue(rag_service._is_retrieval_insufficient({"chunks": []}))
        self.assertTrue(
            rag_service._is_retrieval_insufficient(
                {"chunks": [make_chunk("a", 0.8)], "passed_threshold": False}
            )
        )
        self.assertFalse(
            rag_service._is_retrieval_insufficient(
                {"chunks": [make_chunk("a", 0.8)], "passed_threshold": True}
            )
        )

    def test_conditional_rewrite_fires_only_on_insufficient_retrieval(self):
        llm = FakeLLM("Kubernetes HPA 配置")
        rewritten_chunk = make_chunk("rewritten", 0.9)

        def search_side_effect(query, **kwargs):
            if query == "那它怎么配置？":
                return make_search_result([])
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
        self.assertEqual(context["query_rewrite_reason"], "rewritten")
        self.assertTrue(context["query_fusion_used"])
        self.assertEqual(
            [source["source"] for source in context["sources"]],
            ["rewritten.md"],
        )

    def test_conditional_skips_rewrite_when_retrieval_sufficient(self):
        llm = FakeLLM("不应使用")

        with patch(
            "backend.rag_service.search_relevant_chunks",
            return_value=make_search_result([make_chunk("direct", 0.9)]),
        ) as search:
            context = rag_service.get_rag_context(
                "那它怎么配置？",
                top_k=2,
                query_rewrite_llm=llm,
                history_context="用户：Kubernetes HPA 是什么？",
                query_rewrite_mode="conditional",
            )

        search.assert_called_once()
        self.assertEqual(llm.prompts, [])
        self.assertEqual(context["query_rewrite_reason"], "retrieval_sufficient")
        self.assertFalse(context["query_rewrite_used"])
        self.assertEqual(
            [source["source"] for source in context["sources"]],
            ["direct.md"],
        )

    def test_conditional_skips_rewrite_for_self_contained_question(self):
        llm = FakeLLM("不应使用")

        with patch(
            "backend.rag_service.search_relevant_chunks",
            return_value=make_search_result([]),
        ) as search:
            context = rag_service.get_rag_context(
                "这个 Kubernetes HPA 最小副本数是多少？",
                top_k=2,
                query_rewrite_llm=llm,
                history_context="用户：Kubernetes HPA 是什么？",
                query_rewrite_mode="conditional",
            )

        search.assert_called_once()
        self.assertEqual(llm.prompts, [])
        self.assertEqual(context["query_rewrite_reason"], "self_contained_query")
        self.assertFalse(context["found"])

    def test_search_with_conditional_rewrite_defers_rerank_and_fuses(self):
        calls = []

        def search_side_effect(query, **kwargs):
            calls.append((query, kwargs["apply_reranker"]))
            if query == "那它怎么配置？":
                return make_search_result([])
            result = make_search_result([make_chunk("rewritten", 0.9)])
            result["reranker_top_n"] = 5
            return result

        rewrites = []

        def rewrite_fn(question, history):
            rewrites.append((question, history))
            return {"query": "改写后的查询", "used": True, "error": None, "latency_ms": 1.0}

        with (
            patch.dict(os.environ, {"ENABLE_RERANKER": "true"}, clear=False),
            patch(
                "backend.rag_service.search_relevant_chunks",
                side_effect=search_side_effect,
            ),
            patch(
                "backend.rag_service.rerank_chunks_with_metadata",
                return_value={
                    "chunks": [make_chunk("rewritten", 0.9)],
                    "reranker_used": True,
                    "reranker_model": "mock/model",
                    "reranker_top_n": 5,
                    "reranker_error": None,
                },
            ) as mock_rerank,
        ):
            result, delta = rag_service.search_with_conditional_rewrite(
                "那它怎么配置？",
                history_context="用户：Kubernetes HPA 是什么？",
                query_rewrite_llm=object(),
                top_k=2,
                search_kwargs={"reranker_enabled": True},
                rewrite_fn=rewrite_fn,
            )

        self.assertEqual(calls, [("那它怎么配置？", False), ("改写后的查询", False)])
        self.assertEqual(rewrites, [("那它怎么配置？", "用户：Kubernetes HPA 是什么？")])
        self.assertTrue(delta["query_fusion_used"])
        self.assertEqual(delta["query_rewrite_reason"], "rewritten")
        mock_rerank.assert_called_once()
        self.assertEqual(
            [chunk["source"] for chunk in result["chunks"]],
            ["rewritten.md"],
        )

    def test_search_with_conditional_rewrite_without_history_skips_rewrite(self):
        calls = []

        def search_side_effect(query, **kwargs):
            calls.append((query, kwargs["apply_reranker"]))
            return make_search_result([])

        with patch(
            "backend.rag_service.search_relevant_chunks",
            side_effect=search_side_effect,
        ):
            result, delta = rag_service.search_with_conditional_rewrite(
                "那它怎么配置？",
                history_context=None,
                query_rewrite_llm=object(),
                top_k=2,
                search_kwargs={"reranker_enabled": True},
                rewrite_fn=lambda _q, _h: {"query": "x", "used": True, "error": None, "latency_ms": 0.0},
            )

        self.assertEqual(calls, [("那它怎么配置？", True)])
        self.assertEqual(delta["query_rewrite_reason"], "missing_history")
        self.assertFalse(delta["query_rewrite_attempted"])

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

    def test_always_rewrite_keeps_original_and_rewritten_candidates(self):
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
                query_rewrite_mode="always",
            )

        self.assertTrue(context["found"])
        self.assertEqual(
            {source["source"] for source in context["sources"]},
            {"follow-up.md", "kubernetes.md"},
        )
        self.assertEqual(context["query_rewrite_reason"], "rewritten")

    def test_rewrite_with_reranker_fuses_pool_before_single_rerank(self):
        llm = FakeLLM("Kubernetes HPA 配置")
        original_chunks = [
            make_chunk("o1", 0.9),
            make_chunk("o2", 0.8),
            make_chunk("shared", 0.7),
        ]
        rewritten_chunks = [
            make_chunk("r1", 0.85),
            make_chunk("shared", 0.75),
            make_chunk("r2", 0.6),
        ]

        def search_side_effect(query, **kwargs):
            self.assertFalse(kwargs["apply_reranker"])
            chunks = original_chunks if query == "那它怎么配置？" else rewritten_chunks
            return {
                **make_search_result(chunks),
                "reranker_enabled": True,
                "reranker_used": False,
                "reranker_top_n": 3,
                "reranker_candidate_count": 3,
            }

        with (
            patch.dict(os.environ, {"ENABLE_RERANKER": "true"}, clear=False),
            patch(
                "backend.rag_service.search_relevant_chunks",
                side_effect=search_side_effect,
            ),
            patch(
                "backend.rag_service.rerank_chunks_with_metadata",
                return_value={
                    "chunks": [rewritten_chunks[0], original_chunks[2]],
                    "reranker_enabled": True,
                    "reranker_used": True,
                    "reranker_model": "mock/model",
                    "reranker_top_n": 3,
                    "reranker_error": None,
                },
            ) as mock_rerank,
        ):
            context = rag_service.get_rag_context(
                "那它怎么配置？",
                top_k=2,
                reranker_enabled=True,
                query_rewrite_mode="always",
                query_rewrite_llm=llm,
                history_context="用户：Kubernetes HPA 是什么？",
            )

        self.assertTrue(context["found"])
        self.assertTrue(context["query_fusion_used"])
        mock_rerank.assert_called_once()
        self.assertEqual(mock_rerank.call_args.args[0], "那它怎么配置？")
        self.assertEqual(mock_rerank.call_args.args[2], 2)
        self.assertEqual(
            [item["chunk_id"] for item in mock_rerank.call_args.args[1]],
            ["shared", "o1", "r1"],
        )
        self.assertEqual(
            [source["source"] for source in context["sources"]],
            ["r1.md", "shared.md"],
        )
        self.assertTrue(context["reranker_used"])
        self.assertEqual(context["reranker_candidate_count"], 3)

    def test_fusion_rerank_fallback_applies_hybrid_top1_gate(self):
        llm = FakeLLM("Kubernetes HPA 配置")
        weak_bm25_chunk = {
            "chunk_id": "weak",
            "source": "weak.md",
            "text": "弱关联的 BM25 命中内容",
            "score": 8.2,
            "retrieval": "hybrid",
            "bm25_score": 8.2,
            "bm25_rank": 1,
            "bm25_entity_term_count": 2,
            "bm25_entity_match_count": 1,
        }
        vector_hit_chunk = {
            "chunk_id": "vec",
            "source": "vec.md",
            "text": "向量侧过阈值的内容",
            "score": 0.8,
            "retrieval": "hybrid",
            "vector_score": 0.8,
            "vector_rank": 2,
        }
        other_vector_chunk = {
            "chunk_id": "vec2",
            "source": "vec2.md",
            "text": "另一条向量侧过阈值的内容",
            "score": 0.75,
            "retrieval": "hybrid",
            "vector_score": 0.75,
            "vector_rank": 2,
        }

        def search_side_effect(query, **kwargs):
            self.assertFalse(kwargs["apply_reranker"])
            chunks = (
                [weak_bm25_chunk, vector_hit_chunk]
                if query == "那它怎么配置？"
                else [weak_bm25_chunk, other_vector_chunk]
            )
            return {
                **make_search_result(chunks, retrieval_mode="hybrid"),
                "reranker_enabled": True,
                "reranker_used": False,
                "reranker_top_n": 2,
                "reranker_candidate_count": 2,
            }

        with (
            patch.dict(os.environ, {"ENABLE_RERANKER": "true"}, clear=False),
            patch(
                "backend.rag_service.search_relevant_chunks",
                side_effect=search_side_effect,
            ),
            patch(
                "backend.rag_service.rerank_chunks_with_metadata",
                return_value={
                    "chunks": [weak_bm25_chunk, vector_hit_chunk],
                    "reranker_enabled": True,
                    "reranker_used": False,
                    "reranker_model": "mock/model",
                    "reranker_top_n": 2,
                    "reranker_error": "Reranker model is unavailable",
                },
            ),
        ):
            context = rag_service.get_rag_context(
                "那它怎么配置？",
                top_k=2,
                reranker_enabled=True,
                retrieval_mode="hybrid",
                query_rewrite_llm=llm,
                history_context="用户：Kubernetes HPA 是什么？",
                query_rewrite_mode="conditional",
            )

        self.assertFalse(context["found"])
        self.assertFalse(context["reranker_used"])
        self.assertEqual(context["reranker_error"], "Reranker model is unavailable")

    def test_fusion_rerank_success_skips_hybrid_top1_gate(self):
        llm = FakeLLM("Kubernetes HPA 配置")
        weak_bm25_chunk = {
            "chunk_id": "weak",
            "source": "weak.md",
            "text": "弱关联的 BM25 命中内容",
            "score": 8.2,
            "retrieval": "hybrid",
            "bm25_score": 8.2,
            "bm25_rank": 1,
            "bm25_entity_term_count": 2,
            "bm25_entity_match_count": 1,
        }
        vector_hit_chunk = {
            "chunk_id": "vec",
            "source": "vec.md",
            "text": "向量侧过阈值的内容",
            "score": 0.8,
            "retrieval": "hybrid",
            "vector_score": 0.8,
            "vector_rank": 2,
        }

        def search_side_effect(query, **_kwargs):
            chunks = (
                [weak_bm25_chunk, vector_hit_chunk]
                if query == "那它怎么配置？"
                else [weak_bm25_chunk]
            )
            return {
                **make_search_result(chunks, retrieval_mode="hybrid"),
                "reranker_enabled": True,
                "reranker_used": False,
                "reranker_top_n": 2,
                "reranker_candidate_count": len(chunks),
            }

        with (
            patch.dict(os.environ, {"ENABLE_RERANKER": "true"}, clear=False),
            patch(
                "backend.rag_service.search_relevant_chunks",
                side_effect=search_side_effect,
            ),
            patch(
                "backend.rag_service.rerank_chunks_with_metadata",
                return_value={
                    "chunks": [weak_bm25_chunk, vector_hit_chunk],
                    "reranker_enabled": True,
                    "reranker_used": True,
                    "reranker_model": "mock/model",
                    "reranker_top_n": 2,
                    "reranker_error": None,
                },
            ),
        ):
            context = rag_service.get_rag_context(
                "那它怎么配置？",
                top_k=2,
                reranker_enabled=True,
                retrieval_mode="hybrid",
                query_rewrite_llm=llm,
                history_context="用户：Kubernetes HPA 是什么？",
                query_rewrite_mode="conditional",
            )

        self.assertTrue(context["found"])
        self.assertTrue(context["reranker_used"])
        self.assertEqual(context["sources"][0]["source"], "weak.md")


class ContextWindowExpansionTests(unittest.TestCase):
    def setUp(self):
        self.original_chunks = rag_store.chunks
        rag_store.chunks = [make_store_chunk("course.md", index) for index in range(5)]
        rag_store.chunks.append(make_store_chunk("other.md", 0))

    def tearDown(self):
        rag_store.chunks = self.original_chunks

    @staticmethod
    def _anchor(chunk_index: int, source: str = "course.md") -> dict:
        return {
            "chunk_id": f"{source}:{chunk_index}",
            "source": source,
            "chunk_index": chunk_index,
            "text": f"{source} #{chunk_index} 内容",
            "score": 0.8,
            "retrieval": "hybrid",
        }

    def _run_context(self, anchors):
        with patch(
            "backend.rag_service.search_relevant_chunks",
            return_value=make_search_result(anchors),
        ):
            return rag_service.get_rag_context("问题")

    @staticmethod
    def _block_contents(context) -> list[str]:
        return [
            block.split("内容：\n", 1)[1]
            for block in context["context"].split("\n\n---\n\n")
        ]

    def test_context_expands_prev_and_next_neighbors(self):
        context = self._run_context([self._anchor(2)])

        self.assertEqual(
            self._block_contents(context),
            [
                "course.md #1 内容",
                "course.md #2 内容",
                "course.md #3 内容",
            ],
        )
        blocks = context["context"].split("\n\n---\n\n")
        self.assertIn("检索方式：neighbor", blocks[0])
        self.assertNotIn("得分", blocks[0])
        self.assertIn("检索方式：hybrid", blocks[1])
        self.assertIn("得分：0.8000", blocks[1])
        self.assertEqual(context["neighbor_chunk_count"], 2)
        self.assertEqual(
            [source["chunk_id"] for source in context["sources"]],
            ["course.md:2"],
        )

    def test_adjacent_anchors_share_neighbors_without_duplication(self):
        context = self._run_context([self._anchor(1), self._anchor(2)])

        self.assertEqual(
            self._block_contents(context),
            [
                "course.md #0 内容",
                "course.md #1 内容",
                "course.md #2 内容",
                "course.md #3 内容",
            ],
        )
        self.assertEqual(context["neighbor_chunk_count"], 2)
        self.assertEqual(len(context["sources"]), 2)

    def test_context_expansion_truncates_anchor_over_budget(self):
        anchor = self._anchor(1)
        anchor["text"] = "长" * 2000
        overhead = _context_block_overhead(anchor, "hybrid", include_score=True)
        with patch.object(rag_service, "CONTEXT_BUDGET_CHARS", 500):
            context = self._run_context([anchor])

        self.assertEqual(context["context_truncated_chunk_count"], 1)
        self.assertLessEqual(context["context_chars"], 500)
        content = self._block_contents(context)[0]
        self.assertEqual(content, "长" * (500 - overhead - 1) + "…")

    def test_neighbors_fill_only_leftover_budget(self):
        anchors = [self._anchor(0), self._anchor(4)]
        text_len = len(anchors[0]["text"])
        anchor_overhead = _context_block_overhead(anchors[0], "hybrid", include_score=True)
        neighbor_overhead = _context_block_overhead(
            make_store_chunk("course.md", 1), "neighbor", include_score=False
        )
        budget = (
            2 * (anchor_overhead + text_len)
            + 2 * (neighbor_overhead + text_len)
            - 1
        )

        with patch.object(rag_service, "CONTEXT_BUDGET_CHARS", budget):
            context = self._run_context(anchors)

        self.assertEqual(context["context_anchor_count"], 2)
        self.assertEqual(context["neighbor_chunk_count"], 1)

    def test_duplicate_anchor_text_is_rendered_once(self):
        first = self._anchor(1)
        second = self._anchor(3)
        second["text"] = first["text"]
        context = self._run_context([first, second])

        self.assertEqual(context["context_anchor_count"], 1)
        self.assertEqual(context["context_dropped_chunk_count"], 1)
        self.assertEqual(
            self._block_contents(context),
            [
                "course.md #0 内容",
                "course.md #1 内容",
                "course.md #2 内容",
            ],
        )
        self.assertEqual(len(context["sources"]), 2)

    def test_per_source_cap_limits_context_anchors(self):
        anchors = [self._anchor(index) for index in range(4)]
        context = self._run_context(anchors)

        self.assertEqual(context["context_anchor_count"], 3)
        self.assertEqual(context["context_dropped_chunk_count"], 1)
        self.assertEqual(context["neighbor_chunk_count"], 1)
        contents = self._block_contents(context)
        self.assertEqual(len(contents), 4)
        self.assertEqual(contents[3], "course.md #3 内容")

    def test_adjacent_blocks_share_overlap_seam_once(self):
        seam = "这一段是相邻分块重叠的接缝内容，用于验证去重逻辑。"
        first = self._anchor(1)
        first["text"] = f"第一段 explaining the definition.{seam}"
        second = self._anchor(2)
        second["text"] = f"{seam}第二段 explains the usage."
        context = self._run_context([first, second])

        self.assertEqual(context["context"].count(seam), 1)
        self.assertIn("第二段 explains the usage.", context["context"])

    def test_redundant_header_lines_are_dropped(self):
        anchor = {
            "chunk_id": "flat.md:0",
            "source": "flat.md",
            "chunk_index": 0,
            "text": "flat doc content",
            "score": 0.8,
            "retrieval": "vector",
            "document_title": "flat.md",
            "title": "总览",
            "section": "总览",
        }
        context = self._run_context([anchor])
        block = context["context"].split("\n\n---\n\n")[0]

        self.assertIn("来源文件：flat.md", block)
        self.assertNotIn("文档：", block)
        self.assertNotIn("标题：", block)
        self.assertIn("章节：总览", block)
        self.assertIn("检索方式：vector", block)

    def test_anchor_without_chunk_index_gets_no_neighbors(self):
        anchor = {
            "chunk_id": "legacy:0",
            "source": "legacy.md",
            "text": "legacy anchor",
            "score": 0.7,
            "retrieval": "vector",
        }
        context = self._run_context([anchor])

        self.assertEqual(self._block_contents(context), ["legacy anchor"])
        self.assertEqual(context["neighbor_chunk_count"], 0)
        self.assertEqual(len(context["sources"]), 1)

    def test_rewrite_without_reranker_fuses_top_k_without_rerank(self):
        llm = FakeLLM("Kubernetes HPA 配置")
        original_chunks = [
            make_chunk("o1", 0.9),
            make_chunk("o2", 0.8),
            make_chunk("shared", 0.7),
        ]
        rewritten_chunks = [
            make_chunk("r1", 0.85),
            make_chunk("shared", 0.75),
            make_chunk("r2", 0.6),
        ]

        def search_side_effect(query, **kwargs):
            self.assertTrue(kwargs["apply_reranker"])
            chunks = original_chunks if query == "那它怎么配置？" else rewritten_chunks
            return make_search_result(chunks)

        with (
            patch(
                "backend.rag_service.search_relevant_chunks",
                side_effect=search_side_effect,
            ),
            patch(
                "backend.rag_service.rerank_chunks_with_metadata",
            ) as mock_rerank,
        ):
            context = rag_service.get_rag_context(
                "那它怎么配置？",
                top_k=2,
                reranker_enabled=False,
                query_rewrite_mode="always",
                query_rewrite_llm=llm,
                history_context="用户：Kubernetes HPA 是什么？",
            )

        mock_rerank.assert_not_called()
        self.assertTrue(context["found"])
        self.assertTrue(context["query_fusion_used"])
        self.assertEqual(
            [source["source"] for source in context["sources"]],
            ["shared.md", "o1.md"],
        )
        self.assertFalse(context["reranker_used"])

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
