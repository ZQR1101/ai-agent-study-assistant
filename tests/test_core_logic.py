import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from backend.ai_core import run_chat_request, run_langgraph_chat_request
from backend.agent_core import _extract_json_object, _fallback_agent_plan
from backend.config import DEFAULT_MODEL, get_config, normalize_model
from backend.history_utils import HISTORY_LIMIT, format_history, normalize_history
from backend.judge_service import JudgeEvaluationError, compute_verdict, judge_answer
from backend.llm_service import explain, summarize_usage_records, track_llm_usage
from backend import rag_store, tools as tools_module
from backend.rag_store import expand_query, get_rag_index_status, is_valid_chunk
from backend.schemas import AgentPlan, AgentPlanStep, ChatRequest, ChatResponse, FlashcardItem
from backend.tools import TOOL_REGISTRY, ToolSpec


class HistoryUtilsTests(unittest.TestCase):
    def test_empty_history_returns_empty_values(self):
        self.assertEqual(normalize_history([]), [])
        self.assertEqual(normalize_history(None), [])
        self.assertEqual(format_history([]), "")

    def test_history_is_limited_to_recent_items(self):
        history = [
            {"role": "user", "content": f"message {index}"}
            for index in range(HISTORY_LIMIT + 3)
        ]

        normalized = normalize_history(history)

        self.assertEqual(len(normalized), HISTORY_LIMIT)
        self.assertEqual(normalized[0]["content"], "message 3")
        self.assertEqual(normalized[-1]["content"], f"message {HISTORY_LIMIT + 2}")

    def test_invalid_roles_and_empty_content_are_filtered(self):
        normalized = normalize_history(
            [
                {"role": "system", "content": "do not include"},
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "   "},
                {"role": "user", "content": "include me"},
            ]
        )

        self.assertEqual(normalized, [{"role": "user", "content": "include me"}])

    def test_formatted_history_contains_user_and_assistant_content(self):
        formatted = format_history(
            normalize_history(
                [
                    {"role": "user", "content": "What is RAG?"},
                    {"role": "assistant", "content": "RAG retrieves context first."},
                ]
            )
        )

        self.assertIn("What is RAG?", formatted)
        self.assertIn("RAG retrieves context first.", formatted)


class LLMUsageTrackingTests(unittest.TestCase):
    class FakeResponse:
        def __init__(self, content: str, usage_metadata: dict | None = None):
            self.content = content
            self.usage_metadata = usage_metadata or {}

    class FakeLLM:
        def __init__(self, response):
            self.response = response

        def invoke(self, prompt: str):
            self.prompt = prompt
            return self.response

    def test_explain_prompt_uses_concise_default_for_all_context_modes(self):
        cases = [
            {},
            {"history_context": "用户：什么是 RAG？"},
            {"context": "RAG 会先检索知识，再生成回答。"},
        ]

        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                fake_llm = self.FakeLLM(self.FakeResponse("ok"))
                explain("RAG", custom_llm=fake_llm, **kwargs)

                self.assertIn("否则控制在 300 字以内", fake_llm.prompt)
                self.assertIn("优先遵循用户要求", fake_llm.prompt)
                self.assertIn("3 个核心要点和 1 个简短例子", fake_llm.prompt)


    def test_usage_tracker_prefers_api_token_usage(self):
        tracked_llm = track_llm_usage(
            self.FakeLLM(self.FakeResponse(
                "ok",
                {"input_tokens": 12, "output_tokens": 4, "total_tokens": 16},
            )),
            "deepseek-v4-pro",
        )

        self.assertEqual(tracked_llm.invoke("hello").content, "ok")
        summary = summarize_usage_records(tracked_llm.usage_records)

        self.assertEqual(summary["token_usage"]["input_tokens"], 12)
        self.assertEqual(summary["token_usage"]["output_tokens"], 4)
        self.assertEqual(summary["token_usage"]["total_tokens"], 16)
        self.assertEqual(summary["token_usage"]["source"], "api")
        self.assertGreater(summary["estimated_cost"]["total"], 0)

    def test_usage_tracker_estimates_when_api_usage_is_missing(self):
        tracked_llm = track_llm_usage(
            self.FakeLLM(self.FakeResponse("estimated response")),
            "deepseek-v4-pro",
        )

        tracked_llm.invoke("estimate this prompt")
        summary = summarize_usage_records(tracked_llm.usage_records)

        self.assertEqual(summary["token_usage"]["source"], "estimated")
        self.assertGreater(summary["token_usage"]["input_tokens"], 0)
        self.assertGreater(summary["token_usage"]["output_tokens"], 0)
        self.assertGreater(summary["estimated_cost"]["total"], 0)


class JudgeServiceTests(unittest.TestCase):
    class FakeResponse:
        def __init__(self, content: str):
            self.content = content

    class FakeLLM:
        def __init__(self, content: str):
            self.content = content
            self.prompts = []

        def invoke(self, prompt: str):
            self.prompts.append(prompt)
            return JudgeServiceTests.FakeResponse(self.content)

    def test_judge_answer_parses_scores(self):
        fake_llm = self.FakeLLM("""
        {
          "accuracy": 8,
          "completeness": 7.5,
          "citation_quality": 6,
          "overall_score": 7.2,
          "feedback": "Mostly correct.",
          "deductions": [
            {"metric": "Citation Quality", "points": 4, "reason": "Missing explicit citation."}
          ]
        }
        """)

        result = judge_answer(
            "What is RAG?",
            "RAG retrieves context before generation.",
            sources=[{"source": "rag.md", "snippet": "RAG retrieves relevant context."}],
            trace=[{"title": "Trace", "items": ["used rag"]}],
            runtime_info={"tool_calls": [{"tool": "rag", "latency_ms": 12}]},
            model="judge-model",
            judge_llm=fake_llm,
        )

        self.assertEqual(result["judge_model"], "judge-model")
        self.assertEqual(result["accuracy"], 8)
        self.assertEqual(result["completeness"], 7.5)
        self.assertEqual(result["citation_quality"], 6)
        self.assertEqual(result["overall_score"], 7.2)
        self.assertEqual(result["verdict"], "WEAK_PASS")
        self.assertEqual(result["deductions"][0]["metric"], "Citation Quality")
        self.assertEqual(result["feedback"], "Mostly correct.")
        self.assertIn("User question", fake_llm.prompts[0])
        self.assertIn("Tool calls", fake_llm.prompts[0])

    def test_judge_answer_clamps_scores_and_estimates_overall(self):
        fake_llm = self.FakeLLM("""
        Judge result:
        {"accuracy": 12, "completeness": -2, "citation_quality": 9}
        """)

        result = judge_answer("Q", "A", judge_llm=fake_llm)

        self.assertEqual(result["accuracy"], 10)
        self.assertEqual(result["completeness"], 0)
        self.assertEqual(result["citation_quality"], 9)
        self.assertAlmostEqual(result["overall_score"], 6.33, places=2)
        self.assertEqual(result["verdict"], "WEAK_PASS")
        self.assertTrue(result["deductions"])

    def test_judge_answer_allows_citation_quality_na(self):
        fake_llm = self.FakeLLM("""
        {
          "accuracy": 9,
          "completeness": 8,
          "citation_quality": null,
          "overall_score": 8.5,
          "feedback": "Good answer for a task that does not need citations."
        }
        """)

        result = judge_answer("Say hello", "Hello.", judge_llm=fake_llm)

        self.assertIsNone(result["citation_quality"])
        self.assertEqual(result["verdict"], "PASS")

    def test_compute_verdict_applies_citation_gate(self):
        self.assertEqual(compute_verdict(8.1, 7), "PASS")
        self.assertEqual(compute_verdict(8.1, 5), "WEAK_PASS")
        self.assertEqual(compute_verdict(5.9, 10), "FAIL")

    def test_judge_answer_rejects_invalid_json(self):
        with self.assertRaises(JudgeEvaluationError):
            judge_answer("Q", "A", judge_llm=self.FakeLLM("not json"))


class RagStoreTests(unittest.TestCase):
    def test_invalid_chunks_are_rejected(self):
        for text in ["|", "---", "###", "too short"]:
            with self.subTest(text=text):
                self.assertFalse(is_valid_chunk(text))

    def test_valid_chinese_chunk_is_accepted(self):
        text = (
            "检索增强生成会先从知识库中查找相关内容，"
            "再让语言模型基于这些上下文生成更可靠的回答。"
        )

        self.assertTrue(is_valid_chunk(text))

    def test_valid_english_chunk_is_accepted(self):
        text = (
            "Retrieval augmented generation first retrieves relevant context "
            "from a knowledge base, then asks the language model to answer."
        )

        self.assertTrue(is_valid_chunk(text))

    def test_expand_query_adds_known_expansions(self):
        self.assertIn("代理式RAG", expand_query("agentic rag"))
        self.assertIn("提示工程", expand_query("prompt engineering"))
        self.assertEqual(expand_query("ordinary question"), "ordinary question")

    def test_pure_rag_helpers_do_not_load_embedding_model(self):
        is_valid_chunk("Retrieval augmented generation uses retrieved context before answering.")
        expand_query("agentic rag")

        self.assertIsNone(rag_store.embedding_model)

    def test_missing_local_embedding_model_returns_empty_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_model = Path(tmpdir) / "missing-model"
            with (
                patch.dict(
                    os.environ,
                    {
                        "EMBEDDING_MODEL_PATH": str(missing_model),
                        "EMBEDDING_MODEL_LOCAL_ONLY": "true",
                        "ENABLE_RAG_AUTO_BUILD": "true",
                    },
                    clear=False,
                ),
                patch.object(rag_store, "INDEX_FILE", Path(tmpdir) / "index.faiss"),
                patch.object(rag_store, "CHUNKS_FILE", Path(tmpdir) / "chunks.json"),
                patch.object(
                    rag_store,
                    "build_chunks",
                    return_value=[{"source": "doc.md", "text": "RAG retrieves context before generation."}],
                ),
            ):
                rag_store.embedding_model = None
                rag_store.index = None
                rag_store.chunks = []
                rag_store.rag_index_error = None

                result = rag_store.search_relevant_chunks("what is RAG", include_metadata=True)

                self.assertEqual(result["chunks"], [])
                self.assertFalse(result["passed_threshold"])
                self.assertIn("Embedding model path not found", result["error"])
                self.assertIsNone(rag_store.embedding_model)

            rag_store.embedding_model = None
            rag_store.index = None
            rag_store.chunks = []
            rag_store.rag_index_error = None

    def test_auto_build_disabled_does_not_load_embedding_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.dict(
                    os.environ,
                    {"ENABLE_RAG_AUTO_BUILD": "false"},
                    clear=False,
                ),
                patch.object(rag_store, "INDEX_FILE", Path(tmpdir) / "index.faiss"),
                patch.object(rag_store, "CHUNKS_FILE", Path(tmpdir) / "chunks.json"),
                patch.object(rag_store, "rebuild_rag_index") as mock_rebuild,
                patch.object(rag_store, "get_embedding_model") as mock_model,
            ):
                rag_store.index = None
                rag_store.chunks = []
                rag_store.rag_index_error = None

                result = rag_store.search_relevant_chunks(
                    "what is RAG",
                    include_metadata=True,
                )

                self.assertEqual(result["chunks"], [])
                self.assertEqual(
                    result["error"],
                    "Automatic RAG index building is disabled",
                )
                mock_rebuild.assert_not_called()
                mock_model.assert_not_called()

            rag_store.index = None
            rag_store.chunks = []
            rag_store.rag_index_error = None


class SchemaTests(unittest.TestCase):
    def test_chat_request_can_be_created(self):
        request = ChatRequest(message="What is RAG?", mode="explain", top_k=3)

        self.assertEqual(request.message, "What is RAG?")
        self.assertEqual(request.mode, "explain")
        self.assertEqual(request.model, "deepseek-v4-pro")
        self.assertFalse(request.use_langgraph)
        self.assertEqual(request.planner_mode, "rule")
        self.assertEqual(request.retrieval_mode, "vector")

    def test_chat_request_accepts_retrieval_mode(self):
        request = ChatRequest(message="What is RAG?", retrieval_mode="hybrid")

        self.assertEqual(request.retrieval_mode, "hybrid")

    def test_chat_request_accepts_langgraph_flag(self):
        request = ChatRequest(message="What is RAG?", use_langgraph=True)

        self.assertTrue(request.use_langgraph)

    def test_chat_request_accepts_llm_planner_mode(self):
        request = ChatRequest(message="What is RAG?", planner_mode="llm")

        self.assertEqual(request.planner_mode, "llm")

    def test_chat_response_runtime_info_defaults_to_empty_dict(self):
        response = ChatResponse(answer="ok", mode="chat", model="deepseek-v4-pro")

        self.assertEqual(response.runtime_info, {})

    def test_chat_response_accepts_judge_evaluation(self):
        response = ChatResponse(
            answer="ok",
            mode="chat",
            model="deepseek-v4-pro",
            judge_evaluation={
                "accuracy": 8,
                "completeness": 7,
                "citation_quality": None,
                "overall_score": 7,
                "verdict": "WEAK_PASS",
                "deductions": [
                    {"metric": "Completeness", "points": 3, "reason": "Missing one part."}
                ],
            },
        )

        self.assertEqual(response.judge_evaluation.overall_score, 7)
        self.assertIsNone(response.judge_evaluation.citation_quality)

    def test_chat_request_rejects_invalid_values(self):
        invalid_payloads = [
            {"message": ""},
            {"message": "hello", "temperature": -0.1},
            {"message": "hello", "temperature": 2.1},
            {"message": "hello", "top_k": 0},
            {"message": "hello", "top_k": 11},
            {"message": "hello", "planner_mode": "bad"},
            {"message": "hello", "retrieval_mode": "bad"},
        ]

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(ValidationError):
                    ChatRequest(**payload)

    def test_flashcard_difficulty_is_restricted(self):
        FlashcardItem(front="Q", back="A", difficulty="easy")
        FlashcardItem(front="Q", back="A", difficulty="medium")
        FlashcardItem(front="Q", back="A", difficulty="hard")

        with self.assertRaises(ValidationError):
            FlashcardItem(front="Q", back="A", difficulty="expert")

    def test_agent_plan_step_tool_is_restricted(self):
        AgentPlanStep(tool="chat", input="hello")

        with self.assertRaises(ValidationError):
            AgentPlanStep(tool="unknown", input="hello")


class AgentCoreTests(unittest.TestCase):
    def test_extract_json_object_from_plain_json(self):
        self.assertEqual(_extract_json_object('{"goal": "test"}'), {"goal": "test"})

    def test_extract_json_object_from_surrounding_text(self):
        text = 'Planner result follows: {"goal": "test", "steps": []} done.'

        self.assertEqual(
            _extract_json_object(text),
            {"goal": "test", "steps": []},
        )

    def test_extract_json_object_returns_none_for_invalid_json(self):
        self.assertIsNone(_extract_json_object("not json at all"))
        self.assertIsNone(_extract_json_object('before {"goal": "missing end" after'))

    def test_fallback_agent_plan_returns_valid_agent_plan(self):
        plan = _fallback_agent_plan("hello")

        if hasattr(AgentPlan, "model_validate"):
            validated = AgentPlan.model_validate(plan)
        else:
            validated = AgentPlan.parse_obj(plan)

        self.assertTrue(validated.fallback)
        self.assertEqual(validated.steps[0].tool, "chat")

    def test_fallback_agent_plan_routes_learning_requests(self):
        plan = _fallback_agent_plan("请解释 RAG，并出 3 道题")
        tools = {step["tool"] for step in plan["steps"]}

        self.assertTrue(tools & {"explain", "quiz", "rag"})

    def test_fallback_agent_plan_routes_unknown_input_to_chat(self):
        plan = _fallback_agent_plan("hello there")

        self.assertEqual(plan["steps"][0]["tool"], "chat")


class LangGraphChatRoutingTests(unittest.TestCase):
    def test_auto_agent_langgraph_flag_enters_langgraph_branch(self):
        request = ChatRequest(
            message="use langgraph",
            mode="auto",
            use_agent=True,
            use_langgraph=True,
        )

        with patch("backend.ai_core.run_langgraph_chat_request", return_value={"mode": "langgraph"}) as mock_run:
            result = run_chat_request(request)

        self.assertEqual(result["mode"], "langgraph")
        mock_run.assert_called_once_with(request)

    def test_auto_agent_without_langgraph_flag_uses_existing_agent_branch(self):
        request = ChatRequest(
            message="use agent",
            mode="auto",
            use_agent=True,
            use_langgraph=False,
        )
        agent_result = {
            "answer": "agent answer",
            "sources": [],
            "trace": ["Agent trace"],
            "plan": {"steps": [{"tool": "chat", "input": "use agent"}]},
            "flashcards": [],
            "fallback_used": False,
        }

        with (
            patch("backend.ai_core.run_langgraph_chat_request") as mock_langgraph,
            patch("backend.ai_core.build_llm", return_value=object()),
            patch("backend.ai_core.run_agent", return_value=agent_result) as mock_agent,
        ):
            result = run_chat_request(request)

        self.assertEqual(result["mode"], "agent")
        self.assertEqual(result["answer"], "agent answer")
        self.assertEqual(result["runtime_info"], {})
        mock_langgraph.assert_not_called()
        mock_agent.assert_called_once()

    def test_legacy_agent_ignores_llm_planner_mode_without_langgraph(self):
        request = ChatRequest(
            message="use agent",
            mode="auto",
            use_agent=True,
            use_langgraph=False,
            planner_mode="llm",
        )
        agent_result = {
            "answer": "agent answer",
            "sources": [],
            "trace": ["Agent trace"],
            "plan": {"steps": [{"tool": "chat", "input": "use agent"}]},
            "flashcards": [],
            "fallback_used": False,
        }

        with (
            patch("backend.ai_core.run_langgraph_chat_request") as mock_langgraph,
            patch("backend.ai_core.build_llm", return_value=object()),
            patch("backend.ai_core.run_agent", return_value=agent_result) as mock_agent,
        ):
            result = run_chat_request(request)

        self.assertEqual(result["mode"], "agent")
        mock_langgraph.assert_not_called()
        mock_agent.assert_called_once()

    def test_langgraph_chat_response_wraps_demo_result(self):
        request = ChatRequest(
            message="use langgraph",
            mode="auto",
            model="deepseek-v4-pro",
            use_agent=True,
            use_langgraph=True,
            top_k=5,
        )
        demo_result = {
            "answer": "langgraph answer",
            "sources": [{"source": "demo.md", "score": 0.9}],
            "trace": ["planner: start", "rag: call tool=rag"],
            "plan": [{"tool": "rag", "input": "use langgraph"}],
            "flashcards": [
                {
                    "front": "Q",
                    "back": "A",
                    "tags": ["demo"],
                    "difficulty": "medium",
                }
            ],
            "runtime_info": {
                "runtime": "langgraph",
                "graph_path": ["planner", "rag", "finalizer"],
                "node_count": 3,
                "tool_calls": [],
                "finalizer_used": True,
                "error": None,
            },
        }
        fake_llm = object()

        with (
            patch("backend.langgraph_runtime.build_llm", return_value=fake_llm),
            patch("backend.langgraph_runtime.run_langgraph_workflow", return_value=demo_result) as mock_workflow,
        ):
            result = run_langgraph_chat_request(request)

        self.assertEqual(result["mode"], "langgraph")
        self.assertEqual(result["answer"], "langgraph answer")
        self.assertEqual(result["sources"], [{"source": "demo.md", "score": 0.9}])
        self.assertEqual(result["plan"], [{"tool": "rag", "input": "use langgraph"}])
        self.assertEqual(len(result["flashcards"]), 1)
        self.assertEqual(result["runtime_info"]["runtime"], "langgraph")
        self.assertEqual(result["runtime_info"]["node_count"], 3)
        self.assertTrue(any("LangGraph workflow enabled" in item for block in result["trace"] for item in block["items"]))
        ChatResponse(**result)
        mock_workflow.assert_called_once()
        self.assertEqual(mock_workflow.call_args.kwargs["custom_llm"]._wrapped_llm, fake_llm)
        self.assertEqual(mock_workflow.call_args.kwargs["top_k"], 5)
        self.assertEqual(mock_workflow.call_args.kwargs["planner_mode"], "rule")

    def test_langgraph_chat_request_passes_llm_planner_mode(self):
        request = ChatRequest(
            message="use langgraph",
            mode="auto",
            use_agent=True,
            use_langgraph=True,
            planner_mode="llm",
        )

        with (
            patch("backend.langgraph_runtime.build_llm", return_value=object()),
            patch(
                "backend.langgraph_runtime.run_langgraph_workflow",
                return_value={
                    "answer": "ok",
                    "sources": [],
                    "trace": [],
                    "plan": [],
                    "flashcards": [],
                    "runtime_info": {"runtime": "langgraph"},
                },
            ) as mock_workflow,
        ):
            run_langgraph_chat_request(request)

        self.assertEqual(mock_workflow.call_args.kwargs["planner_mode"], "llm")

    def test_langgraph_unavailable_returns_friendly_response(self):
        from backend.langgraph_runtime import LangGraphRuntimeUnavailableError

        request = ChatRequest(
            message="use langgraph",
            mode="auto",
            use_agent=True,
            use_langgraph=True,
        )

        with (
            patch("backend.langgraph_runtime.build_llm", return_value=object()),
            patch(
                "backend.langgraph_runtime.run_langgraph_workflow",
                side_effect=LangGraphRuntimeUnavailableError("missing langgraph"),
            ),
        ):
            result = run_langgraph_chat_request(request)

        self.assertEqual(result["mode"], "langgraph")
        self.assertIn("missing langgraph", result["answer"])
        self.assertEqual(result["runtime_info"]["runtime"], "langgraph")
        self.assertEqual(result["runtime_info"]["error"], "missing langgraph")


class ToolsTests(unittest.TestCase):
    def test_tool_registry_contains_expected_tools(self):
        expected_tools = {
            "chat",
            "rag_search",
            "study",
            "save_note",
            "save_flashcards",
            "save_quiz",
            "delete_saved_item",
            "delete_knowledge_file",
            "reset_saved_items",
            "reset_rag_index",
            "rebuild_rag_index",
            "run_code_sandbox",
            "delete_run",
        }

        self.assertEqual(set(TOOL_REGISTRY), expected_tools)

    def test_registered_tools_have_required_structure(self):
        for name, tool in TOOL_REGISTRY.items():
            with self.subTest(tool=name):
                self.assertIsInstance(tool, ToolSpec)
                self.assertEqual(tool.name, name)
                self.assertTrue(tool.description.strip())
                self.assertTrue(callable(tool.run))

    def test_rag_tool_keeps_history_out_of_retrieval_query(self):
        history_context = "用户：百度 OCR 是什么？\n助手：百度 OCR 是文字识别服务。"
        rag_context = {
            "sources": [],
            "retrieval_mode": "hybrid",
            "candidate_k": 15,
            "vector_candidates": 0,
            "bm25_candidates": 0,
            "hybrid_used": True,
            "expanded_query": "什么是 skills",
            "max_score": None,
            "threshold": None,
            "raw_count": 0,
            "valid_count": 0,
            "discarded_invalid_count": 0,
            "found": False,
            "error": None,
        }

        with (
            patch("backend.tools.get_rag_context", return_value=rag_context) as mock_retrieval,
            patch("backend.tools.chat", return_value="fallback answer") as mock_chat,
        ):
            tools_module._run_rag_tool(
                "什么是 skills",
                custom_llm=object(),
                top_k=5,
                shared_context={
                    "history_context": history_context,
                    "retrieval_mode": "hybrid",
                },
            )

        mock_retrieval.assert_called_once_with(
            "什么是 skills",
            top_k=5,
            retrieval_mode="hybrid",
        )
        self.assertEqual(mock_chat.call_args.kwargs["history_context"], history_context)

    def test_rag_tool_marks_retrieved_context_as_used(self):
        rag_context = {
            "sources": [{"source": "agent_skills.md", "score": 0.8}],
            "context": "Agent Skill is a reusable workflow knowledge package.",
            "retrieval_mode": "hybrid",
            "candidate_k": 15,
            "vector_candidates": 1,
            "bm25_candidates": 1,
            "hybrid_used": True,
            "expanded_query": "Agent Skill SKILL.md",
            "max_score": 0.03,
            "threshold": None,
            "raw_count": 1,
            "valid_count": 1,
            "discarded_invalid_count": 0,
            "found": True,
            "error": None,
        }

        with (
            patch("backend.tools.get_rag_context", return_value=rag_context),
            patch("backend.tools.chat", return_value="answer"),
        ):
            result = tools_module._run_rag_tool(
                "什么是 skill",
                custom_llm=object(),
                shared_context={"retrieval_mode": "hybrid"},
            )

        self.assertTrue(result["used_context"])
        self.assertEqual(result["context_sources"], ["agent_skills.md"])

    def test_rag_tool_can_skip_answer_generation_for_downstream_node(self):
        rag_context = {
            "sources": [{"source": "agent_skills.md", "score": 0.8}],
            "context": "Agent Skill is a reusable workflow knowledge package.",
            "retrieval_mode": "vector",
            "candidate_k": 3,
            "vector_candidates": 1,
            "bm25_candidates": 0,
            "hybrid_used": False,
            "expanded_query": "Agent Skill",
            "max_score": 0.8,
            "threshold": None,
            "raw_count": 1,
            "valid_count": 1,
            "discarded_invalid_count": 0,
            "found": True,
            "error": None,
        }

        with (
            patch("backend.tools.get_rag_context", return_value=rag_context),
            patch("backend.tools.chat") as mock_chat,
        ):
            result = tools_module._run_rag_tool(
                "什么是 skill",
                custom_llm=object(),
                generate_answer=False,
            )

        mock_chat.assert_not_called()
        self.assertEqual(result["answer"], "")
        self.assertEqual(result["context"], rag_context["context"])
        self.assertIn("RAG answer generation：skipped", result["trace"])


class ConfigTests(unittest.TestCase):
    def test_unknown_model_falls_back_to_default(self):
        self.assertEqual(normalize_model("unknown-model"), DEFAULT_MODEL)

    def test_removed_mimo_model_falls_back_to_deepseek(self):
        self.assertEqual(DEFAULT_MODEL, "deepseek-v4-pro")
        self.assertEqual(normalize_model("mimo-v2.5"), "deepseek-v4-pro")

    def test_config_paths_are_project_local(self):
        config = get_config()

        self.assertEqual(config.docs_path.name, "docs")
        self.assertEqual(config.rag_index_dir.name, "rag_index")


class RagIndexStatusTests(unittest.TestCase):
    def test_rag_index_status_does_not_load_index(self):
        status = get_rag_index_status()

        self.assertIn("ready", status)
        self.assertIn("message", status)
        self.assertIsNone(rag_store.embedding_model)


if __name__ == "__main__":
    unittest.main()
