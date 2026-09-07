import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.ai_core import run_chat_request
from backend.capabilities.learn import LearnResult, learning_workflow, run_learn
from backend.rag_service import LEARN_FALLBACK_PREFIX
from backend.schemas import ChatRequest
from backend.tools import TOOL_REGISTRY


class _FakeResponse:
    def __init__(self, content: str):
        self.content = content


class _FakeLLM:
    def invoke(self, prompt: str):
        self.prompt = prompt
        return _FakeResponse("1. 做练习\n2. 回顾要点")


class _RecordingRegistry:
    def __init__(self, rag_result: dict, study_answers: dict[str, str]):
        self.calls: list[tuple[str, dict]] = []
        self.rag_result = rag_result
        self.study_answers = study_answers

    def execute(self, name: str, **kwargs):
        self.calls.append((name, kwargs))
        if name == "rag_search":
            return dict(self.rag_result)
        if name == "study":
            operation = kwargs.get("operation")
            return {
                "answer": self.study_answers[operation],
                "trace": [f"study operations: {operation}"],
                "flashcards": [],
            }
        raise KeyError(name)


def _hit_rag_result() -> dict:
    return {
        "answer": "",
        "sources": [{"source": "rag.md", "score": 0.9}],
        "context": "RAG retrieves then generates.",
        "trace": ["RAG query：RAG"],
        "used_context": True,
        "fallback_used": False,
        "retrieval_info": {
            "found": True,
            "max_score": 0.9,
            "threshold": 0.3,
            "retrieval_mode": "vector",
        },
    }


def _miss_rag_result() -> dict:
    return {
        "answer": "",
        "sources": [],
        "context": "",
        "trace": ["RAG 是否命中：否"],
        "used_context": False,
        "fallback_used": True,
        "retrieval_info": {
            "found": False,
            "max_score": None,
            "threshold": 0.3,
            "retrieval_mode": "vector",
        },
    }


class LearnCapabilityTests(unittest.TestCase):
    def test_learn_is_not_registered_as_a_tool(self):
        self.assertNotIn("learn", TOOL_REGISTRY)
        self.assertIsNone(TOOL_REGISTRY.get("learn"))

    def test_run_learn_calls_rag_search_then_study_operations(self):
        registry = _RecordingRegistry(
            _hit_rag_result(),
            {
                "explain": "explained RAG",
                "summarize": "summary of RAG",
                "quiz": "1. What is RAG?",
            },
        )

        result = run_learn(
            "RAG",
            custom_llm=_FakeLLM(),
            registry=registry,
            history_context="用户：什么是检索？",
            run_id="run-1",
        )

        names = [name for name, _ in registry.calls]
        self.assertEqual(names, ["rag_search", "study", "study", "study"])
        self.assertFalse(registry.calls[0][1]["generate_answer"])
        self.assertEqual(registry.calls[0][1]["actor"], "learn")
        operations = [kwargs["operation"] for name, kwargs in registry.calls if name == "study"]
        self.assertEqual(operations, ["explain", "summarize", "quiz"])

        explain_kwargs = registry.calls[1][1]
        self.assertEqual(explain_kwargs["step_input"], "RAG")
        self.assertEqual(
            explain_kwargs["shared_context"]["rag_context"],
            "RAG retrieves then generates.",
        )

        summary_kwargs = registry.calls[2][1]
        self.assertEqual(summary_kwargs["step_input"], "explained RAG")
        self.assertNotIn("rag_context", summary_kwargs["shared_context"])

        quiz_kwargs = registry.calls[3][1]
        self.assertEqual(quiz_kwargs["step_input"], "explained RAG")

        self.assertEqual(result.knowledge, "explained RAG")
        self.assertEqual(result.summary, "summary of RAG")
        self.assertEqual(result.quiz, "1. What is RAG?")
        self.assertEqual(result.advice, "1. 做练习\n2. 回顾要点")
        self.assertEqual(result.sources[0]["source"], "rag.md")
        self.assertTrue(result.passed_threshold)
        self.assertFalse(result.fallback_used)
        self.assertEqual([step["tool"] for step in result.plan], ["rag_search", "study", "study", "study"])
        self.assertIn("知识内容：", result.answer)
        self.assertIn("explained RAG", result.answer)

    def test_run_learn_skips_rag_when_context_is_provided(self):
        registry = _RecordingRegistry(
            _hit_rag_result(),
            {
                "explain": "explained from context",
                "summarize": "summary",
                "quiz": "quiz",
            },
        )

        result = run_learn(
            "RAG",
            context="provided context",
            custom_llm=_FakeLLM(),
            registry=registry,
        )

        names = [name for name, _ in registry.calls]
        self.assertEqual(names, ["study", "study", "study"])
        self.assertEqual(
            registry.calls[0][1]["shared_context"]["rag_context"],
            "provided context",
        )
        self.assertTrue(result.passed_threshold)
        self.assertFalse(result.fallback_used)

    def test_run_learn_prefixes_display_knowledge_on_rag_miss(self):
        registry = _RecordingRegistry(
            _miss_rag_result(),
            {
                "explain": "generic explanation",
                "summarize": "summary",
                "quiz": "quiz",
            },
        )

        result = run_learn(
            "RAG",
            custom_llm=_FakeLLM(),
            registry=registry,
            prefix_on_rag_miss=True,
        )

        self.assertFalse(result.passed_threshold)
        self.assertTrue(result.fallback_used)
        self.assertTrue(result.knowledge.startswith(LEARN_FALLBACK_PREFIX))
        self.assertEqual(registry.calls[1][1]["step_input"], "RAG")
        self.assertNotIn("rag_context", registry.calls[1][1]["shared_context"])
        self.assertEqual(registry.calls[2][1]["step_input"], "generic explanation")
        self.assertEqual(result.summary, "summary")

    def test_learning_workflow_keeps_legacy_dict_shape(self):
        registry = _RecordingRegistry(
            _miss_rag_result(),
            {
                "explain": "generic explanation",
                "summarize": "summary",
                "quiz": "quiz",
            },
        )

        with patch("backend.capabilities.learn._default_registry", return_value=registry):
            payload = learning_workflow("RAG", custom_llm=_FakeLLM())

        self.assertEqual(
            set(payload),
            {
                "knowledge",
                "summary",
                "quiz",
                "advice",
                "sources",
                "highest_score",
                "threshold",
                "passed_threshold",
            },
        )
        self.assertEqual(payload["knowledge"], "generic explanation")
        self.assertFalse(payload["passed_threshold"])

    def test_chat_learn_mode_skips_outer_rag_and_uses_capability(self):
        request = ChatRequest(message="RAG", mode="learn", use_rag=True)
        learn_result = LearnResult(
            knowledge="explained",
            summary="summary",
            quiz="quiz",
            advice="advice",
            sources=[{"source": "rag.md"}],
            fallback_used=False,
            passed_threshold=True,
            trace=["Capability learn：started"],
            plan=[{"tool": "rag_search", "input": "RAG", "reason": "Retrieve grounding for the lesson."}],
            retrieval_info={"found": True, "retrieval_mode": "vector"},
        )

        with (
            patch("backend.ai_core.build_llm", return_value=object()),
            patch("backend.ai_core.run_learn", return_value=learn_result) as mock_learn,
        ):
            result = run_chat_request(request)

        mock_learn.assert_called_once()
        kwargs = mock_learn.call_args.kwargs
        self.assertTrue(kwargs["use_rag"])
        self.assertTrue(kwargs["prefix_on_rag_miss"])
        self.assertEqual(result["mode"], "learn")
        self.assertEqual(result["runtime_info"]["capability"], "learn")
        self.assertEqual(result["plan"][0]["tool"], "rag_search")
        self.assertIn("知识内容：", result["answer"])
        self.assertTrue(
            any(
                "交给 Learn capability 的 rag_search 执行" in item
                for block in result["trace"]
                for item in block["items"]
            )
        )


class LearnMemoryWiringTests(unittest.TestCase):
    def setUp(self):
        import backend.memory as memory_module

        self.memory_module = memory_module
        memory_module.reset_memory_engine()

    def tearDown(self):
        self.memory_module.reset_memory_engine()

    def _install_tmp_engine(self):
        from backend.memory import MemoryEngine
        from backend.memory.l1_store import L1Store
        from backend.memory.l2_store import L2Store
        from backend.memory.l3_profile import L3Store

        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        root = Path(tempdir.name)
        engine = MemoryEngine(
            l1=L1Store(root=root / "l1"),
            l2=L2Store(root=root / "l2"),
            l3=L3Store(root=root / "l3"),
        )
        self.memory_module._memory_engine = engine
        return engine

    def test_learn_records_event_extracts_fact_and_refreshes_profile(self):
        engine = self._install_tmp_engine()
        registry = _RecordingRegistry(
            _hit_rag_result(),
            {"explain": "explained RAG", "summarize": "summary", "quiz": "quiz"},
        )

        result = run_learn(
            "RAG",
            custom_llm=_FakeLLM(),
            registry=registry,
            session_id="sess-1",
            run_id="run-1",
        )

        events = engine.get_events(session_id="sess-1")
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.event_type, "lesson_completed")
        self.assertEqual(event.run_id, "run-1")
        self.assertEqual(event.data["topic"], "RAG")

        facts = engine.get_facts()
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].category, "mastery")
        self.assertEqual(facts[0].subject, "RAG")
        self.assertEqual(facts[0].l1_refs, [event.event_id])
        self.assertAlmostEqual(facts[0].confidence, 0.9)

        profile = engine.get_profile()
        self.assertEqual(profile.summary.total_l1_events, 1)
        self.assertEqual(profile.summary.total_l2_facts, 1)
        self.assertEqual(profile.summary.topics_studied, ["RAG"])
        self.assertEqual(profile.summary.current_mastery, {"RAG": 0.9})

        self.assertTrue(any(item.startswith("Memory L1：recorded") for item in result.trace))
        self.assertTrue(any(item.startswith("Memory L2：extracted 1") for item in result.trace))
        self.assertIn("Memory L3：profile refreshed", result.trace)

    def test_learn_without_session_id_records_nothing(self):
        engine = self._install_tmp_engine()
        registry = _RecordingRegistry(
            _hit_rag_result(),
            {"explain": "explained RAG", "summarize": "summary", "quiz": "quiz"},
        )

        result = run_learn("RAG", custom_llm=_FakeLLM(), registry=registry)

        self.assertEqual(engine.get_events(), [])
        self.assertEqual(engine.get_facts(), [])
        self.assertFalse(any(item.startswith("Memory L1") for item in result.trace))

    def test_learn_memory_error_is_swallowed(self):
        engine = self._install_tmp_engine()
        registry = _RecordingRegistry(
            _hit_rag_result(),
            {"explain": "explained RAG", "summarize": "summary", "quiz": "quiz"},
        )

        with patch.object(engine, "record_event", side_effect=RuntimeError("boom")):
            result = run_learn(
                "RAG", custom_llm=_FakeLLM(), registry=registry, session_id="sess-1"
            )

        self.assertTrue(any(item.startswith("Memory L1：skip") for item in result.trace))
        self.assertEqual(result.knowledge, "explained RAG")

    def test_learn_injects_memory_context_into_advice_prompt(self):
        registry = _RecordingRegistry(
            _hit_rag_result(),
            {"explain": "explained RAG", "summarize": "summary", "quiz": "quiz"},
        )
        fake_llm = _FakeLLM()

        run_learn(
            "RAG",
            custom_llm=fake_llm,
            registry=registry,
            memory_context="用户偏好简短讲解",
        )

        self.assertIn("个人学习画像", fake_llm.prompt)
        self.assertIn("用户偏好简短讲解", fake_llm.prompt)


class MemoryDispatchWiringTests(unittest.TestCase):
    def _learn_result(self):
        return LearnResult(
            knowledge="explained",
            summary="summary",
            quiz="quiz",
            advice="advice",
            sources=[],
            fallback_used=False,
            passed_threshold=True,
            trace=[],
            plan=[],
            retrieval_info={},
        )

    def _chat_result(self):
        result = MagicMock()
        result.answer = "hi"
        result.sources = []
        result.plan = []
        result.flashcards = []
        result.fallback_used = False
        result.trace = []
        result.retrieval_info = {}
        result.operation = "chat"
        return result

    def _request_items(self, result):
        for block in result["trace"]:
            if block["title"] == "请求参数":
                return block["items"]
        return []

    def test_dispatch_injects_memory_into_learn_when_enabled(self):
        fake_engine = MagicMock()
        fake_engine.get_context_for_llm.return_value = "profile text"
        request = ChatRequest(message="RAG", mode="learn", use_rag=True, session_id="sess-9")

        with (
            patch.dict(os.environ, {"ENABLE_MEMORY": "true"}),
            patch("backend.memory.get_memory_engine", return_value=fake_engine) as mock_engine,
            patch("backend.ai_core.build_llm", return_value=object()),
            patch("backend.ai_core.run_learn", return_value=self._learn_result()) as mock_learn,
        ):
            result = run_chat_request(request)

        mock_engine.assert_called_once()
        kwargs = mock_learn.call_args.kwargs
        self.assertEqual(kwargs["session_id"], "sess-9")
        self.assertEqual(kwargs["memory_context"], "profile text")
        self.assertIn("memory 注入：是", self._request_items(result))

    def test_dispatch_passes_memory_context_to_chat_modes(self):
        fake_engine = MagicMock()
        fake_engine.get_context_for_llm.return_value = "profile text"
        request = ChatRequest(message="hello", mode="chat")

        with (
            patch.dict(os.environ, {"ENABLE_MEMORY": "true"}),
            patch("backend.memory.get_memory_engine", return_value=fake_engine),
            patch("backend.ai_core.build_llm", return_value=object()),
            patch("backend.ai_core.run_chat", return_value=self._chat_result()) as mock_chat,
        ):
            run_chat_request(request)

        self.assertEqual(mock_chat.call_args.kwargs["memory_context"], "profile text")

    def test_dispatch_skips_memory_when_disabled(self):
        request = ChatRequest(message="RAG", mode="learn", use_rag=True, session_id="sess-9")

        with (
            patch("backend.memory.get_memory_engine") as mock_engine,
            patch("backend.ai_core.build_llm", return_value=object()),
            patch("backend.ai_core.run_learn", return_value=self._learn_result()) as mock_learn,
        ):
            result = run_chat_request(request)

        mock_engine.assert_not_called()
        self.assertIsNone(mock_learn.call_args.kwargs["memory_context"])
        self.assertIn("memory 注入：否", self._request_items(result))


class MemoryToolForwardingTests(unittest.TestCase):
    def test_generation_tools_forward_memory_context(self):
        from backend import tools as tools_module

        shared = {"history_context": "", "memory_context": "profile text"}
        cases = [
            (tools_module._run_chat_tool, "chat"),
            (tools_module._run_explain_tool, "explain"),
            (tools_module._run_summarize_tool, "summarize"),
            (tools_module._run_quiz_tool, "generate_questions"),
        ]
        for runner, fn_name in cases:
            with self.subTest(tool=fn_name):
                with patch.object(tools_module, fn_name, return_value="ok") as mock_fn:
                    runner("输入", shared_context=shared)
                self.assertEqual(mock_fn.call_args.kwargs["memory_context"], "profile text")

    def test_generation_tools_treat_empty_memory_context_as_none(self):
        from backend import tools as tools_module

        with patch.object(tools_module, "chat", return_value="ok") as mock_fn:
            tools_module._run_chat_tool("输入", shared_context={"memory_context": ""})
        self.assertIsNone(mock_fn.call_args.kwargs["memory_context"])


if __name__ == "__main__":
    unittest.main()
