import unittest
from unittest.mock import patch

from backend.ai_core import run_chat_request
from backend.capabilities.chat import CHAT_MODES, ChatResult, run_chat
from backend.rag_service import NO_RAG_ANSWER, RAG_FALLBACK_PREFIX
from backend.schemas import ChatRequest
from backend.tools import TOOL_REGISTRY


class _RecordingRegistry:
    def __init__(self, rag_result: dict | None = None, study_answers: dict | None = None, chat_answer: str = "chat answer"):
        self.calls: list[tuple[str, dict]] = []
        self.rag_result = rag_result
        self.study_answers = study_answers or {}
        self.chat_answer = chat_answer

    def execute(self, name: str, **kwargs):
        self.calls.append((name, kwargs))
        if name == "rag_search":
            return dict(self.rag_result or {})
        if name == "study":
            operation = kwargs.get("operation")
            return {
                "answer": self.study_answers[operation],
                "trace": [f"study operations: {operation}"],
                "flashcards": [],
            }
        if name == "chat":
            return {"answer": self.chat_answer, "trace": [], "flashcards": []}
        raise KeyError(name)


def _hit_rag_result() -> dict:
    return {
        "answer": "",
        "sources": [{"source": "rag.md", "score": 0.9}],
        "context": "retrieved evidence",
        "trace": ["RAG query：RAG"],
        "used_context": True,
        "fallback_used": False,
        "retrieval_info": {
            "found": True,
            "max_score": 0.9,
            "reranker_enabled": True,
            "reranker_used": True,
            "reranker_model": "mock/model",
            "reranker_top_n": 20,
            "reranker_error": None,
            "retrieval_mode": "hybrid",
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
        "retrieval_info": {"found": False, "retrieval_mode": "vector"},
    }


class ChatCapabilityTests(unittest.TestCase):
    def test_chat_modes_are_aliases_not_tools(self):
        for name in CHAT_MODES | {"learn"}:
            if name == "chat":
                continue
            self.assertNotIn(name, TOOL_REGISTRY)
        self.assertIn("chat", TOOL_REGISTRY)
        self.assertIn("study", TOOL_REGISTRY)
        self.assertIn("rag_search", TOOL_REGISTRY)

    def test_explain_mode_calls_study_not_a_legacy_explain_tool(self):
        registry = _RecordingRegistry(study_answers={"explain": "explained RAG"})

        result = run_chat("RAG", mode="explain", registry=registry)

        self.assertEqual([name for name, _ in registry.calls], ["study"])
        self.assertEqual(registry.calls[0][1]["operation"], "explain")
        self.assertEqual(registry.calls[0][1]["actor"], "chat")
        self.assertEqual(result.answer, "explained RAG")
        self.assertEqual(result.operation, "explain")
        self.assertEqual(result.plan[0]["tool"], "study")
        self.assertEqual(result.plan[0]["arguments"]["operation"], "explain")

    def test_summarize_and_quiz_select_study_operations(self):
        for mode in ("summarize", "quiz"):
            registry = _RecordingRegistry(study_answers={mode: f"{mode} output"})
            result = run_chat("RAG", mode=mode, registry=registry)
            self.assertEqual(registry.calls[0][0], "study")
            self.assertEqual(registry.calls[0][1]["operation"], mode)
            self.assertEqual(result.answer, f"{mode} output")

    def test_chat_mode_calls_chat_tool(self):
        registry = _RecordingRegistry(chat_answer="hello")

        result = run_chat("hi", mode="chat", registry=registry)

        self.assertEqual([name for name, _ in registry.calls], ["chat"])
        self.assertNotIn("operation", registry.calls[0][1])
        self.assertEqual(result.answer, "hello")

    def test_rag_hit_retrieves_then_chats_with_context(self):
        registry = _RecordingRegistry(_hit_rag_result(), chat_answer="grounded")

        result = run_chat("RAG", mode="rag", registry=registry)

        self.assertEqual([name for name, _ in registry.calls], ["rag_search", "chat"])
        self.assertFalse(registry.calls[0][1]["generate_answer"])
        self.assertEqual(
            registry.calls[1][1]["shared_context"]["rag_context"],
            "retrieved evidence",
        )
        self.assertEqual(result.answer, "grounded")
        self.assertEqual(result.sources[0]["source"], "rag.md")
        self.assertFalse(result.fallback_used)

    def test_rag_miss_refuses_without_generation(self):
        registry = _RecordingRegistry(_miss_rag_result(), chat_answer="should not run")

        result = run_chat("RAG", mode="rag", registry=registry)

        self.assertEqual([name for name, _ in registry.calls], ["rag_search"])
        self.assertEqual(result.answer, NO_RAG_ANSWER)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.sources, [])

    def test_explain_with_rag_miss_prefixes_fallback(self):
        registry = _RecordingRegistry(
            _miss_rag_result(),
            study_answers={"explain": "generic explanation"},
        )

        result = run_chat("RAG", mode="explain", use_rag=True, registry=registry)

        self.assertEqual([name for name, _ in registry.calls], ["rag_search", "study"])
        self.assertNotIn("rag_context", registry.calls[1][1]["shared_context"])
        self.assertTrue(result.answer.startswith(RAG_FALLBACK_PREFIX))
        self.assertIn("generic explanation", result.answer)
        self.assertTrue(result.fallback_used)

    def test_chat_request_explain_mode_uses_capability(self):
        request = ChatRequest(message="RAG", mode="explain")
        chat_result = ChatResult(
            answer="explained",
            trace=["Capability chat：started (explain/explain)"],
            plan=[{
                "tool": "study",
                "input": "RAG",
                "reason": "Explain the topic.",
                "arguments": {"operation": "explain"},
            }],
            operation="explain",
        )

        with (
            patch("backend.ai_core.build_llm", return_value=object()),
            patch("backend.ai_core.run_chat", return_value=chat_result) as mock_chat,
            patch("backend.ai_core.run_agent") as mock_agent,
        ):
            result = run_chat_request(request)

        mock_agent.assert_not_called()
        mock_chat.assert_called_once()
        self.assertEqual(mock_chat.call_args.kwargs["mode"], "explain")
        self.assertEqual(result["mode"], "explain")
        self.assertEqual(result["runtime_info"]["capability"], "chat")
        self.assertEqual(result["runtime_info"]["operation"], "explain")
        self.assertEqual(result["plan"][0]["tool"], "study")


if __name__ == "__main__":
    unittest.main()
