import unittest
from contextlib import nullcontext
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.run_metadata import build_run_metadata
from backend.run_repository import RunRepository
from backend.schemas import ChatResponse


class RunMetadataTests(unittest.TestCase):
    def test_builds_summary_and_details_for_agent_response(self):
        result = {
            "mode": "agent",
            "sources": [{"source": "rag.md"}],
            "plan": [{"tool": "rag_search", "input": "RAG"}],
            "runtime_info": {
                "runtime": "agent",
                "planner_mode": "agent",
                "tool_calls": [
                    {"tool": "planner", "success": True},
                    {"tool": "rag_search", "success": True},
                ],
                "token_usage": {"total_tokens": 42},
                "estimated_cost": {"total": 0.001},
            },
            "judge_evaluation": {"overall_score": 8.7, "verdict": "PASS"},
        }

        summary, details = build_run_metadata(result, duration_ms=123)

        self.assertEqual(summary["status"], "succeeded")
        self.assertEqual(summary["runtime"], "agent")
        self.assertEqual(summary["duration_ms"], 123)
        self.assertEqual(summary["step_count"], 2)
        self.assertEqual(summary["tool_count"], 1)
        self.assertEqual(summary["source_count"], 1)
        self.assertEqual(details["plan"], result["plan"])
        self.assertEqual(details["tools"], result["runtime_info"]["tool_calls"])
        self.assertNotIn("judge", details)

    def test_failed_tool_marks_run_partial(self):
        summary, _ = build_run_metadata({
            "mode": "agent",
            "runtime_info": {
                "tool_calls": [{"tool": "study", "success": False}],
            },
        })

        self.assertEqual(summary["status"], "partial")

    def test_chat_response_defaults_new_fields(self):
        response = ChatResponse(answer="ok", mode="chat", model="test")

        self.assertEqual(response.run_summary, {})
        self.assertEqual(response.run_details, {})

    def test_chat_endpoint_returns_run_summary_and_details(self):
        from backend.server import app

        backend_result = {
            "answer": "ok",
            "mode": "agent",
            "model": "test-model",
            "sources": [],
            "trace": [],
            "plan": [{"tool": "study", "input": "RAG", "reason": "explain"}],
            "flashcards": [],
            "runtime_info": {
                "runtime": "agent",
                "tool_calls": [
                    {"tool": "planner", "success": True},
                    {"tool": "study", "success": True, "latency_ms": 8},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            repository = RunRepository(Path(directory) / "runs")
            with (
                patch("backend.ai_core.run_chat_request", return_value=backend_result),
                patch("backend.server.is_db_history_enabled", return_value=False),
                patch("backend.server.is_llm_judge_enabled", return_value=False),
                patch("backend.server.get_run_repository", return_value=repository),
            ):
                response = TestClient(app).post("/chat", json={"message": "test"})

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["run_id"])
            self.assertEqual(payload["run_summary"]["tool_count"], 1)
            self.assertEqual(payload["run_details"]["plan"][0]["tool"], "study")
            self.assertEqual(payload["run_details"]["tools"][1]["tool"], "study")
            self.assertNotIn("judge", payload["run_details"])
            stored = repository.get_run(payload["run_id"])
            self.assertEqual(stored.status, "completed")
            self.assertEqual(stored.artifacts["answer"], "ok")

    def test_chat_endpoint_persists_assistant_response_snapshot(self):
        from backend.server import app

        backend_result = {
            "answer": "RAG uses retrieved context.",
            "mode": "agent",
            "model": "test-model",
            "sources": [{"source": "rag.md", "score": 0.9}],
            "trace": [{"title": "Execution", "items": ["used rag"]}],
            "plan": [],
            "flashcards": [],
            "runtime_info": {"runtime": "agent", "tool_calls": []},
        }
        fake_db = object()
        with tempfile.TemporaryDirectory() as directory:
            repository = RunRepository(Path(directory) / "runs")
            with (
                patch("backend.ai_core.run_chat_request", return_value=backend_result),
                patch("backend.server.is_db_history_enabled", return_value=True),
                patch("backend.server.is_llm_judge_enabled", return_value=False),
                patch("backend.server.get_run_repository", return_value=repository),
                patch("backend.server.get_db_session", side_effect=lambda: nullcontext(fake_db)),
                patch("backend.server.create_or_get_session", return_value="session-1"),
                patch("backend.server.get_recent_messages", return_value=[]),
                patch("backend.server.save_message") as save_message_mock,
            ):
                response = TestClient(app).post("/chat", json={"message": "What is RAG?"})

            self.assertEqual(response.status_code, 200)
            self.assertEqual(save_message_mock.call_count, 2)
            assistant_call = save_message_mock.call_args_list[1]
            self.assertEqual(assistant_call.args[2], "assistant")
            snapshot = assistant_call.kwargs["response"]
            self.assertEqual(snapshot["sources"][0]["source"], "rag.md")
            self.assertEqual(snapshot["trace"][0]["items"], ["used rag"])
            self.assertEqual(snapshot["run_summary"]["source_count"], 1)
            self.assertIn("run_details", snapshot)


if __name__ == "__main__":
    unittest.main()
