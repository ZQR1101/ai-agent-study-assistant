import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.agent_core import _execute_agent_tool
from backend.pending_actions import (
    PendingActionRepository,
    set_pending_action_repository,
)
from backend.run_repository import RunRepository, set_run_repository
from backend.tool_registry import AuditLog
from backend.tools import TOOL_REGISTRY


class PendingActionTests(unittest.TestCase):
    def setUp(self):
        from backend.server import app

        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.run_repository = RunRepository(root / "runs")
        self.action_repository = PendingActionRepository(
            root / "pending-actions",
            ttl_seconds=300,
        )
        self.original_audit_log = TOOL_REGISTRY.audit_log
        TOOL_REGISTRY.audit_log = AuditLog(root / "tool-audit.jsonl")
        set_run_repository(self.run_repository)
        set_pending_action_repository(self.action_repository)
        self.client = TestClient(app)

    def tearDown(self):
        TOOL_REGISTRY.audit_log = self.original_audit_log
        set_pending_action_repository(None)
        set_run_repository(None)
        self.temporary_directory.cleanup()

    def _create_delete_run_action(self):
        request_run = self.run_repository.create_run(request={"message": "delete old run"})
        target_run = self.run_repository.create_run(request={"message": "old"})
        self.run_repository.update_run(request_run.id, status="awaiting_action")
        action = self.action_repository.create(
            run_id=request_run.id,
            tool_name="delete_run",
            arguments={"target_run_id": target_run.id},
            request_message="delete old run",
        )
        return request_run, target_run, action

    def test_agent_proposal_is_persisted_without_executing_tool(self):
        request_run = self.run_repository.create_run(request={"message": "reset index"})

        result = _execute_agent_tool(
            "reset_rag_index",
            "重置 RAG 索引",
            shared_context={
                "run_id": request_run.id,
                "original_input": "重置 RAG 索引",
            },
        )

        action = self.action_repository.get(result["pending_action"]["id"])
        self.assertEqual(action.status, "pending")
        self.assertEqual(action.tool_name, "reset_rag_index")
        self.assertIn("尚未执行", result["answer"])

    def test_chat_with_pending_action_leaves_run_awaiting_action(self):
        def pending_chat_result(request):
            action = self.action_repository.create(
                run_id=request.run_id,
                session_id=request.session_id,
                tool_name="reset_rag_index",
                arguments={},
                request_message=request.message,
            )
            payload = action.model_dump(mode="json") if hasattr(action, "model_dump") else action.dict()
            return {
                "answer": "请确认是否重置索引。",
                "mode": "agent",
                "model": "test-model",
                "sources": [],
                "trace": [],
                "plan": [{"tool": "reset_rag_index", "input": request.message}],
                "flashcards": [],
                "pending_actions": [payload],
                "runtime_info": {"runtime": "agent", "awaiting_action": True, "tool_calls": []},
            }

        with (
            patch("backend.ai_core.run_chat_request", side_effect=pending_chat_result),
            patch("backend.server.is_db_history_enabled", return_value=False),
            patch("backend.server.is_llm_judge_enabled", return_value=False),
        ):
            response = self.client.post("/chat", json={"message": "重置 RAG 索引"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["run_summary"]["status"], "awaiting_action")
        self.assertEqual(len(response.json()["pending_actions"]), 1)
        stored = self.run_repository.get_run(response.json()["run_id"])
        self.assertEqual(stored.status, "awaiting_action")
        self.assertIsNone(stored.finished_at)

    def test_approval_executes_exact_action_once_and_finishes_run(self):
        request_run, target_run, action = self._create_delete_run_action()

        response = self.client.post(
            f"/pending-actions/{action.id}/approve",
            json={},
            headers={"X-Requested-With": "AI-Study-Assistant"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"]["status"], "executed")
        self.assertEqual(self.run_repository.get_run(target_run.id).status, "deleted")
        finished = self.run_repository.get_run(request_run.id)
        self.assertEqual(finished.status, "completed")
        self.assertTrue(any(event.get("status") == "executed" for event in finished.audit))

        duplicate = self.client.post(
            f"/pending-actions/{action.id}/approve",
            json={},
            headers={"X-Requested-With": "AI-Study-Assistant"},
        )
        self.assertEqual(duplicate.status_code, 409)

    def test_rejection_does_not_execute_and_cannot_be_retried(self):
        request_run, target_run, action = self._create_delete_run_action()

        response = self.client.post(
            f"/pending-actions/{action.id}/reject",
            json={"reason": "keep it"},
            headers={"X-Requested-With": "AI-Study-Assistant"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"]["status"], "rejected")
        self.assertEqual(self.run_repository.get_run(target_run.id).status, "running")
        self.assertEqual(self.run_repository.get_run(request_run.id).status, "completed")

        retry = self.client.post(
            f"/pending-actions/{action.id}/approve",
            json={},
            headers={"X-Requested-With": "AI-Study-Assistant"},
        )
        self.assertEqual(retry.status_code, 409)

    def test_decision_requires_in_app_request_header(self):
        _, target_run, action = self._create_delete_run_action()

        response = self.client.post(f"/pending-actions/{action.id}/approve", json={})

        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.action_repository.get(action.id).status, "pending")
        self.assertEqual(self.run_repository.get_run(target_run.id).status, "running")

    def test_expired_action_cannot_be_approved(self):
        expired_repository = PendingActionRepository(
            Path(self.temporary_directory.name) / "expired-actions",
            ttl_seconds=-1,
        )
        set_pending_action_repository(expired_repository)
        request_run = self.run_repository.create_run(request={"message": "delete old run"})
        target_run = self.run_repository.create_run(request={"message": "old"})
        self.run_repository.update_run(request_run.id, status="awaiting_action")
        action = expired_repository.create(
            run_id=request_run.id,
            tool_name="delete_run",
            arguments={"target_run_id": target_run.id},
        )

        response = self.client.post(
            f"/pending-actions/{action.id}/approve",
            json={},
            headers={"X-Requested-With": "AI-Study-Assistant"},
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(expired_repository.get(action.id).status, "expired")
        self.assertEqual(self.run_repository.get_run(target_run.id).status, "running")
        self.assertEqual(self.run_repository.get_run(request_run.id).status, "partial")


if __name__ == "__main__":
    unittest.main()
