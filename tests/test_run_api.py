import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.run_repository import RunRepository, set_run_repository
from backend.tool_registry import AuditLog


def _chat_result() -> dict:
    return {
        "answer": "RAG retrieves evidence before generation.",
        "mode": "agent",
        "model": "test-model",
        "sources": [{"source": "rag.md", "score": 0.9}],
        "trace": [{"title": "Execution", "items": ["used rag_search"]}],
        "plan": [
            {"tool": "rag_search", "input": "RAG", "reason": "retrieve evidence"},
        ],
        "flashcards": [],
        "runtime_info": {
            "runtime": "agent",
            "tool_calls": [
                {"tool": "planner", "success": True},
                {"tool": "rag_search", "success": True, "latency_ms": 3},
            ],
        },
    }


class _FailFirstUpdateRepository(RunRepository):
    def __init__(self, root: Path):
        super().__init__(root)
        self._fail_next_update = True

    def update_run(self, run_id: str, **changes):
        if self._fail_next_update:
            self._fail_next_update = False
            raise OSError("metadata write failed")
        return super().update_run(run_id, **changes)


class _FailFinishRepository(RunRepository):
    def finish_run(self, run_id: str, **changes):
        raise OSError("finish write failed")


class RunApiLifecycleTests(unittest.TestCase):
    def setUp(self):
        from backend.server import app

        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.repository = RunRepository(root / "runs")
        self.audit_log = AuditLog(root / "tool-audit.jsonl")
        set_run_repository(self.repository)
        self.client = TestClient(app)

    def tearDown(self):
        set_run_repository(None)
        self.temporary_directory.cleanup()

    def _post_chat(self):
        with (
            patch("backend.ai_core.run_chat_request", return_value=_chat_result()),
            patch("backend.server.get_run_repository", return_value=self.repository),
            patch("backend.server.is_db_history_enabled", return_value=False),
            patch("backend.server.is_llm_judge_enabled", return_value=False),
        ):
            return self.client.post("/chat", json={"message": "What is RAG?"})

    def test_chat_returns_run_id(self):
        response = self._post_chat()

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["run_id"])

    def test_runs_lists_the_run_just_created_by_chat(self):
        chat_response = self._post_chat()
        run_id = chat_response.json()["run_id"]

        with patch("backend.server.get_run_repository", return_value=self.repository):
            response = self.client.get("/runs")

        self.assertEqual(response.status_code, 200)
        self.assertIn(run_id, [run["id"] for run in response.json()["runs"]])

    def test_run_detail_contains_planner_audit_and_artifacts_but_not_judge(self):
        evaluation = {
            "judge_model": "judge-test",
            "accuracy": 9.0,
            "completeness": 8.0,
            "citation_quality": 9.0,
            "overall_score": 8.7,
            "verdict": "PASS",
            "deductions": [],
            "feedback": "Grounded answer.",
        }

        def execute_with_audit(request):
            self.audit_log.record({
                "event": "tool_call",
                "tool": "rag_search",
                "status": "succeeded",
                "actor": "agent",
                "run_id": request.run_id,
            })
            return _chat_result()

        with (
            patch("backend.ai_core.run_chat_request", side_effect=execute_with_audit),
            patch("backend.server.get_run_repository", return_value=self.repository),
            patch("backend.server.is_db_history_enabled", return_value=False),
            patch("backend.server.is_llm_judge_enabled", return_value=True),
            patch("backend.server.judge_answer", return_value=evaluation),
            patch("backend.server.get_database_url", return_value=None),
        ):
            chat_response = self.client.post("/chat", json={"message": "What is RAG?"})
            run_id = chat_response.json()["run_id"]
            detail_response = self.client.get(f"/runs/{run_id}")

        self.assertEqual(detail_response.status_code, 200)
        run = detail_response.json()
        self.assertEqual(run["plan"][0]["tool"], "rag_search")
        self.assertNotIn("judge", run)
        self.assertEqual(chat_response.json()["judge_evaluation"]["verdict"], "PASS")
        self.assertEqual(run["audit"][0]["tool"], "rag_search")
        self.assertEqual(run["artifacts"]["answer"], _chat_result()["answer"])
        self.assertEqual(run["artifacts"]["sources"][0]["source"], "rag.md")

    def test_delete_run_without_approval_key_does_not_receive_confirmation(self):
        run = self.repository.create_run(request={"message": "keep me"})
        request = {
            "arguments": {"target_run_id": run.id},
            "actor": "run-api-test",
        }

        with patch(
            "backend.server.get_config",
            return_value=SimpleNamespace(
                tool_approval_key="requester-secret",
                tool_approver_key="approver-secret",
            ),
        ):
            response = self.client.post(f"/tools/delete_run/invoke", json=request)

        self.assertEqual(response.status_code, 403)
        self.assertNotIn("confirmation_token", response.text)
        stored = self.repository.get_run(run.id)
        self.assertEqual(stored.status, "running")
        self.assertFalse(any(event.get("event") == "run_deleted" for event in stored.audit))

    def test_dangerous_tool_is_unavailable_when_approval_is_not_configured(self):
        run = self.repository.create_run(request={"message": "keep me"})
        request = {
            "arguments": {"target_run_id": run.id},
            "actor": "run-api-test",
        }

        with patch(
            "backend.server.get_config",
            return_value=SimpleNamespace(
                tool_approval_key="existing-requester-secret",
                tool_approver_key=None,
            ),
        ):
            response = self.client.post(
                "/tools/delete_run/invoke",
                json=request,
                headers={"X-Tool-Approval-Key": "any-value"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("confirmation_token", response.text)
        self.assertEqual(self.repository.get_run(run.id).status, "running")

    def test_approver_key_alone_cannot_request_dangerous_tool_execution(self):
        run = self.repository.create_run(request={"message": "keep me"})
        request = {
            "arguments": {"target_run_id": run.id},
            "actor": "run-api-test",
        }
        with patch(
            "backend.server.get_config",
            return_value=SimpleNamespace(
                tool_approval_key="requester-secret",
                tool_approver_key="approver-secret",
            ),
        ):
            response = self.client.post(
                "/tools/delete_run/invoke",
                json=request,
                headers={"X-Tool-Approver-Key": "approver-secret"},
            )

        self.assertEqual(response.status_code, 403)
        self.assertNotIn("approval_request_id", response.text)
        self.assertEqual(self.repository.get_run(run.id).status, "running")

    def test_confirmed_delete_run_soft_deletes_and_records_the_action(self):
        run = self.repository.create_run(request={"message": "delete me"})
        request = {
            "arguments": {"target_run_id": run.id},
            "actor": "run-api-test",
        }
        requester_headers = {"X-Tool-Approval-Key": "requester-secret"}
        approver_headers = {"X-Tool-Approver-Key": "approver-secret"}
        with patch(
            "backend.server.get_config",
            return_value=SimpleNamespace(
                tool_approval_key="requester-secret",
                tool_approver_key="approver-secret",
            ),
        ):
            first_response = self.client.post(
                "/tools/delete_run/invoke",
                json=request,
                headers=requester_headers,
            )
            self.assertEqual(first_response.status_code, 409)
            self.assertNotIn("confirmation_token", first_response.text)
            approval_request_id = first_response.json()["detail"][
                "approval_request_id"
            ]

            missing_approver_response = self.client.post(
                f"/tools/delete_run/approvals/{approval_request_id}",
            )
            self.assertEqual(missing_approver_response.status_code, 403)

            approval_response = self.client.post(
                f"/tools/delete_run/approvals/{approval_request_id}",
                headers=approver_headers,
            )
            self.assertEqual(approval_response.status_code, 200)
            token = approval_response.json()["confirmation_token"]

            rejected_response = self.client.post(
                "/tools/delete_run/invoke",
                json={**request, "confirmation_token": token},
                headers={"X-Tool-Approval-Key": "wrong-secret"},
            )
            self.assertEqual(rejected_response.status_code, 403)

            confirmed_response = self.client.post(
                "/tools/delete_run/invoke",
                json={
                    **request,
                    "actor": "forged-different-actor",
                    "confirmation_token": token,
                },
                headers=requester_headers,
            )

        self.assertEqual(confirmed_response.status_code, 200)
        self.assertTrue(confirmed_response.json()["soft_deleted"])
        stored = self.repository.get_run(run.id)
        self.assertEqual(stored.status, "deleted")
        self.assertIsNotNone(stored.deleted_at)
        self.assertTrue(any(event.get("event") == "run_deleted" for event in stored.audit))
        self.assertTrue(any(
            event.get("tool") == "delete_run"
            and event.get("status") == "succeeded"
            and str(event.get("actor", "")).startswith("requester:")
            and event.get("actor") != "forged-different-actor"
            for event in stored.audit
        ))

    def test_dangerous_tool_rejects_identical_requester_and_approver_keys(self):
        run = self.repository.create_run(request={"message": "keep me"})
        request = {
            "arguments": {"target_run_id": run.id},
            "actor": "run-api-test",
        }
        with patch(
            "backend.server.get_config",
            return_value=SimpleNamespace(
                tool_approval_key="shared-secret",
                tool_approver_key="shared-secret",
            ),
        ):
            response = self.client.post(
                "/tools/delete_run/invoke",
                json=request,
                headers={"X-Tool-Approval-Key": "shared-secret"},
            )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("approval_request_id", response.text)
        self.assertEqual(self.repository.get_run(run.id).status, "running")

    def test_failed_chat_keeps_a_failed_run(self):
        failing_client = TestClient(self.client.app, raise_server_exceptions=False)
        with (
            patch("backend.ai_core.run_chat_request", side_effect=RuntimeError("planner exploded")),
            patch("backend.server.get_run_repository", return_value=self.repository),
            patch("backend.server.is_db_history_enabled", return_value=False),
        ):
            response = failing_client.post("/chat", json={"message": "fail"})

        self.assertEqual(response.status_code, 500)
        runs = self.repository.list_runs()
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0].status, "failed")
        self.assertEqual(runs[0].error, "planner exploded")
        self.assertIsNotNone(runs[0].finished_at)

    def test_update_failure_is_caught_and_fallback_marks_run_failed(self):
        repository = _FailFirstUpdateRepository(
            Path(self.temporary_directory.name) / "fail-first-update-runs"
        )
        with (
            patch("backend.ai_core.run_chat_request", return_value=_chat_result()),
            patch("backend.server.get_run_repository", return_value=repository),
            patch("backend.server.is_db_history_enabled", return_value=False),
            patch("backend.server.is_llm_judge_enabled", return_value=False),
        ):
            response = self.client.post("/chat", json={"message": "persist safely"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["runtime_info"]["run_persistence_error"],
            "metadata write failed",
        )
        stored = repository.get_run(response.json()["run_id"])
        self.assertEqual(stored.status, "failed")
        self.assertEqual(stored.error, "metadata write failed")

    def test_finish_failure_does_not_escape_when_fallback_also_fails(self):
        repository = _FailFinishRepository(
            Path(self.temporary_directory.name) / "fail-finish-runs"
        )
        with (
            patch("backend.ai_core.run_chat_request", return_value=_chat_result()),
            patch("backend.server.get_run_repository", return_value=repository),
            patch("backend.server.is_db_history_enabled", return_value=False),
            patch("backend.server.is_llm_judge_enabled", return_value=False),
            self.assertLogs("backend.server", level="ERROR") as captured_logs,
        ):
            response = self.client.post("/chat", json={"message": "persist safely"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["runtime_info"]["run_persistence_error"],
            "finish write failed",
        )
        self.assertTrue(any("Unable to mark run" in line for line in captured_logs.output))

    def test_persistence_failure_does_not_mask_original_chat_exception(self):
        repository = _FailFinishRepository(
            Path(self.temporary_directory.name) / "original-error-runs"
        )
        with (
            patch(
                "backend.ai_core.run_chat_request",
                side_effect=RuntimeError("planner exploded"),
            ),
            patch("backend.server.get_run_repository", return_value=repository),
            patch("backend.server.is_db_history_enabled", return_value=False),
            self.assertLogs("backend.server", level="ERROR"),
            self.assertRaisesRegex(RuntimeError, "planner exploded"),
        ):
            self.client.post("/chat", json={"message": "fail without masking"})

    def test_unknown_run_summary_status_is_stored_as_partial(self):
        with (
            patch("backend.ai_core.run_chat_request", return_value=_chat_result()),
            patch("backend.server.get_run_repository", return_value=self.repository),
            patch("backend.server.is_db_history_enabled", return_value=False),
            patch("backend.server.is_llm_judge_enabled", return_value=False),
            patch(
                "backend.server.build_run_metadata",
                return_value=({"status": "unexpected"}, {}),
            ),
            self.assertLogs("backend.server", level="WARNING") as captured_logs,
        ):
            response = self.client.post("/chat", json={"message": "unknown status"})

        self.assertEqual(response.status_code, 200)
        stored = self.repository.get_run(response.json()["run_id"])
        self.assertEqual(stored.status, "partial")
        self.assertTrue(any("Unknown run summary status" in line for line in captured_logs.output))


if __name__ == "__main__":
    unittest.main()
