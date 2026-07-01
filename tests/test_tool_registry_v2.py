import json
import tempfile
import unittest
from pathlib import Path

from backend.tool_registry import (
    AuditLog,
    InvalidConfirmation,
    ToolCategory,
    ToolConfirmationRequired,
    ToolRegistry,
    ToolSpec,
)
from backend.run_repository import RunRepository, set_run_repository
from backend.tools import TOOL_REGISTRY


class ToolRegistryV2Tests(unittest.TestCase):
    def _registry(self, directory: str, calls: list) -> ToolRegistry:
        def safe_tool(value=""):
            calls.append(("safe", value))
            return {"value": value}

        def dangerous_tool(value=""):
            calls.append(("dangerous", value))
            return {"value": value}

        return ToolRegistry(
            [
                ToolSpec("safe", "safe", safe_tool),
                ToolSpec(
                    "dangerous",
                    "dangerous",
                    dangerous_tool,
                    ToolCategory.DANGEROUS,
                    requires_confirmation=True,
                ),
            ],
            audit_log=AuditLog(Path(directory) / "audit.jsonl"),
        )

    def test_public_registry_merges_legacy_generation_tools(self):
        public_names = set(TOOL_REGISTRY)
        self.assertIn("study", public_names)
        self.assertIn("rag_search", public_names)
        for legacy_name in ("explain", "summarize", "quiz", "flashcard", "rag"):
            self.assertNotIn(legacy_name, public_names)
        self.assertIsNone(TOOL_REGISTRY.get("explain"))
        self.assertIsNone(TOOL_REGISTRY.get("rag"))

    def test_openapi_exposes_unified_tool_api_not_legacy_duplicates(self):
        from backend.server import app

        paths = app.openapi()["paths"]
        self.assertIn("/chat", paths)
        self.assertIn("/tools", paths)
        self.assertIn("/tools/{tool_name}/invoke", paths)
        self.assertIn("/tools/audit/recent", paths)
        self.assertIn("/runs", paths)
        self.assertIn("/runs/{run_id}", paths)
        for legacy_path in (
            "/explain",
            "/summarize",
            "/quiz",
            "/rag",
            "/agent",
            "/learn",
            "/rebuild-index",
        ):
            self.assertNotIn(legacy_path, paths)

    def test_dangerous_tools_are_confirmation_gated(self):
        dangerous = [
            spec
            for spec in TOOL_REGISTRY.values()
            if spec.category is ToolCategory.DANGEROUS
        ]
        self.assertTrue(dangerous)
        self.assertTrue(all(spec.requires_confirmation for spec in dangerous))
        self.assertIn("delete_run", {spec.name for spec in dangerous})

    def test_tool_audit_is_attached_to_its_run(self):
        with tempfile.TemporaryDirectory() as directory:
            repository = RunRepository(Path(directory) / "runs")
            run = repository.create_run(request={"message": "test"})

            def tool(shared_context=None):
                return {"answer": "ok"}

            registry = ToolRegistry(
                [ToolSpec("safe", "safe", tool)],
                audit_log=AuditLog(Path(directory) / "audit.jsonl"),
            )
            set_run_repository(repository)
            try:
                registry.execute("safe", shared_context={"run_id": run.id})
                stored = repository.get_run(run.id)
            finally:
                set_run_repository(None)

            self.assertIsNotNone(stored)
            self.assertEqual(
                [event["status"] for event in stored.audit],
                ["started", "succeeded"],
            )
            self.assertTrue(all(event["run_id"] == run.id for event in stored.audit))

    def test_confirmation_is_bound_to_actor_arguments_and_single_use(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = []
            registry = self._registry(directory, calls)

            with self.assertRaises(ToolConfirmationRequired) as raised:
                registry.execute("dangerous", value="A", actor="user-1")
            self.assertEqual(calls, [])

            token = raised.exception.token
            result = registry.execute(
                "dangerous",
                value="A",
                actor="user-1",
                confirmation_token=token,
            )
            self.assertEqual(result, {"value": "A"})
            self.assertEqual(calls, [("dangerous", "A")])

            with self.assertRaises(InvalidConfirmation):
                registry.execute(
                    "dangerous",
                    value="A",
                    actor="user-1",
                    confirmation_token=token,
                )

    def test_argument_changes_reject_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = self._registry(directory, [])
            with self.assertRaises(ToolConfirmationRequired) as raised:
                registry.execute("dangerous", value="A", actor="user-1")
            with self.assertRaises(InvalidConfirmation):
                registry.execute(
                    "dangerous",
                    value="B",
                    actor="user-1",
                    confirmation_token=raised.exception.token,
                )

    def test_argument_changes_after_a_long_common_prefix_reject_confirmation(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = self._registry(directory, [])
            common_prefix = "x" * 2000
            with self.assertRaises(ToolConfirmationRequired) as raised:
                registry.execute(
                    "dangerous",
                    value=f"{common_prefix}-approved",
                    actor="user-1",
                )
            with self.assertRaises(InvalidConfirmation):
                registry.execute(
                    "dangerous",
                    value=f"{common_prefix}-changed",
                    actor="user-1",
                    confirmation_token=raised.exception.token,
                )

    def test_audit_records_blocked_success_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            registry = self._registry(directory, [])
            registry.execute("safe", value="ok", actor="test")
            with self.assertRaises(ToolConfirmationRequired):
                registry.execute("dangerous", value="blocked", actor="test")
            with self.assertRaises(KeyError):
                registry.execute("missing", actor="test")

            events = registry.audit_log.recent()
            statuses = [event["status"] for event in events]
            self.assertIn("started", statuses)
            self.assertIn("succeeded", statuses)
            self.assertIn("confirmation_required", statuses)
            self.assertIn("unknown_tool", statuses)
            for line in (Path(directory) / "audit.jsonl").read_text(encoding="utf-8").splitlines():
                json.loads(line)


if __name__ == "__main__":
    unittest.main()
