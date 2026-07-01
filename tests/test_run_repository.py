import tempfile
import unittest
from pathlib import Path

from backend.run_repository import RunRepository


class RunRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.repository = RunRepository(Path(self.temporary_directory.name) / "runs")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_complete_run_lifecycle(self):
        created = self.repository.create_run(
            request={"message": "What is RAG?"},
            session_id="session-1",
            metadata={"entrypoint": "/chat"},
        )
        self.assertEqual(created.status, "running")

        updated = self.repository.update_run(
            created.id,
            plan=[{"tool": "rag_search", "input": "RAG"}],
            artifacts={"sources": [{"source": "rag.md"}]},
            metadata={"planner": {"mode": "rule"}},
        )
        self.assertGreater(updated.version, created.version)
        self.assertEqual(updated.metadata["entrypoint"], "/chat")

        self.repository.append_audit(created.id, {"tool": "rag_search", "status": "succeeded"})
        finished = self.repository.finish_run(
            created.id,
            output={"answer": "Retrieved answer"},
        )
        self.assertEqual(finished.status, "completed")
        self.assertIsNotNone(finished.finished_at)
        self.assertEqual(finished.audit[0]["tool"], "rag_search")

        loaded = self.repository.get_run(created.id)
        self.assertEqual(loaded.output["answer"], "Retrieved answer")
        self.assertEqual(loaded.artifacts["sources"][0]["source"], "rag.md")

        listed = self.repository.list_runs(status="completed", session_id="session-1")
        self.assertEqual([run.id for run in listed], [created.id])

        self.assertTrue(self.repository.delete_run(created.id))
        deleted = self.repository.get_run(created.id)
        self.assertEqual(deleted.status, "deleted")
        self.assertIsNotNone(deleted.deleted_at)
        self.assertEqual(deleted.audit[-1]["event"], "run_deleted")
        self.assertEqual(self.repository.list_runs(), [])
        self.assertEqual(
            [run.id for run in self.repository.list_runs(include_deleted=True)],
            [created.id],
        )
        self.assertFalse(self.repository.delete_run(created.id))

    def test_rejects_duplicate_and_invalid_run_ids(self):
        self.repository.create_run(run_id="known-run")
        with self.assertRaises(ValueError):
            self.repository.create_run(run_id="known-run")
        with self.assertRaises(ValueError):
            self.repository.get_run("../outside")


if __name__ == "__main__":
    unittest.main()
