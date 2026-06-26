import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from backend import rag_store, rag_warmup


class FakeThread:
    created = []

    def __init__(self, target, kwargs=None, daemon=None):
        self.target = target
        self.kwargs = kwargs or {}
        self.daemon = daemon
        self.started = False
        self.__class__.created.append(self)

    def start(self):
        self.started = True


class RagWarmupStateTests(unittest.TestCase):
    def setUp(self):
        rag_warmup._reset_rag_warmup_status_for_tests()
        FakeThread.created = []

    def tearDown(self):
        rag_warmup._reset_rag_warmup_status_for_tests()

    def test_default_status_is_idle(self):
        status = rag_warmup.get_rag_warmup_status()

        self.assertEqual(status["status"], "idle")
        self.assertFalse(status["model_loaded"])
        self.assertFalse(status["index_loaded"])

    def test_start_rag_warmup_enters_loading(self):
        with patch("backend.rag_warmup.threading.Thread", FakeThread):
            result = rag_warmup.start_rag_warmup()

        self.assertTrue(result["started"])
        self.assertEqual(result["warmup"]["status"], "loading")
        self.assertEqual(len(FakeThread.created), 1)
        self.assertTrue(FakeThread.created[0].started)

    def test_loading_warmup_does_not_create_second_thread(self):
        with patch("backend.rag_warmup.threading.Thread", FakeThread):
            first = rag_warmup.start_rag_warmup()
            second = rag_warmup.start_rag_warmup()

        self.assertTrue(first["started"])
        self.assertFalse(second["started"])
        self.assertEqual(second["warmup"]["status"], "loading")
        self.assertEqual(len(FakeThread.created), 1)

    def test_ready_warmup_does_not_start_again(self):
        with (
            patch("backend.rag_warmup._load_embedding_model", return_value=object()),
            patch("backend.rag_warmup._load_existing_rag_index", return_value=False),
        ):
            rag_warmup.run_rag_warmup()

        FakeThread.created = []
        with patch("backend.rag_warmup.threading.Thread", FakeThread):
            result = rag_warmup.start_rag_warmup()

        self.assertFalse(result["started"])
        self.assertEqual(result["warmup"]["status"], "ready")
        self.assertEqual(len(FakeThread.created), 0)

    def test_error_warmup_can_be_retried(self):
        with patch("backend.rag_warmup._load_embedding_model", side_effect=RuntimeError("boom")):
            rag_warmup.run_rag_warmup()

        with patch("backend.rag_warmup.threading.Thread", FakeThread):
            result = rag_warmup.start_rag_warmup()

        self.assertTrue(result["started"])
        self.assertEqual(result["warmup"]["status"], "loading")
        self.assertEqual(len(FakeThread.created), 1)

    def test_successful_warmup_sets_ready_status(self):
        with (
            patch("backend.rag_warmup._load_embedding_model", return_value=object()),
            patch("backend.rag_warmup._load_existing_rag_index", return_value=True),
        ):
            rag_warmup.run_rag_warmup(load_index=True)

        status = rag_warmup.get_rag_warmup_status()
        self.assertEqual(status["status"], "ready")
        self.assertTrue(status["model_loaded"])
        self.assertTrue(status["index_loaded"])
        self.assertIsNotNone(status["elapsed_seconds"])
        self.assertIsNone(status["error"])

    def test_model_load_failure_sets_error_status(self):
        with patch(
            "backend.rag_warmup._load_embedding_model",
            side_effect=RuntimeError("failed with api_key=secret-value"),
        ):
            rag_warmup.run_rag_warmup()

        status = rag_warmup.get_rag_warmup_status()
        self.assertEqual(status["status"], "error")
        self.assertFalse(status["model_loaded"])
        self.assertFalse(status["index_loaded"])
        self.assertIsNotNone(status["elapsed_seconds"])
        self.assertIn("failed", status["error"])
        self.assertNotIn("secret-value", status["error"])

    def test_missing_index_does_not_rebuild(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing_index = Path(tmpdir) / "index.faiss"
            missing_chunks = Path(tmpdir) / "chunks.json"
            with (
                patch("backend.rag_warmup._load_embedding_model", return_value=object()),
                patch.object(rag_store, "INDEX_FILE", missing_index),
                patch.object(rag_store, "CHUNKS_FILE", missing_chunks),
                patch.object(rag_store, "rebuild_rag_index") as mock_rebuild,
            ):
                rag_warmup.run_rag_warmup(load_index=True)

        status = rag_warmup.get_rag_warmup_status()
        self.assertEqual(status["status"], "ready")
        self.assertTrue(status["model_loaded"])
        self.assertFalse(status["index_loaded"])
        mock_rebuild.assert_not_called()


class RagWarmupServerTests(unittest.TestCase):
    def setUp(self):
        rag_warmup._reset_rag_warmup_status_for_tests()

    def tearDown(self):
        rag_warmup._reset_rag_warmup_status_for_tests()

    def test_auto_warmup_disabled_startup_does_not_start_warmup(self):
        from backend import server

        with (
            patch.dict(os.environ, {"ENABLE_RAG_WARMUP": "false"}, clear=False),
            patch("backend.server.get_database_url", return_value=None),
            patch("backend.rag_warmup.start_rag_warmup") as mock_start,
        ):
            server._startup_init_db()

        mock_start.assert_not_called()

    def test_rag_status_endpoint_does_not_load_model(self):
        from backend import server

        with patch("backend.rag_warmup._load_embedding_model") as mock_load:
            status = server.rag_status_api()

        self.assertEqual(status["status"], "idle")
        mock_load.assert_not_called()

    def test_rag_warmup_endpoint_starts_thread_without_loading_model_inline(self):
        from backend import server

        FakeThread.created = []
        with (
            patch("backend.rag_warmup.threading.Thread", FakeThread),
            patch("backend.rag_warmup._load_embedding_model") as mock_load,
        ):
            result = server.rag_warmup_api()

        self.assertTrue(result["started"])
        self.assertEqual(result["warmup"]["status"], "loading")
        self.assertEqual(len(FakeThread.created), 1)
        self.assertTrue(FakeThread.created[0].started)
        mock_load.assert_not_called()

    def test_health_reads_warmup_status_without_loading_model(self):
        from backend import server

        with patch("backend.rag_warmup._load_embedding_model") as mock_load:
            health = server.health_check()

        self.assertIn("rag_warmup", health)
        self.assertEqual(health["rag_warmup"]["status"], "idle")
        mock_load.assert_not_called()


class RagStoreConcurrencyTests(unittest.TestCase):
    def setUp(self):
        rag_store.embedding_model = None

    def tearDown(self):
        rag_store.embedding_model = None

    def test_get_embedding_model_concurrent_calls_initialize_once(self):
        class FakeSentenceTransformer:
            init_count = 0

            def __init__(self, *args, **kwargs):
                type(self).init_count += 1
                time.sleep(0.05)

        results = []
        errors = []

        def worker():
            try:
                results.append(rag_store.get_embedding_model())
            except Exception as exc:
                errors.append(exc)

        with (
            patch.object(rag_store, "get_embedding_model_settings", return_value=("test-model", False)),
            patch.object(rag_store, "_get_sentence_transformer", return_value=FakeSentenceTransformer),
        ):
            threads = [threading.Thread(target=worker) for _ in range(6)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

        self.assertEqual(errors, [])
        self.assertEqual(FakeSentenceTransformer.init_count, 1)
        self.assertEqual(len({id(result) for result in results}), 1)


if __name__ == "__main__":
    unittest.main()
