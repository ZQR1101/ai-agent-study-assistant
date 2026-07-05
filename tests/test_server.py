import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.server import app


def _parse_result(**overrides):
    result = {
        "text": "",
        "method": "failed",
        "need_ocr": True,
        "ocr_used": False,
        "page_count": 1,
        "text_char_count": 0,
        "ocr_error": "OCR is disabled or no OCR engine is configured",
        "warnings": [
            "This PDF appears to be scanned or image-based. Enable OCR to extract text."
        ],
    }
    result.update(overrides)
    return result


class OCRServerTests(unittest.TestCase):
    def setUp(self):
        from backend import server

        self.server = server
        self.server._UPLOAD_RATE_LIMITER.reset()
        self.client = TestClient(app)
        self.tempdir = tempfile.TemporaryDirectory()
        self.docs_path = Path(self.tempdir.name)
        self.docs_patch = patch("backend.server.DOCS_PATH", self.docs_path)
        self.config_patch = patch(
            "backend.server.get_config",
            return_value=SimpleNamespace(
                max_upload_size_bytes=4096,
                max_upload_total_bytes=40960,
                upload_max_concurrency=2,
                max_pdf_pages=10,
                pdf_validation_timeout_seconds=5,
                pdf_validation_max_memory_bytes=256 * 1024 * 1024,
            ),
        )
        self.docs_patch.start()
        self.config_patch.start()

    def tearDown(self):
        self.config_patch.stop()
        self.docs_patch.stop()
        self.tempdir.cleanup()

    def test_upload_returns_ocr_parse_fields(self):
        scan_result = _parse_result()
        with (
            patch("backend.server.validate_pdf_file", return_value=1),
            patch("backend.server.extract_text_from_document", return_value=scan_result),
        ):
            response = self.client.post(
                "/upload",
                files={"file": ("scan.pdf", b"%PDF-valid-test", "application/pdf")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["file"], "scan.pdf")
        self.assertEqual(payload["parse_method"], "failed")
        self.assertTrue(payload["need_ocr"])
        self.assertFalse(payload["ocr_used"])
        self.assertEqual(payload["warnings"], scan_result["warnings"])
        self.assertIn("OCR", payload["message"])

    def test_parse_status_returns_document_state(self):
        path = self.docs_path / "scan.pdf"
        path.write_bytes(b"%PDF-test")
        scan_result = _parse_result()
        with patch(
            "backend.server.extract_text_from_document", return_value=scan_result
        ):
            response = self.client.get("/knowledge-files/scan.pdf/parse-status")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["name"], "scan.pdf")
        self.assertEqual(payload["type"], "pdf")
        self.assertTrue(payload["need_ocr"])
        self.assertEqual(payload["text_char_count"], 0)


if __name__ == "__main__":
    unittest.main()
