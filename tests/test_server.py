import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.pdf_validation import PDFValidationError
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

    def test_parse_status_can_report_successful_rapidocr(self):
        path = self.docs_path / "scan.pdf"
        path.write_bytes(b"%PDF-test")
        result = _parse_result(
            text="recognized OCR content",
            method="ocr",
            ocr_used=True,
            text_char_count=20,
            ocr_error=None,
            warnings=[],
        )
        with patch("backend.server.extract_text_from_document", return_value=result):
            response = self.client.get("/knowledge-files/scan.pdf/parse-status")

        payload = response.json()
        self.assertEqual(payload["parse_method"], "ocr")
        self.assertTrue(payload["need_ocr"])
        self.assertTrue(payload["ocr_used"])
        self.assertEqual(payload["text_char_count"], 20)
        self.assertEqual(payload["warnings"], [])

    def test_upload_parser_exception_is_safe_fallback(self):
        with patch(
            "backend.server.extract_text_from_document",
            side_effect=MemoryError("parser allocation failed"),
        ):
            response = self.client.post(
                "/upload",
                files={"file": ("notes.txt", b"safe content", "text/plain")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["safe_fallback"])
        self.assertIn("parser allocation failed", " ".join(response.json()["warnings"]))
        self.assertTrue((self.docs_path / "notes.txt").exists())

    def test_validation_failure_can_be_repaired_by_fallback_parser(self):
        repaired_result = _parse_result(
            text="repaired PDF text",
            method="text",
            need_ocr=False,
            text_char_count=15,
            ocr_error=None,
            warnings=[],
            safe_fallback=False,
            corrupted_pdf=False,
            pdf_parser="pymupdf",
        )
        with (
            patch(
                "backend.server.validate_pdf_file",
                side_effect=PDFValidationError("Uploaded PDF structure is invalid"),
            ),
            patch(
                "backend.server.extract_text_from_document",
                return_value=repaired_result,
            ),
        ):
            response = self.client.post(
                "/upload",
                files={"file": ("repair.pdf", b"%PDF-repair", "application/pdf")},
            )

        payload = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["parse_method"], "text")
        self.assertTrue(payload["corrupted_pdf"])
        self.assertFalse(payload["safe_fallback"])
        self.assertEqual(payload["pdf_parser"], "pymupdf")

    def test_parse_status_catches_unexpected_parser_exception(self):
        path = self.docs_path / "broken.pdf"
        path.write_bytes(b"%PDF-broken")
        with patch(
            "backend.server.extract_text_from_document",
            side_effect=RuntimeError("unexpected parser crash"),
        ):
            response = self.client.get("/knowledge-files/broken.pdf/parse-status")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["corrupted_pdf"])
        self.assertTrue(response.json()["safe_fallback"])
        self.assertIn("unexpected parser crash", " ".join(response.json()["warnings"]))


if __name__ == "__main__":
    unittest.main()
