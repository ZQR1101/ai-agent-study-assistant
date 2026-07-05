import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from backend import ocr_service


def _config(**overrides):
    values = {
        "enable_ocr": False,
        "ocr_engine": "none",
        "ocr_min_text_chars": 80,
        "ocr_render_dpi": 200,
        "ocr_max_pages": 20,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _Page:
    def __init__(self, text):
        self.text = text

    def extract_text(self):
        return self.text


class OCRServiceTests(unittest.TestCase):
    def test_text_pdf_does_not_need_ocr(self):
        reader = SimpleNamespace(pages=[_Page("A" * 100)])
        with (
            patch("backend.ocr_service.get_config", return_value=_config()),
            patch("pypdf.PdfReader", return_value=reader),
        ):
            result = ocr_service.detect_pdf_text_quality(Path("text.pdf"))

        self.assertFalse(result["need_ocr"])
        self.assertEqual(result["method"], "text")
        self.assertEqual(result["text_char_count"], 100)

    def test_short_or_empty_pdf_needs_ocr(self):
        reader = SimpleNamespace(pages=[_Page("short")])
        with (
            patch("backend.ocr_service.get_config", return_value=_config()),
            patch("pypdf.PdfReader", return_value=reader),
        ):
            result = ocr_service.detect_pdf_text_quality(Path("scan.pdf"))

        self.assertTrue(result["need_ocr"])
        self.assertIn(ocr_service.SCANNED_PDF_WARNING, result["warnings"])

    def test_disabled_ocr_returns_clear_fallback_without_calling_adapter(self):
        quality = {
            "text": "",
            "method": "failed",
            "need_ocr": True,
            "ocr_used": False,
            "page_count": 1,
            "text_char_count": 0,
            "warnings": [ocr_service.SCANNED_PDF_WARNING],
            "page_texts": [""],
            "low_text_pages": [0],
        }
        with (
            patch("backend.ocr_service.get_config", return_value=_config()),
            patch("backend.ocr_service.detect_pdf_text_quality", return_value=quality),
            patch("backend.ocr_service._get_ocr_adapter") as adapter,
        ):
            result = ocr_service.extract_text_with_ocr(Path("scan.pdf"))

        self.assertFalse(result["ocr_used"])
        self.assertEqual(result["method"], "failed")
        self.assertEqual(result["ocr_error"], ocr_service.OCR_NOT_CONFIGURED_ERROR)
        adapter.assert_not_called()

    def test_none_engine_is_safe_even_when_ocr_flag_is_enabled(self):
        quality = {
            "text": "",
            "method": "failed",
            "page_count": 1,
            "warnings": [ocr_service.SCANNED_PDF_WARNING],
        }
        with patch(
            "backend.ocr_service.get_config",
            return_value=_config(enable_ocr=True, ocr_engine="none"),
        ):
            result = ocr_service._extract_text_with_ocr(Path("scan.pdf"), quality)

        self.assertEqual(result["ocr_error"], ocr_service.OCR_NOT_CONFIGURED_ERROR)
        self.assertFalse(result["ocr_used"])

    def test_registered_adapter_can_supply_ocr_text(self):
        class FakeAdapter:
            def extract_text(self, images):
                return ["recognized scan text " * 8 for _image in images]

        quality = {
            "text": "",
            "method": "failed",
            "need_ocr": True,
            "page_count": 1,
            "text_char_count": 0,
            "warnings": [ocr_service.SCANNED_PDF_WARNING],
            "page_texts": [""],
            "low_text_pages": [0],
        }
        with (
            patch(
                "backend.ocr_service.get_config",
                return_value=_config(enable_ocr=True, ocr_engine="fake"),
            ),
            patch("backend.ocr_service.detect_pdf_text_quality", return_value=quality),
            patch("backend.ocr_service._render_pdf_pages", return_value=[object()]),
            patch("backend.ocr_service._get_ocr_adapter", return_value=FakeAdapter()),
        ):
            result = ocr_service.extract_text_with_ocr(Path("scan.pdf"))

        self.assertTrue(result["ocr_used"])
        self.assertEqual(result["method"], "ocr")
        self.assertIn("recognized scan text", result["text"])

    def test_txt_and_md_keep_plain_text_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ("notes.txt", "guide.md"):
                path = Path(tmpdir) / name
                path.write_text("plain document content", encoding="utf-8")
                with patch("backend.ocr_service._get_ocr_adapter") as adapter:
                    result = ocr_service.extract_text_from_document(path)
                self.assertEqual(result["method"], "text")
                self.assertFalse(result["need_ocr"])
                self.assertFalse(result["ocr_used"])
                adapter.assert_not_called()

    def test_missing_optional_ocr_dependency_has_clear_error(self):
        with patch.dict(sys.modules, {"rapidocr_onnxruntime": None}):
            with self.assertRaisesRegex(RuntimeError, "rapidocr-onnxruntime"):
                ocr_service._RapidOCRAdapter()


if __name__ == "__main__":
    unittest.main()
