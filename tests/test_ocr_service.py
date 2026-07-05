import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

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
            patch(
                "backend.ocr_service._extract_pdf_pages_with_pymupdf",
                return_value=None,
            ),
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
            patch(
                "backend.ocr_service._extract_pdf_pages_with_pymupdf",
                return_value=None,
            ),
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
        temporary_directory = tempfile.TemporaryDirectory()
        temporary_path = Path(temporary_directory.name)
        image_path = temporary_path / "page-0001.png"
        image_path.write_bytes(b"png")
        rendered_pages = ocr_service.RenderedPDFPages(
            [image_path],
            temporary_directory,
            page_count=1,
            warnings=[],
        )
        with (
            patch(
                "backend.ocr_service.get_config",
                return_value=_config(enable_ocr=True, ocr_engine="fake"),
            ),
            patch("backend.ocr_service.detect_pdf_text_quality", return_value=quality),
            patch(
                "backend.ocr_service.render_pdf_pages_to_images",
                return_value=rendered_pages,
            ),
            patch("backend.ocr_service._get_ocr_adapter", return_value=FakeAdapter()),
        ):
            result = ocr_service.extract_text_with_ocr(Path("scan.pdf"))

        self.assertTrue(result["ocr_used"])
        self.assertEqual(result["method"], "ocr")
        self.assertIn("recognized scan text", result["text"])
        self.assertFalse(temporary_path.exists())

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

    def test_missing_rapidocr_dependency_returns_failed_result(self):
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
                return_value=_config(enable_ocr=True, ocr_engine="rapidocr"),
            ),
            patch("backend.ocr_service.detect_pdf_text_quality", return_value=quality),
            patch(
                "backend.ocr_service._load_rapidocr_engine",
                side_effect=ocr_service.OCRUnavailableError(
                    ocr_service.RAPIDOCR_NOT_INSTALLED_ERROR
                ),
            ),
            patch("backend.ocr_service.render_pdf_pages_to_images") as render,
        ):
            result = ocr_service.extract_text_with_ocr(Path("scan.pdf"))

        self.assertFalse(result["ocr_used"])
        self.assertEqual(result["method"], "failed")
        self.assertEqual(
            result["ocr_error"], ocr_service.RAPIDOCR_NOT_INSTALLED_ERROR
        )
        render.assert_not_called()

    def test_rapidocr_result_is_joined_in_reading_order(self):
        engine = Mock(
            return_value=(
                [
                    ([[0, 20], [10, 20], [10, 30], [0, 30]], "second line", 0.9),
                    ([[0, 0], [10, 0], [10, 10], [0, 10]], "first line", 0.9),
                ],
                0.01,
            )
        )
        with patch("backend.ocr_service._load_rapidocr_engine", return_value=engine):
            text = ocr_service.run_rapidocr_on_image(Path("page.png"))

        self.assertEqual(text, "first line\nsecond line")
        engine.assert_called_once_with("page.png")

    def test_rapidocr_output_object_structure_is_supported(self):
        output = SimpleNamespace(
            txts=["lower", "upper"],
            boxes=[
                [[0, 30], [10, 30], [10, 40], [0, 40]],
                [[0, 10], [10, 10], [10, 20], [0, 20]],
            ],
        )

        self.assertEqual(
            ocr_service._rapidocr_text_lines(output),
            ["upper", "lower"],
        )

    def test_pdf_render_uses_temporary_directory_and_page_limit(self):
        class FakePixmap:
            def save(self, path):
                Path(path).write_bytes(b"png")

        class FakePage:
            def get_pixmap(self, **_kwargs):
                return FakePixmap()

        class FakeDocument:
            page_count = 3

            def load_page(self, _index):
                return FakePage()

            def close(self):
                return None

        fake_fitz = SimpleNamespace(
            open=Mock(return_value=FakeDocument()),
            Matrix=lambda x, y: (x, y),
        )
        with patch.dict(sys.modules, {"fitz": fake_fitz}):
            rendered = ocr_service.render_pdf_pages_to_images(
                Path("scan.pdf"), dpi=200, max_pages=2
            )

        temporary_path = rendered[0].parent
        self.assertEqual(len(rendered), 2)
        self.assertTrue(all(path.exists() for path in rendered))
        self.assertIn("OCR_MAX_PAGES", rendered.warnings[0])
        rendered.cleanup()
        self.assertFalse(temporary_path.exists())

    def test_ocr_empty_result_is_reported_as_used_with_warning(self):
        class EmptyAdapter:
            def extract_text(self, _image_paths):
                return [""]

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
                return_value=_config(enable_ocr=True, ocr_engine="rapidocr"),
            ),
            patch("backend.ocr_service.detect_pdf_text_quality", return_value=quality),
            patch(
                "backend.ocr_service.render_pdf_pages_to_images",
                return_value=[Path("page-0001.png")],
            ),
            patch("backend.ocr_service._get_ocr_adapter", return_value=EmptyAdapter()),
        ):
            result = ocr_service.extract_text_with_ocr(Path("scan.pdf"))

        self.assertTrue(result["ocr_used"])
        self.assertEqual(result["method"], "ocr")
        self.assertIsNone(result["ocr_error"])
        self.assertIn(ocr_service.OCR_PRODUCED_NO_TEXT_WARNING, result["warnings"])

    def test_pymupdf_failure_marks_pdf_as_corrupted(self):
        native_result = {
            "page_texts": [],
            "warnings": [
                ocr_service.CORRUPTED_PDF_WARNING,
                "PyMuPDF parsing failed: broken xref",
            ],
            "corrupted_pdf": True,
            "pdf_parser": "pymupdf",
            "parse_error": "PyMuPDF parsing failed: broken xref",
        }
        with (
            patch("backend.ocr_service.get_config", return_value=_config()),
            patch(
                "backend.ocr_service._extract_pdf_pages_with_pymupdf",
                return_value=native_result,
            ),
        ):
            result = ocr_service.detect_pdf_text_quality(Path("broken.pdf"))

        self.assertTrue(result["corrupted_pdf"])
        self.assertTrue(result["safe_fallback"])
        self.assertTrue(result["need_ocr"])
        self.assertEqual(result["pdf_parser"], "pymupdf")

    def test_corrupted_pdf_can_recover_through_ocr_bypass(self):
        class FakeAdapter:
            def extract_text(self, _image_paths):
                return ["recovered text from damaged PDF " * 4]

        quality = {
            "text": "",
            "method": "failed",
            "need_ocr": True,
            "page_count": 0,
            "text_char_count": 0,
            "warnings": [ocr_service.CORRUPTED_PDF_WARNING],
            "page_texts": [],
            "low_text_pages": [],
            "corrupted_pdf": True,
            "safe_fallback": True,
            "pdf_parser": "pymupdf",
        }
        class FakeRenderedPages(list):
            warnings = []
            page_count = 1

            def __init__(self):
                super().__init__([Path("page-0001.png")])
                self.cleanup = Mock()

        rendered = FakeRenderedPages()
        with (
            patch(
                "backend.ocr_service.get_config",
                return_value=_config(enable_ocr=True, ocr_engine="rapidocr"),
            ),
            patch("backend.ocr_service.detect_pdf_text_quality", return_value=quality),
            patch(
                "backend.ocr_service.render_pdf_pages_to_images",
                return_value=rendered,
            ),
            patch("backend.ocr_service._get_ocr_adapter", return_value=FakeAdapter()),
        ):
            result = ocr_service.extract_text_with_ocr(Path("broken.pdf"))

        self.assertEqual(result["method"], "ocr")
        self.assertTrue(result["ocr_used"])
        self.assertTrue(result["corrupted_pdf"])
        self.assertFalse(result["safe_fallback"])
        self.assertIn("recovered text", result["text"])
        rendered.cleanup.assert_called_once()

    def test_extreme_pipeline_exception_returns_safe_result(self):
        with patch(
            "backend.ocr_service.detect_pdf_text_quality",
            side_effect=MemoryError("parser exhausted memory"),
        ):
            result = ocr_service.extract_text_from_document(Path("extreme.pdf"))

        self.assertEqual(result["method"], "failed")
        self.assertTrue(result["corrupted_pdf"])
        self.assertTrue(result["safe_fallback"])
        self.assertIn("parser exhausted memory", " ".join(result["warnings"]))


if __name__ == "__main__":
    unittest.main()
