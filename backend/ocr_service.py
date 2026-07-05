"""Document text extraction with an optional, pluggable PDF OCR fallback."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable, Protocol

from backend.config import get_config


SCANNED_PDF_WARNING = (
    "This PDF appears to be scanned or image-based. Enable OCR to extract text."
)
OCR_NOT_CONFIGURED_ERROR = "OCR is disabled or no OCR engine is configured"
OCR_PRODUCED_NO_TEXT_WARNING = "OCR produced no text"
CORRUPTED_PDF_WARNING = "This PDF appears to be corrupted or structurally invalid."
RAPIDOCR_NOT_INSTALLED_ERROR = (
    "RapidOCR is not installed. Install it with: pip install rapidocr-onnxruntime"
)
SUPPORTED_OCR_ENGINES = {"none", "rapidocr", "paddleocr", "tesseract"}


class OCRServiceError(RuntimeError):
    """Expected OCR failure that should be exposed as a safe parse result."""


class OCRUnavailableError(OCRServiceError):
    """The selected optional OCR engine or renderer is not installed."""


class OCRAdapter(Protocol):
    def extract_text(self, image_paths: list[Path]) -> list[str]: ...


OCRAdapterFactory = Callable[[], OCRAdapter]
_OCR_ADAPTER_FACTORIES: dict[str, OCRAdapterFactory] = {}


class RenderedPDFPages(list[Path]):
    """Temporary rendered page paths that remain valid until ``cleanup``."""

    def __init__(
        self,
        paths: list[Path],
        temporary_directory: tempfile.TemporaryDirectory,
        *,
        page_count: int,
        warnings: list[str],
    ) -> None:
        super().__init__(paths)
        self.page_count = page_count
        self.warnings = warnings
        self._temporary_directory = temporary_directory

    def cleanup(self) -> None:
        temporary_directory = self._temporary_directory
        if temporary_directory is not None:
            self._temporary_directory = None
            temporary_directory.cleanup()

    def __enter__(self) -> "RenderedPDFPages":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.cleanup()


def register_ocr_adapter(name: str, factory: OCRAdapterFactory) -> None:
    """Register an OCR adapter factory, primarily for optional plugins and tests."""

    normalized_name = name.strip().lower()
    if not normalized_name or normalized_name == "none":
        raise ValueError("OCR adapter name must not be empty or 'none'")
    _OCR_ADAPTER_FACTORIES[normalized_name] = factory


def is_ocr_enabled() -> bool:
    return bool(getattr(get_config(), "enable_ocr", False))


def _text_char_count(text: str) -> int:
    return sum(1 for character in str(text or "") if not character.isspace())


def _result(
    *,
    text: str = "",
    method: str = "failed",
    need_ocr: bool = False,
    ocr_used: bool = False,
    page_count: int = 0,
    ocr_error: str | None = None,
    warnings: list[str] | None = None,
    corrupted_pdf: bool = False,
    safe_fallback: bool = False,
    pdf_parser: str | None = None,
) -> dict:
    return {
        "text": text,
        "method": method,
        "need_ocr": need_ocr,
        "ocr_used": ocr_used,
        "page_count": page_count,
        "text_char_count": _text_char_count(text),
        "ocr_error": ocr_error,
        "warnings": list(warnings or []),
        "corrupted_pdf": corrupted_pdf,
        "safe_fallback": safe_fallback,
        "pdf_parser": pdf_parser,
    }


def _extract_pdf_pages_with_pymupdf(pdf_path: Path) -> dict | None:
    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError:
        return None

    document = None
    warnings: list[str] = []
    corrupted_pdf = False
    try:
        document = fitz.open(str(pdf_path))
        page_texts = []
        for page_number in range(int(document.page_count)):
            try:
                page_texts.append(document.load_page(page_number).get_text("text") or "")
            except Exception as exc:
                corrupted_pdf = True
                page_texts.append("")
                warnings.append(
                    f"PyMuPDF could not parse PDF page {page_number + 1}: {exc}"
                )
        return {
            "page_texts": page_texts,
            "warnings": warnings,
            "corrupted_pdf": corrupted_pdf,
            "pdf_parser": "pymupdf",
            "parse_error": warnings[0] if warnings else None,
        }
    except Exception as exc:
        error = f"PyMuPDF parsing failed: {exc}"
        return {
            "page_texts": [],
            "warnings": [CORRUPTED_PDF_WARNING, error],
            "corrupted_pdf": True,
            "pdf_parser": "pymupdf",
            "parse_error": error,
        }
    finally:
        if document is not None:
            try:
                document.close()
            except Exception:
                pass


def _extract_pdf_pages_with_pypdf(pdf_path: Path) -> dict:
    warnings: list[str] = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        page_texts = []
        corrupted_pdf = False
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_texts.append(page.extract_text() or "")
            except Exception as exc:
                corrupted_pdf = True
                page_texts.append("")
                warnings.append(f"Could not extract text from PDF page {page_number}: {exc}")
        return {
            "page_texts": page_texts,
            "warnings": warnings,
            "corrupted_pdf": corrupted_pdf,
            "pdf_parser": "pypdf",
            "parse_error": warnings[0] if warnings else None,
        }
    except Exception as exc:
        error = f"PDF fallback parsing failed: {exc}"
        return {
            "page_texts": [],
            "warnings": [CORRUPTED_PDF_WARNING, error],
            "corrupted_pdf": True,
            "pdf_parser": "pypdf",
            "parse_error": error,
        }


def detect_pdf_text_quality(pdf_path: Path) -> dict:
    """Run native PDF parsing and classify scanned or corrupted documents."""

    path = Path(pdf_path)
    min_chars = getattr(get_config(), "ocr_min_text_chars", 80)
    native_result = _extract_pdf_pages_with_pymupdf(path)
    if native_result is None:
        native_result = _extract_pdf_pages_with_pypdf(path)

    page_texts = native_result["page_texts"]
    warnings = list(native_result["warnings"])
    corrupted_pdf = bool(native_result["corrupted_pdf"])
    parse_error = native_result["parse_error"]
    if corrupted_pdf and CORRUPTED_PDF_WARNING not in warnings:
        warnings.insert(0, CORRUPTED_PDF_WARNING)
    page_char_counts = [_text_char_count(page_text) for page_text in page_texts]

    text = "\n".join(page_texts).strip()
    total_chars = _text_char_count(text)
    low_text_pages = [
        index
        for index, char_count in enumerate(page_char_counts)
        if char_count < min_chars
    ]
    need_ocr = corrupted_pdf or total_chars < min_chars or bool(low_text_pages)
    if need_ocr and not corrupted_pdf:
        warnings.append(SCANNED_PDF_WARNING)

    return {
        **_result(
            text=text,
            method="text" if total_chars else "failed",
            need_ocr=need_ocr,
            page_count=len(page_texts),
            ocr_error=parse_error,
            warnings=warnings,
            corrupted_pdf=corrupted_pdf,
            safe_fallback=corrupted_pdf and not bool(total_chars),
            pdf_parser=native_result["pdf_parser"],
        ),
        "page_texts": page_texts,
        "page_char_counts": page_char_counts,
        "low_text_pages": low_text_pages,
    }


def render_pdf_pages_to_images(
    pdf_path: Path,
    dpi: int,
    max_pages: int,
) -> RenderedPDFPages:
    """Render the first PDF pages to temporary PNG files using optional PyMuPDF."""

    try:
        import fitz  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OCRUnavailableError(
            "PyMuPDF is not installed. Install it with: pip install PyMuPDF"
        ) from exc

    temporary_directory = tempfile.TemporaryDirectory(prefix="ai-study-ocr-")
    output_directory = Path(temporary_directory.name)
    document = None
    try:
        document = fitz.open(str(pdf_path))
        page_count = int(document.page_count)
        page_limit = max(1, int(max_pages))
        render_count = min(page_count, page_limit)
        scale = max(1, int(dpi)) / 72
        matrix = fitz.Matrix(scale, scale)
        image_paths: list[Path] = []

        for page_index in range(render_count):
            page = document.load_page(page_index)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = output_directory / f"page-{page_index + 1:04d}.png"
            pixmap.save(str(image_path))
            image_paths.append(image_path)

        warnings = []
        if page_count > page_limit:
            warnings.append(
                f"OCR was limited to the first {page_limit} pages by OCR_MAX_PAGES."
            )
        return RenderedPDFPages(
            image_paths,
            temporary_directory,
            page_count=page_count,
            warnings=warnings,
        )
    except Exception:
        temporary_directory.cleanup()
        raise
    finally:
        if document is not None:
            document.close()


def _box_reading_order(box: object, original_index: int) -> tuple[float, float, int]:
    try:
        points = list(box)  # type: ignore[arg-type]
        xs = [float(point[0]) for point in points]
        ys = [float(point[1]) for point in points]
        return min(ys), min(xs), original_index
    except (TypeError, ValueError, IndexError):
        return float(original_index), 0.0, original_index


def _as_list(value: object) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError:
        return [value]


def _rapidocr_text_lines(raw_result: object) -> list[str]:
    """Normalize common rapidocr-onnxruntime and RapidOCR output structures."""

    result = raw_result
    if isinstance(result, tuple) and len(result) >= 1:
        result = result[0]

    if hasattr(result, "txts"):
        texts = _as_list(getattr(result, "txts"))
        boxes = _as_list(getattr(result, "boxes", None))
        entries = [
            (boxes[index] if index < len(boxes) else None, str(text), index)
            for index, text in enumerate(texts)
        ]
    elif isinstance(result, dict):
        text_value = result.get("txts")
        if text_value is None:
            text_value = result.get("texts")
        if text_value is None:
            text_value = result.get("text")
        texts = _as_list(text_value)
        box_value = result.get("boxes")
        if box_value is None:
            box_value = result.get("dt_boxes")
        boxes = _as_list(box_value)
        entries = [
            (boxes[index] if index < len(boxes) else None, str(text), index)
            for index, text in enumerate(texts)
        ]
    else:
        entries = []
        for index, item in enumerate(_as_list(result)):
            box = None
            text = ""
            if isinstance(item, str):
                text = item
            elif isinstance(item, dict):
                box = item.get("box")
                if box is None:
                    box = item.get("points")
                text = str(item.get("text") or item.get("txt") or "")
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                box = item[0]
                text_value = item[1]
                if isinstance(text_value, (list, tuple)):
                    text_value = text_value[0] if text_value else ""
                text = str(text_value or "")
            entries.append((box, text, index))

    entries.sort(key=lambda entry: _box_reading_order(entry[0], entry[2]))
    return [text.strip() for _box, text, _index in entries if text.strip()]


def _load_rapidocr_engine():
    try:
        from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OCRUnavailableError(RAPIDOCR_NOT_INSTALLED_ERROR) from exc
    return RapidOCR()


def _run_rapidocr_on_image(image_path: Path, engine: object) -> str:
    try:
        raw_result = engine(str(image_path))  # type: ignore[operator]
        return "\n".join(_rapidocr_text_lines(raw_result))
    except OCRServiceError:
        raise
    except Exception as exc:
        raise OCRServiceError(f"RapidOCR failed for {image_path.name}: {exc}") from exc


def run_rapidocr_on_image(image_path: Path) -> str:
    """Run optional RapidOCR on one image and return reading-order plain text."""

    return _run_rapidocr_on_image(Path(image_path), _load_rapidocr_engine())


class _RapidOCRAdapter:
    def __init__(self) -> None:
        self._engine = _load_rapidocr_engine()

    def extract_text(self, image_paths: list[Path]) -> list[str]:
        return [_run_rapidocr_on_image(path, self._engine) for path in image_paths]


class _TesseractAdapter:
    def __init__(self) -> None:
        try:
            import pytesseract  # type: ignore[import-not-found]
            from PIL import Image  # type: ignore[import-not-found]
        except ImportError as exc:
            raise OCRUnavailableError(
                "Install pytesseract and Pillow to use OCR_ENGINE=tesseract"
            ) from exc
        self._pytesseract = pytesseract
        self._image_class = Image

    def extract_text(self, image_paths: list[Path]) -> list[str]:
        texts = []
        for image_path in image_paths:
            with self._image_class.open(image_path) as image:
                texts.append(self._pytesseract.image_to_string(image) or "")
        return texts


class _PaddleOCRAdapter:
    def __init__(self) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-not-found]
        except ImportError as exc:
            raise OCRUnavailableError(
                "Install paddleocr to use OCR_ENGINE=paddleocr"
            ) from exc
        self._engine = PaddleOCR(use_angle_cls=True, lang="ch")

    def extract_text(self, image_paths: list[Path]) -> list[str]:
        texts = []
        for image_path in image_paths:
            pages = self._engine.ocr(str(image_path), cls=True) or []
            lines = []
            for page in pages:
                for item in page or []:
                    if len(item) > 1 and item[1]:
                        lines.append(str(item[1][0]))
            texts.append("\n".join(lines))
        return texts


def _get_ocr_adapter(engine: str) -> OCRAdapter:
    factory = _OCR_ADAPTER_FACTORIES.get(engine)
    if factory is not None:
        return factory()

    builtin_factories: dict[str, OCRAdapterFactory] = {
        "rapidocr": _RapidOCRAdapter,
        "paddleocr": _PaddleOCRAdapter,
        "tesseract": _TesseractAdapter,
    }
    if engine not in builtin_factories:
        supported = ", ".join(sorted(SUPPORTED_OCR_ENGINES))
        raise OCRUnavailableError(
            f"Unknown OCR engine '{engine}'. Supported engines: {supported}"
        )
    return builtin_factories[engine]()


def _without_scanned_warning(warnings: list[str]) -> list[str]:
    return [warning for warning in warnings if warning != SCANNED_PDF_WARNING]


def _extract_text_with_ocr(pdf_path: Path, quality: dict) -> dict:
    config = get_config()
    engine_name = str(getattr(config, "ocr_engine", "none") or "none").strip().lower()
    warnings = list(quality.get("warnings") or [])
    original_text = str(quality.get("text") or "")
    quality_page_count = int(quality.get("page_count", 0))
    corrupted_pdf = bool(quality.get("corrupted_pdf", False))
    pdf_parser = quality.get("pdf_parser")

    if not bool(getattr(config, "enable_ocr", False)) or engine_name == "none":
        return _result(
            text=original_text,
            method=quality.get("method", "failed"),
            need_ocr=True,
            page_count=quality_page_count,
            ocr_error=OCR_NOT_CONFIGURED_ERROR,
            warnings=warnings,
            corrupted_pdf=corrupted_pdf,
            safe_fallback=not bool(original_text.strip()),
            pdf_parser=pdf_parser,
        )

    rendered_pages = None
    try:
        adapter = _get_ocr_adapter(engine_name)
        rendered_pages = render_pdf_pages_to_images(
            Path(pdf_path),
            getattr(config, "ocr_render_dpi", 200),
            getattr(config, "ocr_max_pages", 20),
        )
        warnings.extend(getattr(rendered_pages, "warnings", []))
        image_paths = list(rendered_pages)
        extracted_pages = adapter.extract_text(image_paths)
        if len(extracted_pages) != len(image_paths):
            raise OCRServiceError("OCR adapter returned an unexpected number of pages")
        rendered_page_count = int(
            getattr(rendered_pages, "page_count", quality_page_count)
        )
    except OCRUnavailableError as exc:
        return _result(
            text=original_text,
            method="failed",
            need_ocr=True,
            page_count=quality_page_count,
            ocr_error=str(exc),
            warnings=[*warnings, str(exc)],
            corrupted_pdf=corrupted_pdf,
            safe_fallback=not bool(original_text.strip()),
            pdf_parser=pdf_parser,
        )
    except Exception as exc:
        error = str(exc) if isinstance(exc, OCRServiceError) else f"OCR failed: {exc}"
        return _result(
            text=original_text,
            method="failed",
            need_ocr=True,
            page_count=quality_page_count,
            ocr_error=error,
            warnings=[*warnings, error],
            corrupted_pdf=corrupted_pdf,
            safe_fallback=not bool(original_text.strip()),
            pdf_parser=pdf_parser,
        )
    finally:
        if rendered_pages is not None and hasattr(rendered_pages, "cleanup"):
            rendered_pages.cleanup()

    page_count = quality_page_count or rendered_page_count
    page_texts = list(quality.get("page_texts") or [])
    if len(page_texts) < page_count:
        page_texts.extend([""] * (page_count - len(page_texts)))
    low_text_pages = set(quality.get("low_text_pages") or range(page_count))

    for page_index, ocr_text in enumerate(extracted_pages):
        if page_index in low_text_pages and str(ocr_text or "").strip():
            page_texts[page_index] = str(ocr_text).strip()

    combined_text = "\n".join(page_texts).strip()
    result_warnings = _without_scanned_warning(warnings)
    if not any(_text_char_count(text) for text in extracted_pages):
        result_warnings.append(OCR_PRODUCED_NO_TEXT_WARNING)

    method = "mixed" if quality.get("text_char_count", 0) else "ocr"
    return _result(
        text=combined_text,
        method=method,
        need_ocr=True,
        ocr_used=True,
        page_count=page_count,
        warnings=result_warnings,
        corrupted_pdf=corrupted_pdf,
        safe_fallback=not bool(combined_text.strip()),
        pdf_parser=pdf_parser,
    )


def extract_text_with_ocr(pdf_path: Path) -> dict:
    try:
        quality = detect_pdf_text_quality(Path(pdf_path))
    except Exception as exc:
        return safe_document_parse_result(
            f"PDF parsing pipeline failed unexpectedly: {exc}",
        )
    return _extract_text_with_ocr(Path(pdf_path), quality)


def safe_document_parse_result(
    warning: str,
    *,
    corrupted_pdf: bool = True,
) -> dict:
    """Build a non-throwing result when parsing must be bypassed entirely."""

    return _result(
        method="failed",
        need_ocr=True,
        warnings=[warning],
        corrupted_pdf=corrupted_pdf,
        safe_fallback=True,
    )


def extract_text_from_document(file_path: Path) -> dict:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return _result(
                ocr_error=str(exc),
                warnings=[f"Document text extraction failed: {exc}"],
            )
        return _result(text=text, method="text", page_count=1)

    if suffix != ".pdf":
        return _result(
            ocr_error=f"Unsupported document type: {suffix or '<none>'}",
            warnings=[f"Unsupported document type: {suffix or '<none>'}"],
        )

    try:
        quality = detect_pdf_text_quality(path)
    except Exception as exc:
        return safe_document_parse_result(
            f"PDF parsing pipeline failed unexpectedly: {exc}",
        )
    if not quality["need_ocr"]:
        return _result(
            text=quality["text"],
            method="text",
            page_count=quality["page_count"],
            warnings=quality["warnings"],
            corrupted_pdf=quality.get("corrupted_pdf", False),
            safe_fallback=False,
            pdf_parser=quality.get("pdf_parser"),
        )
    return _extract_text_with_ocr(path, quality)
