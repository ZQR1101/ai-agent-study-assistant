"""Document text extraction with an optional, pluggable PDF OCR fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Protocol

from backend.config import get_config


SCANNED_PDF_WARNING = (
    "This PDF appears to be scanned or image-based. Enable OCR to extract text."
)
OCR_NOT_CONFIGURED_ERROR = "OCR is disabled or no OCR engine is configured"
SUPPORTED_OCR_ENGINES = {"none", "rapidocr", "paddleocr", "tesseract"}


class OCRAdapter(Protocol):
    def extract_text(self, images: list[object]) -> list[str]: ...


OCRAdapterFactory = Callable[[], OCRAdapter]
_OCR_ADAPTER_FACTORIES: dict[str, OCRAdapterFactory] = {}


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
    }


def detect_pdf_text_quality(pdf_path: Path) -> dict:
    """Extract native PDF text and report whether any page looks image-based."""

    path = Path(pdf_path)
    config = get_config()
    min_chars = getattr(config, "ocr_min_text_chars", 80)
    warnings: list[str] = []

    try:
        from pypdf import PdfReader

        reader = PdfReader(path)
        page_texts: list[str] = []
        page_char_counts: list[int] = []
        for page_number, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:  # A damaged page should not abort an index rebuild.
                page_text = ""
                warnings.append(f"Could not extract text from PDF page {page_number}: {exc}")
            page_texts.append(page_text)
            page_char_counts.append(_text_char_count(page_text))
    except Exception as exc:
        return {
            **_result(
                need_ocr=True,
                ocr_error=f"PDF text extraction failed: {exc}",
                warnings=[f"PDF text extraction failed: {exc}"],
            ),
            "page_texts": [],
            "page_char_counts": [],
            "low_text_pages": [],
        }

    text = "\n".join(page_texts).strip()
    total_chars = _text_char_count(text)
    low_text_pages = [
        index
        for index, char_count in enumerate(page_char_counts)
        if char_count < min_chars
    ]
    need_ocr = total_chars < min_chars or bool(low_text_pages)
    if need_ocr:
        warnings.append(SCANNED_PDF_WARNING)

    return {
        **_result(
            text=text,
            method="text" if total_chars else "failed",
            need_ocr=need_ocr,
            page_count=len(page_texts),
            warnings=warnings,
        ),
        "page_texts": page_texts,
        "page_char_counts": page_char_counts,
        "low_text_pages": low_text_pages,
    }


def _render_pdf_pages(pdf_path: Path, page_indices: list[int], dpi: int) -> list[object]:
    try:
        import pypdfium2 as pdfium  # type: ignore[import-not-found]

        document = pdfium.PdfDocument(str(pdf_path))
        scale = dpi / 72
        try:
            return [document[index].render(scale=scale).to_pil() for index in page_indices]
        finally:
            document.close()
    except ImportError:
        pass

    try:
        import fitz  # type: ignore[import-not-found]
        from PIL import Image  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PDF rendering is unavailable; install pypdfium2 or PyMuPDF with Pillow"
        ) from exc

    document = fitz.open(str(pdf_path))
    scale = dpi / 72
    try:
        images = []
        for index in page_indices:
            pixmap = document[index].get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
            images.append(Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples))
        return images
    finally:
        document.close()


class _TesseractAdapter:
    def __init__(self) -> None:
        try:
            import pytesseract  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("install pytesseract to use OCR_ENGINE=tesseract") from exc
        self._pytesseract = pytesseract

    def extract_text(self, images: list[object]) -> list[str]:
        return [self._pytesseract.image_to_string(image) or "" for image in images]


class _RapidOCRAdapter:
    def __init__(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "install rapidocr-onnxruntime to use OCR_ENGINE=rapidocr"
            ) from exc
        self._engine = RapidOCR()

    def extract_text(self, images: list[object]) -> list[str]:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("RapidOCR requires numpy") from exc

        texts = []
        for image in images:
            result, _elapsed = self._engine(np.asarray(image))
            texts.append("\n".join(str(item[1]) for item in (result or []) if len(item) > 1))
        return texts


class _PaddleOCRAdapter:
    def __init__(self) -> None:
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("install paddleocr to use OCR_ENGINE=paddleocr") from exc
        self._engine = PaddleOCR(use_angle_cls=True, lang="ch")

    def extract_text(self, images: list[object]) -> list[str]:
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("PaddleOCR requires numpy") from exc

        texts = []
        for image in images:
            pages = self._engine.ocr(np.asarray(image), cls=True) or []
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
        raise RuntimeError(f"Unknown OCR engine '{engine}'. Supported engines: {supported}")
    return builtin_factories[engine]()


def _extract_text_with_ocr(pdf_path: Path, quality: dict) -> dict:
    config = get_config()
    engine = str(getattr(config, "ocr_engine", "none") or "none").strip().lower()
    base_warnings = list(quality.get("warnings") or [])

    if not bool(getattr(config, "enable_ocr", False)) or engine == "none":
        return {
            **_result(
                text=quality.get("text", ""),
                method=quality.get("method", "failed"),
                need_ocr=True,
                page_count=quality.get("page_count", 0),
                ocr_error=OCR_NOT_CONFIGURED_ERROR,
                warnings=base_warnings,
            )
        }

    page_count = int(quality.get("page_count", 0))
    max_pages = getattr(config, "ocr_max_pages", 20)
    low_text_pages = list(quality.get("low_text_pages") or range(page_count))
    page_indices = [index for index in low_text_pages if index < max_pages]
    if page_count > max_pages:
        base_warnings.append(
            f"OCR was limited to the first {max_pages} pages by OCR_MAX_PAGES."
        )
    if not page_indices:
        error = "No PDF pages are eligible for OCR within OCR_MAX_PAGES"
        return _result(
            text=quality.get("text", ""),
            method=quality.get("method", "failed"),
            need_ocr=True,
            page_count=page_count,
            ocr_error=error,
            warnings=[*base_warnings, error],
        )

    try:
        adapter = _get_ocr_adapter(engine)
        images = _render_pdf_pages(
            Path(pdf_path),
            page_indices,
            getattr(config, "ocr_render_dpi", 200),
        )
        extracted_pages = adapter.extract_text(images)
        if len(extracted_pages) != len(page_indices):
            raise RuntimeError("OCR adapter returned an unexpected number of pages")
    except Exception as exc:
        error = f"OCR engine '{engine}' is unavailable or failed: {exc}"
        return _result(
            text=quality.get("text", ""),
            method=quality.get("method", "failed"),
            need_ocr=True,
            page_count=page_count,
            ocr_error=error,
            warnings=[*base_warnings, error],
        )

    page_texts = list(quality.get("page_texts") or [""] * page_count)
    for index, ocr_text in zip(page_indices, extracted_pages):
        if str(ocr_text or "").strip():
            page_texts[index] = str(ocr_text).strip()

    combined_text = "\n".join(page_texts).strip()
    ocr_used = any(_text_char_count(text) for text in extracted_pages)
    if not ocr_used:
        error = f"OCR engine '{engine}' returned no text"
        return _result(
            text=combined_text,
            method="text" if _text_char_count(combined_text) else "failed",
            need_ocr=True,
            page_count=page_count,
            ocr_error=error,
            warnings=[*base_warnings, error],
        )

    method = "mixed" if quality.get("text_char_count", 0) else "ocr"
    success_warnings = [
        warning for warning in base_warnings if warning != SCANNED_PDF_WARNING
    ]
    return _result(
        text=combined_text,
        method=method,
        need_ocr=True,
        ocr_used=True,
        page_count=page_count,
        warnings=success_warnings,
    )


def extract_text_with_ocr(pdf_path: Path) -> dict:
    quality = detect_pdf_text_quality(Path(pdf_path))
    return _extract_text_with_ocr(Path(pdf_path), quality)


def extract_text_from_document(file_path: Path) -> dict:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            return _result(ocr_error=str(exc), warnings=[f"Document text extraction failed: {exc}"])
        return _result(text=text, method="text", page_count=1)

    if suffix != ".pdf":
        return _result(
            ocr_error=f"Unsupported document type: {suffix or '<none>'}",
            warnings=[f"Unsupported document type: {suffix or '<none>'}"],
        )

    quality = detect_pdf_text_quality(path)
    if not quality["need_ocr"]:
        return _result(
            text=quality["text"],
            method="text",
            page_count=quality["page_count"],
            warnings=quality["warnings"],
        )
    return _extract_text_with_ocr(path, quality)
