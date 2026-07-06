from __future__ import annotations

from io import BytesIO
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = PROJECT_ROOT / "docs"
FAISS_SOURCE = DOCS_PATH / "paper_faiss_2017.pdf"
RAG_SOURCE = DOCS_PATH / "paper_rag_2020.pdf"
SCANNED_OUTPUT = DOCS_PATH / "paper_faiss_2017_scanned_pages.pdf"
MIXED_OUTPUT = DOCS_PATH / "paper_rag_2020_mixed_pages.pdf"


def _load_banner_font(size: int):
    from PIL import ImageFont

    for candidate in (
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _image_page(output, source_page, *, scale: float = 2.0, banner: str | None = None) -> None:
    import fitz
    from PIL import Image, ImageDraw, ImageOps

    pixmap = source_page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    image = Image.open(BytesIO(pixmap.tobytes("png"))).convert("RGB")
    if banner:
        image = ImageOps.expand(image, border=(0, 150, 0, 0), fill="white")
        draw = ImageDraw.Draw(image)
        draw.text((60, 38), banner, fill="black", font=_load_banner_font(62))
    encoded = BytesIO()
    image.save(encoded, format="PNG")
    target = output.new_page(width=image.width / scale, height=image.height / scale)
    target.insert_image(target.rect, stream=encoded.getvalue())


def create_scanned_fixture() -> Path:
    import fitz

    source = fitz.open(FAISS_SOURCE)
    output = fitz.open()
    try:
        for page_number in range(min(2, source.page_count)):
            _image_page(
                output,
                source.load_page(page_number),
                banner="FAISSSCAN 2017 OCR ONLY" if page_number == 0 else None,
            )
        output.save(SCANNED_OUTPUT, garbage=4, deflate=True)
    finally:
        output.close()
        source.close()
    return SCANNED_OUTPUT


def create_mixed_fixture() -> Path:
    import fitz

    source = fitz.open(RAG_SOURCE)
    output = fitz.open()
    try:
        output.insert_pdf(source, from_page=0, to_page=0)
        if source.page_count > 1:
            _image_page(
                output,
                source.load_page(1),
                banner="RAGMIX 2026 OCR PAGE",
            )
        output.save(MIXED_OUTPUT, garbage=4, deflate=True)
    finally:
        output.close()
        source.close()
    return MIXED_OUTPUT


def main() -> int:
    scanned = create_scanned_fixture()
    mixed = create_mixed_fixture()
    print(f"Scanned fixture: {scanned}")
    print(f"Mixed fixture: {mixed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
