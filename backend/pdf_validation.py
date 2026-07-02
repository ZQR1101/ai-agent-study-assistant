"""Validate untrusted PDFs in a short-lived subprocess."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


class PDFValidationError(RuntimeError):
    pass


class PDFValidationTimeout(PDFValidationError):
    pass


class PDFPageLimitExceeded(PDFValidationError):
    pass


def validate_pdf_file(
    path: Path,
    *,
    max_pages: int,
    timeout_seconds: int,
    max_memory_bytes: int,
) -> int:
    command = [
        sys.executable,
        "-I",
        str(Path(__file__).resolve()),
        "--worker",
        str(path.resolve()),
        "--max-pages",
        str(max_pages),
        "--max-memory-bytes",
        str(max_memory_bytes),
    ]
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=max(1, timeout_seconds),
            check=False,
            creationflags=creation_flags,
        )
    except subprocess.TimeoutExpired as exc:
        raise PDFValidationTimeout("PDF validation timed out") from exc

    try:
        result = json.loads(completed.stdout)
    except (TypeError, json.JSONDecodeError) as exc:
        raise PDFValidationError("PDF validation process failed") from exc

    if completed.returncode == 0 and result.get("valid") is True:
        return int(result["pages"])
    if result.get("error") == "page_limit":
        raise PDFPageLimitExceeded(
            f"PDF exceeds the {max_pages} page limit"
        )
    if result.get("error") == "encrypted":
        raise PDFValidationError("Encrypted PDFs are not supported")
    raise PDFValidationError("Uploaded PDF structure is invalid")


def _apply_memory_limit(max_memory_bytes: int) -> None:
    if max_memory_bytes <= 0:
        return
    try:
        import resource
    except ImportError:
        return
    try:
        resource.setrlimit(
            resource.RLIMIT_AS,
            (max_memory_bytes, max_memory_bytes),
        )
    except (OSError, ValueError):
        return


def _worker(path: Path, max_pages: int, max_memory_bytes: int) -> int:
    _apply_memory_limit(max_memory_bytes)
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path), strict=True)
        if reader.is_encrypted:
            print(json.dumps({"valid": False, "error": "encrypted"}))
            return 2
        pages = len(reader.pages)
        if pages > max_pages:
            print(
                json.dumps(
                    {"valid": False, "error": "page_limit", "pages": pages}
                )
            )
            return 3
        for page in reader.pages:
            page.get("/Type")
    except Exception:
        print(json.dumps({"valid": False, "error": "invalid"}))
        return 2

    print(json.dumps({"valid": True, "pages": pages}))
    return 0


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("path", type=Path)
    parser.add_argument("--max-pages", type=int, required=True)
    parser.add_argument("--max-memory-bytes", type=int, required=True)
    arguments = parser.parse_args()
    if not arguments.worker:
        return 2
    return _worker(
        arguments.path,
        max(1, arguments.max_pages),
        max(1, arguments.max_memory_bytes),
    )


if __name__ == "__main__":
    raise SystemExit(_main())
