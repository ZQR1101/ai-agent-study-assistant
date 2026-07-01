import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
import os

from fastapi.testclient import TestClient

from backend.server import app
from backend.config import read_cors_allowed_origins


class _RedirectingOpener:
    def open(self, request, timeout):
        raise HTTPError(
            request.full_url,
            302,
            "redirect",
            {"Location": "http://127.0.0.1/private-image"},
            BytesIO(b""),
        )


class ServerSecurityTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_cors_rejects_unlisted_origin(self):
        response = self.client.options(
            "/chat",
            headers={
                "Origin": "https://attacker.example",
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertNotIn("access-control-allow-origin", response.headers)

    def test_cors_allows_default_local_frontend(self):
        response = self.client.options(
            "/chat",
            headers={
                "Origin": "http://127.0.0.1:5500",
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertEqual(
            response.headers.get("access-control-allow-origin"),
            "http://127.0.0.1:5500",
        )

    def test_cors_configuration_rejects_wildcard(self):
        with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": "*"}):
            with self.assertRaises(ValueError):
                read_cors_allowed_origins()

    def test_image_proxy_revalidates_redirect_target(self):
        with (
            patch("backend.server._build_image_opener", return_value=_RedirectingOpener()),
            patch(
                "backend.server._is_public_image_url",
                side_effect=lambda url: not url.startswith("http://127.0.0.1"),
            ),
        ):
            response = self.client.get(
                "/image-proxy",
                params={"url": "https://public.example/image"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Unsupported image URL")


class UploadSecurityTests(unittest.TestCase):
    def setUp(self):
        from backend import server

        self.server = server
        self.client = TestClient(app)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.docs_path = Path(self.temporary_directory.name)
        self.docs_patch = patch("backend.server.DOCS_PATH", self.docs_path)
        self.config_patch = patch(
            "backend.server.get_config",
            return_value=SimpleNamespace(max_upload_size_bytes=16),
        )
        self.docs_patch.start()
        self.config_patch.start()

    def tearDown(self):
        self.config_patch.stop()
        self.docs_patch.stop()
        self.temporary_directory.cleanup()

    def test_upload_accepts_valid_pdf(self):
        content = b"%PDF-1.4\nbody"
        response = self.client.post(
            "/upload",
            files={"file": ("guide.pdf", content, "application/pdf")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual((self.docs_path / "guide.pdf").read_bytes(), content)

    def test_upload_rejects_unsupported_extension_and_mime(self):
        extension_response = self.client.post(
            "/upload",
            files={"file": ("payload.exe", b"data", "application/octet-stream")},
        )
        mime_response = self.client.post(
            "/upload",
            files={"file": ("fake.pdf", b"%PDF-data", "text/plain")},
        )

        self.assertEqual(extension_response.status_code, 415)
        self.assertEqual(mime_response.status_code, 415)
        self.assertEqual(list(self.docs_path.iterdir()), [])

    def test_upload_rejects_invalid_pdf_signature(self):
        response = self.client.post(
            "/upload",
            files={"file": ("fake.pdf", b"not-a-pdf", "application/pdf")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertFalse((self.docs_path / "fake.pdf").exists())

    def test_upload_rejects_oversized_file_without_leaving_partial_file(self):
        response = self.client.post(
            "/upload",
            files={"file": ("large.txt", b"x" * 17, "text/plain")},
        )

        self.assertEqual(response.status_code, 413)
        self.assertFalse((self.docs_path / "large.txt").exists())

    def test_upload_does_not_overwrite_existing_file(self):
        target = self.docs_path / "notes.txt"
        target.write_text("original", encoding="utf-8")

        response = self.client.post(
            "/upload",
            files={"file": ("notes.txt", b"replacement", "text/plain")},
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(target.read_text(encoding="utf-8"), "original")


if __name__ == "__main__":
    unittest.main()
