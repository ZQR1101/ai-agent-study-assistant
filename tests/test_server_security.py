import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import URLError
from urllib.parse import urlparse
import os

from fastapi.testclient import TestClient
from pypdf import PdfWriter

from backend.server import (
    _PinnedHTTPSConnection,
    _open_public_image,
    _resolve_public_image_url,
    app,
)
from backend.config import read_cors_allowed_origins, read_tool_secret
from backend.pdf_validation import PDFValidationTimeout


def _pdf_bytes(page_count=1):
    buffer = BytesIO()
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=72, height=72)
    writer.write(buffer)
    return buffer.getvalue()


class _FakeImageResponse:
    def __init__(self, status=200, headers=None, content=b"image"):
        self.status = status
        self.headers = headers or {"content-type": "image/png"}
        self.content = content
        self.closed = False
        self.offset = 0

    def read(self, amount=-1):
        if amount < 0:
            result = self.content[self.offset :]
            self.offset = len(self.content)
            return result
        result = self.content[self.offset : self.offset + amount]
        self.offset += len(result)
        return result

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class ServerSecurityTests(unittest.TestCase):
    def setUp(self):
        from backend import server

        self.server = server
        self.server._IMAGE_PROXY_RATE_LIMITER.reset()
        self.server._UPLOAD_RATE_LIMITER.reset()
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

    def test_cors_does_not_allow_browser_approver_credentials(self):
        response = self.client.options(
            "/tools/delete_run/approvals/request-id",
            headers={
                "Origin": "http://127.0.0.1:5500",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-Tool-Approver-Key",
            },
        )

        self.assertEqual(response.status_code, 400)
        allowed_headers = response.headers.get("access-control-allow-headers", "").lower()
        self.assertNotIn("x-tool-approver-key", allowed_headers)

    def test_cors_allows_upload_csrf_header_for_trusted_frontend(self):
        response = self.client.options(
            "/upload",
            headers={
                "Origin": "http://127.0.0.1:5500",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "X-Requested-With",
            },
        )

        self.assertEqual(response.status_code, 200)
        allowed_headers = response.headers.get("access-control-allow-headers", "").lower()
        self.assertIn("x-requested-with", allowed_headers)

    def test_cors_configuration_rejects_wildcard(self):
        with patch.dict(os.environ, {"CORS_ALLOWED_ORIGINS": "*"}):
            with self.assertRaises(ValueError):
                read_cors_allowed_origins()

    def test_tool_secrets_reject_short_and_example_values(self):
        with patch.dict(
            os.environ,
            {
                "TOOL_APPROVAL_KEY": "short",
                "TOOL_APPROVER_KEY": "replace-with-a-different-long-random-approver-secret",
            },
        ):
            self.assertIsNone(read_tool_secret("TOOL_APPROVAL_KEY"))
            self.assertIsNone(read_tool_secret("TOOL_APPROVER_KEY"))

        strong_secret = "a-strong-random-tool-secret-value-123456"
        with patch.dict(os.environ, {"TOOL_APPROVAL_KEY": strong_secret}):
            self.assertEqual(read_tool_secret("TOOL_APPROVAL_KEY"), strong_secret)

    def test_image_proxy_revalidates_redirect_target(self):
        redirect_response = _FakeImageResponse(
            status=302,
            headers={"Location": "http://127.0.0.1/private-image"},
        )
        with (
            patch(
                "backend.server._resolve_public_image_url",
                side_effect=[
                    (
                        urlparse("https://public.example/image"),
                        ("93.184.216.34",),
                    ),
                    None,
                ],
            ),
            patch(
                "backend.server._open_pinned_image_response",
                return_value=redirect_response,
            ),
        ):
            response = self.client.get(
                "/image-proxy",
                params={"url": "https://public.example/image"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Unsupported image URL")
        self.assertTrue(redirect_response.closed)

    def test_image_proxy_rejects_non_global_and_mixed_dns_results(self):
        blocked_addresses = (
            "127.0.0.1",
            "10.0.0.1",
            "169.254.1.1",
            "100.64.0.1",
            "192.0.2.1",
            "::1",
            "fc00::1",
        )
        for address in blocked_addresses:
            with self.subTest(address=address), patch(
                "backend.server.socket.getaddrinfo",
                return_value=[(2, 1, 6, "", (address, 443))],
            ):
                self.assertIsNone(_resolve_public_image_url("https://images.example/a.png"))

        with patch(
            "backend.server.socket.getaddrinfo",
            return_value=[
                (2, 1, 6, "", ("93.184.216.34", 443)),
                (2, 1, 6, "", ("127.0.0.1", 443)),
            ],
        ):
            self.assertIsNone(_resolve_public_image_url("https://images.example/a.png"))

    def test_image_proxy_connects_to_the_validated_ip(self):
        final_response = _FakeImageResponse()
        with (
            patch(
                "backend.server.socket.getaddrinfo",
                return_value=[(2, 1, 6, "", ("93.184.216.34", 443))],
            ) as resolve,
            patch(
                "backend.server._open_pinned_image_response",
                return_value=final_response,
            ) as open_pinned,
        ):
            response = _open_public_image("https://images.example/a.png")

        self.assertIs(response, final_response)
        self.assertEqual(resolve.call_count, 1)
        self.assertEqual(open_pinned.call_args.args[1], "93.184.216.34")

    def test_image_proxy_caps_resolved_addresses_and_total_deadline(self):
        dns_results = [
            (2, 1, 6, "", (f"93.184.216.{index}", 443))
            for index in range(1, 13)
        ]
        with patch("backend.server.socket.getaddrinfo", return_value=dns_results):
            resolved = _resolve_public_image_url("https://images.example/a.png")

        self.assertIsNotNone(resolved)
        self.assertEqual(len(resolved[1]), 8)

        parsed = urlparse("https://images.example/a.png")
        with (
            patch(
                "backend.server._resolve_public_image_url",
                return_value=(parsed, ("93.184.216.1", "93.184.216.2")),
            ),
            patch("backend.server.perf_counter", side_effect=[1, 11]),
            patch(
                "backend.server._open_pinned_image_response",
                side_effect=OSError("connect failed"),
            ) as open_pinned,
        ):
            with self.assertRaises(URLError):
                _open_public_image("https://images.example/a.png", deadline=10)

        self.assertEqual(open_pinned.call_count, 1)

    def test_image_proxy_revalidates_each_relative_redirect(self):
        first_redirect = _FakeImageResponse(
            status=302,
            headers={"Location": "/second.png"},
        )
        second_redirect = _FakeImageResponse(
            status=307,
            headers={"Location": "https://cdn.example/final.png"},
        )
        final_response = _FakeImageResponse()
        resolved_targets = [
            (urlparse("https://images.example/first.png"), ("93.184.216.34",)),
            (urlparse("https://images.example/second.png"), ("93.184.216.35",)),
            (urlparse("https://cdn.example/final.png"), ("93.184.216.36",)),
        ]
        with (
            patch(
                "backend.server._resolve_public_image_url",
                side_effect=resolved_targets,
            ) as resolve,
            patch(
                "backend.server._open_pinned_image_response",
                side_effect=[first_redirect, second_redirect, final_response],
            ) as open_pinned,
        ):
            response = _open_public_image("https://images.example/first.png")

        self.assertIs(response, final_response)
        self.assertEqual(
            [call.args[0] for call in resolve.call_args_list],
            [
                "https://images.example/first.png",
                "https://images.example/second.png",
                "https://cdn.example/final.png",
            ],
        )
        self.assertEqual(
            [call.args[1] for call in open_pinned.call_args_list],
            ["93.184.216.34", "93.184.216.35", "93.184.216.36"],
        )
        self.assertTrue(first_redirect.closed)
        self.assertTrue(second_redirect.closed)

    def test_pinned_https_connection_preserves_hostname_for_sni(self):
        connection = _PinnedHTTPSConnection(
            "images.example",
            "93.184.216.34",
            443,
            timeout=20,
        )
        raw_socket = Mock()
        wrapped_socket = Mock()
        connection._create_connection = Mock(return_value=raw_socket)
        connection._context = Mock()
        connection._context.wrap_socket.return_value = wrapped_socket

        connection.connect()

        connection._create_connection.assert_called_once_with(
            ("93.184.216.34", 443),
            20,
            None,
        )
        connection._context.wrap_socket.assert_called_once_with(
            raw_socket,
            server_hostname="images.example",
        )
        self.assertIs(connection.sock, wrapped_socket)

    def test_image_proxy_enforces_rate_and_concurrency_limits(self):
        config = SimpleNamespace(
            image_proxy_rate_limit=1,
            image_proxy_rate_window_seconds=60,
            image_proxy_max_concurrency=1,
            image_proxy_max_response_bytes=16,
        )
        with (
            patch("backend.server.get_config", return_value=config),
            patch.object(
                self.server._IMAGE_PROXY_RATE_LIMITER,
                "allow",
                return_value=(False, 7),
            ),
            patch("backend.server._open_public_image") as open_image,
        ):
            rate_response = self.client.get(
                "/image-proxy",
                params={"url": "https://images.example/a.png"},
            )

        self.assertEqual(rate_response.status_code, 429)
        self.assertEqual(rate_response.headers.get("retry-after"), "7")
        open_image.assert_not_called()

        with (
            patch("backend.server.get_config", return_value=config),
            patch.object(
                self.server._IMAGE_PROXY_RATE_LIMITER,
                "allow",
                return_value=(True, 0),
            ),
            patch.object(
                self.server._IMAGE_PROXY_CONCURRENCY_GATE,
                "try_acquire",
                return_value=False,
            ),
            patch("backend.server._open_public_image") as open_image,
        ):
            busy_response = self.client.get(
                "/image-proxy",
                params={"url": "https://images.example/a.png"},
            )

        self.assertEqual(busy_response.status_code, 503)
        open_image.assert_not_called()

    def test_image_proxy_uses_configured_response_size_limit(self):
        config = SimpleNamespace(
            image_proxy_rate_limit=10,
            image_proxy_rate_window_seconds=60,
            image_proxy_max_concurrency=1,
            image_proxy_max_response_bytes=4,
        )
        image_response = _FakeImageResponse(content=b"12345")
        with (
            patch("backend.server.get_config", return_value=config),
            patch.object(
                self.server._IMAGE_PROXY_RATE_LIMITER,
                "allow",
                return_value=(True, 0),
            ),
            patch("backend.server._open_public_image", return_value=image_response),
        ):
            response = self.client.get(
                "/image-proxy",
                params={"url": "https://images.example/a.png"},
            )

        self.assertEqual(response.status_code, 413)
        self.assertTrue(image_response.closed)

    def test_image_proxy_rejects_cross_site_browser_requests(self):
        with patch("backend.server._open_public_image") as open_image:
            fetch_metadata_response = self.client.get(
                "/image-proxy",
                params={"url": "https://images.example/a.png"},
                headers={"Sec-Fetch-Site": "cross-site"},
            )
            referer_response = self.client.get(
                "/image-proxy",
                params={"url": "https://images.example/a.png"},
                headers={"Referer": "https://attacker.example/page"},
            )

        self.assertEqual(fetch_metadata_response.status_code, 403)
        self.assertEqual(referer_response.status_code, 403)
        open_image.assert_not_called()

    def test_image_proxy_rejects_active_or_mislabeled_image_content(self):
        config = SimpleNamespace(
            cors_allowed_origins=("http://127.0.0.1:5500",),
            image_proxy_rate_limit=10,
            image_proxy_rate_window_seconds=60,
            image_proxy_max_concurrency=1,
            image_proxy_max_response_bytes=1024,
            image_proxy_timeout_seconds=20,
        )
        valid_png = _FakeImageResponse(
            headers={"content-type": "image/png"},
            content=b"\x89PNG\r\n\x1a\ncontent",
        )
        with (
            patch("backend.server.get_config", return_value=config),
            patch.object(
                self.server._IMAGE_PROXY_RATE_LIMITER,
                "allow",
                return_value=(True, 0),
            ),
            patch("backend.server._open_public_image", return_value=valid_png),
        ):
            valid_response = self.client.get(
                "/image-proxy",
                params={"url": "https://images.example/a.png"},
            )

        self.assertEqual(valid_response.status_code, 200)
        self.assertEqual(valid_response.headers.get("x-content-type-options"), "nosniff")

        for content_type, content in (
            ("image/svg+xml", b"<svg></svg>"),
            ("image/png", b"not-a-png"),
        ):
            with (
                self.subTest(content_type=content_type),
                patch("backend.server.get_config", return_value=config),
                patch.object(
                    self.server._IMAGE_PROXY_RATE_LIMITER,
                    "allow",
                    return_value=(True, 0),
                ),
                patch(
                    "backend.server._open_public_image",
                    return_value=_FakeImageResponse(
                        headers={"content-type": content_type},
                        content=content,
                    ),
                ),
            ):
                response = self.client.get(
                    "/image-proxy",
                    params={"url": "https://images.example/a.png"},
                )
            self.assertEqual(response.status_code, 400)


class UploadSecurityTests(unittest.TestCase):
    def setUp(self):
        from backend import server

        self.server = server
        self.server._UPLOAD_RATE_LIMITER.reset()
        self.client = TestClient(app)
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.docs_path = Path(self.temporary_directory.name)
        self.docs_patch = patch("backend.server.DOCS_PATH", self.docs_path)
        self.config_patch = patch(
            "backend.server.get_config",
            return_value=SimpleNamespace(
                max_upload_size_bytes=16,
                max_upload_total_bytes=160,
                upload_max_concurrency=2,
                upload_rate_limit=10,
                upload_rate_window_seconds=60,
                cors_allowed_origins=("http://127.0.0.1:5500",),
            ),
        )
        self.docs_patch.start()
        self.config_patch.start()

    def tearDown(self):
        self.config_patch.stop()
        self.docs_patch.stop()
        self.temporary_directory.cleanup()

    def test_upload_accepts_valid_pdf(self):
        content = _pdf_bytes()
        with patch(
            "backend.server.get_config",
            return_value=SimpleNamespace(
                max_upload_size_bytes=4096,
                max_upload_total_bytes=40960,
                upload_max_concurrency=2,
                max_pdf_pages=10,
                pdf_validation_timeout_seconds=5,
                pdf_validation_max_memory_bytes=256 * 1024 * 1024,
            ),
        ):
            response = self.client.post(
                "/upload",
                files={"file": ("guide.pdf", content, "application/pdf")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["pages"], 1)
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

    def test_upload_rejects_reserved_and_overlong_filenames(self):
        reserved_response = self.client.post(
            "/upload",
            files={"file": ("CON.txt", b"data", "text/plain")},
        )
        long_response = self.client.post(
            "/upload",
            files={"file": (f"{'a' * 125}.txt", b"data", "text/plain")},
        )

        self.assertEqual(reserved_response.status_code, 400)
        self.assertEqual(long_response.status_code, 400)
        self.assertEqual(list(self.docs_path.iterdir()), [])

    def test_upload_rejects_invalid_pdf_signature(self):
        response = self.client.post(
            "/upload",
            files={"file": ("fake.pdf", b"not-a-pdf", "application/pdf")},
        )

        self.assertEqual(response.status_code, 415)
        self.assertFalse((self.docs_path / "fake.pdf").exists())

    def test_upload_safely_stores_malformed_pdf_structure(self):
        response = self.client.post(
            "/upload",
            files={"file": ("fake.pdf", b"%PDF-broken", "application/pdf")},
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["corrupted_pdf"])
        self.assertTrue(response.json()["safe_fallback"])
        self.assertIn("structure is invalid", " ".join(response.json()["warnings"]))
        self.assertTrue((self.docs_path / "fake.pdf").exists())

    def test_upload_rejects_pdf_page_limit_and_validation_timeout(self):
        content = _pdf_bytes(page_count=2)
        config = SimpleNamespace(
            max_upload_size_bytes=4096,
            max_upload_total_bytes=40960,
            upload_max_concurrency=2,
            max_pdf_pages=1,
            pdf_validation_timeout_seconds=5,
            pdf_validation_max_memory_bytes=256 * 1024 * 1024,
        )
        with patch("backend.server.get_config", return_value=config):
            page_response = self.client.post(
                "/upload",
                files={"file": ("pages.pdf", content, "application/pdf")},
            )

        self.assertEqual(page_response.status_code, 422)
        self.assertIn("page limit", page_response.json()["detail"])
        self.assertFalse((self.docs_path / "pages.pdf").exists())

        with (
            patch("backend.server.get_config", return_value=config),
            patch(
                "backend.server.validate_pdf_file",
                side_effect=PDFValidationTimeout("PDF validation timed out"),
            ),
        ):
            timeout_response = self.client.post(
                "/upload",
                files={"file": ("timeout.pdf", content, "application/pdf")},
            )

        self.assertEqual(timeout_response.status_code, 200)
        self.assertEqual(timeout_response.json()["pdf_validation_status"], "timeout")
        self.assertTrue(timeout_response.json()["safe_fallback"])
        self.assertTrue((self.docs_path / "timeout.pdf").exists())

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

    def test_upload_rejects_oversized_request_before_multipart_parsing(self):
        response = self.client.post(
            "/upload",
            content=b"x" * (16 + self.server.UPLOAD_MULTIPART_OVERHEAD_BYTES + 1),
            headers={"Content-Type": "application/octet-stream"},
        )

        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], "Upload request body is too large")

    def test_upload_rejects_when_concurrency_gate_is_busy(self):
        with patch.object(
            self.server._UPLOAD_CONCURRENCY_GATE,
            "try_acquire",
            return_value=False,
        ):
            response = self.client.post(
                "/upload",
                files={"file": ("notes.txt", b"hello", "text/plain")},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers.get("retry-after"), "1")
        self.assertFalse((self.docs_path / "notes.txt").exists())

    def test_upload_enforces_total_storage_quota(self):
        (self.docs_path / "existing.txt").write_bytes(b"x" * 150)

        response = self.client.post(
            "/upload",
            files={"file": ("new.txt", b"y" * 11, "text/plain")},
        )

        self.assertEqual(response.status_code, 507)
        self.assertFalse((self.docs_path / "new.txt").exists())
        self.assertEqual(
            [path.name for path in self.docs_path.iterdir()],
            ["existing.txt"],
        )

    def test_upload_rejects_untrusted_browser_origin_and_requires_csrf_header(self):
        untrusted_response = self.client.post(
            "/upload",
            files={"file": ("untrusted.txt", b"hello", "text/plain")},
            headers={"Origin": "https://attacker.example"},
        )
        missing_header_response = self.client.post(
            "/upload",
            files={"file": ("missing.txt", b"hello", "text/plain")},
            headers={"Origin": "http://127.0.0.1:5500"},
        )
        allowed_response = self.client.post(
            "/upload",
            files={"file": ("allowed.txt", b"hello", "text/plain")},
            headers={
                "Origin": "http://127.0.0.1:5500",
                "X-Requested-With": "AI-Study-Assistant",
            },
        )

        self.assertEqual(untrusted_response.status_code, 403)
        self.assertEqual(missing_header_response.status_code, 403)
        self.assertEqual(allowed_response.status_code, 200)
        self.assertFalse((self.docs_path / "untrusted.txt").exists())
        self.assertFalse((self.docs_path / "missing.txt").exists())
        self.assertTrue((self.docs_path / "allowed.txt").exists())

    def test_upload_enforces_per_client_rate_limit(self):
        with patch.object(
            self.server._UPLOAD_RATE_LIMITER,
            "allow",
            return_value=(False, 9),
        ):
            response = self.client.post(
                "/upload",
                files={"file": ("limited.txt", b"hello", "text/plain")},
            )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers.get("retry-after"), "9")
        self.assertFalse((self.docs_path / "limited.txt").exists())


if __name__ == "__main__":
    unittest.main()
