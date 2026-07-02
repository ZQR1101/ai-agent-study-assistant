"""Small in-process resource guards for public HTTP endpoints."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Callable


class ConcurrencyGate:
    def __init__(self) -> None:
        self._active = 0
        self._lock = threading.Lock()

    def try_acquire(self, limit: int) -> bool:
        with self._lock:
            if self._active >= max(1, limit):
                return False
            self._active += 1
            return True

    def release(self) -> None:
        with self._lock:
            if self._active > 0:
                self._active -= 1


@dataclass
class _TokenBucket:
    tokens: float
    updated_at: float
    limit: int
    window_seconds: int


class TokenBucketRateLimiter:
    def __init__(self, max_clients: int = 4096) -> None:
        self._buckets: dict[str, _TokenBucket] = {}
        self._max_clients = max_clients
        self._lock = threading.Lock()

    def allow(
        self,
        key: str,
        limit: int,
        window_seconds: int,
        *,
        now: float | None = None,
    ) -> tuple[bool, int]:
        limit = max(1, limit)
        window_seconds = max(1, window_seconds)
        current_time = monotonic() if now is None else now
        refill_rate = limit / window_seconds

        with self._lock:
            bucket = self._buckets.get(key)
            if (
                bucket is None
                or bucket.limit != limit
                or bucket.window_seconds != window_seconds
            ):
                bucket = _TokenBucket(
                    tokens=float(limit),
                    updated_at=current_time,
                    limit=limit,
                    window_seconds=window_seconds,
                )
                self._buckets[key] = bucket
            else:
                elapsed = max(0.0, current_time - bucket.updated_at)
                bucket.tokens = min(float(limit), bucket.tokens + elapsed * refill_rate)
                bucket.updated_at = current_time

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                self._trim_locked()
                return True, 0

            retry_after = max(1, int((1.0 - bucket.tokens) / refill_rate) + 1)
            self._trim_locked()
            return False, retry_after

    def reset(self) -> None:
        with self._lock:
            self._buckets.clear()

    def _trim_locked(self) -> None:
        if len(self._buckets) <= self._max_clients:
            return
        oldest = sorted(
            self._buckets.items(),
            key=lambda item: item[1].updated_at,
        )[: len(self._buckets) - self._max_clients]
        for key, _ in oldest:
            self._buckets.pop(key, None)


class UploadBodyTooLarge(RuntimeError):
    pass


class UploadQuotaReservations:
    def __init__(self) -> None:
        self._reserved_bytes = 0
        self._lock = threading.Lock()

    def reserve(
        self,
        root: Path,
        expected_bytes: int,
        total_limit: int,
    ) -> int | None:
        expected_bytes = max(0, expected_bytes)
        with self._lock:
            used_bytes = self._used_bytes(root)
            if used_bytes + self._reserved_bytes + expected_bytes > total_limit:
                return None
            self._reserved_bytes += expected_bytes
            return expected_bytes

    def grow(
        self,
        root: Path,
        reserved_bytes: int,
        required_bytes: int,
        total_limit: int,
    ) -> int | None:
        if required_bytes <= reserved_bytes:
            return reserved_bytes
        additional_bytes = required_bytes - reserved_bytes
        with self._lock:
            used_bytes = self._used_bytes(root)
            if used_bytes + self._reserved_bytes + additional_bytes > total_limit:
                return None
            self._reserved_bytes += additional_bytes
            return required_bytes

    def release(self, reserved_bytes: int) -> None:
        with self._lock:
            self._reserved_bytes = max(0, self._reserved_bytes - max(0, reserved_bytes))

    @staticmethod
    def _used_bytes(root: Path) -> int:
        if not root.exists():
            return 0
        return sum(
            path.stat().st_size
            for path in root.iterdir()
            if path.is_file()
        )


class UploadBodyLimitMiddleware:
    def __init__(
        self,
        app,
        *,
        config_provider: Callable,
        concurrency_gate: ConcurrencyGate,
        rate_limiter: TokenBucketRateLimiter,
        multipart_overhead_bytes: int,
    ) -> None:
        self.app = app
        self.config_provider = config_provider
        self.concurrency_gate = concurrency_gate
        self.rate_limiter = rate_limiter
        self.multipart_overhead_bytes = multipart_overhead_bytes

    async def __call__(self, scope, receive, send) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != "/upload"
        ):
            await self.app(scope, receive, send)
            return

        config = self.config_provider()
        file_limit = config.max_upload_size_bytes
        body_limit = file_limit + self.multipart_overhead_bytes
        concurrency_limit = getattr(config, "upload_max_concurrency", 2)

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        origin = headers.get(b"origin", b"").decode("latin-1").rstrip("/")
        if origin:
            allowed_origins = set(getattr(config, "cors_allowed_origins", ()))
            requested_with = headers.get(b"x-requested-with", b"").decode("latin-1")
            if origin not in allowed_origins or requested_with != "AI-Study-Assistant":
                await self._send_error(send, 403, "Upload browser request denied")
                return

        client = scope.get("client")
        client_host = str(client[0]) if client else "unknown"
        allowed, retry_after = self.rate_limiter.allow(
            client_host,
            getattr(config, "upload_rate_limit", 10),
            getattr(config, "upload_rate_window_seconds", 60),
        )
        if not allowed:
            await self._send_error(
                send,
                429,
                "Upload rate limit exceeded",
                retry_after=retry_after,
            )
            return

        if not self.concurrency_gate.try_acquire(concurrency_limit):
            await self._send_error(
                send,
                503,
                "Upload capacity is busy; retry later",
                retry_after=1,
            )
            return

        try:
            content_length = headers.get(b"content-length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    await self._send_error(send, 400, "Invalid Content-Length")
                    return
                if declared_length < 0:
                    await self._send_error(send, 400, "Invalid Content-Length")
                    return
                if declared_length > body_limit:
                    await self._send_error(send, 413, "Upload request body is too large")
                    return

            received = 0

            async def limited_receive():
                nonlocal received
                message = await receive()
                if message.get("type") == "http.request":
                    received += len(message.get("body", b""))
                    if received > body_limit:
                        raise UploadBodyTooLarge
                return message

            try:
                await self.app(scope, limited_receive, send)
            except UploadBodyTooLarge:
                await self._send_error(send, 413, "Upload request body is too large")
        finally:
            self.concurrency_gate.release()

    @staticmethod
    async def _send_error(send, status: int, detail: str, retry_after: int | None = None):
        body = json.dumps({"detail": detail}).encode("utf-8")
        headers = [
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode("ascii")),
        ]
        if retry_after is not None:
            headers.append((b"retry-after", str(retry_after).encode("ascii")))
        await send({"type": "http.response.start", "status": status, "headers": headers})
        await send({"type": "http.response.body", "body": body})
