import unittest
from types import SimpleNamespace

from backend.resource_limits import (
    ConcurrencyGate,
    TokenBucketRateLimiter,
    UploadBodyLimitMiddleware,
)


class ResourceLimitTests(unittest.TestCase):
    def test_concurrency_gate_releases_capacity(self):
        gate = ConcurrencyGate()

        self.assertTrue(gate.try_acquire(1))
        self.assertFalse(gate.try_acquire(1))
        gate.release()
        self.assertTrue(gate.try_acquire(1))
        gate.release()

    def test_token_bucket_recovers_after_time_passes(self):
        limiter = TokenBucketRateLimiter()

        self.assertEqual(limiter.allow("client", 2, 10, now=0), (True, 0))
        self.assertEqual(limiter.allow("client", 2, 10, now=0), (True, 0))
        allowed, retry_after = limiter.allow("client", 2, 10, now=0)
        self.assertFalse(allowed)
        self.assertGreaterEqual(retry_after, 1)
        self.assertEqual(limiter.allow("client", 2, 10, now=5), (True, 0))


class UploadBodyLimitTests(unittest.IsolatedAsyncioTestCase):
    async def test_streamed_body_without_content_length_is_limited(self):
        gate = ConcurrencyGate()
        sent_messages = []
        incoming_messages = iter(
            [
                {"type": "http.request", "body": b"123", "more_body": True},
                {"type": "http.request", "body": b"456", "more_body": False},
            ]
        )

        async def receive():
            return next(incoming_messages)

        async def send(message):
            sent_messages.append(message)

        async def consume_body(scope, receive_body, send_response):
            while True:
                message = await receive_body()
                if not message.get("more_body"):
                    break

        middleware = UploadBodyLimitMiddleware(
            consume_body,
            config_provider=lambda: SimpleNamespace(
                max_upload_size_bytes=4,
                upload_max_concurrency=1,
            ),
            concurrency_gate=gate,
            rate_limiter=TokenBucketRateLimiter(),
            multipart_overhead_bytes=0,
        )
        await middleware(
            {
                "type": "http",
                "method": "POST",
                "path": "/upload",
                "headers": [],
            },
            receive,
            send,
        )

        self.assertEqual(sent_messages[0]["status"], 413)
        self.assertTrue(gate.try_acquire(1))
        gate.release()


if __name__ == "__main__":
    unittest.main()
