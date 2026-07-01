"""Typed tool registry with confirmation and audit enforcement.

Every tool invocation must pass through :meth:`ToolRegistry.execute`.  This is
the single enforcement point for dangerous-tool confirmation and audit logs.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any, Callable
from uuid import uuid4


class ToolCategory(str, Enum):
    READ = "read"
    WRITE = "write"
    DANGEROUS = "dangerous"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    run: Callable[..., dict]
    category: ToolCategory = ToolCategory.READ
    requires_confirmation: bool = False
    agent_visible: bool = True

    def __post_init__(self) -> None:
        if self.category is ToolCategory.DANGEROUS and not self.requires_confirmation:
            raise ValueError(f"Dangerous tool {self.name!r} must require confirmation")


class ToolConfirmationRequired(RuntimeError):
    def __init__(self, tool_name: str, token: str, expires_in_seconds: int):
        self.tool_name = tool_name
        self.token = token
        self.expires_in_seconds = expires_in_seconds
        super().__init__(f"Tool {tool_name!r} requires explicit confirmation")

    def as_dict(self) -> dict:
        return {
            "error": "confirmation_required",
            "tool": self.tool_name,
            "confirmation_token": self.token,
            "expires_in_seconds": self.expires_in_seconds,
        }


class InvalidConfirmation(RuntimeError):
    pass


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return "<max-depth>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 2000 else f"{value[:2000]}...<truncated>"
    if isinstance(value, Mapping):
        safe = {}
        for key, item in list(value.items())[:50]:
            key_text = str(key)
            if any(marker in key_text.lower() for marker in ("password", "secret", "api_key")):
                safe[key_text] = "<redacted>"
            elif key_text in {"custom_llm"}:
                safe[key_text] = f"<{type(item).__name__}>"
            else:
                safe[key_text] = _json_safe(item, depth=depth + 1)
        return safe
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item, depth=depth + 1) for item in list(value)[:50]]
    return f"<{type(value).__name__}>"


def _arguments_digest(tool_name: str, arguments: Mapping[str, Any]) -> str:
    payload = json.dumps(
        {"tool": tool_name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=lambda value: {
            "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
            "__repr__": repr(value),
        },
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _event_run_id(event: Mapping[str, Any]) -> str | None:
    if event.get("run_id"):
        return str(event["run_id"])
    arguments = event.get("arguments")
    if not isinstance(arguments, Mapping):
        return None
    if arguments.get("run_id"):
        return str(arguments["run_id"])
    if arguments.get("target_run_id"):
        return str(arguments["target_run_id"])
    shared_context = arguments.get("shared_context")
    if isinstance(shared_context, Mapping) and shared_context.get("run_id"):
        return str(shared_context["run_id"])
    return None


class AuditLog:
    """Append-only JSONL audit log."""

    def __init__(self, path: Path | None = None):
        default_path = Path(__file__).parent.parent / "logs" / "tool_audit.jsonl"
        self.path = Path(path or os.getenv("TOOL_AUDIT_LOG_PATH", default_path))
        self._lock = threading.Lock()

    def record(self, event: Mapping[str, Any]) -> None:
        run_id = _event_run_id(event)
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **_json_safe(dict(event)),
        }
        if run_id:
            entry["run_id"] = run_id
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if run_id:
            from backend.run_repository import get_run_repository

            get_run_repository().append_audit(run_id, entry)

    def recent(self, limit: int = 100) -> list[dict]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        entries = []
        for line in lines[-max(1, min(limit, 1000)):]:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries


@dataclass
class _PendingConfirmation:
    tool_name: str
    arguments_digest: str
    actor: str
    expires_at: float


class ConfirmationManager:
    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        self._pending: dict[str, _PendingConfirmation] = {}
        self._lock = threading.Lock()

    def issue(self, tool_name: str, arguments: Mapping[str, Any], actor: str) -> str:
        token = secrets.token_urlsafe(32)
        pending = _PendingConfirmation(
            tool_name=tool_name,
            arguments_digest=_arguments_digest(tool_name, arguments),
            actor=actor,
            expires_at=monotonic() + self.ttl_seconds,
        )
        with self._lock:
            self._prune_locked()
            self._pending[token] = pending
        return token

    def consume(self, token: str, tool_name: str, arguments: Mapping[str, Any], actor: str) -> None:
        with self._lock:
            self._prune_locked()
            pending = self._pending.pop(token, None)
        if pending is None:
            raise InvalidConfirmation("Confirmation token is invalid, expired, or already used")
        if pending.tool_name != tool_name:
            raise InvalidConfirmation("Confirmation token was issued for a different tool")
        if pending.actor != actor:
            raise InvalidConfirmation("Confirmation token was issued for a different actor")
        if pending.arguments_digest != _arguments_digest(tool_name, arguments):
            raise InvalidConfirmation("Tool arguments changed after confirmation was requested")

    def _prune_locked(self) -> None:
        now = monotonic()
        expired = [token for token, item in self._pending.items() if item.expires_at <= now]
        for token in expired:
            self._pending.pop(token, None)


class ToolRegistry(Mapping[str, ToolSpec]):
    def __init__(
        self,
        specs: list[ToolSpec],
        *,
        aliases: Mapping[str, str] | None = None,
        audit_log: AuditLog | None = None,
        confirmation_manager: ConfirmationManager | None = None,
    ):
        self._specs = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("Tool names must be unique")
        self._aliases = dict(aliases or {})
        self.audit_log = audit_log or AuditLog()
        self.confirmations = confirmation_manager or ConfirmationManager()

    def __getitem__(self, name: str) -> ToolSpec:
        canonical = self._aliases.get(name, name)
        return self._specs[canonical]

    def __iter__(self) -> Iterator[str]:
        return iter(self._specs)

    def __len__(self) -> int:
        return len(self._specs)

    def get(self, name: str, default=None):
        try:
            return self[name]
        except KeyError:
            return default

    def canonical_name(self, name: str) -> str:
        return self._aliases.get(name, name)

    def agent_specs(self) -> list[ToolSpec]:
        return [spec for spec in self._specs.values() if spec.agent_visible]

    def execute(
        self,
        name: str,
        *,
        confirmation_token: str | None = None,
        actor: str = "system",
        **arguments: Any,
    ) -> dict:
        requested_name = name
        canonical_name = self.canonical_name(name)
        spec = self._specs.get(canonical_name)
        invocation_id = str(uuid4())
        run_id = _event_run_id({"arguments": arguments})

        if spec is None:
            self.audit_log.record({
                "event": "tool_call",
                "invocation_id": invocation_id,
                "tool": requested_name,
                "status": "unknown_tool",
                "actor": actor,
                "run_id": run_id,
                "arguments": arguments,
            })
            raise KeyError(f"Unknown tool: {requested_name}")

        if requested_name != canonical_name and canonical_name == "study":
            arguments.setdefault("operation", requested_name)

        if spec.requires_confirmation:
            if confirmation_token:
                try:
                    self.confirmations.consume(
                        confirmation_token, canonical_name, arguments, actor
                    )
                except InvalidConfirmation as exc:
                    self.audit_log.record({
                        "event": "tool_call",
                        "invocation_id": invocation_id,
                        "tool": canonical_name,
                        "category": spec.category.value,
                        "status": "confirmation_rejected",
                        "actor": actor,
                        "run_id": run_id,
                        "arguments": arguments,
                        "error": str(exc),
                    })
                    raise
            else:
                token = self.confirmations.issue(canonical_name, arguments, actor)
                self.audit_log.record({
                    "event": "tool_call",
                    "invocation_id": invocation_id,
                    "tool": canonical_name,
                    "category": spec.category.value,
                    "status": "confirmation_required",
                    "actor": actor,
                    "run_id": run_id,
                    "arguments": arguments,
                })
                raise ToolConfirmationRequired(
                    canonical_name, token, self.confirmations.ttl_seconds
                )

        started_at = perf_counter()
        self.audit_log.record({
            "event": "tool_call",
            "invocation_id": invocation_id,
            "tool": canonical_name,
            "requested_tool": requested_name,
            "category": spec.category.value,
            "status": "started",
            "actor": actor,
            "run_id": run_id,
            "confirmed": bool(spec.requires_confirmation),
            "arguments": arguments,
        })
        try:
            result = spec.run(**arguments) or {}
        except Exception as exc:
            self.audit_log.record({
                "event": "tool_call",
                "invocation_id": invocation_id,
                "tool": canonical_name,
                "category": spec.category.value,
                "status": "failed",
                "actor": actor,
                "run_id": run_id,
                "duration_ms": round((perf_counter() - started_at) * 1000),
                "error": str(exc),
            })
            raise

        self.audit_log.record({
            "event": "tool_call",
            "invocation_id": invocation_id,
            "tool": canonical_name,
            "category": spec.category.value,
            "status": "succeeded",
            "actor": actor,
            "run_id": run_id,
            "duration_ms": round((perf_counter() - started_at) * 1000),
            "result": result,
        })
        return result

    invoke = execute
