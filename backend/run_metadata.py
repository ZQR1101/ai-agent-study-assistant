"""Build stable per-response runtime metadata for API and UI consumers."""

from __future__ import annotations

from typing import Any


def _as_dict(value: Any) -> dict:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def build_run_metadata(result: dict, *, duration_ms: int | None = None) -> tuple[dict, dict]:
    runtime_info = _as_dict(result.get("runtime_info"))
    tool_calls = _as_list(runtime_info.get("tool_calls"))
    plan = _as_list(result.get("plan"))
    actual_tools = [call for call in tool_calls if call.get("tool") != "planner"]
    failed_tools = [call for call in actual_tools if call.get("success") is False]
    runtime_error = runtime_info.get("error")

    if runtime_error or result.get("mode") == "error":
        status = "failed"
    elif failed_tools:
        status = "partial"
    else:
        status = "succeeded"

    summary = {
        "status": status,
        "runtime": runtime_info.get("runtime") or result.get("mode") or "chat",
        "mode": result.get("mode") or "",
        "planner_mode": runtime_info.get("planner_mode"),
        "duration_ms": duration_ms,
        "step_count": len(tool_calls),
        "tool_count": len(actual_tools),
        "successful_tool_count": sum(call.get("success") is not False for call in actual_tools),
        "source_count": len(_as_list(result.get("sources"))),
        "token_usage": _as_dict(runtime_info.get("token_usage")),
        "estimated_cost": _as_dict(runtime_info.get("estimated_cost")),
    }
    details = {
        "plan": plan,
        "tools": tool_calls,
    }
    return summary, details
