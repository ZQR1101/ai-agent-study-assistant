"""Persistent human-in-the-loop actions proposed by the study agent."""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


PendingActionStatus = Literal[
    "pending",
    "approved",
    "executing",
    "executed",
    "rejected",
    "expired",
    "failed",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime) -> str:
    return value.isoformat()


def _model_dump(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


ACTION_PRESENTATION = {
    "delete_saved_item": {
        "summary": "删除一条已保存的学习内容",
        "impact": "目标笔记、卡片集或练习题会被永久删除。",
        "reversible": False,
    },
    "delete_run": {
        "summary": "删除一条 Run 记录",
        "impact": "Run 会被软删除并保留审计记录。",
        "reversible": True,
    },
    "delete_knowledge_file": {
        "summary": "删除知识库文件",
        "impact": "目标文件会从本地知识库永久删除，现有索引不会自动重建。",
        "reversible": False,
    },
    "reset_saved_items": {
        "summary": "清空已保存的学习数据",
        "impact": "指定集合中的学习数据会被永久清空。",
        "reversible": False,
    },
    "reset_rag_index": {
        "summary": "重置 RAG 索引",
        "impact": "当前持久化索引和内存索引会被删除，需要重建后才能检索。",
        "reversible": False,
    },
    "rebuild_rag_index": {
        "summary": "重建 RAG 索引",
        "impact": "当前索引会被知识库文件的最新内容替换，执行可能耗时。",
        "reversible": False,
    },
}

SUPPORTED_PENDING_ACTION_TOOLS = frozenset(ACTION_PRESENTATION)


class PendingAction(BaseModel):
    id: str
    run_id: str
    session_id: str | None = None
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    arguments_digest: str
    summary: str
    impact: str
    reversible: bool = False
    risk_level: Literal["high"] = "high"
    status: PendingActionStatus = "pending"
    requested_by: str = "agent"
    request_message: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    decision_reason: str | None = None
    created_at: str
    updated_at: str
    expires_at: str
    decided_at: str | None = None
    executed_at: str | None = None
    version: int = 1


def normalize_action_arguments(
    tool_name: str,
    arguments: dict[str, Any] | None,
    tool_input: str = "",
) -> dict[str, Any]:
    """Keep only supported arguments and infer obvious values from the request."""
    supplied = dict(arguments or {})
    text = str(tool_input or "").strip()

    if tool_name == "delete_saved_item":
        collection = supplied.get("collection") or _infer_collection(text)
        item_id = str(supplied.get("item_id") or "").strip()
        if not collection or not item_id:
            raise ValueError("请提供要删除内容的 collection 和 item_id。")
        return {"collection": collection, "item_id": item_id}

    if tool_name == "delete_run":
        target_run_id = str(supplied.get("target_run_id") or "").strip()
        if not target_run_id:
            match = re.search(r"\b[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b", text)
            target_run_id = match.group(0) if match else ""
        if not target_run_id:
            raise ValueError("请提供要删除的 Run ID。")
        return {"target_run_id": target_run_id}

    if tool_name == "delete_knowledge_file":
        filename = str(supplied.get("filename") or "").strip()
        if not filename:
            match = re.search(r"([^\\/:*?\"<>|\s]+\.(?:md|txt|pdf))", text, re.IGNORECASE)
            filename = match.group(1) if match else ""
        if not filename:
            raise ValueError("请提供要删除的知识库文件名。")
        return {"filename": filename}

    if tool_name == "reset_saved_items":
        collection = supplied.get("collection") or _infer_collection(text)
        if collection:
            return {"collection": collection}
        return {}

    if tool_name in {"reset_rag_index", "rebuild_rag_index"}:
        return {}

    raise ValueError(f"Unsupported pending-action tool: {tool_name}")


def _infer_collection(text: str) -> str | None:
    lowered = text.lower()
    if any(marker in lowered for marker in ("note", "笔记")):
        return "notes"
    if any(marker in lowered for marker in ("flashcard", "card", "卡片")):
        return "flashcards"
    if any(marker in lowered for marker in ("quiz", "练习", "题库", "测验")):
        return "quizzes"
    return None


class PendingActionRepository:
    """Filesystem-backed repository; one JSON document per pending action."""

    _UPDATABLE_FIELDS = {
        "status",
        "result",
        "error",
        "decision_reason",
        "decided_at",
        "executed_at",
    }

    def __init__(self, root: str | Path | None = None, *, ttl_seconds: int | None = None):
        project_root = Path(__file__).parent.parent
        self.root = Path(root or os.getenv("PENDING_ACTIONS_DIR", project_root / "data" / "pending_actions"))
        self.ttl_seconds = int(ttl_seconds or os.getenv("PENDING_ACTION_TTL_SECONDS", "300"))
        self._lock = threading.RLock()

    def create(
        self,
        *,
        run_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str | None = None,
        request_message: str = "",
        requested_by: str = "agent",
    ) -> PendingAction:
        presentation = ACTION_PRESENTATION.get(tool_name)
        if presentation is None:
            raise ValueError(f"Unsupported pending-action tool: {tool_name}")
        now = _utc_now()
        canonical_arguments = json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        action = PendingAction(
            id=str(uuid4()),
            run_id=run_id,
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            arguments_digest=hashlib.sha256(
                f"{tool_name}:{canonical_arguments}".encode("utf-8")
            ).hexdigest(),
            **presentation,
            requested_by=requested_by,
            request_message=request_message,
            created_at=_iso(now),
            updated_at=_iso(now),
            expires_at=_iso(now + timedelta(seconds=self.ttl_seconds)),
        )
        with self._lock:
            self._write(action)
        return action

    def get(self, action_id: str, *, expire: bool = True) -> PendingAction | None:
        with self._lock:
            path = self._path(action_id)
            if not path.exists():
                return None
            action = self._read(path)
            if expire:
                action = self._expire_if_needed(action)
            return action

    def list(
        self,
        *,
        status: str | None = None,
        run_id: str | None = None,
        limit: int = 100,
    ) -> list[PendingAction]:
        safe_limit = max(1, min(int(limit), 500))
        with self._lock:
            if not self.root.exists():
                return []
            actions = []
            for path in self.root.glob("*.json"):
                try:
                    action = self._expire_if_needed(self._read(path))
                except (OSError, ValueError):
                    continue
                if status and action.status != status:
                    continue
                if run_id and action.run_id != run_id:
                    continue
                actions.append(action)
            actions.sort(key=lambda item: item.created_at, reverse=True)
            return actions[:safe_limit]

    def transition(
        self,
        action_id: str,
        *,
        expected: set[str],
        status: PendingActionStatus,
        **changes: Any,
    ) -> PendingAction:
        unknown = set(changes) - self._UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"Unsupported PendingAction field(s): {', '.join(sorted(unknown))}")
        with self._lock:
            action = self.get(action_id)
            if action is None:
                raise KeyError(f"Pending action not found: {action_id}")
            if action.status not in expected:
                raise ValueError(f"Pending action is already {action.status}")
            payload = _model_dump(action)
            payload.update(changes)
            payload["status"] = status
            payload["updated_at"] = _iso(_utc_now())
            payload["version"] = action.version + 1
            updated = PendingAction(**payload)
            self._write(updated)
            return updated

    def _expire_if_needed(self, action: PendingAction) -> PendingAction:
        if action.status != "pending":
            return action
        expires_at = datetime.fromisoformat(action.expires_at)
        if expires_at > _utc_now():
            return action
        payload = _model_dump(action)
        payload.update(
            status="expired",
            updated_at=_iso(_utc_now()),
            decided_at=_iso(_utc_now()),
            version=action.version + 1,
        )
        expired = PendingAction(**payload)
        self._write(expired)
        return expired

    def _path(self, action_id: str) -> Path:
        safe_id = str(action_id).strip()
        if not safe_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in safe_id):
            raise ValueError("Invalid pending action id")
        return self.root / f"{safe_id}.json"

    def _read(self, path: Path) -> PendingAction:
        text = path.read_text(encoding="utf-8")
        if hasattr(PendingAction, "model_validate_json"):
            return PendingAction.model_validate_json(text)
        return PendingAction.parse_raw(text)

    def _write(self, action: PendingAction) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(action.id)
        temporary = path.with_suffix(".tmp")
        if hasattr(action, "model_dump_json"):
            content = action.model_dump_json(indent=2)
        else:
            content = action.json(indent=2, ensure_ascii=False)
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)


_repository: PendingActionRepository | None = None
_repository_lock = threading.Lock()


def get_pending_action_repository() -> PendingActionRepository:
    global _repository
    if _repository is None:
        with _repository_lock:
            if _repository is None:
                _repository = PendingActionRepository()
    return _repository


def set_pending_action_repository(repository: PendingActionRepository | None) -> None:
    global _repository
    with _repository_lock:
        _repository = repository
