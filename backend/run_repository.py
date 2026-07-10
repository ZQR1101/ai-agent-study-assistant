"""Persistent Run aggregate and its single repository boundary."""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


RunStatus = Literal[
    "created",
    "running",
    "awaiting_action",
    "completed",
    "partial",
    "failed",
    "deleted",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_dump(model: BaseModel) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model.dict()


class Run(BaseModel):
    id: str
    status: RunStatus = "created"
    session_id: str | None = None
    request: dict = Field(default_factory=dict)
    plan: list[dict] = Field(default_factory=list)
    tools: list[dict] = Field(default_factory=list)
    audit: list[dict] = Field(default_factory=list)
    artifacts: dict = Field(default_factory=dict)
    output: dict = Field(default_factory=dict)
    error: str | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    finished_at: str | None = None
    deleted_at: str | None = None
    version: int = 1


class RunRepository:
    """Filesystem-backed repository; one JSON document per Run aggregate."""

    _UPDATABLE_FIELDS = {
        "status",
        "session_id",
        "request",
        "plan",
        "tools",
        "audit",
        "artifacts",
        "output",
        "error",
        "metadata",
        "finished_at",
        "deleted_at",
    }

    def __init__(self, root: str | Path | None = None):
        project_root = Path(__file__).parent.parent
        self.root = Path(root or os.getenv("RUNS_DIR", project_root / "data" / "runs"))
        self._lock = threading.RLock()

    def create_run(
        self,
        *,
        request: dict | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        run_id: str | None = None,
    ) -> Run:
        run = Run(
            id=run_id or str(uuid4()),
            status="running",
            session_id=session_id,
            request=request or {},
            metadata=metadata or {},
        )
        with self._lock:
            path = self._path(run.id)
            if path.exists():
                raise ValueError(f"Run already exists: {run.id}")
            self._write(run)
        return run

    def update_run(self, run_id: str, **changes: Any) -> Run:
        unknown = set(changes) - self._UPDATABLE_FIELDS
        if unknown:
            raise ValueError(f"Unsupported Run field(s): {', '.join(sorted(unknown))}")
        with self._lock:
            run = self._require(run_id)
            payload = _model_dump(run)
            for field in ("metadata", "artifacts", "output"):
                if field in changes and isinstance(changes[field], dict):
                    changes[field] = {**payload.get(field, {}), **changes[field]}
            payload.update(changes)
            payload["updated_at"] = _utc_now()
            payload["version"] = run.version + 1
            updated = Run(**payload)
            self._write(updated)
            return updated

    def finish_run(
        self,
        run_id: str,
        *,
        output: dict | None = None,
        status: Literal["completed", "partial", "failed"] = "completed",
        error: str | None = None,
    ) -> Run:
        return self.update_run(
            run_id,
            status=status,
            output=output or {},
            error=error,
            finished_at=_utc_now(),
        )

    def get_run(self, run_id: str) -> Run | None:
        with self._lock:
            path = self._path(run_id)
            return self._read(path) if path.exists() else None

    def list_runs(
        self,
        *,
        limit: int = 50,
        status: str | None = None,
        session_id: str | None = None,
        include_deleted: bool = False,
    ) -> list[Run]:
        safe_limit = max(1, min(int(limit), 500))
        with self._lock:
            if not self.root.exists():
                return []
            runs = []
            for path in self.root.glob("*.json"):
                try:
                    run = self._read(path)
                except (OSError, ValueError):
                    continue
                if status and run.status != status:
                    continue
                if run.status == "deleted" and not include_deleted and status != "deleted":
                    continue
                if session_id and run.session_id != session_id:
                    continue
                runs.append(run)
            runs.sort(key=lambda item: item.created_at, reverse=True)
            return runs[:safe_limit]

    def delete_run(self, run_id: str) -> bool:
        """Soft-delete a Run while retaining a tombstone and deletion audit."""
        with self._lock:
            run = self.get_run(run_id)
            if run is None or run.status == "deleted":
                return False
            deleted_at = _utc_now()
            audit = [
                *run.audit,
                {
                    "timestamp": deleted_at,
                    "event": "run_deleted",
                    "tool": "delete_run",
                    "status": "succeeded",
                    "actor": "run_repository",
                    "run_id": run_id,
                },
            ]
            self.update_run(
                run_id,
                status="deleted",
                deleted_at=deleted_at,
                audit=audit,
            )
            return True

    def append_audit(self, run_id: str, event: dict) -> Run:
        with self._lock:
            run = self._require(run_id)
            audit = [*run.audit, event]
            return self.update_run(run_id, audit=audit)

    def _path(self, run_id: str) -> Path:
        safe_id = str(run_id).strip()
        if not safe_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in safe_id):
            raise ValueError("Invalid run id")
        return self.root / f"{safe_id}.json"

    def _require(self, run_id: str) -> Run:
        path = self._path(run_id)
        if not path.exists():
            raise KeyError(f"Run not found: {run_id}")
        return self._read(path)

    def _read(self, path: Path) -> Run:
        text = path.read_text(encoding="utf-8")
        if hasattr(Run, "model_validate_json"):
            return Run.model_validate_json(text)
        return Run.parse_raw(text)

    def _write(self, run: Run) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(run.id)
        temporary = path.with_suffix(".tmp")
        if hasattr(run, "model_dump_json"):
            content = run.model_dump_json(indent=2)
        else:
            content = run.json(indent=2, ensure_ascii=False)
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(path)


_repository: RunRepository | None = None
_repository_lock = threading.Lock()


def get_run_repository() -> RunRepository:
    global _repository
    if _repository is None:
        with _repository_lock:
            if _repository is None:
                _repository = RunRepository()
    return _repository


def set_run_repository(repository: RunRepository | None) -> None:
    """Override/reset the process repository, primarily for tests."""
    global _repository
    with _repository_lock:
        _repository = repository
