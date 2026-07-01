"""Persistence and dangerous actions exposed by the tool registry."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


PROJECT_ROOT = Path(__file__).parent.parent
SAVED_ITEMS_DIR = Path(os.getenv("SAVED_ITEMS_DIR", PROJECT_ROOT / "data" / "saved_items"))
_store_lock = threading.Lock()
_COLLECTIONS = {"notes", "flashcards", "quizzes"}


def _collection_path(collection: str) -> Path:
    if collection not in _COLLECTIONS:
        raise ValueError(f"Unsupported collection: {collection}")
    return SAVED_ITEMS_DIR / f"{collection}.json"


def _read_collection(collection: str) -> list[dict]:
    path = _collection_path(collection)
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Invalid saved-item store: {path}")
    return data


def _write_collection(collection: str, items: list[dict]) -> None:
    path = _collection_path(collection)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _save(collection: str, payload: dict) -> dict:
    item = {
        "id": str(uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    with _store_lock:
        items = _read_collection(collection)
        items.append(item)
        _write_collection(collection, items)
    return {
        "answer": f"Saved {collection[:-1]} {item['id']}",
        "saved_item": item,
        "collection": collection,
    }


def save_note(
    step_input: str = "",
    *,
    title: str = "",
    content: str = "",
    tags: list[str] | None = None,
    **_: Any,
) -> dict:
    body = str(content or step_input).strip()
    if not body:
        raise ValueError("Note content cannot be empty")
    return _save("notes", {"title": title.strip(), "content": body, "tags": tags or []})


def save_flashcards(
    step_input: str = "",
    *,
    flashcards: list[dict] | None = None,
    title: str = "",
    **_: Any,
) -> dict:
    cards = flashcards or []
    if not cards and step_input:
        cards = [{"front": step_input, "back": ""}]
    if not cards:
        raise ValueError("At least one flashcard is required")
    return _save("flashcards", {"title": title.strip(), "cards": cards})


def save_quiz(
    step_input: str = "",
    *,
    title: str = "",
    questions: list[dict] | None = None,
    content: str = "",
    **_: Any,
) -> dict:
    quiz_content = str(content or step_input).strip()
    if not questions and not quiz_content:
        raise ValueError("Quiz content or questions are required")
    return _save(
        "quizzes",
        {"title": title.strip(), "content": quiz_content, "questions": questions or []},
    )


def delete_saved_item(*, collection: str, item_id: str, **_: Any) -> dict:
    with _store_lock:
        items = _read_collection(collection)
        kept = [item for item in items if item.get("id") != item_id]
        if len(kept) == len(items):
            raise KeyError(f"Saved item not found: {item_id}")
        _write_collection(collection, kept)
    return {"answer": f"Deleted {item_id}", "deleted": True, "id": item_id}


def delete_run(*, target_run_id: str, **_: Any) -> dict:
    from backend.run_repository import get_run_repository

    if not get_run_repository().delete_run(target_run_id):
        raise KeyError(f"Run not found: {target_run_id}")
    return {
        "answer": f"Soft-deleted run {target_run_id}",
        "deleted": True,
        "soft_deleted": True,
        "run_id": target_run_id,
    }


def reset_saved_items(*, collection: str | None = None, **_: Any) -> dict:
    collections = [collection] if collection else sorted(_COLLECTIONS)
    counts = {}
    with _store_lock:
        for name in collections:
            items = _read_collection(name)
            counts[name] = len(items)
            _write_collection(name, [])
    return {"answer": "Saved study data reset", "deleted_counts": counts}


def delete_knowledge_file(*, filename: str, **_: Any) -> dict:
    docs_path = (PROJECT_ROOT / "docs").resolve()
    target = (docs_path / Path(filename).name).resolve()
    if target.parent != docs_path or target.suffix.lower() not in {".md", ".txt", ".pdf"}:
        raise ValueError("Unsupported knowledge file path")
    if not target.exists():
        raise FileNotFoundError(filename)
    target.unlink()
    return {"answer": f"Deleted knowledge file {target.name}", "deleted": target.name}


def rebuild_rag_index_tool(**_: Any) -> dict:
    from backend.rag_store import get_rag_index_status, rebuild_rag_index

    rebuild_rag_index()
    return {"answer": "RAG index rebuilt", "rag_index": get_rag_index_status()}


def reset_rag_index(**_: Any) -> dict:
    from backend import rag_store

    with rag_store._rag_index_lock:
        rag_store.index = None
        rag_store.chunks = []
        for path in (rag_store.INDEX_FILE, rag_store.CHUNKS_FILE):
            if path.exists():
                path.unlink()
    rag_store._reset_bm25_index()
    return {"answer": "RAG index reset", "reset": True}


_BLOCKED_AST_NODES = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
    ast.With,
    ast.AsyncWith,
)
_BLOCKED_CALLS = {
    "__import__",
    "breakpoint",
    "compile",
    "delattr",
    "dir",
    "eval",
    "exec",
    "getattr",
    "globals",
    "input",
    "locals",
    "memoryview",
    "open",
    "setattr",
    "type",
    "vars",
}


def _validate_sandbox_code(code: str) -> None:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, _BLOCKED_AST_NODES):
            raise ValueError(f"Sandbox code cannot use {type(node).__name__}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BLOCKED_CALLS:
                raise ValueError(f"Sandbox code cannot call {node.func.id}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("Sandbox code cannot access dunder names")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("Sandbox code cannot access dunder attributes")


def run_code_sandbox(
    step_input: str = "",
    *,
    code: str = "",
    timeout_seconds: int = 3,
    **_: Any,
) -> dict:
    source = str(code or step_input)
    if not source.strip():
        raise ValueError("Python code cannot be empty")
    if len(source) > 20_000:
        raise ValueError("Python code exceeds the 20 KB limit")
    _validate_sandbox_code(source)
    timeout = max(1, min(int(timeout_seconds), 10))
    environment = {"PYTHONIOENCODING": "utf-8", "PYTHONHASHSEED": "0"}
    with tempfile.TemporaryDirectory(prefix="study-sandbox-") as directory:
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", source],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "answer": "Code execution timed out",
                "stdout": (exc.stdout or "")[-20_000:],
                "stderr": (exc.stderr or "")[-20_000:],
                "exit_code": None,
                "timed_out": True,
            }
    return {
        "answer": "Code execution finished",
        "stdout": completed.stdout[-20_000:],
        "stderr": completed.stderr[-20_000:],
        "exit_code": completed.returncode,
        "timed_out": False,
    }
