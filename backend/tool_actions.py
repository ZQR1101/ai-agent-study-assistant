"""Persistence and dangerous actions exposed by the tool registry."""

from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from backend.config import read_obsidian_vault_path


PROJECT_ROOT = Path(__file__).parent.parent
SAVED_ITEMS_DIR = Path(os.getenv("SAVED_ITEMS_DIR", PROJECT_ROOT / "data" / "saved_items"))
_store_lock = threading.Lock()
_COLLECTIONS = {"notes", "flashcards", "quizzes"}

# Obsidian vault integration
OBSIDIAN_VAULT_PATH = read_obsidian_vault_path()
VAULT_NOTES_DIR = "AI Study Assistant"


def _sanitize_filename(name: str) -> str:
    """Sanitize a string into a safe Obsidian filename."""
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    return (name.strip("._") or "untitled")[:80]


def _vault_note_path(title: str) -> Path | None:
    """Generate a unique note path in the vault (always creates new file)."""
    if OBSIDIAN_VAULT_PATH is None or not OBSIDIAN_VAULT_PATH.is_dir():
        return None

    vault_root = OBSIDIAN_VAULT_PATH.resolve()
    notes_dir = vault_root / VAULT_NOTES_DIR
    safe_title = _sanitize_filename(title)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = notes_dir / f"{safe_title}_{timestamp}.md"
    suffix = 2
    while path.exists():
        path = notes_dir / f"{safe_title}_{timestamp}_{suffix}.md"
        suffix += 1

    resolved = path.resolve()
    if not resolved.is_relative_to(vault_root):
        raise ValueError("Refusing to write outside the Obsidian vault")
    return path


def _yaml_scalar(value: str) -> str:
    return json.dumps(str(value), ensure_ascii=False)


def _write_vault_note(
    title: str,
    content: str,
    tags: list[str] | None = None,
) -> Path | None:
    """Write a note to the Obsidian vault with YAML frontmatter."""
    vault_path = _vault_note_path(title)
    if vault_path is None:
        return None

    vault_path.parent.mkdir(parents=True, exist_ok=True)

    frontmatter_lines = [
        "---",
        f"title: {_yaml_scalar(title)}",
        f"created: {_yaml_scalar(datetime.now(timezone.utc).isoformat())}",
        "source: ai-study-assistant",
    ]
    if tags:
        rendered = ", ".join(_yaml_scalar(tag) for tag in tags)
        frontmatter_lines.append(f"tags: [{rendered}]")

    frontmatter = "\n".join(frontmatter_lines) + "\n---\n"
    vault_path.write_text(frontmatter + content.strip() + "\n", encoding="utf-8")
    return vault_path


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

    # Save to JSON store (existing behavior)
    result = _save("notes", {"title": title.strip(), "content": body, "tags": tags or []})

    # Write to Obsidian vault (if configured)
    vault_path = _write_vault_note(title or "untitled", body, tags)
    if vault_path:
        result["answer"] += f"\nObsidian: {vault_path.relative_to(OBSIDIAN_VAULT_PATH).as_posix()}"

    return result


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
