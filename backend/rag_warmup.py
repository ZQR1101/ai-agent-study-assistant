from __future__ import annotations

from datetime import datetime, timezone
import re
import threading
import time
from typing import Any


_STATUS_IDLE = {
    "status": "idle",
    "model_loaded": False,
    "index_loaded": False,
    "started_at": None,
    "finished_at": None,
    "elapsed_seconds": None,
    "error": None,
}

_status_lock = threading.Lock()
_run_lock = threading.Lock()
_warmup_status = dict(_STATUS_IDLE)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_error_message(error: BaseException) -> str:
    message = str(error) or error.__class__.__name__
    message = re.sub(
        r"(?i)(api[_-]?key|token|password|secret)\s*=\s*[^,\s]+",
        r"\1=<redacted>",
        message,
    )
    return message[:1000]


def get_rag_warmup_status() -> dict[str, Any]:
    with _status_lock:
        return dict(_warmup_status)


def _replace_status(**updates: Any) -> dict[str, Any]:
    with _status_lock:
        _warmup_status.update(updates)
        return dict(_warmup_status)


def _load_embedding_model() -> object:
    from backend.rag_store import get_embedding_model

    return get_embedding_model()


def _load_existing_rag_index() -> bool:
    from backend import rag_store

    if not rag_store.INDEX_FILE.exists() or not rag_store.CHUNKS_FILE.exists():
        return False

    return rag_store.load_rag_index()


def run_rag_warmup(load_index: bool = True) -> None:
    if not _run_lock.acquire(blocking=False):
        return

    started = time.monotonic()
    try:
        _replace_status(
            status="loading",
            model_loaded=False,
            index_loaded=False,
            started_at=_utc_now_iso(),
            finished_at=None,
            elapsed_seconds=None,
            error=None,
        )

        _load_embedding_model()
        _replace_status(model_loaded=True)

        index_loaded = False
        index_error = None
        if load_index:
            try:
                index_loaded = _load_existing_rag_index()
            except Exception as exc:
                index_error = f"Index warmup failed: {_safe_error_message(exc)}"

        _replace_status(
            status="ready",
            index_loaded=index_loaded,
            finished_at=_utc_now_iso(),
            elapsed_seconds=time.monotonic() - started,
            error=index_error,
        )
    except Exception as exc:
        _replace_status(
            status="error",
            model_loaded=False,
            index_loaded=False,
            finished_at=_utc_now_iso(),
            elapsed_seconds=time.monotonic() - started,
            error=_safe_error_message(exc),
        )
    finally:
        _run_lock.release()


def start_rag_warmup(load_index: bool = True) -> dict[str, Any]:
    with _status_lock:
        if _warmup_status["status"] in {"loading", "ready"}:
            return {"started": False, "warmup": dict(_warmup_status)}

        _warmup_status.update(
            status="loading",
            model_loaded=False,
            index_loaded=False,
            started_at=_utc_now_iso(),
            finished_at=None,
            elapsed_seconds=None,
            error=None,
        )

    try:
        thread = threading.Thread(
            target=run_rag_warmup,
            kwargs={"load_index": load_index},
            daemon=True,
        )
        thread.start()
    except Exception as exc:
        _replace_status(
            status="error",
            model_loaded=False,
            index_loaded=False,
            finished_at=_utc_now_iso(),
            elapsed_seconds=0.0,
            error=_safe_error_message(exc),
        )
        return {"started": False, "warmup": get_rag_warmup_status()}

    return {"started": True, "warmup": get_rag_warmup_status()}


def _reset_rag_warmup_status_for_tests() -> None:
    with _status_lock:
        _warmup_status.clear()
        _warmup_status.update(_STATUS_IDLE)
