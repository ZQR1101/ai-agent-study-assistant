from __future__ import annotations

from pathlib import Path
import threading
from time import monotonic
from typing import Any

from backend.config import get_config


_reranker_model = None
_reranker_model_name: str | None = None
_reranker_model_error: str | None = None
_reranker_last_failure_at: float | None = None
_reranker_lock = threading.Lock()
RERANKER_RETRY_COOLDOWN_SECONDS = 30.0


def _get_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder


def _is_path_like(model_name: str) -> bool:
    model_path = Path(model_name)
    return (
        model_path.is_absolute()
        or model_name.startswith((".", "/", "\\"))
        or "\\" in model_name
    )


def is_reranker_enabled(requested: bool | None = None) -> bool:
    configured = get_config().enable_reranker
    return configured if requested is None else configured and bool(requested)


def _retry_cooldown_active(model_name: str) -> bool:
    return bool(
        _reranker_model_name == model_name
        and _reranker_model is None
        and _reranker_model_error
        and _reranker_last_failure_at is not None
        and monotonic() - _reranker_last_failure_at < RERANKER_RETRY_COOLDOWN_SECONDS
    )


def get_reranker_model():
    global _reranker_model, _reranker_model_name, _reranker_model_error
    global _reranker_last_failure_at

    config = get_config()
    model_name = config.reranker_model
    if not config.enable_reranker or not model_name:
        return None

    if _reranker_model_name == model_name and _reranker_model is not None:
        return _reranker_model
    if _retry_cooldown_active(model_name):
        return None

    with _reranker_lock:
        if _reranker_model_name == model_name and _reranker_model is not None:
            return _reranker_model
        if _retry_cooldown_active(model_name):
            return None

        _reranker_model = None
        _reranker_model_name = model_name
        _reranker_model_error = None
        _reranker_last_failure_at = None
        try:
            model_path = Path(model_name)
            if _is_path_like(model_name) and not model_path.exists():
                raise FileNotFoundError(f"Reranker model path not found: {model_name}")

            CrossEncoder = _get_cross_encoder()
            _reranker_model = CrossEncoder(model_name)
        except Exception as exc:
            _reranker_model_error = str(exc)
            _reranker_last_failure_at = monotonic()

    return _reranker_model


def get_reranker_error() -> str | None:
    return _reranker_model_error


def get_reranker_settings() -> dict[str, Any]:
    config = get_config()
    return {
        "reranker_model": config.reranker_model or None,
        "reranker_top_n": config.reranker_top_n,
    }


def _reranker_document_text(chunk: dict) -> str:
    metadata_parts = [
        str(chunk.get("document_title") or ""),
        str(chunk.get("section") or ""),
        str(chunk.get("title") or ""),
        " ".join(str(item) for item in chunk.get("headings", []) if str(item).strip()),
    ]
    metadata_text = "\n".join(part for part in metadata_parts if part.strip())
    text = str(chunk.get("text") or "")
    return f"{metadata_text}\n{text}".strip()


def rerank_chunks_with_metadata(
    query: str,
    chunks: list[dict],
    top_k: int,
    *,
    enabled: bool | None = None,
) -> dict:
    config = get_config()
    requested = config.enable_reranker if enabled is None else bool(enabled)
    fallback_chunks = list(chunks[:top_k])
    metadata = {
        "chunks": fallback_chunks,
        "reranker_enabled": requested,
        "reranker_used": False,
        "reranker_model": config.reranker_model or None,
        "reranker_top_n": config.reranker_top_n,
        "reranker_error": None,
    }

    if not requested:
        return metadata
    if not config.enable_reranker:
        metadata["reranker_error"] = "Reranker is disabled by ENABLE_RERANKER"
        return metadata
    if not config.reranker_model:
        metadata["reranker_error"] = "RERANKER_MODEL is empty"
        return metadata
    if not chunks:
        return metadata

    model = get_reranker_model()
    if model is None:
        metadata["reranker_error"] = get_reranker_error() or "Reranker model is unavailable"
        return metadata

    try:
        pairs = [(query, _reranker_document_text(chunk)) for chunk in chunks]
        scores = model.predict(pairs, show_progress_bar=False)
        if len(scores) != len(chunks):
            raise ValueError("Reranker returned an unexpected number of scores")

        ranked = []
        for chunk, score in zip(chunks, scores):
            ranked.append({
                **chunk,
                "rerank_score": float(score),
                "reranker_used": True,
            })
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        for rank, chunk in enumerate(ranked, start=1):
            chunk["rerank_rank"] = rank

        metadata["chunks"] = ranked[:top_k]
        metadata["reranker_used"] = True
        return metadata
    except Exception as exc:
        metadata["reranker_error"] = str(exc)
        return metadata


def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    return rerank_chunks_with_metadata(query, chunks, top_k)["chunks"]


def _reset_reranker_state() -> None:
    global _reranker_model, _reranker_model_name, _reranker_model_error
    global _reranker_last_failure_at

    with _reranker_lock:
        _reranker_model = None
        _reranker_model_name = None
        _reranker_model_error = None
        _reranker_last_failure_at = None
