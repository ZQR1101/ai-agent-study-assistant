"""Three-layer memory system: L1 (events) → L2 (facts) → L3 (profile).

This module provides a human-editable, traceable memory system without
hidden vector stores:

- L1: append-only JSONL raw events (session/run-scoped)
- L2: editable facts, each referencing at least one L1 event
- L3: synthesized learning profile, each insight referencing at least one L2 fact

Design principles:
- No embeddings — plain JSON/JSONL for readability and manual editing
- Traceability: L3 → L2 → L1 references are mandatory
- Human-in-the-loop: L2 facts can be edited at any time
- Separation: events are never modified after append
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

# Re-export shared models
from backend.memory.models import (
    L1Event,
    L1EventType,
    L2Category,
    L2Fact,
    L3Insight,
    L3InsightType,
    L3Profile,
    L3Summary,
)

# Re-export stores
from backend.memory.l1_store import L1Store
from backend.memory.l2_store import L2Store
from backend.memory.l3_profile import L3Store

# Re-export engine
from backend.memory.engine import MemoryEngine

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_memory_engine: MemoryEngine | None = None
_memory_engine_lock = threading.Lock()


def get_memory_engine() -> MemoryEngine:
    """Get or create the singleton MemoryEngine instance.

    The engine is process-global: all three stores are shared across
    requests within the same process.
    """
    global _memory_engine
    if _memory_engine is None:
        with _memory_engine_lock:
            if _memory_engine is None:
                _memory_engine = MemoryEngine()
    return _memory_engine


def reset_memory_engine() -> None:
    """Reset the singleton — primarily for tests."""
    global _memory_engine
    with _memory_engine_lock:
        _memory_engine = None
