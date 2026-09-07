"""L2 Store — editable fact storage that must reference L1 events."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.memory.models import L2Category, L2Fact


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class L2Store:
    """Editable fact store backed by one JSON file per fact.

    Each fact MUST reference at least one L1 event_id.
    Facts are stored in individual JSON files for easy human editing.
    """

    def __init__(self, root: str | Path | None = None):
        from backend.config import get_config

        project_root = Path(__file__).parent.parent.parent
        config = get_config()
        self.root = Path(root or config.project_root / "data" / "memory" / "l2_facts")
        self._lock = threading.RLock()

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------

    def _path(self, fact_id: str) -> Path:
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in fact_id)
        return self.root / f"{safe_id}.json"

    def _write(self, fact: L2Fact) -> L2Fact:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(fact.fact_id)
        content = fact.model_dump_json(indent=2, ensure_ascii=False)
        path.write_text(content, encoding="utf-8")
        return fact

    def add_fact(
        self,
        content: str,
        category: L2Category,
        l1_refs: list[str],
        subject: str = "",
        confidence: float = 0.5,
    ) -> L2Fact:
        """Create and persist a new fact, referencing one or more L1 events.

        Args:
            content: Human-readable fact content.
            category: The category of the fact.
            l1_refs: List of L1 event_ids this fact is derived from.
            subject: Optional subject (e.g., topic name).
            confidence: 0.0-1.0 confidence score.

        Returns:
            The persisted L2Fact.

        Raises:
            ValueError: If l1_refs is empty.
        """
        if not l1_refs:
            raise ValueError("L2 fact must reference at least one L1 event")

        fact = L2Fact(
            category=category,
            subject=subject,
            content=content,
            confidence=confidence,
            l1_refs=list(l1_refs),
        )
        with self._lock:
            return self._write(fact)

    def update_fact(
        self,
        fact_id: str,
        content: str | None = None,
        category: L2Category | None = None,
        subject: str | None = None,
        confidence: float | None = None,
    ) -> L2Fact | None:
        """Update an existing fact's content and/or metadata.

        Preserves l1_refs (L1→L2 relationship is immutable).
        Updates the updated_at timestamp.
        """
        with self._lock:
            path = self._path(fact_id)
            if not path.exists():
                return None

            raw = path.read_text(encoding="utf-8")
            fact = L2Fact.model_validate_json(raw)

            if content is not None:
                fact.content = content
            if category is not None:
                fact.category = category
            if subject is not None:
                fact.subject = subject
            if confidence is not None:
                fact.confidence = max(0.0, min(1.0, confidence))

            fact.updated_at = _utc_now()
            return self._write(fact)

    def delete_fact(self, fact_id: str) -> bool:
        """Delete a fact by ID. Returns True if deleted, False if not found."""
        with self._lock:
            path = self._path(fact_id)
            if not path.exists():
                return False
            path.unlink(missing_ok=True)
            return True

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------

    def get(self, fact_id: str) -> L2Fact | None:
        """Get a fact by ID."""
        with self._lock:
            path = self._path(fact_id)
            if not path.exists():
                return None
            raw = path.read_text(encoding="utf-8")
            return L2Fact.model_validate_json(raw)

    def query(
        self,
        category: L2Category | None = None,
        subject: str | None = None,
    ) -> list[L2Fact]:
        """Query facts, optionally filtered by category and/or subject substring."""
        results: list[L2Fact] = []
        for path in self.root.glob("*.json"):
            try:
                raw = path.read_text(encoding="utf-8")
                fact = L2Fact.model_validate_json(raw)
            except (json.JSONDecodeError, ValueError, OSError):
                continue

            if category and fact.category != category:
                continue
            if subject and subject.lower() not in fact.subject.lower():
                continue

            results.append(fact)

        # Sort by updated_at descending
        results.sort(key=lambda f: f.updated_at, reverse=True)
        return results

    def get_all(self) -> list[L2Fact]:
        """Return all facts sorted by updated_at descending."""
        return self.query()

    def count(self) -> int:
        """Total number of facts."""
        return len(list(self.root.glob("*.json")))

    def get_by_l1_ref(self, event_id: str) -> list[L2Fact]:
        """Find all L2 facts that reference a given L1 event."""
        return [
            fact
            for fact in self.query()
            if event_id in fact.l1_refs
        ]

    def get_mastery_facts(self, subject: str | None = None) -> list[L2Fact]:
        """Get all mastery facts, optionally filtered by subject."""
        return self.query(category="mastery", subject=subject)

    def get_preference_facts(self) -> list[L2Fact]:
        """Get all learning preference facts."""
        return self.query(category="preference")

    def get_learning_goals(self) -> list[L2Fact]:
        """Get all active learning goals."""
        return self.query(category="learning_goal")

    def get_knowledge_gaps(self) -> list[L2Fact]:
        """Get all identified knowledge gaps."""
        return self.query(category="knowledge_gap")

    # -------------------------------------------------------------------------
    # L1 reference tracking
    # -------------------------------------------------------------------------

    def get_all_l1_refs(self) -> set[str]:
        """Collect all L1 event IDs referenced by any L2 fact."""
        refs: set[str] = set()
        for fact in self.query():
            refs.update(fact.l1_refs)
        return refs

    def get_unreferenced_l1_ids(
        self,
        known_l1_ids: set[str],
    ) -> list[str]:
        """Return L1 event IDs that have not yet been turned into facts."""
        referenced = self.get_all_l1_refs()
        return sorted(known_l1_ids - referenced)
