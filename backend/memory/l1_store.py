"""L1 Store — append-only JSONL event storage."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from backend.memory.models import L1Event, L1EventType


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class L1Store:
    """Append-only event store backed by session-scoped JSONL files.

    Each session gets its own JSONL file: {session_id}_{timestamp}.jsonl.
    Events are never deleted or modified after append.
    """

    def __init__(self, root: str | Path | None = None):
        from backend.config import get_config

        config = get_config()
        self.root = Path(root or config.project_root / "data" / "memory" / "l1_events")
        self._lock = threading.RLock()

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------

    def append(self, event: L1Event) -> L1Event:
        """Append an event to the session's JSONL file.

        Creates the file and parent directories if needed.
        """
        with self._lock:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self._session_path(event.session_id)
            line = event.to_jsonl_line()
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        return event

    def record(
        self,
        session_id: str,
        run_id: str | None,
        event_type: L1EventType,
        data: dict[str, Any] | None = None,
    ) -> L1Event:
        """Convenience: create and append an event in one call."""
        event = L1Event(
            session_id=session_id,
            run_id=run_id,
            event_type=event_type,
            data=data or {},
        )
        return self.append(event)

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------

    def _session_path(self, session_id: str) -> Path:
        # Sanitize session_id to avoid path traversal
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self.root / f"{safe_id}.jsonl"

    def _parse_line(self, line: str) -> L1Event | None:
        try:
            data = json.loads(line)
            return L1Event(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            return None

    def _iter_lines(self, path: Path) -> Iterator[str]:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

    def query(
        self,
        session_id: str | None = None,
        event_type: L1EventType | None = None,
        since: str | None = None,
        limit: int | None = None,
    ) -> list[L1Event]:
        """Query events, optionally filtered by session, type, and timestamp.

        If session_id is provided, reads only that session's file.
        Otherwise scans all JSONL files in the root directory.
        """
        events: list[L1Event] = []

        if session_id:
            path = self._session_path(session_id)
            if not path.exists():
                return events
            paths = [path]
        else:
            paths = list(self.root.glob("*.jsonl"))

        for path in sorted(paths):
            for line in self._iter_lines(path):
                event = self._parse_line(line)
                if event is None:
                    continue
                if event_type and event.event_type != event_type:
                    continue
                if since and event.timestamp < since:
                    continue
                events.append(event)
                if limit and len(events) >= limit:
                    return events

        return events

    def get(self, event_id: str) -> L1Event | None:
        """Retrieve a single event by its ID by scanning all files."""
        for path in self.root.glob("*.jsonl"):
            for line in self._iter_lines(path):
                event = self._parse_line(line)
                if event and event.event_id == event_id:
                    return event
        return None

    def count(self) -> int:
        """Total number of events across all sessions."""
        total = 0
        for path in self.root.glob("*.jsonl"):
            with open(path, encoding="utf-8") as f:
                total += sum(1 for line in f if line.strip())
        return total

    def count_by_session(self, session_id: str) -> int:
        """Number of events for a specific session."""
        path = self._session_path(session_id)
        if not path.exists():
            return 0
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())

    def get_event_ids(self) -> set[str]:
        """Return all known event IDs (for deduplication checks)."""
        ids: set[str] = set()
        for path in self.root.glob("*.jsonl"):
            for line in self._iter_lines(path):
                event = self._parse_line(line)
                if event:
                    ids.add(event.event_id)
        return ids
