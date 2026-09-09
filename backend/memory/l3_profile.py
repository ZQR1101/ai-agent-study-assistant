"""L3 Store — synthesized learning profile that must reference L2 facts."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.memory.models import L3Insight, L3InsightType, L3Profile, L3Summary


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class L3Store:
    """Single-file learning profile store (L3).

    The profile MUST reference L2 facts (for traceability).
    Insights within the profile also reference specific L2 facts.
    """

    def __init__(self, root: str | Path | None = None):
        from backend.config import get_config

        config = get_config()
        self.root = Path(root or config.project_root / "data" / "memory" / "l3_profile")
        self.profile_path = self.root / "profile.json"
        self._lock = threading.RLock()

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------

    def _write(self, profile: L3Profile) -> L3Profile:
        self.root.mkdir(parents=True, exist_ok=True)
        content = profile.model_dump_json(indent=2, ensure_ascii=False)
        self.profile_path.write_text(content, encoding="utf-8")
        return profile

    def _load(self) -> L3Profile:
        if not self.profile_path.exists():
            return L3Profile()
        try:
            raw = self.profile_path.read_text(encoding="utf-8")
            return L3Profile.model_validate_json(raw)
        except (json.JSONDecodeError, ValueError, OSError):
            return L3Profile()

    def save(self, profile: L3Profile) -> L3Profile:
        """Persist a profile (replaces any existing profile)."""
        with self._lock:
            return self._write(profile)

    def add_insight(
        self,
        insight_type: L3InsightType,
        content: str,
        l2_refs: list[str],
    ) -> L3Profile:
        """Add a new insight to the profile.

        Args:
            insight_type: strength | weakness | recommendation
            content: The insight text.
            l2_refs: List of L2 fact_ids this insight is based on.

        Returns:
            The updated profile.

        Raises:
            ValueError: If l2_refs is empty.
        """
        if not l2_refs:
            raise ValueError("L3 insight must reference at least one L2 fact")

        with self._lock:
            profile = self._load()
            insight = L3Insight(
                type=insight_type,
                content=content,
                l2_refs=list(l2_refs),
            )
            profile.insights.append(insight)
            # Ensure the referenced L2 fact IDs are tracked at the profile level
            for ref in l2_refs:
                if ref not in profile.l2_refs:
                    profile.l2_refs.append(ref)
            profile.updated_at = _utc_now()
            return self._write(profile)

    def update_summary(
        self,
        summary: L3Summary | dict[str, Any] | None = None,
        *,
        total_l1_events: int | None = None,
        total_l2_facts: int | None = None,
        topics_studied: list[str] | None = None,
        current_mastery: dict[str, float] | None = None,
        learning_preferences: dict[str, Any] | None = None,
    ) -> L3Profile:
        """Update the profile summary (partial update)."""
        with self._lock:
            profile = self._load()
            if summary is not None:
                if isinstance(summary, dict):
                    profile.summary = L3Summary(**summary)
                else:
                    profile.summary = summary
            else:
                s = profile.summary
                if total_l1_events is not None:
                    s.total_l1_events = total_l1_events
                if total_l2_facts is not None:
                    s.total_l2_facts = total_l2_facts
                if topics_studied is not None:
                    s.topics_studied = topics_studied
                if current_mastery is not None:
                    s.current_mastery = current_mastery
                if learning_preferences is not None:
                    s.learning_preferences = learning_preferences
            profile.updated_at = _utc_now()
            return self._write(profile)

    def remove_insight(self, insight_id: str) -> L3Profile:
        """Remove an insight by its ID."""
        with self._lock:
            profile = self._load()
            profile.insights = [i for i in profile.insights if i.id != insight_id]
            profile.updated_at = _utc_now()
            return self._write(profile)

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------

    def get(self) -> L3Profile:
        """Get the current profile (creates empty if none exists)."""
        with self._lock:
            return self._load()

    def exists(self) -> bool:
        """Check if a profile file exists."""
        return self.profile_path.exists()

    # -------------------------------------------------------------------------
    # Convenience queries
    # -------------------------------------------------------------------------

    def get_strengths(self) -> list[L3Insight]:
        """Get all strength insights."""
        return [i for i in self.get().insights if i.type == "strength"]

    def get_weaknesses(self) -> list[L3Insight]:
        """Get all weakness insights."""
        return [i for i in self.get().insights if i.type == "weakness"]

    def get_recommendations(self) -> list[L3Insight]:
        """Get all recommendation insights."""
        return [i for i in self.get().insights if i.type == "recommendation"]

    # -------------------------------------------------------------------------
    # Text format for LLM consumption
    # -------------------------------------------------------------------------

    def format_for_llm(self) -> str:
        """Format the profile as human-readable text for LLM context injection."""
        profile = self.get()
        lines = ["=== 学习画像 ==="]

        summary = profile.summary
        if summary.topics_studied:
            lines.append(f"- 已学习主题：{', '.join(summary.topics_studied[:10])}")
        if summary.current_mastery:
            mastery_lines = [
                f"  - {topic}: {level:.0%}" for topic, level in summary.current_mastery.items()
            ]
            lines.append("- 当前掌握度：")
            lines.extend(mastery_lines)
        if summary.learning_preferences:
            prefs = summary.learning_preferences
            if isinstance(prefs, dict):
                pref_items = [f"{k}={v}" for k, v in prefs.items()]
                if pref_items:
                    lines.append(f"- 学习偏好：{', '.join(pref_items)}")

        if profile.insights:
            lines.append("\n=== 洞察 ===")
            for insight in profile.insights:
                type_label = {"strength": "优势", "weakness": "薄弱", "recommendation": "建议"}.get(
                    insight.type, insight.type
                )
                lines.append(f"- [{type_label}] {insight.content}")

        if not lines or len(lines) == 1:
            lines.append("（暂无画像数据）")

        return "\n".join(lines)
