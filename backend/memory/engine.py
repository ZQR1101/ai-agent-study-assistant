"""Memory Engine — orchestrates L1/L2/L3 and provides the public API."""

from __future__ import annotations

from typing import Any

from backend.memory.l1_store import L1Store
from backend.memory.l2_store import L2Store
from backend.memory.l3_profile import L3Store
from backend.memory.models import (
    L1Event,
    L1EventType,
    L2Category,
    L2Fact,
    L3InsightType,
    L3Profile,
)


class MemoryEngine:
    """Unified interface over the three-layer memory system.

    Usage:
        engine = get_memory_engine()
        engine.record_event(session_id, "lesson_completed", data={...}, run_id=run_id)
        engine.refresh_profile()
        context = engine.get_context_for_llm()
    """

    def __init__(
        self,
        l1: L1Store | None = None,
        l2: L2Store | None = None,
        l3: L3Store | None = None,
    ):
        self.l1 = l1 or L1Store()
        self.l2 = l2 or L2Store()
        self.l3 = l3 or L3Store()

    # -------------------------------------------------------------------------
    # L1 Operations
    # -------------------------------------------------------------------------

    def record_event(
        self,
        session_id: str,
        event_type: L1EventType,
        data: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> L1Event:
        """Record a raw event to L1 (append-only).

        This is the main entry point for capturing learning activity.
        Call this after each meaningful user interaction.

        Args:
            session_id: The current chat session ID.
            event_type: One of the defined L1EventType values.
            data: Arbitrary event payload (topic, score, etc.).
            run_id: Optional run ID if this event is part of an agent run.

        Returns:
            The persisted L1Event.
        """
        return self.l1.record(
            session_id=session_id,
            run_id=run_id,
            event_type=event_type,
            data=data or {},
        )

    def get_events(
        self,
        session_id: str | None = None,
        event_type: L1EventType | None = None,
        since: str | None = None,
        limit: int | None = None,
    ) -> list[L1Event]:
        """Query L1 events with optional filters."""
        return self.l1.query(
            session_id=session_id,
            event_type=event_type,
            since=since,
            limit=limit,
        )

    # -------------------------------------------------------------------------
    # L2 Operations
    # -------------------------------------------------------------------------

    def extract_fact(
        self,
        event_id: str,
        content: str,
        category: L2Category,
        subject: str = "",
        confidence: float = 0.5,
    ) -> L2Fact | None:
        """Create an L2 fact from an L1 event.

        The fact will reference the given event_id.
        Returns None if the event doesn't exist.

        Args:
            event_id: The L1 event to reference.
            content: The fact content (human-editable).
            category: The fact category.
            subject: Optional subject label.
            confidence: 0.0-1.0 confidence score.

        Returns:
            The persisted L2Fact, or None if event_id not found.
        """
        event = self.l1.get(event_id)
        if event is None:
            return None
        return self.l2.add_fact(
            content=content,
            category=category,
            l1_refs=[event_id],
            subject=subject,
            confidence=confidence,
        )

    def add_fact(
        self,
        content: str,
        category: L2Category,
        l1_refs: list[str],
        subject: str = "",
        confidence: float = 0.5,
    ) -> L2Fact:
        """Create an L2 fact referencing one or more L1 events.

        Unlike extract_fact, this allows multi-event aggregation.
        """
        return self.l2.add_fact(
            content=content,
            category=category,
            l1_refs=l1_refs,
            subject=subject,
            confidence=confidence,
        )

    def update_fact(
        self,
        fact_id: str,
        content: str | None = None,
        category: L2Category | None = None,
        subject: str | None = None,
        confidence: float | None = None,
    ) -> L2Fact | None:
        """Update an L2 fact's content and/or metadata."""
        return self.l2.update_fact(
            fact_id=fact_id,
            content=content,
            category=category,
            subject=subject,
            confidence=confidence,
        )

    def get_facts(
        self,
        category: L2Category | None = None,
        subject: str | None = None,
    ) -> list[L2Fact]:
        """Query L2 facts with optional filters."""
        return self.l2.query(category=category, subject=subject)

    def delete_fact(self, fact_id: str) -> bool:
        """Delete an L2 fact."""
        return self.l2.delete_fact(fact_id)

    def get_unreferenced_events(self) -> list[str]:
        """Get L1 event IDs that haven't been turned into L2 facts yet."""
        known_ids = self.l1.get_event_ids()
        return self.l2.get_unreferenced_l1_ids(known_ids)

    # -------------------------------------------------------------------------
    # L3 Operations
    # -------------------------------------------------------------------------

    def refresh_profile(
        self,
        topics_studied: list[str] | None = None,
        current_mastery: dict[str, float] | None = None,
        learning_preferences: dict[str, Any] | None = None,
    ) -> L3Profile:
        """Regenerate the L3 profile from current L1/L2 state.

        Recomputes summary statistics and saves the updated profile.
        Does NOT regenerate insights — those are added explicitly via add_insight.

        Args:
            topics_studied: Override the topics list.
            current_mastery: Override the mastery map.
            learning_preferences: Override the preferences dict.

        Returns:
            The updated L3Profile.
        """
        all_facts = self.l2.get_all()
        all_events = self.l1.query(limit=None)

        # Collect topics from mastery facts
        if topics_studied is None:
            topics_studied = sorted(
                set(f.subject for f in all_facts if f.category == "mastery" and f.subject)
            )

        # Aggregate mastery scores by subject
        if current_mastery is None:
            mastery_scores: dict[str, list[float]] = {}
            for fact in all_facts:
                if fact.category == "mastery" and fact.subject:
                    mastery_scores.setdefault(fact.subject, []).append(fact.confidence)
            current_mastery = {
                subject: sum(scores) / len(scores)
                for subject, scores in mastery_scores.items()
            }

        # Collect preferences
        if learning_preferences is None:
            learning_preferences = {}
            for fact in all_facts:
                if fact.category == "preference":
                    learning_preferences[fact.subject or "general"] = fact.content

        # Collect all referenced L2 IDs
        l2_refs = [f.fact_id for f in all_facts]

        return self.l3.update_summary(
            total_l1_events=len(all_events),
            total_l2_facts=len(all_facts),
            topics_studied=topics_studied,
            current_mastery=current_mastery,
            learning_preferences=learning_preferences,
        )

    def add_insight(
        self,
        insight_type: L3InsightType,
        content: str,
        l2_refs: list[str],
    ) -> L3Profile:
        """Add an insight to the profile (must reference L2 facts)."""
        return self.l3.add_insight(
            insight_type=insight_type,
            content=content,
            l2_refs=l2_refs,
        )

    def get_profile(self) -> L3Profile:
        """Get the current learning profile."""
        return self.l3.get()

    def get_profile_text(self) -> str:
        """Get the profile formatted as LLM-consumable text."""
        return self.l3.format_for_llm()

    # -------------------------------------------------------------------------
    # Unified Context
    # -------------------------------------------------------------------------

    def get_context_for_llm(
        self,
        include_profile: bool = True,
        max_facts: int = 20,
    ) -> str:
        """Build a comprehensive memory context string for LLM injection.

        Args:
            include_profile: Whether to include the L3 profile section.
            max_facts: Maximum number of L2 facts to include.

        Returns:
            A formatted string suitable for use as LLM system context.
        """
        parts: list[str] = []

        if include_profile:
            profile_text = self.get_profile_text()
            if profile_text and "（暂无画像数据）" not in profile_text:
                parts.append(profile_text)

        # Include recent L2 facts as evidence
        all_facts = self.l2.get_all()[:max_facts]
        if all_facts:
            fact_lines = ["\n=== 相关事实 ==="]
            category_labels = {
                "mastery": "掌握",
                "preference": "偏好",
                "knowledge_gap": "缺口",
                "project_state": "状态",
                "learning_goal": "目标",
            }
            for fact in all_facts:
                label = category_labels.get(fact.category, fact.category)
                fact_lines.append(
                    f"- [{label}] {fact.content} (confidence: {fact.confidence:.0%})"
                )
            parts.append("\n".join(fact_lines))

        if not parts:
            return ""

        return "\n\n".join(parts)

    # -------------------------------------------------------------------------
    # Auto-extraction helpers
    # -------------------------------------------------------------------------

    def auto_extract_from_lesson(
        self,
        event_id: str,
        topic: str,
        knowledge_score: float | None = None,
    ) -> list[L2Fact]:
        """Extract L2 facts from a lesson_completed L1 event.

        Creates mastery and knowledge_gap facts based on the event data.

        Returns:
            List of created L2Fact objects.
        """
        created: list[L2Fact] = []

        # Mastery fact
        mastery = self.extract_fact(
            event_id=event_id,
            content=f"已学习主题：{topic}",
            category="mastery",
            subject=topic,
            confidence=knowledge_score if knowledge_score is not None else 0.6,
        )
        if mastery:
            created.append(mastery)

        return created

    def auto_extract_from_quiz(
        self,
        event_id: str,
        topic: str,
        score: float,
        total: float,
    ) -> list[L2Fact]:
        """Extract L2 facts from a quiz_result L1 event.

        Creates mastery (if passed) or knowledge_gap (if failed) facts.

        Returns:
            List of created L2Fact objects.
        """
        created: list[L2Fact] = []
        accuracy = score / total if total > 0 else 0.0

        if accuracy >= 0.7:
            fact = self.extract_fact(
                event_id=event_id,
                content=f"练习正确率 {accuracy:.0%}，已掌握",
                category="mastery",
                subject=topic,
                confidence=accuracy,
            )
        else:
            fact = self.extract_fact(
                event_id=event_id,
                content=f"练习正确率 {accuracy:.0%}，需要加强",
                category="knowledge_gap",
                subject=topic,
                confidence=1.0 - accuracy,
            )
        if fact:
            created.append(fact)

        return created
