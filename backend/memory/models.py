"""Shared Pydantic models for the 3-layer memory system."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# L1 Event Types
# ---------------------------------------------------------------------------

L1EventType = Literal[
    "run_completed",
    "lesson_completed",
    "quiz_result",
    "topic_discussed",
    "preference_expressed",
]


# ---------------------------------------------------------------------------
# L2 Fact Categories
# ---------------------------------------------------------------------------

L2Category = Literal[
    "mastery",
    "preference",
    "knowledge_gap",
    "project_state",
    "learning_goal",
]


# ---------------------------------------------------------------------------
# L3 Insight Types
# ---------------------------------------------------------------------------

L3InsightType = Literal[
    "strength",
    "weakness",
    "recommendation",
]


# ---------------------------------------------------------------------------
# L1 - Raw Event
# ---------------------------------------------------------------------------

class L1Event(BaseModel):
    """A raw, append-only event stored in JSONL."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(default_factory=_utc_now)
    session_id: str
    run_id: str | None = None
    event_type: L1EventType
    data: dict[str, Any] = Field(default_factory=dict)
    # L1 is the base layer — no refs to other L1 events
    l1_ref: None = None

    def to_jsonl_line(self) -> str:
        return self.model_dump_json(ensure_ascii=False)


# ---------------------------------------------------------------------------
# L2 - Refined Fact
# ---------------------------------------------------------------------------

class L2Fact(BaseModel):
    """An editable fact derived from one or more L1 events."""

    fact_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: str = Field(default_factory=_utc_now)
    updated_at: str = Field(default_factory=_utc_now)
    category: L2Category
    subject: str = ""  # e.g. topic name, project name
    content: str  # human-editable
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    # Required: must reference at least one L1 event
    l1_refs: list[str] = Field(default_factory=list)
    # L2 does not reference other L2 facts
    l2_ref: None = None


# ---------------------------------------------------------------------------
# L3 - Learning Profile
# ---------------------------------------------------------------------------

class L3Insight(BaseModel):
    """A synthesized insight in the learning profile."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: L3InsightType
    content: str
    # Required: must reference at least one L2 fact
    l2_refs: list[str] = Field(default_factory=list)


class L3Summary(BaseModel):
    """Aggregated statistics for the profile."""

    total_l1_events: int = 0
    total_l2_facts: int = 0
    topics_studied: list[str] = Field(default_factory=list)
    current_mastery: dict[str, float] = Field(default_factory=dict)
    learning_preferences: dict[str, Any] = Field(default_factory=dict)


class L3Profile(BaseModel):
    """The synthesized learning profile (L3)."""

    profile_id: str = "user_default"
    updated_at: str = Field(default_factory=_utc_now)
    summary: L3Summary = Field(default_factory=L3Summary)
    insights: list[L3Insight] = Field(default_factory=list)
    # All L2 fact IDs referenced by this profile
    l2_refs: list[str] = Field(default_factory=list)
    # L3 does not reference other L3 profiles
    l3_ref: None = None
