"""Tests for the 3-layer memory system."""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

import pytest

from backend.memory import (
    L1EventType,
    L2Category,
    L3InsightType,
    MemoryEngine,
    get_memory_engine,
    reset_memory_engine,
)
from backend.memory.l1_store import L1Store
from backend.memory.l2_store import L2Store
from backend.memory.l3_profile import L3Store


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton before each test."""
    reset_memory_engine()
    yield
    reset_memory_engine()


@pytest.fixture
def temp_root(tmp_path):
    return tmp_path


@pytest.fixture
def l1(temp_root):
    return L1Store(root=temp_root / "l1")


@pytest.fixture
def l2(temp_root):
    return L2Store(root=temp_root / "l2")


@pytest.fixture
def l3(temp_root):
    return L3Store(root=temp_root / "l3")


@pytest.fixture
def engine(temp_root):
    return MemoryEngine(
        l1=L1Store(root=temp_root / "l1"),
        l2=L2Store(root=temp_root / "l2"),
        l3=L3Store(root=temp_root / "l3"),
    )


# ---------------------------------------------------------------------------
# L1 Tests
# ---------------------------------------------------------------------------

class TestL1Store:
    def test_record_and_query(self, l1):
        event = l1.record("session-abc", "run-1", "run_completed", {"step": 1})
        assert event.session_id == "session-abc"
        assert event.run_id == "run-1"
        assert event.event_type == "run_completed"
        assert event.data == {"step": 1}
        assert event.l1_ref is None

        events = l1.query(session_id="session-abc")
        assert len(events) == 1
        assert events[0].event_id == event.event_id

    def test_append_only(self, l1):
        """Events are never modified after append."""
        event1 = l1.record("session-x", None, "topic_discussed", {"topic": "RAG"})
        l1.record("session-x", None, "topic_discussed", {"topic": "LangGraph"})

        # Query and check first event is unchanged
        events = l1.query(session_id="session-x")
        assert len(events) == 2
        assert events[0].data["topic"] == "RAG"

    def test_query_by_type(self, l1):
        l1.record("s1", None, "run_completed", {})
        l1.record("s1", None, "lesson_completed", {})
        l1.record("s1", None, "quiz_result", {})

        assert len(l1.query(event_type="run_completed")) == 1
        assert len(l1.query(event_type="lesson_completed")) == 1
        assert len(l1.query(event_type="quiz_result")) == 1
        assert len(l1.query()) == 3

    def test_query_by_session(self, l1):
        l1.record("s1", None, "run_completed", {})
        l1.record("s2", None, "run_completed", {})

        assert len(l1.query(session_id="s1")) == 1
        assert len(l1.query(session_id="s2")) == 1
        assert len(l1.query()) == 2

    def test_count(self, l1):
        assert l1.count() == 0
        l1.record("s1", None, "run_completed", {})
        l1.record("s1", None, "lesson_completed", {})
        assert l1.count() == 2

    def test_get_by_id(self, l1):
        event = l1.record("s1", None, "topic_discussed", {"topic": "RAG"})
        found = l1.get(event.event_id)
        assert found is not None
        assert found.event_id == event.event_id

        not_found = l1.get("non-existent-id")
        assert not_found is None

    def test_get_event_ids(self, l1):
        e1 = l1.record("s1", None, "run_completed", {})
        e2 = l1.record("s2", None, "run_completed", {})
        ids = l1.get_event_ids()
        assert e1.event_id in ids
        assert e2.event_id in ids

    def test_session_id_sanitization(self, l1):
        """Session IDs with special chars are sanitized to safe filenames."""
        event = l1.record("session/with\\chars", None, "run_completed", {})
        events = l1.query(session_id="session/with\\chars")
        assert len(events) == 1
        assert events[0].event_id == event.event_id


# ---------------------------------------------------------------------------
# L2 Tests
# ---------------------------------------------------------------------------

class TestL2Store:
    def test_add_fact_requires_l1_ref(self, l2):
        with pytest.raises(ValueError, match="must reference at least one L1"):
            l2.add_fact("test", category="mastery", l1_refs=[])

    def test_add_and_get(self, l2):
        l1_event_id = "fake-l1-event-id"
        fact = l2.add_fact(
            content="RAG 概念已理解",
            category="mastery",
            l1_refs=[l1_event_id],
            subject="RAG",
            confidence=0.8,
        )
        assert fact.l1_refs == [l1_event_id]
        assert fact.category == "mastery"
        assert fact.confidence == 0.8

        found = l2.get(fact.fact_id)
        assert found is not None
        assert found.content == "RAG 概念已理解"

    def test_update_fact(self, l2):
        fact = l2.add_fact("原始内容", category="mastery", l1_refs=["l1-1"])
        updated = l2.update_fact(fact.fact_id, content="修改后内容", confidence=0.9)
        assert updated is not None
        assert updated.content == "修改后内容"
        assert updated.confidence == 0.9
        assert updated.l1_refs == ["l1-1"]  # preserved

    def test_update_nonexistent(self, l2):
        result = l2.update_fact("nonexistent-id", content="x")
        assert result is None

    def test_query_by_category(self, l2):
        l2.add_fact("f1", category="mastery", l1_refs=["l1"])
        l2.add_fact("f2", category="preference", l1_refs=["l1"])
        l2.add_fact("f3", category="mastery", l1_refs=["l1"])

        mastery = l2.query(category="mastery")
        assert len(mastery) == 2
        assert all(f.category == "mastery" for f in mastery)

    def test_query_by_subject(self, l2):
        l2.add_fact("f1", category="mastery", l1_refs=["l1"], subject="RAG")
        l2.add_fact("f2", category="mastery", l1_refs=["l1"], subject="LangGraph")
        l2.add_fact("f3", category="mastery", l1_refs=["l1"], subject="RAG Advanced")

        rag_facts = l2.query(subject="RAG")
        assert len(rag_facts) == 2

    def test_get_by_l1_ref(self, l2):
        l1_id = "l1-target"
        fact1 = l2.add_fact("f1", category="mastery", l1_refs=[l1_id])
        fact2 = l2.add_fact("f2", category="preference", l1_refs=[l1_id, "l1-other"])

        found = l2.get_by_l1_ref(l1_id)
        assert len(found) == 2
        ids = {f.fact_id for f in found}
        assert fact1.fact_id in ids
        assert fact2.fact_id in ids

    def test_delete_fact(self, l2):
        fact = l2.add_fact("to delete", category="mastery", l1_refs=["l1"])
        assert l2.get(fact.fact_id) is not None

        deleted = l2.delete_fact(fact.fact_id)
        assert deleted is True
        assert l2.get(fact.fact_id) is None

        # Delete again returns False
        assert l2.delete_fact(fact.fact_id) is False

    def test_get_unreferenced_l1_ids(self, l2):
        all_ids = {"l1-a", "l1-b", "l1-c"}
        # l1-a is referenced
        l2.add_fact("f1", category="mastery", l1_refs=["l1-a"])

        unreferenced = l2.get_unreferenced_l1_ids(all_ids)
        assert set(unreferenced) == {"l1-b", "l1-c"}

    def test_updated_at_changes(self, l2):
        import time
        fact = l2.add_fact("v1", category="mastery", l1_refs=["l1"])
        original_updated = fact.updated_at
        time.sleep(0.01)
        updated = l2.update_fact(fact.fact_id, content="v2")
        assert updated is not None
        assert updated.updated_at > original_updated


# ---------------------------------------------------------------------------
# L3 Tests
# ---------------------------------------------------------------------------

class TestL3Store:
    def test_get_empty_profile(self, l3):
        profile = l3.get()
        assert profile.profile_id == "user_default"
        assert profile.summary.total_l1_events == 0
        assert profile.insights == []

    def test_add_insight_requires_l2_ref(self, l3):
        with pytest.raises(ValueError, match="must reference at least one L2"):
            l3.add_insight("strength", "content", l2_refs=[])

    def test_add_insight(self, l3):
        profile = l3.add_insight(
            insight_type="strength",
            content="RAG 基础掌握良好",
            l2_refs=["fact-1", "fact-2"],
        )
        assert len(profile.insights) == 1
        assert profile.insights[0].type == "strength"
        assert profile.insights[0].l2_refs == ["fact-1", "fact-2"]
        assert "fact-1" in profile.l2_refs
        assert "fact-2" in profile.l2_refs

    def test_update_summary(self, l3):
        profile = l3.update_summary(
            total_l1_events=100,
            total_l2_facts=20,
            topics_studied=["RAG", "LangGraph"],
        )
        assert profile.summary.total_l1_events == 100
        assert profile.summary.total_l2_facts == 20
        assert "RAG" in profile.summary.topics_studied

    def test_remove_insight(self, l3):
        profile = l3.add_insight("strength", "s1", l2_refs=["f1"])
        profile = l3.add_insight("weakness", "w1", l2_refs=["f2"])
        assert len(profile.insights) == 2

        profile = l3.remove_insight(profile.insights[0].id)
        assert len(profile.insights) == 1
        assert profile.insights[0].type == "weakness"

    def test_format_for_llm(self, l3):
        l3.update_summary(topics_studied=["RAG", "LangGraph"], current_mastery={"RAG": 0.8})
        l3.add_insight("strength", "RAG 基础掌握良好", l2_refs=["f1"])

        text = l3.format_for_llm()
        assert "RAG" in text
        assert "LangGraph" in text
        assert "优势" in text

    def test_convenience_queries(self, l3):
        l3.add_insight("strength", "s1", l2_refs=["f1"])
        l3.add_insight("weakness", "w1", l2_refs=["f2"])
        l3.add_insight("recommendation", "r1", l2_refs=["f3"])

        assert len(l3.get_strengths()) == 1
        assert len(l3.get_weaknesses()) == 1
        assert len(l3.get_recommendations()) == 1


# ---------------------------------------------------------------------------
# MemoryEngine Integration Tests
# ---------------------------------------------------------------------------

class TestMemoryEngine:
    def test_record_and_query(self, engine):
        event = engine.record_event(
            session_id="s1",
            event_type="run_completed",
            run_id="r1",
            data={"steps": 3},
        )
        assert event.event_type == "run_completed"

        events = engine.get_events(session_id="s1")
        assert len(events) == 1
        assert events[0].event_id == event.event_id

    def test_extract_fact_from_event(self, engine):
        event = engine.record_event(
            session_id="s1",
            event_type="lesson_completed",
            data={"topic": "RAG"},
        )

        fact = engine.extract_fact(
            event_id=event.event_id,
            content="已完成 RAG 课程",
            category="mastery",
            subject="RAG",
            confidence=0.7,
        )
        assert fact is not None
        assert event.event_id in fact.l1_refs

        # Non-existent event
        missing = engine.extract_fact(
            event_id="does-not-exist",
            content="x",
            category="mastery",
        )
        assert missing is None

    def test_refresh_profile(self, engine):
        # Record some events
        engine.record_event("s1", event_type="lesson_completed", data={"topic": "RAG"})
        engine.record_event("s1", event_type="quiz_result", data={"topic": "RAG", "score": 8, "total": 10})

        # Extract facts
        events = engine.get_events()
        for ev in events:
            if ev.event_type == "lesson_completed":
                engine.extract_fact(
                    event_id=ev.event_id,
                    content="已学习 RAG",
                    category="mastery",
                    subject="RAG",
                )

        # Refresh profile
        profile = engine.refresh_profile()
        assert profile.summary.total_l1_events == 2
        assert profile.summary.total_l2_facts == 1
        assert "RAG" in profile.summary.topics_studied

    def test_auto_extract_from_lesson(self, engine):
        event = engine.record_event("s1", event_type="lesson_completed", data={"topic": "LangGraph"})
        facts = engine.auto_extract_from_lesson(event.event_id, "LangGraph", knowledge_score=0.8)
        assert len(facts) >= 1
        assert facts[0].category == "mastery"

    def test_auto_extract_from_quiz_pass(self, engine):
        event = engine.record_event("s1", event_type="quiz_result", data={"score": 8, "total": 10})
        facts = engine.auto_extract_from_quiz(event.event_id, "RAG", score=8, total=10)
        assert len(facts) == 1
        assert facts[0].category == "mastery"
        assert facts[0].confidence == 0.8

    def test_auto_extract_from_quiz_fail(self, engine):
        event = engine.record_event("s1", event_type="quiz_result", data={"score": 3, "total": 10})
        facts = engine.auto_extract_from_quiz(event.event_id, "RAG", score=3, total=10)
        assert len(facts) == 1
        assert facts[0].category == "knowledge_gap"
        assert facts[0].confidence == 0.7  # 1.0 - 0.3

    def test_get_context_for_llm(self, engine):
        engine.record_event("s1", event_type="lesson_completed", data={"topic": "RAG"})
        events = engine.get_events()
        if events:
            engine.extract_fact(
                event_id=events[0].event_id,
                content="已学习 RAG",
                category="mastery",
                subject="RAG",
            )
        engine.refresh_profile()

        context = engine.get_context_for_llm()
        assert "RAG" in context or "学习画像" in context

    def test_get_unreferenced_events(self, engine):
        e1 = engine.record_event("s1", event_type="run_completed")
        e2 = engine.record_event("s1", event_type="lesson_completed")

        # Only extract from e1
        engine.extract_fact(e1.event_id, "extracted", "mastery")

        unreferenced = engine.get_unreferenced_events()
        assert e2.event_id in unreferenced
        assert e1.event_id not in unreferenced

    def test_singleton(self):
        reset_memory_engine()
        e1 = get_memory_engine()
        e2 = get_memory_engine()
        assert e1 is e2
        reset_memory_engine()


# ---------------------------------------------------------------------------
# Traceability Tests (L3 → L2 → L1)
# ---------------------------------------------------------------------------

class TestTraceability:
    def test_full_chain(self, engine):
        # L1: record a lesson event
        event = engine.record_event("s1", event_type="lesson_completed", data={"topic": "RAG"})

        # L2: extract a fact referencing the L1 event
        fact = engine.extract_fact(
            event_id=event.event_id,
            content="已完成 RAG 基础课程",
            category="mastery",
            subject="RAG",
            confidence=0.8,
        )

        # L3: add insight referencing the L2 fact
        profile = engine.add_insight(
            insight_type="strength",
            content="RAG 基础知识已掌握",
            l2_refs=[fact.fact_id],
        )

        # Verify the chain
        assert event.event_id in fact.l1_refs  # L2 → L1 ✓
        assert fact.fact_id in profile.l2_refs  # L3 → L2 ✓
        assert profile.insights[0].l2_refs == [fact.fact_id]  # insight → L2 ✓
