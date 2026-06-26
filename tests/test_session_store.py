"""Tests for session_store and database module using SQLite in-memory."""

import os
import unittest
from unittest.mock import patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.db_models import Base, ChatMessage, ChatSession
from backend.evaluation_store import (
    list_recent_judge_results,
    save_judge_result,
    update_judge_feedback,
)
from backend.session_store import (
    create_or_get_session,
    get_recent_messages,
    get_session_messages,
    list_sessions,
    save_message,
)


def _create_test_db():
    """Create an in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


class SessionStoreTests(unittest.TestCase):
    def setUp(self):
        self.SessionLocal = _create_test_db()

    def test_create_session_generates_id(self):
        with self.SessionLocal() as db:
            session_id = create_or_get_session(db, session_id=None, title="Hello world")

        self.assertIsNotNone(session_id)
        self.assertEqual(len(session_id), 36)  # UUID format

    def test_create_session_with_provided_id(self):
        with self.SessionLocal() as db:
            session_id = create_or_get_session(db, session_id="test-123", title="Test")

        self.assertEqual(session_id, "test-123")

    def test_get_existing_session_returns_same_id(self):
        with self.SessionLocal() as db:
            first_id = create_or_get_session(db, session_id="my-session", title="First")
            second_id = create_or_get_session(db, session_id="my-session", title="Second")

        self.assertEqual(first_id, second_id)

    def test_title_is_truncated_to_30_chars(self):
        long_title = "A" * 50
        with self.SessionLocal() as db:
            session_id = create_or_get_session(db, session_id=None, title=long_title)
            session = db.get(ChatSession, session_id)

        self.assertEqual(len(session.title), 30)

    def test_save_and_retrieve_messages(self):
        with self.SessionLocal() as db:
            session_id = create_or_get_session(db, session_id="s1", title="Test")
            save_message(db, session_id, "user", "What is RAG?")
            save_message(db, session_id, "assistant", "RAG is retrieval augmented generation.")

            messages = get_recent_messages(db, session_id, limit=10)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "What is RAG?")
        self.assertEqual(messages[1]["role"], "assistant")

    def test_get_recent_messages_respects_limit(self):
        with self.SessionLocal() as db:
            session_id = create_or_get_session(db, session_id="s2", title="Test")
            for i in range(20):
                save_message(db, session_id, "user", f"Message {i}")

            messages = get_recent_messages(db, session_id, limit=5)

        self.assertEqual(len(messages), 5)
        # Should be the most recent 5
        self.assertEqual(messages[-1]["content"], "Message 19")

    def test_get_recent_messages_ordered_ascending(self):
        with self.SessionLocal() as db:
            session_id = create_or_get_session(db, session_id="s3", title="Test")
            save_message(db, session_id, "user", "First")
            save_message(db, session_id, "assistant", "Second")
            save_message(db, session_id, "user", "Third")

            messages = get_recent_messages(db, session_id, limit=10)

        self.assertEqual(messages[0]["content"], "First")
        self.assertEqual(messages[1]["content"], "Second")
        self.assertEqual(messages[2]["content"], "Third")

    def test_list_sessions(self):
        with self.SessionLocal() as db:
            create_or_get_session(db, session_id="s1", title="Session 1")
            save_message(db, "s1", "user", "Hello")
            create_or_get_session(db, session_id="s2", title="Session 2")

            sessions = list_sessions(db, limit=50)

        self.assertEqual(len(sessions), 2)
        # Each session dict has expected keys
        for s in sessions:
            self.assertIn("id", s)
            self.assertIn("title", s)
            self.assertIn("message_count", s)

    def test_get_session_messages_returns_full_details(self):
        with self.SessionLocal() as db:
            create_or_get_session(db, session_id="s1", title="Test")
            save_message(db, "s1", "user", "Hello")
            save_message(db, "s1", "assistant", "Hi there")

            messages = get_session_messages(db, "s1", limit=50)

        self.assertEqual(len(messages), 2)
        self.assertIn("id", messages[0])
        self.assertIn("session_id", messages[0])
        self.assertIn("created_at", messages[0])

    def test_save_and_list_judge_evaluations(self):
        with self.SessionLocal() as db:
            first = save_judge_result(
                db,
                session_id="s1",
                question="Question 1",
                answer="Answer 1",
                evaluation={
                    "judge_model": "judge-model",
                    "accuracy": 8,
                    "completeness": 7,
                    "citation_quality": 6,
                    "overall_score": 7,
                    "verdict": "WEAK_PASS",
                    "deductions": [
                        {"metric": "Citation Quality", "points": 4, "reason": "Not enough source support."}
                    ],
                    "feedback": "ok",
                    "raw_output": "{}",
                },
            )
            second = save_judge_result(
                db,
                session_id="s1",
                question="Question 2",
                answer="Answer 2",
                evaluation={
                    "judge_model": "judge-model",
                    "accuracy": 9,
                    "completeness": 8,
                    "citation_quality": None,
                    "overall_score": 8,
                    "verdict": "PASS",
                },
            )

            evaluations = list_recent_judge_results(db, limit=20)
            updated = update_judge_feedback(
                db,
                result_id=first["id"],
                judge_feedback="bad",
                reason="score too high",
            )

        self.assertEqual(len(evaluations), 2)
        self.assertEqual(evaluations[0]["id"], second["id"])
        self.assertEqual(evaluations[1]["id"], first["id"])
        self.assertEqual(evaluations[0]["overall_score"], 8)
        self.assertIsNone(evaluations[0]["citation_quality"])
        self.assertEqual(evaluations[0]["verdict"], "PASS")
        self.assertEqual(evaluations[1]["feedback"], "ok")
        self.assertEqual(evaluations[1]["deductions"][0]["metric"], "Citation Quality")
        self.assertEqual(updated["judge_feedback"], "bad")
        self.assertEqual(updated["judge_feedback_reason"], "score too high")


class DatabaseModuleTests(unittest.TestCase):
    @patch.dict(os.environ, {"ENABLE_DB_HISTORY": "false"}, clear=False)
    def test_db_history_disabled_by_default(self):
        from backend.database import is_db_history_enabled

        self.assertFalse(is_db_history_enabled())

    @patch.dict(os.environ, {"ENABLE_DB_HISTORY": "true"}, clear=False)
    def test_db_history_enabled_when_set(self):
        from backend.database import is_db_history_enabled

        self.assertTrue(is_db_history_enabled())


class ChatResponseSessionIdTests(unittest.TestCase):
    def test_chat_response_accepts_session_id(self):
        from backend.schemas import ChatResponse

        response = ChatResponse(
            answer="test",
            mode="chat",
            model="test-model",
            session_id="abc-123",
        )
        self.assertEqual(response.session_id, "abc-123")

    def test_chat_response_session_id_defaults_to_none(self):
        from backend.schemas import ChatResponse

        response = ChatResponse(
            answer="test",
            mode="chat",
            model="test-model",
        )
        self.assertIsNone(response.session_id)


if __name__ == "__main__":
    unittest.main()
