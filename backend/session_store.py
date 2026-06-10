"""CRUD operations for chat sessions and messages."""

from __future__ import annotations

import uuid

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db_models import ChatMessage, ChatSession


def create_or_get_session(
    db: Session, session_id: str | None = None, title: str | None = None
) -> str:
    """Get existing session or create a new one. Returns session_id."""
    if session_id:
        existing = db.get(ChatSession, session_id)
        if existing:
            return existing.id

    # Create new session
    new_id = session_id or str(uuid.uuid4())
    session_title = title[:30] if title else None
    new_session = ChatSession(id=new_id, title=session_title)
    db.add(new_session)
    db.commit()
    return new_id


def save_message(db: Session, session_id: str, role: str, content: str) -> None:
    """Save a single message to the database."""
    message = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(message)
    db.commit()


def get_recent_messages(db: Session, session_id: str, limit: int = 10) -> list[dict]:
    """Get the most recent messages for a session, ordered by id ascending."""
    # Get the last N messages by id desc, then reverse for chronological order
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.desc())
        .limit(limit)
        .all()
    )
    messages.reverse()
    return [
        {"role": msg.role, "content": msg.content}
        for msg in messages
    ]


def list_sessions(db: Session, limit: int = 50) -> list[dict]:
    """List sessions with message count, most recent first."""
    sessions = (
        db.query(ChatSession)
        .order_by(ChatSession.updated_at.desc())
        .limit(limit)
        .all()
    )
    result = []
    for session in sessions:
        message_count = (
            db.query(func.count(ChatMessage.id))
            .filter(ChatMessage.session_id == session.id)
            .scalar()
        )
        result.append({
            "id": session.id,
            "title": session.title,
            "created_at": session.created_at.isoformat() if session.created_at else None,
            "updated_at": session.updated_at.isoformat() if session.updated_at else None,
            "message_count": message_count or 0,
        })
    return result


def get_session_messages(db: Session, session_id: str, limit: int = 50) -> list[dict]:
    """Get all messages for a session, ordered by id ascending."""
    messages = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id.asc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": msg.id,
            "session_id": msg.session_id,
            "role": msg.role,
            "content": msg.content,
            "created_at": msg.created_at.isoformat() if msg.created_at else None,
        }
        for msg in messages
    ]
