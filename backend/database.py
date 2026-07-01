"""Database connection and session management.

Provides SQLAlchemy engine, session factory, and helpers.
All DB operations are gated behind ENABLE_DB_HISTORY env var.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import Session, sessionmaker

_engine = None
_SessionLocal: sessionmaker | None = None


def is_db_history_enabled() -> bool:
    """Return True if database history persistence is enabled."""
    return os.getenv("ENABLE_DB_HISTORY", "false").lower() in ("true", "1", "yes")


def get_database_url() -> str | None:
    """Return DATABASE_URL from environment, or None if not set."""
    return os.getenv("DATABASE_URL")


def get_engine():
    """Lazily create and return the SQLAlchemy engine."""
    global _engine
    if _engine is None:
        url = get_database_url()
        if not url:
            raise RuntimeError("DATABASE_URL is not configured")
        engine_options = {"pool_pre_ping": True}
        if url.startswith(("postgresql://", "postgresql+")):
            timeout = int(os.getenv("DATABASE_CONNECT_TIMEOUT", "1"))
            engine_options["connect_args"] = {"connect_timeout": timeout}
        _engine = create_engine(url, **engine_options)
    return _engine


def get_session_factory() -> sessionmaker:
    """Return the session factory, creating it if needed."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, expire_on_commit=False)
    return _SessionLocal


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Context manager for use outside FastAPI dependency injection."""
    factory = get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables and apply small additive compatibility migrations."""
    from backend.db_models import Base

    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    ensure_schema_compatibility(engine)


def ensure_schema_compatibility(engine) -> None:
    """Add columns introduced after the initial schema without dropping data."""
    inspector = inspect(engine)
    if "chat_messages" not in inspector.get_table_names():
        return

    message_columns = {column["name"] for column in inspector.get_columns("chat_messages")}
    if "response_json" not in message_columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE chat_messages ADD COLUMN response_json JSON"))


def reset_engine() -> None:
    """Reset engine and session factory. Used in tests."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
