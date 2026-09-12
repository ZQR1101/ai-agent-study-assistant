"""FastAPI dependencies for authentication and role checks."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from backend.auth.models import User
from backend.platform_db import platform_session

SESSION_COOKIE_NAME = "rulebook_session"


def _extract_token(request: Request) -> str | None:
    auth_header = request.headers.get("Authorization") or ""
    if auth_header.startswith("Bearer "):
        return auth_header.removeprefix("Bearer ").strip()
    return request.cookies.get(SESSION_COOKIE_NAME)


def get_current_user(request: Request) -> User | None:
    from backend.auth.service import decode_session_token

    token = _extract_token(request)
    if not token:
        return None
    payload = decode_session_token(token)
    if not payload:
        return None
    session: Session = platform_session()
    try:
        user = session.get(User, payload.get("sub"))
        if user is None or not user.active:
            return None
        # Detach from the short-lived session so the instance stays usable.
        session.expunge(user)
        return user
    finally:
        session.close()


def require_user(user: User | None = Depends(get_current_user)) -> User:
    if user is None:
        raise HTTPException(status_code=401, detail="未登录或会话已过期")
    return user


def require_reviewer(user: User = Depends(require_user)) -> User:
    """Admin and expert roles may review, approve, and export."""

    return user


def require_admin(user: User = Depends(require_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="需要管理员权限")
    return user
