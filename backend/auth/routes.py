"""Auth API routes: login/logout/me and admin user management."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from backend.auth.dependencies import (
    SESSION_COOKIE_NAME,
    get_current_user,
    require_admin,
    require_user,
)
from backend.auth.models import ROLES, User
from backend.auth.service import (
    SESSION_TTL_SECONDS,
    create_session_token,
    hash_password,
    normalize_role,
    verify_password,
)
from backend.platform_db import platform_session
from backend.documents.audit import record_audit

router = APIRouter(prefix="/auth", tags=["Platform Auth"])

SESSION_COOKIE_MAX_AGE = SESSION_TTL_SECONDS


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=50)
    password: str = Field(min_length=1, max_length=200)


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=2, max_length=50, pattern=r"^[A-Za-z0-9_.-]+$")
    display_name: str = Field(default="", max_length=100)
    role: str = "expert"
    password: str = Field(min_length=8, max_length=200)


class UpdateUserRequest(BaseModel):
    display_name: str | None = Field(default=None, max_length=100)
    role: str | None = None
    active: bool | None = None
    password: str | None = Field(default=None, min_length=8, max_length=200)


def _user_payload(user: User) -> dict:
    return {
        "id": user.id,
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
        "active": user.active,
    }


@router.post("/login")
def login(body: LoginRequest, response: Response) -> dict:
    session: Session = platform_session()
    try:
        user = session.query(User).filter(User.username == body.username).first()
        if user is None or not user.active or not verify_password(body.password, user.password_hash):
            raise HTTPException(status_code=401, detail="用户名或密码错误")
        token = create_session_token(user.id, user.role)
        record_audit(
            session,
            correlation_id="auth",
            event="user.login",
            actor=user.username,
            payload={"username": user.username, "role": user.role},
        )
        session.commit()
    finally:
        session.close()

    response.set_cookie(
        SESSION_COOKIE_NAME,
        token,
        max_age=SESSION_COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return {"token": token, "user": _user_payload(user)}


@router.post("/logout")
def logout(response: Response) -> dict:
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@router.get("/me")
def me(user: User | None = Depends(get_current_user)) -> dict:
    if user is None:
        return {"user": None}
    return {"user": _user_payload(user)}


@router.get("/users")
def list_users(_: User = Depends(require_admin)) -> dict:
    session: Session = platform_session()
    try:
        users = session.query(User).order_by(User.created_at).all()
        return {"users": [_user_payload(u) for u in users]}
    finally:
        session.close()


@router.post("/users", status_code=201)
def create_user(body: CreateUserRequest, admin: User = Depends(require_admin)) -> dict:
    session: Session = platform_session()
    try:
        exists = session.query(User).filter(User.username == body.username).first()
        if exists:
            raise HTTPException(status_code=409, detail="用户名已存在")
        user = User(
            id=str(uuid.uuid4()),
            username=body.username,
            display_name=body.display_name or body.username,
            role=normalize_role(body.role),
            password_hash=hash_password(body.password),
        )
        session.add(user)
        record_audit(
            session,
            correlation_id="auth",
            event="user.created",
            actor=admin.username,
            payload={"username": user.username, "role": user.role},
        )
        session.commit()
        return {"user": _user_payload(user)}
    finally:
        session.close()


@router.patch("/users/{user_id}")
def update_user(
    user_id: str, body: UpdateUserRequest, admin: User = Depends(require_admin)
) -> dict:
    session: Session = platform_session()
    try:
        user = session.get(User, user_id)
        if user is None:
            raise HTTPException(status_code=404, detail="用户不存在")
        if body.role is not None:
            user.role = normalize_role(body.role)
        if body.active is not None:
            if user.id == admin.id and body.active is False:
                raise HTTPException(status_code=400, detail="不能停用自己的账号")
            user.active = body.active
        if body.display_name is not None:
            user.display_name = body.display_name
        if body.password is not None:
            user.password_hash = hash_password(body.password)
        record_audit(
            session,
            correlation_id="auth",
            event="user.updated",
            actor=admin.username,
            payload={"target": user.username, "role": user.role, "active": user.active},
        )
        session.commit()
        return {"user": _user_payload(user)}
    finally:
        session.close()
