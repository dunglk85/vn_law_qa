from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt

from app.config import config


TOKEN_SCHEMA = "Bearer"


def create_access_token(user_id: str, tenant_id: str = "", roles: list[str] | None = None) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "iat": now,
        "exp": now + timedelta(minutes=config.access_token_expire_minutes),
        "tenant_id": tenant_id,
        "roles": roles or [],
    }
    return jwt.encode(payload, config.jwt_secret, algorithm=config.jwt_algorithm)


def create_refresh_token() -> str:
    return str(uuid.uuid4())


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, config.jwt_secret, algorithms=[config.jwt_algorithm])
    except JWTError:
        raise ValueError("Invalid token")
    return payload


def validate_claims(payload: dict) -> None:
    if "sub" not in payload:
        raise ValueError("Token missing required claim: sub")
    if "tenant_id" not in payload:
        raise ValueError("Token missing required claim: tenant_id")
    if "roles" not in payload:
        raise ValueError("Token missing required claim: roles")
