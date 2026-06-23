from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from app.auth.jwt import TOKEN_SCHEMA, create_access_token, create_refresh_token
from app.config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


# ---------------------------------------------------------------------------
# Refresh token store (Redis-backed with in-memory fallback)
# ---------------------------------------------------------------------------

class _RefreshTokenStore:
    """Persists refresh tokens in Redis when available, otherwise in-memory."""

    def __init__(self) -> None:
        self._memory: dict[str, dict[str, Any]] = {}
        self._redis = None
        self._redis_url: str | None = getattr(config, "redis_url", None)

    async def _get_redis(self):
        if self._redis is None and self._redis_url:
            try:
                import redis.asyncio as aioredis
                self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
                logger.info("Refresh token store: Redis-backed")
            except Exception as exc:
                logger.warning("Redis unavailable for refresh tokens, using in-memory: %s", exc)
                self._redis = False  # sentinel — don't retry
        return self._redis if self._redis is not False else None

    def _key(self, token: str) -> str:
        return f"refresh:{token}"

    async def add(self, token: str, user_id: str, expires: datetime) -> None:
        data = {"user_id": user_id, "expires": expires.isoformat()}
        r = await self._get_redis()
        if r is not None:
            try:
                ttl = max(int((expires - datetime.now(UTC)).total_seconds()), 1)
                await r.setex(self._key(token), ttl, json.dumps(data))
                return
            except Exception as exc:
                logger.warning("Redis write failed for refresh token, using in-memory: %s", exc)
        self._memory[token] = {"user_id": user_id, "expires": expires}

    async def pop(self, token: str) -> dict[str, Any] | None:
        r = await self._get_redis()
        if r is not None:
            try:
                raw = await r.getdel(self._key(token))
                if raw is None:
                    return None
                data = json.loads(raw)
                data["expires"] = datetime.fromisoformat(data["expires"])
                return data
            except Exception as exc:
                logger.warning("Redis read failed for refresh token, using in-memory: %s", exc)
        return self._memory.pop(token, None)


_refresh_store = _RefreshTokenStore()


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = TOKEN_SCHEMA


class RefreshRequest(BaseModel):
    refresh_token: str


@router.post("/token", response_model=TokenResponse)
async def login(req: TokenRequest) -> TokenResponse:
    if req.username != config.admin_username or req.password != config.admin_password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    access_token = create_access_token(user_id=req.username, tenant_id="default", roles=["admin"])
    refresh_token = create_refresh_token()
    expires = datetime.now(UTC) + timedelta(days=config.refresh_token_expire_days)
    await _refresh_store.add(refresh_token, req.username, expires)
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest) -> TokenResponse:
    stored = await _refresh_store.pop(req.refresh_token)
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    if datetime.now(UTC) > stored["expires"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired, please log in again",
        )
    user_id = stored["user_id"]
    new_access = create_access_token(user_id=user_id, tenant_id="default", roles=["admin"])
    new_refresh = create_refresh_token()
    expires = datetime.now(UTC) + timedelta(days=config.refresh_token_expire_days)
    await _refresh_store.add(new_refresh, user_id, expires)
    return TokenResponse(access_token=new_access, refresh_token=new_refresh)
