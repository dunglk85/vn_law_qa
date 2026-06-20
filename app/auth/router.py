from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.auth.jwt import TOKEN_SCHEMA, create_access_token, create_refresh_token, decode_token, validate_claims
from app.config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# In-memory refresh token store (DB persistence deferred to later story)
_refresh_tokens: dict[str, dict[str, Any]] = {}


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
    if req.username != "admin" or req.password != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )
    access_token = create_access_token(user_id=req.username, tenant_id="default", roles=["admin"])
    refresh_token = create_refresh_token()
    _refresh_tokens[refresh_token] = {
        "user_id": req.username,
        "expires": datetime.now(timezone.utc) + timedelta(days=config.refresh_token_expire_days),
    }
    return TokenResponse(access_token=access_token, refresh_token=refresh_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(req: RefreshRequest) -> TokenResponse:
    stored = _refresh_tokens.pop(req.refresh_token, None)
    if stored is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    if datetime.now(timezone.utc) > stored["expires"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token expired, please log in again",
        )
    user_id = stored["user_id"]
    new_access = create_access_token(user_id=user_id, tenant_id="default", roles=["admin"])
    new_refresh = create_refresh_token()
    _refresh_tokens[new_refresh] = {
        "user_id": user_id,
        "expires": datetime.now(timezone.utc) + timedelta(days=config.refresh_token_expire_days),
    }
    return TokenResponse(access_token=new_access, refresh_token=new_refresh)
