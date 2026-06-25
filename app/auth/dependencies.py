from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config import config


async def require_api_key(x_api_key: str = Header("", alias="X-API-Key")) -> str:
    if not config.app_api_key:
        return "mvp-mode"
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header",
        )
    if x_api_key != config.app_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    return x_api_key
