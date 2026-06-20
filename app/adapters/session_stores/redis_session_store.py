from __future__ import annotations

import json
import logging

import redis.asyncio as aioredis

from app.ports.session_store import SessionStorePort

logger = logging.getLogger(__name__)


class RedisSessionStore(SessionStorePort):
    def __init__(self, redis_url: str, ttl_seconds: int = 3600, prefix: str = "session:") -> None:
        self._ttl = ttl_seconds
        self._prefix = prefix
        self._redis: aioredis.Redis | None = None
        self._redis_url = redis_url

    async def _get_redis(self) -> aioredis.Redis:
        if self._redis is None:
            self._redis = aioredis.from_url(self._redis_url, decode_responses=True)
        return self._redis

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    async def load(self, session_id: str) -> list[dict]:
        r = await self._get_redis()
        raw = await r.get(self._key(session_id))
        if raw is None:
            return []
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Corrupt session data for %s, resetting", session_id)
            return []

    async def save(self, session_id: str, history: list[dict]) -> None:
        r = await self._get_redis()
        raw = json.dumps(history, default=str)
        await r.setex(self._key(session_id), self._ttl, raw)

    async def delete(self, session_id: str) -> None:
        r = await self._get_redis()
        await r.delete(self._key(session_id))

    async def exists(self, session_id: str) -> bool:
        r = await self._get_redis()
        return await r.exists(self._key(session_id)) > 0
