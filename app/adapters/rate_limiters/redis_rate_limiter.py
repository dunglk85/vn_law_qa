from __future__ import annotations

import logging
import time

import redis.asyncio as aioredis

from app.ports.rate_limiter import RateLimiterPort

logger = logging.getLogger(__name__)


class RedisRateLimiterAdapter(RateLimiterPort):
    def __init__(self, max_requests: int, window_seconds: float, redis_url: str) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._redis = aioredis.from_url(redis_url, decode_responses=True)

    async def check(self, client_ip: str) -> bool:
        now = time.time()
        window = int(now // self._window_seconds)
        key = f"ratelimit:{client_ip}:{window}"
        try:
            count = await self._redis.incr(key)
            if count == 1:
                await self._redis.expire(key, int(self._window_seconds))
            return count <= self._max_requests
        except Exception as exc:
            logger.warning("Redis rate limiter check failed, allowing request: %s", exc)
            return True
