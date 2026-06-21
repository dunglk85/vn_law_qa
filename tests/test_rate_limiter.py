from __future__ import annotations

import time
from unittest.mock import AsyncMock, patch

import pytest

from app.adapters.rate_limiters.memory_rate_limiter import MemoryRateLimiterAdapter
from app.adapters.rate_limiters.redis_rate_limiter import RedisRateLimiterAdapter


class TestMemoryRateLimiter:
    @pytest.mark.asyncio
    async def test_allows_requests_under_limit(self):
        limiter = MemoryRateLimiterAdapter(max_requests=5, window_seconds=60.0)
        ip = "192.168.1.1"
        for _ in range(5):
            assert await limiter.check(ip) is True

    @pytest.mark.asyncio
    async def test_blocks_requests_over_limit(self):
        limiter = MemoryRateLimiterAdapter(max_requests=3, window_seconds=60.0)
        ip = "192.168.1.2"
        for _ in range(3):
            assert await limiter.check(ip) is True
        assert await limiter.check(ip) is False

    @pytest.mark.asyncio
    async def test_window_slides_after_time_elapses(self):
        limiter = MemoryRateLimiterAdapter(max_requests=2, window_seconds=0.05)
        ip = "192.168.1.3"
        assert await limiter.check(ip) is True
        assert await limiter.check(ip) is True
        assert await limiter.check(ip) is False
        time.sleep(0.06)
        assert await limiter.check(ip) is True

    @pytest.mark.asyncio
    async def test_different_ips_have_independent_counters(self):
        limiter = MemoryRateLimiterAdapter(max_requests=1, window_seconds=60.0)
        assert await limiter.check("ip-a") is True
        assert await limiter.check("ip-a") is False
        assert await limiter.check("ip-b") is True

    @pytest.mark.asyncio
    async def test_zero_max_requests_blocks_everything(self):
        limiter = MemoryRateLimiterAdapter(max_requests=0, window_seconds=60.0)
        assert await limiter.check("any-ip") is False

    @pytest.mark.asyncio
    async def test_unknown_ip_returns_unknown(self):
        limiter = MemoryRateLimiterAdapter(max_requests=5, window_seconds=60.0)
        assert await limiter.check("new-ip") is True


class TestRedisRateLimiter:
    @pytest.mark.asyncio
    async def test_allows_under_limit(self):
        with patch("app.adapters.rate_limiters.redis_rate_limiter.aioredis.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.incr = AsyncMock(return_value=1)
            mock_redis.expire = AsyncMock(return_value=True)
            mock_from_url.return_value = mock_redis

            limiter = RedisRateLimiterAdapter(max_requests=5, window_seconds=60.0, redis_url="redis://localhost:6379/0")
            result = await limiter.check("192.168.1.1")
            assert result is True

    @pytest.mark.asyncio
    async def test_blocks_over_limit(self):
        with patch("app.adapters.rate_limiters.redis_rate_limiter.aioredis.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.incr = AsyncMock(return_value=6)
            mock_redis.expire = AsyncMock(return_value=True)
            mock_from_url.return_value = mock_redis

            limiter = RedisRateLimiterAdapter(max_requests=5, window_seconds=60.0, redis_url="redis://localhost:6379/0")
            result = await limiter.check("192.168.1.1")
            assert result is False

    @pytest.mark.asyncio
    async def test_sets_expire_on_first_request(self):
        with patch("app.adapters.rate_limiters.redis_rate_limiter.aioredis.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.incr = AsyncMock(return_value=1)
            mock_redis.expire = AsyncMock(return_value=True)
            mock_from_url.return_value = mock_redis

            limiter = RedisRateLimiterAdapter(max_requests=5, window_seconds=60.0, redis_url="redis://localhost:6379/0")
            await limiter.check("192.168.1.1")
            mock_redis.expire.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_set_expire_on_subsequent_requests(self):
        with patch("app.adapters.rate_limiters.redis_rate_limiter.aioredis.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.incr = AsyncMock(return_value=3)
            mock_redis.expire = AsyncMock(return_value=True)
            mock_from_url.return_value = mock_redis

            limiter = RedisRateLimiterAdapter(max_requests=5, window_seconds=60.0, redis_url="redis://localhost:6379/0")
            await limiter.check("192.168.1.1")
            mock_redis.expire.assert_not_called()

    @pytest.mark.asyncio
    async def test_key_format_includes_ip_and_window(self):
        with patch("app.adapters.rate_limiters.redis_rate_limiter.aioredis.from_url") as mock_from_url:
            mock_redis = AsyncMock()
            mock_redis.incr = AsyncMock(return_value=1)
            mock_redis.expire = AsyncMock(return_value=True)
            mock_from_url.return_value = mock_redis

            limiter = RedisRateLimiterAdapter(max_requests=5, window_seconds=60.0, redis_url="redis://localhost:6379/0")
            with patch("time.time", return_value=1000.0):
                await limiter.check("10.0.0.1")
                expected_key = "ratelimit:10.0.0.1:16"
                mock_redis.incr.assert_awaited_once_with(expected_key)
