from __future__ import annotations

from app.ports.cache import CachePort


class NoneCacheAdapter(CachePort):
    """No-op cache adapter — disables LLM caching entirely.

    Use by setting CACHE_TYPE=none in .env.
    Useful for development, testing, or when Redis is not available.
    """

    def apply(self) -> None:
        print("CACHE: No cache configured (CACHE_TYPE=none).")
