from __future__ import annotations
import logging

from app.ports.cache import CachePort

logger = logging.getLogger(__name__)


class NoneCacheAdapter(CachePort):
    """No-op cache adapter — disables LLM caching entirely.

    Use by setting CACHE_TYPE=none in .env.
    Useful for development, testing, or when Redis is not available.
    """

    def apply(self) -> None:
        logger.info("CACHE: No cache configured (CACHE_TYPE=none).")
