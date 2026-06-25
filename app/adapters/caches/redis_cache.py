from __future__ import annotations

import logging

from langchain_core.globals import set_llm_cache
from langchain_redis import RedisSemanticCache

from app.ports.cache import CachePort
from app.ports.embeddings import EmbeddingsPort

logger = logging.getLogger(__name__)


class RedisCacheAdapter(CachePort):
    """Concrete adapter for Redis Semantic Cache via langchain-redis."""

    def __init__(
        self,
        redis_url: str,
        embeddings_port: EmbeddingsPort,
        distance_threshold: float = 0.98,
    ) -> None:
        self._redis_url = redis_url
        self._embeddings = embeddings_port.get_embeddings()
        self._distance_threshold = distance_threshold
        self._cache_instance: RedisSemanticCache | None = None

    def apply(self) -> None:
        """Activate Redis semantic cache globally for all LangChain LLM calls."""
        if self._cache_instance is not None:
            logger.info("CACHE: Redis semantic cache already active, skipping re-init.")
            return
        self._cache_instance = RedisSemanticCache(
            redis_url=self._redis_url,
            embeddings=self._embeddings,
            distance_threshold=self._distance_threshold,
        )
        set_llm_cache(self._cache_instance)
        logger.info("CACHE: Redis semantic cache activated.")
