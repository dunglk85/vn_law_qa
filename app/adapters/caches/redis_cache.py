from __future__ import annotations

from langchain_core.globals import set_llm_cache
from langchain_redis import RedisSemanticCache

from app.ports.cache import CachePort
from app.ports.embeddings import EmbeddingsPort


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

    def apply(self) -> None:
        """Activate Redis semantic cache globally for all LangChain LLM calls."""
        cache = RedisSemanticCache(
            redis_url=self._redis_url,
            embeddings=self._embeddings,
            distance_threshold=self._distance_threshold,
        )
        set_llm_cache(cache)
        print("CACHE: Redis semantic cache activated.")
