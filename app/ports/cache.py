from __future__ import annotations
from abc import ABC, abstractmethod


class CachePort(ABC):
    """Abstract interface for LLM caching backends.

    Swap Redis → InMemory/DynamoDB by creating a new adapter that implements
    this interface, then setting CACHE_TYPE in .env.
    Setting CACHE_TYPE=none uses the NoneCacheAdapter (no caching).
    """

    @abstractmethod
    def apply(self) -> None:
        """Activate the cache globally via LangChain's set_llm_cache().

        Implementations should call:
            from langchain_core.globals import set_llm_cache
            set_llm_cache(<concrete_cache_instance>)
        """
        ...
