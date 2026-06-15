"""app/factory.py

Config-driven dependency factory.
─────────────────────────────────
To add a new provider:
  1. Create an adapter in app/adapters/ that implements the relevant Port.
  2. Add a `case` block here.
  3. Set the corresponding *_TYPE env var in .env.

No business-logic files need to change.
"""
from __future__ import annotations

from app.config import config
from app.ports.embeddings import EmbeddingsPort
from app.ports.vector_store import VectorStorePort
from app.ports.llm import LLMPort
from app.ports.reranker import RerankerPort
from app.ports.cache import CachePort


# --------------------------------------------------------------------------- #
# Embeddings                                                                   #
# --------------------------------------------------------------------------- #

def create_embeddings() -> EmbeddingsPort:
    match config.embeddings_type:
        case "openai":
            from app.adapters.embeddings.openai_embeddings import OpenAIEmbeddingsAdapter
            return OpenAIEmbeddingsAdapter(model=config.embeddings_model)
        # ── add new providers below ──────────────────────────────────────────
        # case "huggingface":
        #     from app.adapters.embeddings.hf_embeddings import HFEmbeddingsAdapter
        #     return HFEmbeddingsAdapter(model=config.embeddings_model)
        case _:
            raise ValueError(
                f"Unknown EMBEDDINGS_TYPE='{config.embeddings_type}'. "
                "Supported: openai"
            )


# --------------------------------------------------------------------------- #
# Vector Store                                                                 #
# --------------------------------------------------------------------------- #

def create_vector_store(embeddings: EmbeddingsPort | None = None) -> VectorStorePort:
    embeddings = embeddings or create_embeddings()
    match config.vector_store_type:
        case "pgvector":
            from app.adapters.vector_stores.pgvector_store import PGVectorStoreAdapter
            return PGVectorStoreAdapter(config.database_url, embeddings)
        # ── add new providers below ──────────────────────────────────────────
        # case "chroma":
        #     from app.adapters.vector_stores.chroma_store import ChromaStoreAdapter
        #     return ChromaStoreAdapter(embeddings)
        case _:
            raise ValueError(
                f"Unknown VECTOR_STORE_TYPE='{config.vector_store_type}'. "
                "Supported: pgvector"
            )


# --------------------------------------------------------------------------- #
# LLM                                                                          #
# --------------------------------------------------------------------------- #

def create_llm() -> LLMPort:
    match config.llm_type:
        case "openai":
            from app.adapters.llms.openai_llm import OpenAILLMAdapter
            return OpenAILLMAdapter(model=config.llm_model)
        # ── add new providers below ──────────────────────────────────────────
        # case "gemini":
        #     from app.adapters.llms.gemini_llm import GeminiLLMAdapter
        #     return GeminiLLMAdapter(model=config.llm_model)
        # case "ollama":
        #     from app.adapters.llms.ollama_llm import OllamaLLMAdapter
        #     return OllamaLLMAdapter(model=config.llm_model)
        case _:
            raise ValueError(
                f"Unknown LLM_TYPE='{config.llm_type}'. "
                "Supported: openai"
            )


# --------------------------------------------------------------------------- #
# Reranker                                                                     #
# --------------------------------------------------------------------------- #

def create_reranker() -> RerankerPort:
    match config.reranker_type:
        case "cohere":
            from app.adapters.rerankers.cohere_reranker import CohereRerankerAdapter
            return CohereRerankerAdapter(
                model=config.reranker_model,
                top_n=config.reranker_top_n,
            )
        case "none":
            from app.adapters.rerankers.none_reranker import NoneRerankerAdapter
            return NoneRerankerAdapter()
        # ── add new providers below ──────────────────────────────────────────
        # case "jina":
        #     from app.adapters.rerankers.jina_reranker import JinaRerankerAdapter
        #     return JinaRerankerAdapter(model=config.reranker_model, top_n=config.reranker_top_n)
        case _:
            raise ValueError(
                f"Unknown RERANKER_TYPE='{config.reranker_type}'. "
                "Supported: cohere, none"
            )


# --------------------------------------------------------------------------- #
# Cache                                                                        #
# --------------------------------------------------------------------------- #

def create_cache(embeddings: EmbeddingsPort | None = None) -> CachePort:
    match config.cache_type:
        case "redis":
            from app.adapters.caches.redis_cache import RedisCacheAdapter
            return RedisCacheAdapter(
                redis_url=config.redis_url,
                embeddings_port=embeddings or create_embeddings(),
                distance_threshold=config.cache_distance_threshold,
            )
        case "none":
            from app.adapters.caches.none_cache import NoneCacheAdapter
            return NoneCacheAdapter()
        # ── add new providers below ──────────────────────────────────────────
        case _:
            raise ValueError(
                f"Unknown CACHE_TYPE='{config.cache_type}'. "
                "Supported: redis, none"
            )
