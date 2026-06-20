from __future__ import annotations
import os
import logging
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def _int_env(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid %s='%s', falling back to %s", key, raw, default)
        return default


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid %s='%s', falling back to %s", key, raw, default)
        return default


def _str_env(key: str, default: str) -> str:
    raw = os.getenv(key)
    return raw if raw is not None else default


@dataclass(frozen=True)
class AppConfig:
    # ------------------------------------------------------------------
    # Provider selection — change these in .env to swap implementations
    # ------------------------------------------------------------------
    vector_store_type: str = _str_env("VECTOR_STORE_TYPE", "pgvector")
    # pgvector | chroma | qdrant

    llm_type: str = _str_env("LLM_TYPE", "openai")
    # openai | gemini | ollama

    embeddings_type: str = _str_env("EMBEDDINGS_TYPE", "openai")
    # openai | huggingface

    reranker_type: str = _str_env("RERANKER_TYPE", "cohere")
    # cohere | cross_encoder | mmr | none

    cache_type: str = _str_env("CACHE_TYPE", "redis")
    # redis | none

    chunker_type: str = _str_env("CHUNKER_TYPE", "recursive")
    # recursive | semantic

    retriever_type: str = _str_env("RETRIEVER_TYPE", "dense")
    # dense | bm25 | hybrid_interleaving | hybrid_rrf

    query_transformer_type: str = _str_env("QUERY_TRANSFORMER_TYPE", "none")
    # none | hyde | decomposition | hyde_decomposition

    metadata_enricher_type: str = _str_env("METADATA_ENRICHER_TYPE", "basic")
    # basic | llm | none

    rag_mode: str = _str_env("RAG_MODE", "legacy")
    # legacy (RAGService) | agentic (AgenticService)

    # ------------------------------------------------------------------
    # Connection strings
    # ------------------------------------------------------------------
    database_url: str = _str_env("DATABASE_URL", "")
    redis_url: str = _str_env("REDIS_URL", "")

    # ------------------------------------------------------------------
    # API Keys
    # ------------------------------------------------------------------
    openai_api_key: str = _str_env("OPENAI_API_KEY", "")
    cohere_api_key: str = _str_env("CO_API_KEY", "")
    langsmith_api_key: str = _str_env("LANGSMITH_API_KEY", "")

    # ------------------------------------------------------------------
    # Model names
    # ------------------------------------------------------------------
    llm_model: str = _str_env("LLM_MODEL", "gpt-4o-mini")
    embeddings_model: str = _str_env("EMBEDDINGS_MODEL", "text-embedding-3-small")
    reranker_model: str = _str_env("RERANKER_MODEL", "rerank-multilingual-v3.0")

    # ------------------------------------------------------------------
    # RAG tuning parameters
    # ------------------------------------------------------------------
    retrieval_k: int = _int_env("RETRIEVAL_K", 5)
    rrf_k: int = _int_env("RRF_K", 60)
    reranker_top_n: int = _int_env("RERANKER_TOP_N", 3)
    mmr_lambda_mult: float = _float_env("MMR_LAMBDA_MULT", 0.5)
    chunk_size: int = _int_env("CHUNK_SIZE", 900)
    chunk_overlap: int = _int_env("CHUNK_OVERLAP", 120)
    semantic_breakpoint_threshold_type: str = _str_env("SEMANTIC_BREAKPOINT_THRESHOLD_TYPE", "percentile")
    semantic_breakpoint_threshold_amount: float = _float_env("SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT", 95.0)
    cache_distance_threshold: float = _float_env("CACHE_DISTANCE_THRESHOLD", 0.98)

    data_dir: str = _str_env("DATA_DIR", "data")

    # ------------------------------------------------------------------
    # Index parameters (pgvector)
    # ------------------------------------------------------------------
    index_type: str = _str_env("INDEX_TYPE", "hnsw")
    # hnsw | ivfflat

    hnsw_m: int = _int_env("HNSW_M", 16)
    hnsw_ef_construction: int = _int_env("HNSW_EF_CONSTRUCTION", 64)

    ivfflat_lists: int = _int_env("IVFFLAT_LISTS", 100)
    ivfflat_probes: int = _int_env("IVFFLAT_PROBES", 10)


# Singleton — import this everywhere instead of reading os.getenv directly
config = AppConfig()
