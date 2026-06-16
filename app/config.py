from __future__ import annotations
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    # ------------------------------------------------------------------
    # Provider selection — change these in .env to swap implementations
    # ------------------------------------------------------------------
    vector_store_type: str = os.getenv("VECTOR_STORE_TYPE", "pgvector")
    # pgvector | chroma | qdrant

    llm_type: str = os.getenv("LLM_TYPE", "openai")
    # openai | gemini | ollama

    embeddings_type: str = os.getenv("EMBEDDINGS_TYPE", "openai")
    # openai | huggingface

    reranker_type: str = os.getenv("RERANKER_TYPE", "cohere")
    # cohere | cross_encoder | mmr | none

    cache_type: str = os.getenv("CACHE_TYPE", "redis")
    # redis | none

    chunker_type: str = os.getenv("CHUNKER_TYPE", "recursive")
    # recursive | semantic

    retriever_type: str = os.getenv("RETRIEVER_TYPE", "dense")
    # dense | bm25 | hybrid_interleaving | hybrid_rrf

    query_transformer_type: str = os.getenv("QUERY_TRANSFORMER_TYPE", "none")
    # none | hyde | decomposition | hyde_decomposition

    # ------------------------------------------------------------------
    # Connection strings
    # ------------------------------------------------------------------
    database_url: str = os.getenv("DATABASE_URL", "")
    redis_url: str = os.getenv("REDIS_URL", "")

    # ------------------------------------------------------------------
    # API Keys
    # ------------------------------------------------------------------
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    cohere_api_key: str = os.getenv("CO_API_KEY", "")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")

    # ------------------------------------------------------------------
    # Model names
    # ------------------------------------------------------------------
    llm_model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    embeddings_model: str = os.getenv("EMBEDDINGS_MODEL", "text-embedding-3-small")
    reranker_model: str = os.getenv("RERANKER_MODEL", "rerank-multilingual-v3.0")

    # ------------------------------------------------------------------
    # RAG tuning parameters
    # ------------------------------------------------------------------
    retrieval_k: int = int(os.getenv("RETRIEVAL_K", "5"))
    rrf_k: int = int(os.getenv("RRF_K", "60"))
    reranker_top_n: int = int(os.getenv("RERANKER_TOP_N", "3"))
    mmr_lambda_mult: float = float(os.getenv("MMR_LAMBDA_MULT", "0.5"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    semantic_breakpoint_threshold_type: str = os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD_TYPE", "percentile")
    semantic_breakpoint_threshold_amount: float = float(os.getenv("SEMANTIC_BREAKPOINT_THRESHOLD_AMOUNT", "95.0"))
    cache_distance_threshold: float = float(os.getenv("CACHE_DISTANCE_THRESHOLD", "0.98"))

    data_dir: str = os.getenv("DATA_DIR", "data")

    # ------------------------------------------------------------------
    # HNSW index parameters (pgvector)
    # ------------------------------------------------------------------
    hnsw_m: int = int(os.getenv("HNSW_M", "16"))
    hnsw_ef_construction: int = int(os.getenv("HNSW_EF_CONSTRUCTION", "64"))


# Singleton — import this everywhere instead of reading os.getenv directly
config = AppConfig()
