from __future__ import annotations

import logging
import os
from dataclasses import dataclass

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


def _pos_int_env(key: str, default: int) -> int:
    return max(1, _int_env(key, default))


def _float_env(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid %s='%s', falling back to %s", key, raw, default)
        return default


def _pos_float_env(key: str, default: float) -> float:
    return max(0.1, _float_env(key, default))


def _str_env(key: str, default: str) -> str:
    raw = os.getenv(key)
    return raw if raw is not None else default


@dataclass(frozen=True)
class AppConfig:
    # ------------------------------------------------------------------
    # Provider selection — change these in .env to swap implementations
    # ------------------------------------------------------------------
    vector_store_type: str = _str_env("VECTOR_STORE_TYPE", "pgvector")

    llm_type: str = _str_env("LLM_TYPE", "openai")
    # openai | gemini | ollama

    embeddings_type: str = _str_env("EMBEDDINGS_TYPE", "openai")
    # openai | huggingface

    reranker_type: str = _str_env("RERANKER_TYPE", "cohere")
    # cohere | cross_encoder | mmr | none

    cache_type: str = _str_env("CACHE_TYPE", "redis")
    # redis | none

    retriever_type: str = _str_env("RETRIEVER_TYPE", "dense")
    # dense | bm25 | hybrid_interleaving | hybrid_rrf

    query_transformer_type: str = _str_env("QUERY_TRANSFORMER_TYPE", "none")
    # none | hyde | decomposition | hyde_decomposition

    document_loader_type: str = _str_env("DOCUMENT_LOADER_TYPE", "parquet")
    # parquet

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
    retrieval_k: int = _pos_int_env("RETRIEVAL_K", 5)
    rrf_k: int = _pos_int_env("RRF_K", 60)
    reranker_top_n: int = _int_env("RERANKER_TOP_N", 3)
    mmr_lambda_mult: float = _float_env("MMR_LAMBDA_MULT", 0.5)
    cache_distance_threshold: float = _float_env("CACHE_DISTANCE_THRESHOLD", 0.98)

    data_dir: str = _str_env("DATA_DIR", "data")

    # ------------------------------------------------------------------
    # Index parameters (pgvector)
    # ------------------------------------------------------------------
    index_type: str = _str_env("INDEX_TYPE", "hnsw")
    # hnsw | ivfflat

    hnsw_m: int = _int_env("HNSW_M", 16)
    hnsw_ef_construction: int = _int_env("HNSW_EF_CONSTRUCTION", 200)
    hnsw_ef_search: int = _int_env("HNSW_EF_SEARCH", 50)

    ivfflat_lists: int = _int_env("IVFFLAT_LISTS", 100)
    ivfflat_probes: int = _int_env("IVFFLAT_PROBES", 10)

    # ------------------------------------------------------------------
    # Agent / Agentic RAG parameters
    # ------------------------------------------------------------------
    llm_timeout: float = _float_env("LLM_TIMEOUT", 30.0)
    agent_timeout: float = _float_env("AGENT_TIMEOUT", 90.0)
    ask_timeout: float = _float_env("ASK_TIMEOUT", 120.0)
    rate_limit_max: int = _pos_int_env("RATE_LIMIT_MAX", 30)
    rate_limit_window: float = _pos_float_env("RATE_LIMIT_WINDOW", 60.0)

    max_retries: int = _pos_int_env("MAX_RETRIES", 2)
    quality_threshold: float = _float_env("QUALITY_THRESHOLD", 0.75)
    n_results_per_vector: int = _int_env("N_RESULTS_PER_VECTOR", 5)
    top_k_research: int = _int_env("TOP_K_RESEARCH", 5)
    top_k_llm_score: int = _int_env("TOP_K_LLM_SCORE", 8)
    hyde_enabled: bool = os.getenv("HYDE_ENABLED", "true").lower() == "true"
    subquery_count: int = _int_env("SUBQUERY_COUNT", 3)
    relevance_threshold: float = _float_env("RELEVANCE_THRESHOLD", 0.5)

    # ------------------------------------------------------------------
    # Retry / Error Recovery
    # ------------------------------------------------------------------
    tool_retry_max_attempts: int = _int_env("TOOL_RETRY_MAX_ATTEMPTS", 2)
    tool_retry_base_delay: float = _float_env("TOOL_RETRY_BASE_DELAY", 1.0)

    # ------------------------------------------------------------------
    # Session / Memory
    # ------------------------------------------------------------------
    session_ttl_seconds: int = _int_env("SESSION_TTL_SECONDS", 3600)
    max_history_tokens: int = _int_env("MAX_HISTORY_TOKENS", 4096)
    recent_turns_to_keep: int = _int_env("RECENT_TURNS_TO_KEEP", 4)

    # ------------------------------------------------------------------
    # Auth / JWT
    # ------------------------------------------------------------------
    jwt_secret: str = _str_env("JWT_SECRET", "change-me-in-production")
    jwt_algorithm: str = _str_env("JWT_ALGORITHM", "HS256")
    access_token_expire_minutes: int = _int_env("ACCESS_TOKEN_EXPIRE_MINUTES", 30)
    refresh_token_expire_days: int = _int_env("REFRESH_TOKEN_EXPIRE_DAYS", 7)
    admin_username: str = _str_env("ADMIN_USERNAME", "admin")
    admin_password: str = _str_env("ADMIN_PASSWORD", "admin")

    # ------------------------------------------------------------------
    # MCP (Model Context Protocol)
    # ------------------------------------------------------------------
    mcp_enabled: bool = os.getenv("MCP_ENABLED", "false").lower() == "true"
    mcp_server_timeout: int = _int_env("MCP_SERVER_TIMEOUT", 30)
    mcp_max_restarts: int = _int_env("MCP_MAX_RESTARTS", 3)

    # ------------------------------------------------------------------
    # A2A (Agent-to-Agent Protocol)
    # ------------------------------------------------------------------
    a2a_legal_research_url: str = _str_env("A2A_LEGAL_RESEARCH_URL", "")
    a2a_citation_checker_url: str = _str_env("A2A_CITATION_CHECKER_URL", "")
    a2a_response_synthesizer_url: str = _str_env("A2A_RESPONSE_SYNTHESIZER_URL", "")
    a2a_task_timeout: int = _int_env("A2A_TASK_TIMEOUT", 25)
    a2a_max_retries: int = _int_env("A2A_MAX_RETRIES", 1)

    # ------------------------------------------------------------------
    # LangSmith (Observability)
    # ------------------------------------------------------------------
    langsmith_tracing: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    langsmith_project: str = _str_env("LANGSMITH_PROJECT", "company-knowledge-assistant")

    # ------------------------------------------------------------------
    # Pipeline stage config (agentic pipeline per-stage model control)
    # ------------------------------------------------------------------
    pipeline_stage_order: str = _str_env(
        "PIPELINE_STAGE_ORDER",
        "router,planner,reasoner,tool_caller,code_writer,evaluator",
    )

    pipeline_model_router: str = _str_env("PIPELINE_MODEL_ROUTER", "")
    pipeline_model_planner: str = _str_env("PIPELINE_MODEL_PLANNER", "")
    pipeline_model_reasoner: str = _str_env("PIPELINE_MODEL_REASONER", "")
    pipeline_model_tool_caller: str = _str_env("PIPELINE_MODEL_TOOL_CALLER", "")
    pipeline_model_code_writer: str = _str_env("PIPELINE_MODEL_CODE_WRITER", "")
    pipeline_model_evaluator: str = _str_env("PIPELINE_MODEL_EVALUATOR", "")

    pipeline_prompt_router: str = _str_env("PIPELINE_PROMPT_ROUTER", "")
    pipeline_prompt_planner: str = _str_env("PIPELINE_PROMPT_PLANNER", "")
    pipeline_prompt_reasoner: str = _str_env("PIPELINE_PROMPT_REASONER", "")
    pipeline_prompt_tool_caller: str = _str_env("PIPELINE_PROMPT_TOOL_CALLER", "")
    pipeline_prompt_code_writer: str = _str_env("PIPELINE_PROMPT_CODE_WRITER", "")
    pipeline_prompt_evaluator: str = _str_env("PIPELINE_PROMPT_EVALUATOR", "")


# Singleton — import this everywhere instead of reading os.getenv directly
config = AppConfig()

# --- Startup validation ---
if config.jwt_secret == "change-me-in-production":
    from app.exceptions import ConfigurationError
    raise ConfigurationError(
        "JWT_SECRET is using the default value 'change-me-in-production'. "
        "Set a strong secret via the JWT_SECRET env var."
    )

if config.admin_password == "admin":
    from app.exceptions import ConfigurationError
    raise ConfigurationError(
        "ADMIN_PASSWORD is 'admin'. "
        "Set a strong password via the ADMIN_PASSWORD env var."
    )
