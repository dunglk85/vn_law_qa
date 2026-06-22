"""app/factory.py

Config-driven dependency factory.
─────────────────────────────────
To add a new provider:
  1. Create an adapter in app/adapters/ that implements the relevant Port.
  2. Add a ``@_register("kind", "key")`` factory function below.
  3. Set the corresponding *_TYPE env var in .env.

No business-logic files need to change.
"""
from __future__ import annotations

import logging
from collections.abc import Callable

from app.config import config
from app.ports.cache import CachePort
from app.ports.document_loader import DocumentLoaderPort
from app.ports.embeddings import EmbeddingsPort
from app.ports.llm import LLMPort
from app.ports.query_transformer import QueryTransformerPort
from app.ports.rate_limiter import RateLimiterPort
from app.ports.reranker import RerankerPort
from app.ports.retriever import RetrieverPort
from app.ports.session_store import SessionStorePort
from app.ports.vector_store import VectorStorePort

logger = logging.getLogger(__name__)

# ── registry ────────────────────────────────────────────────────────────────

_registry: dict[tuple[str, str], Callable] = {}


def _register(kind: str, key: str):
    """Decorator: register an adapter factory for the given kind + key.

    The decorated function is called lazily — imports happen at call time,
    preserving the existing lazy-import pattern and avoiding circular deps.
    """

    def decorator(fn: Callable):
        _registry[(kind, key)] = fn
        return fn

    return decorator


def _resolve(kind: str, key: str, **kwargs):
    """Look up an adapter factory by kind + key and invoke it."""
    supported = {k: fn for (kk, k), fn in _registry.items() if kk == kind}
    fn = supported.get(key)
    if fn is None:
        raise ValueError(
            f"Unknown {kind.upper()}_TYPE='{key}'. "
            f"Supported: {', '.join(sorted(supported.keys()))}"
        )
    return fn(**kwargs)


# ── adapter factories (registered) ──────────────────────────────────────────


@_register("embeddings", "openai")
def _create_openai_embeddings(model: str, api_key: str | None) -> EmbeddingsPort:
    from app.adapters.embeddings.openai_embeddings import OpenAIEmbeddingsAdapter

    return OpenAIEmbeddingsAdapter(model=model, api_key=api_key)


@_register("vector_store", "pgvector")
def _create_pgvector_store(
    embeddings: EmbeddingsPort,
    database_url: str,
    index_type: str,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_ef_search: int,
    ivfflat_lists: int,
    ivfflat_probes: int,
) -> VectorStorePort:
    from app.adapters.vector_stores.pgvector_store import PGVectorStoreAdapter

    return PGVectorStoreAdapter(
        database_url, embeddings,
        index_type=index_type,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_ef_search=hnsw_ef_search,
        ivfflat_lists=ivfflat_lists,
        ivfflat_probes=ivfflat_probes,
    )


@_register("llm", "openai")
def _create_openai_llm(model: str, api_key: str | None) -> LLMPort:
    from app.adapters.llms.openai_llm import OpenAILLMAdapter

    return OpenAILLMAdapter(model=model, api_key=api_key)


@_register("reranker", "cohere")
def _create_cohere_reranker(model: str, top_n: int, api_key: str | None) -> RerankerPort:
    from app.adapters.rerankers.cohere_reranker import CohereRerankerAdapter

    return CohereRerankerAdapter(model=model, top_n=top_n, api_key=api_key)


@_register("reranker", "cross_encoder")
def _create_cross_encoder_reranker(model: str, top_n: int) -> RerankerPort:
    from app.adapters.rerankers.cross_encoder_reranker import CrossEncoderRerankerAdapter

    return CrossEncoderRerankerAdapter(model=model, top_n=top_n)


@_register("reranker", "mmr")
def _create_mmr_reranker(embeddings, top_n: int, lambda_mult: float) -> RerankerPort:
    from app.adapters.rerankers.mmr_reranker import MMRRerankerAdapter

    return MMRRerankerAdapter(embeddings=embeddings, top_n=top_n, lambda_mult=lambda_mult)


@_register("reranker", "none")
def _create_none_reranker() -> RerankerPort:
    from app.adapters.rerankers.none_reranker import NoneRerankerAdapter

    return NoneRerankerAdapter()


@_register("cache", "redis")
def _create_redis_cache(
    redis_url: str,
    embeddings_port: EmbeddingsPort,
    distance_threshold: float,
) -> CachePort:
    from app.adapters.caches.redis_cache import RedisCacheAdapter

    return RedisCacheAdapter(
        redis_url=redis_url,
        embeddings_port=embeddings_port,
        distance_threshold=distance_threshold,
    )


@_register("cache", "none")
def _create_none_cache() -> CachePort:
    from app.adapters.caches.none_cache import NoneCacheAdapter

    return NoneCacheAdapter()


@_register("retriever", "dense")
def _create_dense_retriever(vector_store: VectorStorePort, k: int) -> RetrieverPort:
    from app.adapters.retrievers.dense_retriever import DenseRetrieverAdapter

    return DenseRetrieverAdapter(vector_store, k=k)


@_register("retriever", "bm25")
def _create_bm25_retriever(k: int) -> RetrieverPort:
    from app.adapters.retrievers.bm25_retriever import BM25RetrieverAdapter

    return BM25RetrieverAdapter(k=k)


@_register("retriever", "hybrid_interleaving")
def _create_hybrid_interleaving_retriever(vector_store: VectorStorePort, k: int) -> RetrieverPort:
    from app.adapters.retrievers.hybrid_interleaving_retriever import HybridInterleavingRetrieverAdapter

    return HybridInterleavingRetrieverAdapter(vector_store, k=k)


@_register("retriever", "hybrid_rrf")
def _create_hybrid_rrf_retriever(vector_store: VectorStorePort, k: int, rrf_k: int) -> RetrieverPort:
    from app.adapters.retrievers.hybrid_rrf_retriever import HybridRRFRetrieverAdapter

    return HybridRRFRetrieverAdapter(vector_store, k=k, rrf_k=rrf_k)


@_register("query_transformer", "none")
def _create_none_query_transformer() -> QueryTransformerPort:
    from app.adapters.query_transformers.none_transformer import NoneQueryTransformerAdapter

    return NoneQueryTransformerAdapter()


@_register("query_transformer", "hyde")
def _create_hyde_query_transformer(chat_model) -> QueryTransformerPort:
    from app.adapters.query_transformers.hyde_transformer import HyDEQueryTransformerAdapter

    return HyDEQueryTransformerAdapter(chat_model)


@_register("query_transformer", "decomposition")
def _create_decomposition_query_transformer(chat_model) -> QueryTransformerPort:
    from app.adapters.query_transformers.decomposition_transformer import DecompositionQueryTransformerAdapter

    return DecompositionQueryTransformerAdapter(chat_model)


@_register("query_transformer", "hyde_decomposition")
def _create_hyde_decomposition_query_transformer(chat_model) -> QueryTransformerPort:
    from app.adapters.query_transformers.hyde_decomposition_transformer import HyDEDecompositionQueryTransformerAdapter

    return HyDEDecompositionQueryTransformerAdapter(chat_model)


# ── public factory functions ────────────────────────────────────────────────


def create_embeddings() -> EmbeddingsPort:
    return _resolve("embeddings", config.embeddings_type,
                    model=config.embeddings_model,
                    api_key=config.openai_api_key)


def create_vector_store(embeddings: EmbeddingsPort | None = None) -> VectorStorePort:
    embeddings = embeddings or create_embeddings()
    return _resolve("vector_store", config.vector_store_type,
                    embeddings=embeddings,
                    database_url=config.database_url,
                    index_type=config.index_type,
                    hnsw_m=config.hnsw_m,
                    hnsw_ef_construction=config.hnsw_ef_construction,
                    hnsw_ef_search=config.hnsw_ef_search,
                    ivfflat_lists=config.ivfflat_lists,
                    ivfflat_probes=config.ivfflat_probes)


def create_llm() -> LLMPort:
    return _resolve("llm", config.llm_type,
                    model=config.llm_model,
                    api_key=config.openai_api_key)


def create_reranker(embeddings: EmbeddingsPort | None = None) -> RerankerPort:
    kw: dict = dict(model=config.reranker_model, top_n=config.reranker_top_n)

    if config.reranker_type == "cohere":
        kw["api_key"] = config.cohere_api_key
    elif config.reranker_type == "mmr":
        embeddings_port = embeddings or create_embeddings()
        kw["embeddings"] = embeddings_port.get_embeddings()
        kw["lambda_mult"] = config.mmr_lambda_mult

    return _resolve("reranker", config.reranker_type, **kw)


def create_cache(embeddings: EmbeddingsPort | None = None) -> CachePort:
    kw: dict = {}
    if config.cache_type == "redis":
        kw["redis_url"] = config.redis_url
        kw["embeddings_port"] = embeddings or create_embeddings()
        kw["distance_threshold"] = config.cache_distance_threshold

    return _resolve("cache", config.cache_type, **kw)


def create_retriever(vector_store: VectorStorePort) -> RetrieverPort:
    kw: dict = dict(k=config.retrieval_k)

    if config.retriever_type in ("dense", "hybrid_interleaving", "hybrid_rrf"):
        kw["vector_store"] = vector_store
    if config.retriever_type == "hybrid_rrf":
        kw["rrf_k"] = config.rrf_k

    return _resolve("retriever", config.retriever_type, **kw)


def create_query_transformer(llm: LLMPort) -> QueryTransformerPort:
    kw: dict = {}
    if config.query_transformer_type != "none":
        kw["chat_model"] = llm.get_chat_model()

    return _resolve("query_transformer", config.query_transformer_type, **kw)


def create_document_loader() -> DocumentLoaderPort:
    from app.adapters.document_loaders.parquet_loader import ParquetLoaderAdapter

    return ParquetLoaderAdapter()


# --------------------------------------------------------------------------- #
# Agents (LangGraph-based)                                                     #
# --------------------------------------------------------------------------- #


def create_legal_research_agent(retriever: RetrieverPort, llm: LLMPort):
    from app.agents.legal_research_agent import LegalResearchAgent

    return LegalResearchAgent(retriever, llm.get_chat_model())


def create_citation_checker_agent(vector_store: VectorStorePort, llm: LLMPort):
    from app.agents.citation_checker_agent import CitationCheckerAgent

    return CitationCheckerAgent(vector_store, llm.get_chat_model())


def create_response_synthesizer_agent(llm: LLMPort):
    from app.agents.response_synthesizer_agent import ResponseSynthesizerAgent

    return ResponseSynthesizerAgent(llm.get_chat_model())


def create_supervisor_agent(
    research_agent,
    citation_agent,
    synthesis_agent,
    llm: LLMPort,
    knowledge_search_tool=None,
    a2a_client=None,
):
    from app.agents.supervisor_agent import SupervisorAgent

    return SupervisorAgent(
        research_agent=research_agent,
        citation_agent=citation_agent,
        synthesis_agent=synthesis_agent,
        llm=llm.get_chat_model(),
        knowledge_search_tool=knowledge_search_tool,
        a2a_client=a2a_client,
    )


# --------------------------------------------------------------------------- #
# A2A Client (Agent-to-Agent Protocol)                                        #
# --------------------------------------------------------------------------- #


def create_a2a_client(research_agent=None, citation_agent=None, synthesis_agent=None):
    from app.adapters.agents.a2a_fallback_client import InProcessFallbackClient
    from app.adapters.agents.a2a_remote_client import A2ARemoteClient

    legal_url = config.a2a_legal_research_url
    citation_url = config.a2a_citation_checker_url
    synthesis_url = config.a2a_response_synthesizer_url

    if legal_url or citation_url or synthesis_url:
        agent_map = {}
        if legal_url:
            agent_map["legal-research-agent"] = legal_url
        if citation_url:
            agent_map["citation-checker-agent"] = citation_url
        if synthesis_url:
            agent_map["response-synthesizer-agent"] = synthesis_url

        logger.info("A2A client: remote mode (%s)", agent_map)
        return A2ARemoteClient(
            agent_map=agent_map,
            timeout=config.a2a_task_timeout,
        )

    logger.info("A2A client: in-process fallback")
    if not all([research_agent, citation_agent, synthesis_agent]):
        raise ValueError("All agents required for in-process fallback")
    return InProcessFallbackClient(
        research_agent=research_agent,
        citation_agent=citation_agent,
        synthesis_agent=synthesis_agent,
    )


def create_session_store() -> SessionStorePort:
    if config.redis_url:
        try:
            from app.adapters.session_stores.redis_session_store import RedisSessionStore

            store = RedisSessionStore(
                redis_url=config.redis_url,
                ttl_seconds=config.session_ttl_seconds,
            )
            logger.info("Session store: Redis-backed")
            return store
        except Exception:
            logger.warning("Redis connection failed for session store, falling back to in-memory")

    from app.adapters.session_stores.memory_session_store import MemorySessionStore

    logger.warning("No REDIS_URL configured, session store using in-memory (single-instance only)")
    return MemorySessionStore(ttl_seconds=config.session_ttl_seconds)


# --------------------------------------------------------------------------- #
# Knowledge Search Tool (MCP-backed with direct fallback)                     #
# --------------------------------------------------------------------------- #


def create_knowledge_search_tool(retriever_port: RetrieverPort | None = None):
    """Return a knowledge_search callable — either MCP-backed or direct.

    When MCP_ENABLED=true, returns an MCP-backed tool that connects lazily
    on first invocation (no asyncio.run() at import time).
    When MCP_ENABLED=false (default), returns the existing direct @tool.
    """
    if not config.mcp_enabled:
        from app.agents.tools.knowledge_search import create_knowledge_search_tool as _direct

        if retriever_port is None:
            raise ValueError("retriever_port required when MCP_ENABLED=false")
        return _direct(retriever_port, k=config.retrieval_k)

    from app.adapters.tools.mcp_tool_adapter import create_mcp_knowledge_search_tool_lazy

    logger.info("Knowledge search tool: MCP-backed (lazy initialization)")
    return create_mcp_knowledge_search_tool_lazy(
        server_timeout=config.mcp_server_timeout,
        max_restarts=config.mcp_max_restarts,
    )


def create_rate_limiter() -> RateLimiterPort:
    from app.adapters.rate_limiters.memory_rate_limiter import MemoryRateLimiterAdapter
    from app.adapters.rate_limiters.redis_rate_limiter import RedisRateLimiterAdapter

    if config.redis_url:
        try:
            limiter = RedisRateLimiterAdapter(
                max_requests=config.rate_limit_max,
                window_seconds=config.rate_limit_window,
                redis_url=config.redis_url,
            )
            logger.info("Rate limiter: Redis-backed (multi-instance ready)")
            return limiter
        except Exception:
            logger.warning("Redis connection failed for rate limiter, falling back to in-memory")

    logger.warning("No REDIS_URL configured, rate limiter falling back to in-memory (single-instance only)")
    return MemoryRateLimiterAdapter(
        max_requests=config.rate_limit_max,
        window_seconds=config.rate_limit_window,
    )
