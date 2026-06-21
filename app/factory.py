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

import logging

from app.config import config
from app.ports.cache import CachePort
from app.ports.chunking import ChunkingPort
from app.ports.embeddings import EmbeddingsPort
from app.ports.llm import LLMPort
from app.ports.metadata_enrichment import MetadataEnrichmentPort
from app.ports.query_transformer import QueryTransformerPort
from app.ports.rate_limiter import RateLimiterPort
from app.ports.reranker import RerankerPort
from app.ports.retriever import RetrieverPort
from app.ports.session_store import SessionStorePort
from app.ports.vector_store import VectorStorePort

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Embeddings                                                                   #
# --------------------------------------------------------------------------- #

def create_embeddings() -> EmbeddingsPort:
    match config.embeddings_type:
        case "openai":
            from app.adapters.embeddings.openai_embeddings import OpenAIEmbeddingsAdapter
            return OpenAIEmbeddingsAdapter(model=config.embeddings_model, api_key=config.openai_api_key)
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
            return PGVectorStoreAdapter(
                config.database_url, embeddings,
                index_type=config.index_type,
                hnsw_m=config.hnsw_m,
                hnsw_ef_construction=config.hnsw_ef_construction,
                ivfflat_lists=config.ivfflat_lists,
                ivfflat_probes=config.ivfflat_probes,
            )
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
            return OpenAILLMAdapter(model=config.llm_model, api_key=config.openai_api_key)
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

def create_reranker(embeddings: EmbeddingsPort | None = None) -> RerankerPort:
    match config.reranker_type:
        case "cohere":
            from app.adapters.rerankers.cohere_reranker import CohereRerankerAdapter
            return CohereRerankerAdapter(
                model=config.reranker_model,
                top_n=config.reranker_top_n,
                api_key=config.cohere_api_key,
            )
        case "cross_encoder":
            from app.adapters.rerankers.cross_encoder_reranker import CrossEncoderRerankerAdapter
            return CrossEncoderRerankerAdapter(
                model=config.reranker_model,
                top_n=config.reranker_top_n,
            )
        case "mmr":
            from app.adapters.rerankers.mmr_reranker import MMRRerankerAdapter
            embeddings_port = embeddings or create_embeddings()
            return MMRRerankerAdapter(
                embeddings=embeddings_port.get_embeddings(),
                top_n=config.reranker_top_n,
                lambda_mult=config.mmr_lambda_mult,
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
                "Supported: cohere, cross_encoder, mmr, none"
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


# --------------------------------------------------------------------------- #
# Chunking                                                                     #
# --------------------------------------------------------------------------- #

def create_chunker(embeddings: EmbeddingsPort | None = None) -> ChunkingPort:
    match config.chunker_type:
        case "recursive":
            from app.adapters.chunkers.recursive_chunker import RecursiveChunkerAdapter
            return RecursiveChunkerAdapter(
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap,
            )
        case "semantic":
            from app.adapters.chunkers.semantic_chunker import SemanticChunkerAdapter
            embeddings_port = embeddings or create_embeddings()
            return SemanticChunkerAdapter(
                embeddings=embeddings_port.get_embeddings(),
                breakpoint_threshold_type=config.semantic_breakpoint_threshold_type,
                breakpoint_threshold_amount=config.semantic_breakpoint_threshold_amount,
            )
        # ── add new providers below ──────────────────────────────────────────
        case _:
            raise ValueError(
                f"Unknown CHUNKER_TYPE='{config.chunker_type}'. "
                "Supported: recursive, semantic"
            )


# --------------------------------------------------------------------------- #
# Retriever                                                                    #
# --------------------------------------------------------------------------- #

def create_retriever(vector_store: VectorStorePort) -> RetrieverPort:
    match config.retriever_type:
        case "dense":
            from app.adapters.retrievers.dense_retriever import DenseRetrieverAdapter
            return DenseRetrieverAdapter(vector_store, k=config.retrieval_k)
        case "bm25":
            from app.adapters.retrievers.bm25_retriever import BM25RetrieverAdapter
            return BM25RetrieverAdapter(k=config.retrieval_k)
        case "hybrid_interleaving":
            from app.adapters.retrievers.hybrid_interleaving_retriever import HybridInterleavingRetrieverAdapter
            return HybridInterleavingRetrieverAdapter(vector_store, k=config.retrieval_k)
        case "hybrid_rrf":
            from app.adapters.retrievers.hybrid_rrf_retriever import HybridRRFRetrieverAdapter
            return HybridRRFRetrieverAdapter(vector_store, k=config.retrieval_k, rrf_k=config.rrf_k)
        # ── add new providers below ──────────────────────────────────────────
        case _:
            raise ValueError(
                f"Unknown RETRIEVER_TYPE='{config.retriever_type}'. "
                "Supported: dense, bm25, hybrid_interleaving, hybrid_rrf"
            )


# --------------------------------------------------------------------------- #
# Query Transformer                                                            #
# --------------------------------------------------------------------------- #

def create_query_transformer(llm: LLMPort) -> QueryTransformerPort:
    match config.query_transformer_type:
        case "none":
            from app.adapters.query_transformers.none_transformer import NoneQueryTransformerAdapter
            return NoneQueryTransformerAdapter()
        case "hyde":
            from app.adapters.query_transformers.hyde_transformer import HyDEQueryTransformerAdapter
            return HyDEQueryTransformerAdapter(llm.get_chat_model())
        case "decomposition":
            from app.adapters.query_transformers.decomposition_transformer import DecompositionQueryTransformerAdapter
            return DecompositionQueryTransformerAdapter(llm.get_chat_model())
        case "hyde_decomposition":
            from app.adapters.query_transformers.hyde_decomposition_transformer import HyDEDecompositionQueryTransformerAdapter
            return HyDEDecompositionQueryTransformerAdapter(llm.get_chat_model())
        # ── add new providers below ──────────────────────────────────────────
        case _:
            raise ValueError(
                f"Unknown QUERY_TRANSFORMER_TYPE='{config.query_transformer_type}'. "
                "Supported: none, hyde, decomposition, hyde_decomposition"
            )


# --------------------------------------------------------------------------- #
# Metadata Enrichment                                                          #
# --------------------------------------------------------------------------- #

def create_metadata_enricher(llm: LLMPort) -> MetadataEnrichmentPort:
    match config.metadata_enricher_type:
        case "none":
            from app.adapters.metadata_enrichers.none_enricher import NoneEnricherAdapter
            return NoneEnricherAdapter()
        case "basic":
            from app.adapters.metadata_enrichers.basic_enricher import BasicEnricherAdapter
            return BasicEnricherAdapter()
        case "llm":
            from app.adapters.metadata_enrichers.llm_enricher import LLMEnricherAdapter
            return LLMEnricherAdapter(llm.get_chat_model())
        # ── add new providers below ──────────────────────────────────────────
        case _:
            raise ValueError(
                f"Unknown METADATA_ENRICHER_TYPE='{config.metadata_enricher_type}'. "
                "Supported: none, basic, llm"
            )


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
):
    from app.agents.supervisor_agent import SupervisorAgent
    return SupervisorAgent(
        research_agent=research_agent,
        citation_agent=citation_agent,
        synthesis_agent=synthesis_agent,
        llm=llm.get_chat_model(),
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
