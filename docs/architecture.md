# Architecture — Company Knowledge Assistant

> **Paige's note**: This document describes the production state of the codebase.
> Architecture boundary rules are enforced by automated tests in
> `tests/test_architecture.py` — 9 tests, 9 passing. All tests pass
> and lint check (`ruff check .`) is clean on every CI run.

---

## 1. Guiding Architecture: Ports & Adapters (Hexagonal)

The project follows strict hexagonal architecture:

```
                    ┌─────────────┐
                    │    API      │  app/api.py
                    │  (FastAPI)  │  ─── config + factory only
                    └──────┬──────┘
                           │ ports
                    ┌──────▼──────┐
                    │   Core +    │  app/core/
                    │  Services   │  ─── pure ports/domain logic
                    └──────┬──────┘
                           │ ports
               ┌───────────┼───────────┐
               │           │           │
        ┌──────▼────┐ ┌───▼────┐ ┌───▼──────┐
        │ Adapter   │ │ Adapter│ │ Adapter   │  app/adapters/
        │ (Vector)  │ │ (LLM)  │ │ (Cache)   │  ─── concrete backends
        └───────────┘ └────────┘ └──────────┘
```

```mermaid
graph TB
    subgraph APILayer["API Layer"]
        API["FastAPI /ask<br/>app/api.py"]
        AUTH["Auth & RBAC<br/>JWT + require_role"]
        RL["Rate Limiter<br/>Redis / In-Memory"]
    end

    subgraph CoreLayer["Core & Agent Layer"]
        RAG["RAG Service<br/>app/core/rag_service.py"]
        AS["Agentic Service<br/>app/core/agentic_service.py"]
        IS["Ingest Service<br/>app/core/ingest_service.py"]
        SA["Supervisor Agent<br/>app/agents/supervisor_agent.py"]
    end

    subgraph PortLayer["Port Layer (10 interfaces)"]
        VP["VectorStorePort"]
        LP["LLMPort"]
        EP["EmbeddingsPort"]
        RP["RetrieverPort"]
        RER["RerankerPort"]
        CP["CachePort"]
        DLP["DocumentLoaderPort"]
        QTP["QueryTransformerPort"]
        SSP["SessionStorePort"]
        RLP["RateLimiterPort"]
    end

    subgraph AdapterLayer["Adapter Implementations"]
        PG["PGVectorStoreAdapter"]
        OA["OpenAILLMAdapter"]
        OE["OpenAIEmbeddingsAdapter"]
        DR["DenseRetrieverAdapter"]
        CO["CohereRerankerAdapter"]
        RC["RedisCacheAdapter"]
        PL["ParquetLoaderAdapter"]
        A2A_REMOTE["A2ARemoteClient"]
        A2A_FALLBACK["InProcessFallbackClient"]
        MCP_ADAPTER["MCPKnowledgeSearchTool"]
    end

    subgraph StorageLayer["Storage"]
        CH[("ChromaDB<br/>Vector Index")]
        PSQL[("PostgreSQL + pgvector<br/>Metadata & ACLs")]
        RD[("Redis<br/>Cache + Sessions")]
        BM["BM25 Fallback<br/>rank_bm25"]
    end

    API --> AUTH --> RL
    RL --> RAG
    RL --> AS
    RAG --> RP
    RAG --> RER
    RAG --> LP
    AS --> SA
    SA --> RP
    IS --> DLP
    IS --> VP

    RP -.->|implements| DR
    LP -.->|implements| OA
    EP -.->|implements| OE
    RER -.->|implements| CO
    CP -.->|implements| RC
    VP -.->|implements| PG
    DLP -.->|implements| PL
    SSP -.->|implements| RC
    RLP -.->|implements| RL

    DR --> CH
    DR --> PSQL
    DR -.->|degraded| BM
    PG --> PSQL
    RC --> RD

    A2A_REMOTE --> PSQL
    A2A_FALLBACK --> PSQL

    style API fill:#fff3e0,stroke:#e65100
    style RL fill:#fff3e0,stroke:#e65100
    style AUTH fill:#fff3e0,stroke:#e65100
    style VP fill:#e1f5fe,stroke:#01579b
    style RP fill:#e1f5fe,stroke:#01579b
    style SA fill:#f3e5f5,stroke:#6a1b9a
    style AS fill:#f3e5f5,stroke:#6a1b9a
    style CH fill:#e8f5e9,stroke:#1b5e20
    style PSQL fill:#e8f5e9,stroke:#1b5e20
    style RD fill:#e8f5e9,stroke:#1b5e20
    style BM fill:#fce4ec,stroke:#b71c1c
```

> Each port has one or more adapter implementations wired by the Factory (see [§5](#5-factory--dependency-injection)). Solid arrows: request flow. Dashed arrows: interface implementation or degraded fallback.

### Layer Rules (Enforced by Tests) (see [§9](#9-architecture-boundary-tests))

| Layer | Imports allowed | Imports forbidden |
|-------|----------------|-------------------|
| **Ports** (`app/ports/`) | Nothing | `app.adapters`, `app.core`, `app.factory`, `app.api` |
| **Core** (`app/core/`) | Ports, Config | `app.adapters`, `app.factory`, `app.api`, provider SDKs |
| **Adapters** (`app/adapters/`) | Ports, Config | `app.core`, `app.api` |
| **Factory** (`app/factory.py`) | Ports, Config (see [§5](#5-factory--dependency-injection)) | `app.core`, `app.api` |
| **API** (`app/api.py`) | Factory, Config, Core services | `app.adapters` directly |
| **Agents** (`app/agents/`) | `app.core.models`, Ports | `shared`, `app.adapters` |
| **Auth** (`app/auth/`) | Ports, Config, `app.core.models` | `app.adapters` directly |
| **A2A Servers** (`app/agents/a2a_servers/`) | Agents, Config, `app.core.models` | `app.adapters` directly |
| **A2A Clients** (`app/adapters/agents/`) | Ports, Config, `app.core.a2a_client` | `app.core` (business logic), `app.api` |

---

## 2. Project Map

```
company-knowledge-assistant/
├── app/
│   ├── __init__.py
│   ├── api.py                 # FastAPI bootstrap + endpoints
│   ├── config.py              # AppConfig dataclass (60+ env vars)
│   ├── exceptions.py          # AppError, ConfigurationError
│   ├── factory.py             # Config-driven DI factory (registry pattern)
│   ├── core/
│   │   ├── agentic_service.py # LangGraph multi-agent RAG service
│   │   ├── rag_service.py     # Traditional single-pass RAG service
│   │   ├── ingest_service.py  # Parquet loading + embedding pipeline
│   │   ├── models.py          # Article, Citation, Task dataclasses + utilities
│   │   ├── a2a_client.py      # Abstract A2A client interface
│   │   ├── retry.py           # Exponential backoff retry wrapper
│   │   ├── reasoning_step.py  # Agent trace data shape
│   │   └── token_tracker.py   # LLM token usage tracking
│   ├── agents/
│   │   ├── supervisor_agent.py           # LangGraph state machine (6 nodes)
│   │   ├── legal_research_agent.py       # HyDE + decomposition + rerank
│   │   ├── citation_checker_agent.py     # 3-gate hallucination firewall
│   │   ├── response_synthesizer_agent.py # Vietnamese legal response gen
│   │   ├── tools/
│   │   │   └── knowledge_search.py       # LangChain @tool wrapper
│   │   └── a2a_servers/
│   │       ├── legal_research_server.py       # A2A agent server (port 8101)
│   │       ├── citation_checker_server.py     # A2A agent server (port 8102)
│   │       └── response_synthesizer_server.py # A2A agent server (port 8103)
│   ├── ports/                 # 10 abstract interfaces
│   │   ├── cache.py, document_loader.py, embeddings.py, llm.py
│   │   ├── query_transformer.py, rate_limiter.py, reranker.py
│   │   ├── retriever.py, session_store.py, vector_store.py
│   │   └── (10 total)
│   ├── adapters/              # 25+ concrete implementations
│   │   ├── agents/            # A2ARemoteClient, InProcessFallbackClient
│   │   ├── caches/            # RedisCacheAdapter, NoneCacheAdapter
│   │   ├── document_loaders/  # ParquetLoaderAdapter
│   │   ├── embeddings/        # OpenAIEmbeddingsAdapter
│   │   ├── llms/              # OpenAILLMAdapter
│   │   ├── rate_limiters/     # RedisRateLimiterAdapter, MemoryRateLimiterAdapter
│   │   ├── rerankers/         # CohereRerankerAdapter, CrossEncoder*, MMR*, None*
│   │   ├── retrievers/        # Dense*, BM25*, HybridInterleaving*, HybridRRF*
│   │   ├── session_stores/    # RedisSessionStore, MemorySessionStore
│   │   ├── tools/             # MCP tool adapter
│   │   └── vector_stores/     # PGVectorStoreAdapter
│   ├── auth/
│   │   ├── router.py          # /auth/token, /auth/refresh (JWT + refresh tokens)
│   │   ├── jwt.py             # Token create/decode/validate
│   │   └── dependencies.py    # require_role(), get_current_user()
│   └── static/                # Demo-only frontend (not production UI)
│       ├── index.html         # Single-page query interface for testing
│       └── style.css          # Basic styling — no UX spec exists
├── tests/
│   ├── test_architecture.py   # 9 architectural boundary tests
│   ├── test_rate_limiter.py   # Unit tests for memory & Redis rate limiters
│   ├── test_a2a_client.py     # A2A client unit tests
│   ├── test_a2a_phase2.py     # A2A phase 2 integration tests
│   ├── test_mcp_tool.py       # MCP tool adapter tests
│   ├── conftest.py            # Shared test fixtures (env setup)
│   ├── unit/
│   │   ├── test_config.py             # Config validation
│   │   ├── test_factory.py            # Factory resolution
│   │   ├── test_agentic_service.py    # AgenticService logic
│   │   ├── test_supervisor_agent.py   # Supervisor agent
│   │   ├── test_llm_ainvoke.py        # LLM adapter invoke
│   │   ├── test_auth_jwt.py           # JWT token logic
│   │   ├── test_exceptions.py         # Error handling
│   │   └── test_token_tracker.py      # Token usage tracking
│   └── integration/
│       └── test_api.py        # FastAPI integration tests (auth, /ask, /health)
├── scripts/
│   ├── a2a-entrypoint.sh      # A2A agent container entrypoint
│   ├── generate_test_parquet.py # Test data generator
│   └── eval_ragas.py          # RAGAS evaluation script
├── Dockerfile                 # Multi-stage build, python:3.12-slim, non-root
├── Dockerfile.a2a             # A2A agent server image (shared base)
├── docker-compose.yml         # postgres + redis + app + a2a agents + pgweb
├── .env.example               # All config keys with placeholder values
├── pyproject.toml             # Ruff config (line-length 120, py311 target)
├── requirements.txt           # Production deps (fastapi, langchain, pgvector, etc.)
└── requirements-dev.txt       # Dev deps (ruff, pytest, pytest-asyncio)
```

---

## 3. Request Flow

### 3a. Traditional RAG (Default)

```
Client → FastAPI /ask
  → rate_limiter.check(client_ip)
  → RAGService.answer(question, category)
    → QueryTransformer.transform(question)
    → RetrieverPort.get_retriever().invoke(transformed_question)
    → RerankerPort.get_compressor().compress(docs, query)
    → LLMPort.get_chat_model().invoke(prompt + context)
  → JSON {answer, sources, contexts}
```

### 3b. Agentic RAG (RAG_MODE=agentic)

```mermaid
sequenceDiagram
    participant Client
    participant API as FastAPI /ask
    participant RL as RateLimiter
    participant AS as AgenticService
    participant SS as SessionStore
    participant QT as QueryTransformer
    participant SA as SupervisorAgent
    participant LRA as LegalResearchAgent
    participant CCA as CitationCheckerAgent
    participant RSA as ResponseSynthesizerAgent

    Client->>API: POST /ask {question}
    API->>RL: check(client_ip)
    RL-->>API: allowed
    API->>AS: answer(question, category, session_id)
    AS->>SS: load(session_id)
    SS-->>AS: {history, summary}
    AS->>QT: transform(question)
    QT-->>AS: [transformed_query]
    AS->>SA: run(transformed_query, history, summary)

    SA->>LRA: research(query)
    LRA-->>SA: [Article]
    SA->>CCA: verify(articles, query)
    CCA-->>SA: [Citation] (3-gate check)
    SA->>RSA: synthesize(query, citations)
    RSA-->>SA: {response}
    SA->>SA: validate_quality(score)
    alt score >= 0.75
        SA-->>AS: {final_response, citations, steps}
    else score < 0.75
        SA->>RSA: retry_synthesis
        RSA-->>SA: {response}
        SA-->>AS: {final_response, citations, steps}
    end

    AS->>SS: save(session_id, {history, summary})
    AS-->>API: (answer, sources, contexts, reasoning_trace)
    API-->>Client: JSON {answer, sources, contexts, reasoning_trace}
```

### 3c. Ingestion

Ingestion is handled by the `law-crawler/` pipeline (see [ADR-0001](adr/0001-law-crawler-integration.md)).
The crawler produces pre-chunked Parquet files in `law-crawler/data/gold/`.
The main app loads these via `ParquetLoaderAdapter` and embeds them into pgvector.

```mermaid
flowchart TD
    A["law-crawler/"] -->|"dvc repro"| B["Bronze → Silver → Gold"]
    B --> C["data/gold/*.parquet"]
    C -->|"ParquetLoaderAdapter.load()"| D["_load_docs()"]
    D --> E["VectorStorePort.add_documents()"]
    E --> F["VectorStorePort.create_index()"]
    F --> G["RetrieverPort.build_index()"]
```

The `/ingest` API endpoint has been removed. Ingestion is triggered by running
the law-crawler pipeline or by calling `run_ingest()` directly from a script.

---

## 4. Configuration

All config lives in `app/config.py` as a frozen `AppConfig` dataclass singleton.
Every field is backed by an environment variable with a sensible default.

| Category | Env Vars | Purpose / Default |
|----------|----------|-------------------|
| **Provider selection** | `VECTOR_STORE_TYPE`, `LLM_TYPE`, `EMBEDDINGS_TYPE`, `RERANKER_TYPE`, `CACHE_TYPE`, `RETRIEVER_TYPE`, `QUERY_TRANSFORMER_TYPE`, `DOCUMENT_LOADER_TYPE`, `RAG_MODE` | Select which adapter each port binds to. Default: `VECTOR_STORE_TYPE=pgvector`, `DOCUMENT_LOADER_TYPE=parquet`, `RAG_MODE=legacy` |
| **Connection strings** | `DATABASE_URL`, `REDIS_URL` | Infrastructure endpoints. Default: `DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db` |
| **API keys** | `OPENAI_API_KEY`, `CO_API_KEY`, `LANGSMITH_API_KEY` | Provider credentials. Startup warning issued if unset. |
| **Model names** | `LLM_MODEL`, `EMBEDDINGS_MODEL`, `RERANKER_MODEL` | Model identifiers. Default: `LLM_MODEL=gpt-4o-mini` |
| **RAG tuning** | `RETRIEVAL_K`, `RRF_K`, `RERANKER_TOP_N`, `MMR_LAMBDA_MULT`, `CACHE_DISTANCE_THRESHOLD`, `DATA_DIR` | Retrieval parameters. Chunking is handled by the law-crawler pipeline. |
| **Index params** | `INDEX_TYPE`, `HNSW_M`, `HNSW_EF_CONSTRUCTION`, `HNSW_EF_SEARCH`, `IVFFLAT_LISTS`, `IVFFLAT_PROBES` | Vector index configuration. Default: `INDEX_TYPE=hnsw` |
| **Agent timeouts** | `LLM_TIMEOUT`, `AGENT_TIMEOUT`, `ASK_TIMEOUT`, `RATE_LIMIT_MAX`, `RATE_LIMIT_WINDOW` | Timeouts and rate limit thresholds. Default: `ASK_TIMEOUT=120`, `RATE_LIMIT_MAX=30` |
| **Agent tuning** | `MAX_RETRIES`, `QUALITY_THRESHOLD`, `N_RESULTS_PER_VECTOR`, `TOP_K_RESEARCH`, `TOP_K_LLM_SCORE`, `HYDE_ENABLED`, `SUBQUERY_COUNT`, `RELEVANCE_THRESHOLD` | Agent behavior knobs. Default: `HYDE_ENABLED=true`, `QUALITY_THRESHOLD=0.75` |
| **Retry / Error Recovery** | `TOOL_RETRY_MAX_ATTEMPTS`, `TOOL_RETRY_BASE_DELAY` | Tool retry with exponential backoff. Default: `TOOL_RETRY_MAX_ATTEMPTS=2` |
| **Session / Memory** | `SESSION_TTL_SECONDS`, `MAX_HISTORY_TOKENS`, `RECENT_TURNS_TO_KEEP` | Session and history management. Default: `SESSION_TTL_SECONDS=3600`, `MAX_HISTORY_TOKENS=4096` |
| **Auth** | `JWT_SECRET`, `JWT_ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`, `REFRESH_TOKEN_EXPIRE_DAYS`, `ADMIN_USERNAME`, `ADMIN_PASSWORD` | JWT and admin credentials. Default: `ACCESS_TOKEN_EXPIRE_MINUTES=30`, `REFRESH_TOKEN_EXPIRE_DAYS=7` |
| **MCP (Model Context Protocol)** | `MCP_ENABLED`, `MCP_SERVER_TIMEOUT`, `MCP_MAX_RESTARTS` | MCP-backed knowledge search tool. Default: `MCP_ENABLED=false` |
| **A2A (Agent-to-Agent Protocol)** | `A2A_LEGAL_RESEARCH_URL`, `A2A_CITATION_CHECKER_URL`, `A2A_RESPONSE_SYNTHESIZER_URL`, `A2A_TASK_TIMEOUT`, `A2A_MAX_RETRIES` | Remote agent endpoints for A2A protocol. Default: empty (in-process fallback) |
| **LangSmith (Observability)** | `LANGCHAIN_TRACING_V2`, `LANGSMITH_PROJECT` | LLM call tracing. Default: `LANGSMITH_PROJECT=company-knowledge-assistant` |

---

## 5. Factory — Dependency Injection

`app/factory.py` is the central DI container (see [§1](#1-guiding-architecture-ports--adapters-hexagonal) for layering rules). It uses a **registry pattern** with Python decorators:

1. Each adapter factory is registered via `@_register("kind", "key")` decorator
2. Public `create_*(**)` functions delegate to `_resolve("kind", config.xxx_type, **kwargs)`
3. Lazy imports happen inside the registered function body (no circular deps)
4. Unknown types raise `ValueError` with all supported options listed

```python
_registry: dict[tuple[str, str], Callable] = {}

def _register(kind: str, key: str):
    def decorator(fn: Callable):
        _registry[(kind, key)] = fn
        return fn
    return decorator

def _resolve(kind: str, key: str, **kwargs):
    supported = {k: fn for (kk, k), fn in _registry.items() if kk == kind}
    fn = supported.get(key)
    if fn is None:
        raise ValueError(f"Unknown {kind}='{key}'. Supported: {sorted(supported)}")
    return fn(**kwargs)

@_register("llm", "openai")
def _create_openai_llm(model: str, api_key: str | None) -> LLMPort:
    from app.adapters.llms.openai_llm import OpenAILLMAdapter
    return OpenAILLMAdapter(model=model, api_key=api_key)

def create_llm() -> LLMPort:
    return _resolve("llm", config.llm_type,
                    model=config.llm_model,
                    api_key=config.openai_api_key)
```

**Key property**: No adapter imports `app.config` directly. All configuration reaches adapters through constructor parameters from the factory. Adding a new provider = write the adapter + register it — no switch/match changes needed.

---

## 6. Agent Architecture (LangGraph)

The `SupervisorAgent` (in `app/agents/supervisor_agent.py`) builds a
LangGraph `StateGraph` with 6 nodes. Agents receive their dependencies
through the Factory (see [§5](#5-factory--dependency-injection)):

```mermaid
stateDiagram-v2
    [*] --> analyze_query
    analyze_query --> plan_tasks
    plan_tasks --> execute_legal_research

    execute_legal_research --> execute_citation_check : success
    execute_legal_research --> [*] : error

    execute_citation_check --> execute_response_synthesis : citations ok
    execute_citation_check --> execute_legal_research : retry (empty citations)
    execute_citation_check --> [*] : max retries exhausted

    execute_response_synthesis --> validate_quality

    validate_quality --> [*] : score ≥ 0.75
    validate_quality --> execute_response_synthesis : score < 0.75 (retry synthesis)
    validate_quality --> [*] : max retries reached, deliver best-effort
```

### Sub-agents

- **LegalResearchAgent** — Pipeline: HyDE generation → sub-query decomposition → parallel retrieval (deduplicated) → LLM relevance scoring → blended ranking
- **CitationCheckerAgent** — Three-gate firewall: existence (vector store `mget`), relevance (blended score ≥ threshold), contradiction (LLM cross-reference)
- **ResponseSynthesizerAgent** — Single LLM call with prompt + verified citations

---

## 7. Security

| Concern | Implementation |
|---------|---------------|
| **API auth** | JWT bearer tokens via `/auth/token`, verified in `require_role()` dependency |
| **Admin creds** | `ADMIN_USERNAME`/`ADMIN_PASSWORD` env vars (not hardcoded). Defaults warned at startup. |
| **JWT secret** | Configurable via `JWT_SECRET`. Startup warning issued if default used. |
| **Refresh tokens** | Redis-backed `_RefreshTokenStore` with in-memory fallback. Tokens expire per `REFRESH_TOKEN_EXPIRE_DAYS`. |
| **Rate limiting** | Redis-backed sliding window rate limiter per IP, with in-memory fallback. |
| **Container user** | Dockerfile runs as `appuser` (non-root). |

---

## 8. Deployment

### Docker Compose

```
services:
  postgres:                   pgvector/pgvector:pg17  # PostgreSQL + vector extension
  redis:                      redis/redis-stack:7.4.0  # Caching + rate limiting + sessions
  app:                        build: ./Dockerfile       # python:3.12-slim (multi-stage), port 8000
  legal-research-agent:       build: ./Dockerfile.a2a   # A2A agent (port 8101)
  citation-checker-agent:     build: ./Dockerfile.a2a   # A2A agent (port 8102)
  response-synthesizer-agent: build: ./Dockerfile.a2a   # A2A agent (port 8103)
  pgweb:                      sosedoff/pgweb           # Admin UI (no auth — dev only)
```

### Dockerfile

```
Multi-stage build (builder → runtime):
  builder: python:3.12-slim
    → system deps (build-essential)
    → pip install requirements.txt (incl. torch --cpu)
  runtime: python:3.12-slim
    → system deps (curl, ca-certificates)
    → copy site-packages from builder
    → COPY app/ directory
    → groupadd + useradd appuser
    → CMD uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### Dockerfile.a2a (Agent-to-Agent)

```
Shares same base as main app:
  Multi-stage build (python:3.12-slim, same deps)
  → ENTRYPOINT ["/app/entrypoint.sh"]
  → A2A_AGENT env selects server module
  → Exposes ports 8101–8103
```

### Volume Strategy (docker-compose)

- `./.env:/app/.env` — runtime env vars (read-only)
- `./data:/app/data` — hot-reloadable document corpus
- **No** `.:/app` — container uses the built image, not host source
- **Persistent volumes**: `pgdata` (PostgreSQL), `redis-data` (Redis)

### Deployment Topology

```mermaid
graph LR
    subgraph External
        Client
    end
    subgraph Docker
        App["app :8000"]
        A2A_LR["legal-research-agent :8101"]
        A2A_CC["citation-checker-agent :8102"]
        A2A_RS["response-synthesizer-agent :8103"]
        PG["postgres :5432<br/>(pgvector)"]
        Redis["redis :6379"]
        PGWeb["pgweb :8081<br/>(dev only)"]
    end
    Client -->|HTTP| App
    App -->|asyncpg| PG
    App -->|redis-py| Redis
    App -->|HTTP / A2A| A2A_LR
    App -->|HTTP / A2A| A2A_CC
    App -->|HTTP / A2A| A2A_RS
    A2A_LR -->|asyncpg| PG
    A2A_CC -->|asyncpg| PG
    A2A_RS -->|asyncpg| PG
    PGWeb -->|SQL| PG
    App -->|reads| Vol1["./.env"]
    App -->|reads/writes| Vol2["./data"]
```

---

## 9. Architecture Boundary Tests

`tests/test_architecture.py` uses AST parsing (no runtime imports) to verify
the layering rules defined in [§1](#1-guiding-architecture-ports--adapters-hexagonal):

| Test | What it prevents |
|------|-----------------|
| `test_ports_isolation` | Ports importing adapters/core/factory/api |
| `test_core_isolation` | Core importing adapters/factory/api |
| `test_adapters_isolation` | Adapters importing core/api |
| `test_factory_isolation` | Factory importing core/api |
| `test_api_isolation` | API importing adapters directly |
| `test_no_concrete_providers_in_api_and_core` | API/core importing provider SDKs |
| `test_adapters_implement_ports` | Adapter class without Port base |
| `test_agents_do_not_import_shared` | Agents importing root `shared` instead of `app.core.models` |
| `test_agents_do_not_import_adapters` | Agents importing adapters directly |

---

## 10. Patch History

The cycle that established the current architecture resolved 15 items
(P0–P2), including startup crash fixes, config wiring cleanup,
adapter isolation, agent import rules, and Docker hardening.
See the `INTEGRATION_SUMMARY.md` and code-level docstrings for detailed change tracking.

## 11. Known Gaps

| Issue | Severity | Status |
|-------|----------|--------|
| Live API keys in `.env` (not git-tracked, but on disk) | Critical | Set as env vars, not file |
| No Alembic auto-migration on container startup | Medium | Add entrypoint script |
| No unit tests for adapters (LLM, vector store, cache) | Medium | Coverage gap |
| `pgweb` exposed without auth on port 8081 | Low | Dev-only; add basic auth for staging |
| `app/static/` frontend has no UX spec | Low | Demo-only; not intended for production end users |

## 12. Data Model

### Vector Store Schema

The system uses a single `documents` table in PostgreSQL (via PGVector):

| Column | Type | Description |
|--------|------|-------------|
| `id` | `UUID` (PK) | Unique document chunk identifier |
| `content` | `TEXT` | Chunk text content |
| `metadata` | `JSONB` | Enriched metadata (source, category, timestamps) |
| `embedding` | `vector(1536)` | OpenAI `text-embedding-3-small` embedding |

Index types are configurable via `INDEX_TYPE`:
- **HNSW** — `hnsw` (default): `M=16`, `ef_construction=200`, `ef_search=50`
- **IVFFlat** — `ivfflat`: `lists=100`, `probes=10`

### Session Store Schema (Redis)

| Key pattern | Value type | TTL |
|-------------|------------|-----|
| `session:{session_id}` | Hash (history, summary) | `SESSION_TTL_SECONDS` |
| `refresh_token:{token_hash}` | String (username) | `REFRESH_TOKEN_EXPIRE_DAYS` |

## 13. Error Handling & Retry Strategy

The architecture applies retry logic at two levels:

| Layer | Mechanism | Fallback |
|-------|-----------|----------|
| **Core services** (`app/core/retry.py`) | Exponential backoff with jitter; configurable `MAX_RETRIES` | Fail-fast after exhaustion |
| **Agent quality gate** (§6) | Up to 2 synthesis retries when `score < 0.75` | Best-effort response |
| **Rate limiter** | Sliding window per IP; returns `429` | `MemoryRateLimiterAdapter` fallback if Redis is down |
| **Cache** | Silent miss on Redis failure | `NoneCacheAdapter` no-op fallback |

Provider SDK errors (OpenAI, Cohere) propagate as `HTTP 502` after retry exhaustion.

## 14. Testing Strategy

| Category | Location | Scope |
|----------|----------|-------|
| **Architecture boundary** | `tests/test_architecture.py` | AST-level import rule enforcement (9 tests) |
| **Unit — config** | `tests/unit/test_config.py` | AppConfig validation and env parsing |
| **Unit — factory** | `tests/unit/test_factory.py` | Registry resolution and error cases |
| **Unit — services** | `tests/unit/test_agentic_service.py` | AgenticService orchestration logic |
| **Unit — agents** | `tests/unit/test_supervisor_agent.py` | Supervisor agent state machine |
| **Unit — LLM** | `tests/unit/test_llm_ainvoke.py` | LLM adapter async invoke |
| **Unit — auth** | `tests/unit/test_auth_jwt.py` | JWT create/decode/validate |
| **Unit — exceptions** | `tests/unit/test_exceptions.py` | Error handling and custom exceptions |
| **Unit — token tracking** | `tests/unit/test_token_tracker.py` | Token usage counting and reset |
| **Unit — rate limiter** | `tests/test_rate_limiter.py` | Memory + Redis rate limiter logic |
| **Unit — A2A client** | `tests/test_a2a_client.py` | A2A remote and fallback client |
| **Integration — API** | `tests/integration/test_api.py` | FastAPI endpoints (auth, /ask, /health) |
| **Integration — A2A** | `tests/test_a2a_phase2.py` | A2A agent-to-agent protocol |
| **Integration — MCP** | `tests/test_mcp_tool.py` | MCP tool adapter lifecycle |

All tests run on every CI push via `python -m pytest tests/ --ignore=tests/test_rate_limiter.py -v` (rate limiter tests excluded in CI because they require Redis). To run all tests locally including Redis-dependent: `python -m pytest tests/ -v`.

Linting: `ruff check .` — configured in `pyproject.toml` (line-length 120, py311 target).

## 15. See Also

- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — Contribution guidelines
- [`tests/test_architecture.py`](../tests/test_architecture.py) — Architecture boundary test source
- `INTEGRATION_SUMMARY.md` — Integration change history
