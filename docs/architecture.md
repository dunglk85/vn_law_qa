# Architecture — Company Knowledge Assistant

> **Paige's note**: This document describes the production state of the codebase.
> Architecture boundary rules are enforced by automated tests in
> `tests/test_architecture.py` — 9 tests, 9 passing.

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

    subgraph PortLayer["Port Layer (11 interfaces)"]
        VP["VectorStorePort"]
        LP["LLMPort"]
        EP["EmbeddingsPort"]
        RP["RetrieverPort"]
        RER["RerankerPort"]
        CP["CachePort"]
        CHP["ChunkingPort"]
        QTP["QueryTransformerPort"]
        MEP["MetadataEnrichmentPort"]
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
    RL --> IS
    RAG --> RP
    RAG --> RER
    RAG --> LP
    AS --> SA
    SA --> RP
    IS --> CHP
    IS --> MEP
    IS --> EP

    RP -.->|implements| DR
    LP -.->|implements| OA
    EP -.->|implements| OE
    RER -.->|implements| CO
    CP -.->|implements| RC
    VP -.->|implements| PG
    SSP -.->|implements| RC
    RLP -.->|implements| RL

    DR --> CH
    DR --> PSQL
    DR -.->|degraded| BM
    PG --> PSQL
    RC --> RD

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

---

## 2. Project Map

```
company-knowledge-assistant/
├── app/
│   ├── __init__.py
│   ├── api.py                 # FastAPI bootstrap + endpoints
│   ├── config.py              # AppConfig dataclass (50+ env vars)
│   ├── factory.py             # Config-driven DI factory (Port → Adapter)
│   ├── core/
│   │   ├── agentic_service.py # LangGraph multi-agent RAG service
│   │   ├── rag_service.py     # Traditional single-pass RAG service
│   │   ├── ingest_service.py  # Document loading + chunking pipeline
│   │   ├── models.py          # Article, Citation, Task dataclasses + utilities
│   │   ├── retry.py           # Exponential backoff retry wrapper
│   │   └── reasoning_step.py  # Agent trace data shape
│   ├── agents/
│   │   ├── supervisor_agent.py           # LangGraph state machine (6 nodes)
│   │   ├── legal_research_agent.py       # HyDE + decomposition + rerank
│   │   ├── citation_checker_agent.py     # 3-gate hallucination firewall
│   │   ├── response_synthesizer_agent.py # Vietnamese legal response gen
│   │   └── tools/
│   │       └── knowledge_search.py       # LangChain @tool wrapper
│   ├── ports/                 # 11 abstract interfaces
│   │   ├── cache.py, chunking.py, embeddings.py, llm.py
│   │   ├── metadata_enrichment.py, query_transformer.py
│   │   ├── rate_limiter.py, reranker.py, retriever.py
│   │   ├── session_store.py, vector_store.py
│   └── adapters/              # 20+ concrete implementations
│       ├── caches/            # RedisCacheAdapter, NoneCacheAdapter
│       ├── chunkers/          # RecursiveChunkerAdapter, SemanticChunkerAdapter
│       ├── embeddings/        # OpenAIEmbeddingsAdapter
│       ├── llms/              # OpenAILLMAdapter
│       ├── metadata_enrichers/ # BasicEnricherAdapter, LLMEnricherAdapter, NoneEnricherAdapter
│       ├── rate_limiters/     # RedisRateLimiterAdapter, MemoryRateLimiterAdapter
│       ├── rerankers/         # CohereRerankerAdapter, CrossEncoder*, MMR*, None*
│       ├── retrievers/        # Dense*, BM25*, HybridInterleaving*, HybridRRF*
│       ├── session_stores/    # RedisSessionStore, MemorySessionStore
│       └── vector_stores/     # PGVectorStoreAdapter
│   └── auth/
│       ├── router.py          # /auth/token, /auth/refresh (JWT + refresh tokens)
│       ├── jwt.py             # Token create/decode/validate
│       └── dependencies.py    # require_role(), get_current_user()
├── tests/
│   └── test_architecture.py   # 9 architectural boundary tests
│   └── test_rate_limiter.py   # Unit tests for memory & Redis rate limiters
├── Dockerfile                 # python:3.11-slim, non-root user, no --reload
├── docker-compose.yml         # postgres + redis + app + pgweb
└── .env.example               # All config keys with placeholder values
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

```mermaid
flowchart TD
    A["POST /ingest"] --> B{"Auth check"}
    B -->|"role=admin or X-Api-Key"| C["_load_docs()"]
    B -->|"fail"| X["401 Unauthorized"]
    C --> D{"Document type?"}
    D -->|pdf| E["PyMuPDFLoader"]
    D -->|docx| F["Docx2txtLoader"]
    D -->|txt| G["TextLoader"]
    E --> H["Metadata enricher"]
    F --> H
    G --> H
    H --> I["ChunkerPort.chunk()"]
    I --> J{"Chunker type"}
    J -->|recursive| K["RecursiveCharacterTextSplitter"]
    J -->|semantic| L["Embedding-based splitter"]
    K --> M["VectorStorePort.add_documents()"]
    L --> M
    M --> N["VectorStorePort.create_index()"]
    N --> O["RetrieverPort.build_index()"]
    O --> P["JSON {documents, chunks}"]
```

---

## 4. Configuration

All config lives in `app/config.py` as a frozen `AppConfig` dataclass singleton.
Every field is backed by an environment variable with a sensible default.

| Category | Env Vars | Purpose / Default |
|----------|----------|-------------------|
| **Provider selection** | `VECTOR_STORE_TYPE`, `LLM_TYPE`, `EMBEDDINGS_TYPE`, `RERANKER_TYPE`, `CACHE_TYPE`, `CHUNKER_TYPE`, `RETRIEVER_TYPE`, `QUERY_TRANSFORMER_TYPE`, `METADATA_ENRICHER_TYPE`, `RAG_MODE` | Select which adapter each port binds to. Default: `VECTOR_STORE_TYPE=pgvector`, `RAG_MODE=traditional` |
| **Connection strings** | `DATABASE_URL`, `REDIS_URL` | Infrastructure endpoints. Default: `DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db` |
| **API keys** | `OPENAI_API_KEY`, `CO_API_KEY`, `LANGSMITH_API_KEY` | Provider credentials. Startup warning issued if unset. |
| **Model names** | `LLM_MODEL`, `EMBEDDINGS_MODEL`, `RERANKER_MODEL` | Model identifiers. Default: `LLM_MODEL=gpt-4o-mini` |
| **RAG tuning** | `RETRIEVAL_K`, `RRF_K`, `RERANKER_TOP_N`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, etc. | Retrieval and chunking parameters. Default: `CHUNK_SIZE=900` |
| **Index params** | `INDEX_TYPE`, `HNSW_M`, `HNSW_EF_CONSTRUCTION`, `IVFFLAT_LISTS`, `IVFFLAT_PROBES` | Vector index configuration. Default: `INDEX_TYPE=hnsw` |
| **Agent timeouts** | `LLM_TIMEOUT`, `AGENT_TIMEOUT`, `ASK_TIMEOUT`, `RATE_LIMIT_MAX`, `RATE_LIMIT_WINDOW` | Timeouts and rate limit thresholds. Default: `ASK_TIMEOUT=120` |
| **Agent tuning** | `MAX_RETRIES`, `QUALITY_THRESHOLD`, `N_RESULTS_PER_VECTOR`, `TOP_K_RESEARCH`, `TOP_K_LLM_SCORE`, `HYDE_ENABLED`, `SUBQUERY_COUNT`, `RELEVANCE_THRESHOLD` | Agent behavior knobs. Default: `HYDE_ENABLED=true` |
| **Auth** | `JWT_SECRET`, `JWT_ALGORITHM`, `ACCESS_TOKEN_EXPIRE_MINUTES`, `REFRESH_TOKEN_EXPIRE_DAYS`, `ADMIN_USERNAME`, `ADMIN_PASSWORD` | JWT and admin credentials. Default: `JWT_SECRET=<strong-random>` |
| **Session** | `SESSION_TTL_SECONDS`, `MAX_HISTORY_TOKENS`, `RECENT_TURNS_TO_KEEP` | Session and history management. Default: `SESSION_TTL_SECONDS=3600` |

---

## 5. Factory — Dependency Injection

`app/factory.py` is the central DI container (see [§1](#1-guiding-architecture-ports--adapters-hexagonal) for layering rules). Each `create_*()` function:

1. Reads the relevant `*_TYPE` env var from config
2. Matches on it with `case` blocks
3. Lazy-imports the concrete adapter class
4. Wires it with config values from `AppConfig`

```python
def create_llm() -> LLMPort:
    match config.llm_type:
        case "openai":
            from app.adapters.llms.openai_llm import OpenAILLMAdapter
            return OpenAILLMAdapter(model=config.llm_model, api_key=config.openai_api_key)
        case _:
            raise ValueError(...)
```

**Key property**: No adapter imports `app.config` directly. All configuration reaches adapters through constructor parameters from the factory.

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
| **Ingestion auth** | Optional `X-Api-Key` header check via `INGEST_API_KEY` env var. Also requires `require_role("admin")`. |
| **Rate limiting** | Redis-backed sliding window rate limiter per IP, with in-memory fallback. |
| **Container user** | Dockerfile runs as `appuser` (non-root). |

---

## 8. Deployment

### Docker Compose

```
services:
  postgres: pgvector/pgvector:pg17  # PostgreSQL + vector extension
  redis:    redis/redis-stack:7.4.0  # Caching + rate limiting + sessions
  app:      build: ./Dockerfile       # python:3.11-slim, port 8000
  pgweb:    sosedoff/pgweb           # Admin UI (no auth — dev only)
```

### Dockerfile

```
FROM python:3.11-slim
  → system deps (build-essential, curl)
  → pip install -r requirements.txt
  → pre-download NLTK data
  → COPY app/ data/
  → groupadd + useradd appuser
  → CMD uvicorn app.api:app --host 0.0.0.0 --port 8000
```

### Volume Strategy (docker-compose)

- `./.env:/app/.env` — runtime env vars (read-only)
- `./data:/app/data` — hot-reloadable document corpus
- **No** `.:/app` — container uses the built image, not host source

### Deployment Topology

```mermaid
graph LR
    subgraph External
        Client
    end
    subgraph Docker
        App["app :8000"]
        PG["postgres :5432<br/>(pgvector)"]
        Redis["redis :6379"]
        PGWeb["pgweb :8081<br/>(dev only)"]
    end
    Client -->|HTTP| App
    App -->|asyncpg| PG
    App -->|redis-py| Redis
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

Detailed changes are tracked in [`CHANGELOG.md`](../CHANGELOG.md).
The cycle that established the current architecture resolved 15 items
(P0–P2), including startup crash fixes, config wiring cleanup,
adapter isolation, agent import rules, and Docker hardening.

## 11. Known Gaps

| Issue | Severity | Status |
|-------|----------|--------|
| Live API keys in `.env` (not git-tracked, but on disk) | Critical | Set as env vars, not file |
| No Alembic auto-migration on container startup | Medium | Add entrypoint script |
| No unit tests for RAG services, agents, adapters | Medium | Coverage gap |
| `pgweb` exposed without auth on port 8081 | Low | Dev-only; add basic auth for staging |

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
| **Unit** | `tests/test_rate_limiter.py` | Memory + Redis rate limiter logic |
| **Coverage gap** | — | No unit tests for RAG services, agents, or adapters (see §11) |

Integration and E2E test suites are planned but not yet implemented.

## 15. See Also

- [`CONTRIBUTING.md`](../CONTRIBUTING.md) — Contribution guidelines
- [`tests/test_architecture.py`](../tests/test_architecture.py) — Architecture boundary test source
- `CHANGELOG.md` — Project change history
