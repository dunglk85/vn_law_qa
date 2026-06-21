---
stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
workflowType: 'architecture'
lastStep: 8
status: 'complete'
completedAt: '2026-06-21'
inputDocuments:
  - "docs/architecture.md"
  - "_bmad-output/planning-artifacts/architecture.md"
  - "_bmad-output/planning-artifacts/epics.md"
workflowType: 'architecture'
project_name: 'company-knowledge-assistant'
user_name: 'Admin'
date: '2026-06-21'
---

# Architecture Decision Document

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

---

## Core Architectural Decisions

### Data Architecture

Decision: Postgres for canonical metadata + ChromaDB for vector storage (hybrid)

Summary:
- Postgres (+pgvector) remains the authoritative store for transactional metadata, user/session state, ACLs, and canonical records.
- ChromaDB holds embeddings and vector indexes used for semantic retrieval and hybrid search.

Implications & Patterns:
- Ingest pipeline: write metadata to Postgres, emit an ingest event, produce embeddings, and upsert to Chroma (event-driven, eventual consistency).
- Query path: similarity/search calls go to Chroma; results are joined with Postgres by document ID for authoritative fields.
- Fallback: on Chroma outage use Postgres + BM25 (`rank_bm25`) for degraded retrieval quality.

Operational Notes:
- Chroma introduces additional operational surface (scaling, backups, reconciliation). Consider managed Chroma or sidecar deployment depending on budget.
- Monitor sync lag, vector count vs metadata, and reconciliation job health.

Libraries & Integration:
- Use `langchain-postgres` and `langchain-chroma` connectors with async clients and pooled connections (asyncpg/pgvector client patterns).

Failure Modes & Mitigations:
- Chroma outage: fallback to BM25 and surface degraded mode to users.
- Sync drift: implement idempotent reconciliation job and alerting on lag thresholds.

Acceptance Tests:
- Ingest E2E: ingest a document → verify Postgres record exists and Chroma vector present.
- Fallback test: simulate Chroma downtime and verify fallback retrieval correctness.
- Reconciliation test: run reconciliation job on synthetic drift and assert eventual consistency.

Trade-offs (short):
- Pros: superior semantic retrieval and hybrid search capabilities.
- Cons: increased system complexity and operational overhead.

### Authentication & Security

Decision: OAuth2 / OIDC with short-lived JWTs, RBAC, and per-tenant scopes (FastAPI-backed initially; option to delegate to Keycloak/Auth0 later).

Summary:
- Use OAuth2/OIDC for user authentication and issue short-lived JWT access tokens with refresh tokens. Enforce RBAC scopes in service handlers and at tool access boundaries.
- For service-to-service or internal agent traffic, use token introspection or mTLS for stronger identity guarantees; require introspection for any cross-service tool invocation.

Implications & Patterns:
- Token lifecycle: short-lived access tokens (5–15m), rotating signing keys (JWKS), refresh tokens with PKCE for browser flows.
- Enforce least-privilege: tools and agents receive scoped service accounts; map agent capabilities to minimal scopes.
- Per-tenant isolation: include tenant claim in tokens and validate against Postgres ACLs on every sensitive retrieval.

Libraries & Integration:
- FastAPI + `authlib` / `python-jose` for JWT handling; `Authlib` or `fastapi-security` for OIDC flows.
- Consider delegating to Keycloak/Auth0 for enterprise SSO, OIDC, and admin UIs.

Failure Modes & Mitigations:
- Token theft: short token TTLs + rotation + revocation list in Redis; emergency revoke endpoint.
- Replay & CSRF: use PKCE for browser flows, require fresh tokens for high-risk actions and HITL approvals.
- Compromised agent: service account rotation, scoped credentials, and mandatory audit trails in LangSmith traces.

Acceptance Tests:
- Auth flow test: complete user login → receive access + refresh tokens → access protected endpoint.
- RBAC test: assert role-based restrictions for tool invocations and document access.
- Token revocation test: revoke a token → ensure access denied immediately.

Operational Notes:
- Publish JWKS endpoint; rotate keys regularly and automate rotation in deployment.
- Log auth events to LangSmith/audit sink and correlate with trace IDs for full auditability.

Next actions: proceed to API & Communication decisions.

stepsCompleted: [1, 2, 3, 4, 5, 6, 7, 8]
lastStep: 8
status: 'complete'
completedAt: '2026-06-20'
appendedAt: '2026-06-21'
appendedSections:
  - "MCP Decoupling: Knowledge Search Tool"
inputDocuments: 
  - "api.py (FastAPI bootstrap, dependency wiring)"
  - "docker-compose.yml (service architecture)"
  - "requirements.txt (technology stack)"
workflowType: 'architecture'
project_name: 'company-knowledge-assistant'
user_name: 'Admin'
date: '2026-06-19'
context: 'Convert existing RAG system into agentic system'
---

# Architecture Decision Document: Agentic Conversion

_This document builds collaboratively through step-by-step discovery. Sections are appended as we work through each architectural decision together._

## Project Overview

**Current System:** Company Knowledge Assistant — a FastAPI-based RAG (Retrieval-Augmented Generation) system that answers questions using a knowledge base.

**Current Stack:**
- FastAPI + Uvicorn (API server)
- LangChain ecosystem (orchestration)
- PostgreSQL + pgvector (vector storage)
- Redis (semantic caching)
- Modular adapters (embeddings, chunkers, retrievers, rerankers, query transformers)

**Conversion Goal:** Add agentic capabilities — autonomous reasoning, multi-step planning, tool use, and decision-making.

---

## Visual Architecture Overview

```mermaid
graph TB
    subgraph Client["Client Layer"]
        U[User / Browser]
    end

    subgraph API["API Layer"]
        F[FastAPI /ask endpoint]
        A[Auth & RBAC<br/>JWT + require_role]
        R[Rate Limiter<br/>Redis / In-Memory]
    end

    subgraph Agentic["Agentic Orchestration Layer"]
        AS[Agentic Service<br/>app/core/agentic_service.py]
        S[Session Store<br/>Redis / In-Memory]
        HC[History Compression<br/>Summarization on overflow]
        SA[Supervisor Agent<br/>app/agents/supervisor_agent.py]
        LA[Legal Research Agent]
        CA[Citation Checker Agent]
        RA[Response Synthesizer]
        ES[Error Recovery<br/>retry_with_backoff + fallback]
        RT[Reasoning Steps<br/>Per-node trace collection]
    end

    subgraph Tools["Tool & Port Layer"]
        RS[Retriever Port<br/>app/ports/retriever.py]
        KS[Knowledge Search Tool<br/>app/agents/tools/knowledge_search.py]
    end

    subgraph Storage["Storage Layer"]
        CH[(ChromaDB<br/>Vector Index)]
        PG[(PostgreSQL + pgvector<br/>Metadata & ACLs)]
        RD[(Redis<br/>Cache + Rate Limit + Session)]
        BM[BM25 Fallback<br/>rank_bm25 on Chroma outage]
    end

    U -- HTTP Request --> F
    F --> A
    A --> R
    R --> AS
    AS --> S
    AS --> HC
    AS --> SA
    SA --> LA
    SA --> CA
    SA --> RA
    SA --> ES
    SA --> RT
    LA --> KS
    LA --> RS
    CA --> RS
    RA --> RS
    KS --> CH
    RS --> CH
    RS --> PG
    RS -. degraded .-> BM
    S --> RD
    R --> RD

    style U fill:#e1f5fe,stroke:#01579b
    style F fill:#fff3e0,stroke:#e65100
    style SA fill:#f3e5f5,stroke:#6a1b9a
    style AS fill:#f3e5f5,stroke:#6a1b9a
    style CH fill:#e8f5e9,stroke:#1b5e20
    style PG fill:#e8f5e9,stroke:#1b5e20
    style RD fill:#e8f5e9,stroke:#1b5e20
```

**Request Flow:** User → FastAPI → Auth/RBAC → Rate Limiter → Agentic Service → Session/History → Supervisor Agent → Sub-Agents → Tools → Storage → Response

---

## Project Context Analysis

### Agentic Conversion Requirements

**Current Architecture:**
- Single-turn RAG: question → retrieve → rank → generate → answer
- Stateless endpoints with isolated adapter components
- Knowledge base: announcements, FAQs, guides, policies
- Modular pattern enables flexible swapping of embeddings, chunkers, rerankers

**Target Architecture:**
Add multi-step autonomous reasoning with:
- **Agent reasoning loop:** Plan → Act → Observe → Reflect
- **Tool use:** Query knowledge base, search specific documents, fetch details, check availability
- **State management:** Remember reasoning context across multi-turn conversations
- **Decision-making:** Agent autonomously selects which tools to invoke and when
- **Chain-of-thought:** Explicit reasoning steps visible and auditable

### Functional Requirements (Implied by Agentic Conversion)

1. Multi-turn agent conversation (session state persistence)
2. Tool definitions for agent actions (knowledge search, document retrieval, lookups, checks)
3. Reasoning step tracking and visualization
4. Agent memory (conversation history, decisions, context window management)
5. Error recovery (graceful handling of tool failures, invalid outputs)

### Non-Functional Requirements

- **Latency:** Multi-step reasoning loops increase response time vs. single-turn RAG (trade-off: accuracy/autonomy for speed)
- **Token efficiency:** Reasoning traces and multi-turn context consume more LLM tokens (cost/budget implications)
- **Observability:** Need comprehensive tracing of agent decisions for debugging, audit, and user transparency
- **Reliability:** Multiple sequential LLM calls increase failure surface area; need robust error handling
- **Scalability:** Stateful conversations require persistent storage; impact on database and cache load
- **Coherence:** Agent decisions must remain consistent across tool calls and reasoning steps

### Technical Constraints & Current Advantages

**Constraints:**
- Existing single-turn API surface needs evolution to support multi-turn state
- LLM token budget impacts reasoning depth and context window size
- Tool call latency compounds across multi-step reasoning chains

**Existing Advantages:**
- Modular adapter pattern is ideal foundation for tool abstraction
- FastAPI supports stateful session management via Redis or database
- PostgreSQL + pgvector already handle vector/semantic operations
- LangChain 0.3+ provides mature agent frameworks (AgentExecutor, ReAct, tool protocols)
- Redis semantic cache can reduce token consumption on repeated reasoning patterns

### Cross-Cutting Architectural Concerns

1. **Tool Layer:** How do adapters (retrievers, rerankers, embeddings) expose themselves as agent tools?
2. **State Management:** Conversation storage design (single-turn vs. long-context vs. summarization)?
3. **API Evolution:** Backward compatibility with existing `/ask` endpoint?
4. **Observability:** What gets logged, traced, and displayed to users?
5. **Safety & Constraints:** How to prevent infinite loops, token overruns, or incorrect tool use?

### Complexity Assessment

- **Project Complexity:** Medium-to-High (multi-agent reasoning with external tools)
- **Primary Domain:** Backend API / AI orchestration
- **Estimated New Components:** Tool registry, session store, agent executor wrapper, observability middleware
- **Risk Areas:** Tool reliability, token budget management, session cleanup/memory leaks

---

_Architectural decisions and design choices will be appended as we work through each decision point._

## Starter Template Evaluation

### Primary Technology Domain

Identified domain: API / Backend (Python + FastAPI) with LangChain-based agentic extensions.

### Starter Options Considered

1. Minimal FastAPI Starter — small, easy to integrate with existing codebase; minimal ceremony. Best if you want to retain full control and make incremental changes.
2. FastAPI + LangChain Scaffold — includes agent hooks, tool registry, and tracing integrations (LangSmith/Ragas). Best for rapid agentic feature development and easier onboarding for agent patterns.
3. Dockerized Monorepo Starter — includes CI, Docker, and basic K8s manifests. Best when you want an immediately production-ready infra layout.

### Recommendation

For this migration, use the FastAPI + LangChain scaffold (option 2) as the primary starter for the `CitationCheckerAgent` vertical slice. It reduces friction when implementing agent tool registration, tracing, and self-correction loops while remaining compatible with the repository's existing Docker/K8s artifacts.

If you prefer minimal changes or want to limit risk, choose the Minimal FastAPI Starter (option 1) and incrementally add LangChain/agent scaffolding.

### Rationale & Starter Implications

- Language & Runtime: Python 3.11+ (align with existing environment)
- Project layout: `app/` for agents and core, `tests/` for unit/integration, `scripts/` for infra helpers
- Testing: pytest + Ragas integration tests for faithfulness gating
- Lint/format: ruff + black + isort with pre-commit hooks
- Observability: LangSmith traces integrated at agent execution boundaries; export trace IDs in API responses for debugging
- Deployment: Keep existing Docker/K8s manifests; scaffold starter must be compatible with those artifacts

### Next Steps (if you confirm)

- I will append this evaluation to the architecture document and we can proceed to Step 4 (Core Architectural Decisions).
- If you want, I can look up concrete starter repos/CLI commands and add exact initialization commands.

---

## Implementation Patterns & Consistency Rules

### Naming Patterns

**Database Naming Conventions:**
- Table/column naming: snake_case — `users`, `documents`, `citation_results`
- Foreign key format: `referenced_table_id` — e.g., `document_id`, `user_id`
- Index naming: `idx_{table}_{column}` — e.g., `idx_documents_created_at`

**API Naming Conventions:**
- REST endpoints: plural nouns — `/users`, `/documents`, `/ingest`
- Route parameters: `{param}` format — `/documents/{document_id}`
- Query parameters: snake_case — `question`, `category`, `user_id`
- Response JSON field naming: snake_case — `answer`, `sources`, `contexts`

**Code Naming Conventions:**
- Python: snake_case for functions/variables, PascalCase for classes — `create_embeddings()`, `AgenticService`, `LLMPort`
- Type aliases: PascalCase — `AskRequest`, `RetrieverPort`
- File naming: snake_case matching primary class/function — `agentic_service.py`, `supervisor_agent.py`
- Test files: `test_{module}.py` — `test_architecture.py`, `test_factory.py`

### Structure Patterns

**Project Organization:**
- Adapter ports: `app/ports/{domain}.py` — interfaces only, no implementation
- Adapter implementations: `app/adapters/{category}/{provider}_store.py`
- Core services: `app/core/{domain}_service.py` — business orchestration
- Agents: `app/agents/{role}_agent.py` — LangGraph agent definitions
- API layer: `app/api.py` — FastAPI routes and bootstrap
- Factory/wiring: `app/factory.py` — config-driven DI (no core imports)
- Tests: `tests/test_{module}.py` — flat structure, mirroring app/ hierarchy

**File Structure Patterns:**
- One class per file for adapters and agents
- Exceptions defined inline or in a shared `exceptions.py` if reused across modules
- Config lives in `app/config.py` loaded from environment variables
- Static frontend files in `app/static/`

### Format Patterns

**API Response Formats:**
- Success: `{"answer": str, "sources": list[str], "contexts": list[str]}`
- Error: `{"ok": false, "message": str}` with appropriate HTTP status code
- Health: `{"status": "ok"}`
- Status endpoints: `{"ok": true, ...data...}` — consistency with `/ingest/status`

**Data Exchange Formats:**
- JSON field naming: snake_case always
- Dates: ISO 8601 strings (`datetime.isoformat()`)
- Boolean representations: `true`/`false` in JSON, Python `True`/`False`
- Null handling: omit null fields from responses where reasonable, include with `None` where schema expects them

### Communication Patterns

**Logging Patterns:**
- Logger declaration: `logger = logging.getLogger(__name__)` at module level
- Message format: `%s`-style — `logger.info("Processing %s for user %s", doc_id, user_id)` (not f-strings)
- Log levels: `INFO` for lifecycle events, `WARNING` for recoverable issues, `ERROR` for failures
- Structured context: include `client=%s`, `question=%.80s` patterns for traceability

**Async Patterns:**
- All IO-bound operations: async/await throughout
- Timeouts required on all external calls — `asyncio.wait_for(coro, timeout=30)` with `try/except asyncio.TimeoutError`
  - DB query timeout: configurable via env var (`DB_QUERY_TIMEOUT`, default 30s)
- LLM invocations: use shared `llm_ainvoke()` wrapper with 30s timeout
- Cache activation: wrapped in `try/except` with fallback logging

### Process Patterns

**Error Handling Patterns:**
- **Factories and configuration:** raise `ValueError` with descriptive message on unknown type
- **Agent code:** graceful fallback with logging, return safe defaults (`_NO_CONTEXT_ANSWER`, empty lists)
- **API layer:** wrap in outer `try/except asyncio.TimeoutError`, return 504 JSON response
- **Ingestion:** capture exceptions into status dict, never crash the process
- **Validation:** Pydantic models with `Field(min_length=1, max_length=2000)` constraints at API boundary
- **Rate limiting:** 429 response with in-memory sliding window, integrated at middleware level
  - **Note:** In-memory rate limiting is single-instance only. For multi-instance deployments, replace with Redis-backed counter

**Retry & Recovery Patterns:**
- LLM calls: max retries with exponential backoff (1s, 2s, 4s, 8s, ...), route to error state on exhaustion
- Supervisor agent: retry iteration up to configured max (default 3), fall through to final synthesis
- Cache activation: soft failure — log warning and continue without cache
- Ingestion: single attempt with status reporting, no automatic retry

### Configuration & DI Patterns

- Environment variable naming: `{MODULE}_{KEY}` uppercase, underscore-separated — e.g., `LLM_TYPE`, `VECTOR_STORE_TYPE`, `DB_QUERY_TIMEOUT`
- All new env vars MUST have a typed accessor in `app/config.py` with a sensible default and a warning log on fallback
- External dependency wiring: adapter selection goes in `app/factory.py` as a `match`/`case` block driven by a config env var
- Agent-level orchestration wiring: lives in `app/core/agentic_service.py` (not `app/factory.py`)
- New adapter providers: implement the relevant Port interface, add a `case` in the corresponding factory function, set the env var

### Database Schema Patterns

- Use Alembic for all schema migrations
- Revision naming: `{seq}_{short_description}.py` — e.g., `0001_add_documents_table.py`
- All new tables: include `created_at` and `updated_at` timestamp columns with `func.now()` defaults

### Exception Hierarchy

- Define `class AppError(Exception)` in `app/exceptions.py` as the project-wide base
- API middleware maps `AppError` subclasses to appropriate HTTP status codes and JSON error bodies
- Agent code: catch `AppError` for graceful fallback; raise `ValueError` in factory/config code for misconfiguration
- New error types: subclass `AppError` rather than raising bare `Exception`

### Enforcement Guidelines

**All AI Agents MUST:**
- Use `from __future__ import annotations` at the top of every Python file
- Add `logger = logging.getLogger(__name__)` for all new modules
- Use `asyncio.wait_for()` with appropriate timeout for any external call
- Follow existing adapter port interface contracts exactly when implementing new providers
- Place tests in `tests/` organized by type: `tests/unit/`, `tests/integration/`, `tests/architecture/`
- Name test files `test_{module_name}.py` within the appropriate subdirectory

**Pattern Enforcement:**
- Architecture tests in `tests/test_architecture.py` verify isolation boundaries
- `ruff check` in CI (covers lint and format)
- Code review must verify naming conventions and error handling patterns match this document

---

## Project Structure & Boundaries

### Complete Project Directory Structure

```
company-knowledge-assistant/
├── app/                          # Application source
│   ├── __init__.py
│   ├── api.py                    # FastAPI routes, bootstrap, rate limiting
│   ├── config.py                 # Env-var configuration with typed accessors
│   ├── factory.py                # Config-driven DI for adapters (no core imports)
│   ├── shared.py                 # Shared helpers (llm_ainvoke, parse_json, parse_list)
│   ├── agents/                   # LangGraph agent definitions
│   │   ├── supervisor_agent.py
│   │   ├── legal_research_agent.py
│   │   ├── citation_checker_agent.py
│   │   └── response_synthesizer_agent.py
│   ├── core/                     # Business orchestration services
│   │   ├── agentic_service.py    # LangGraph workflow + factory function
│   │   ├── ingest_service.py     # Document ingestion pipeline
│   │   └── rag_service.py        # Legacy single-turn RAG
│   ├── ports/                    # Abstract interfaces (no implementation)
│   │   ├── cache.py / chunking.py / embeddings.py / llm.py
│   │   ├── metadata_enrichment.py / query_transformer.py
│   │   └── reranker.py / retriever.py / vector_store.py
│   ├── adapters/                 # Port implementations (one subdir per category)
│   │   ├── caches/ / chunkers/ / embeddings/ / llms/
│   │   ├── metadata_enrichers/ / query_transformers/
│   │   ├── rerankers/ / retrievers/ / vector_stores/
│   └── static/                   # Frontend assets
│       ├── index.html / style.css
├── tests/
│   └── test_architecture.py      # Isolation boundary tests
├── scripts/                      # Utility scripts
├── alembic/                      # DB migrations
│   ├── env.py / alembic.ini
│   └── versions/
├── data/                         # Knowledge base documents
├── seed/                         # Test data
├── init-db/                      # PostgreSQL schema init
├── law-crawler/                  # Legal batch ingestion tool
├── .env / Dockerfile / docker-compose.yml / requirements.txt
```

### Architectural Boundaries

- **API Boundary:** `app/api.py` only — rate limiting, request validation, route definitions
- **Port Boundary:** `app/ports/` — abstract interfaces only, zero implementation imports
- **Adapter Boundary:** `app/adapters/` — concrete implementations, no cross-adapter imports
- **Factory Boundary:** `app/factory.py` — wires ports to adapters via config; must not import `app.core`
- **Agent Boundary:** `app/agents/` — LangGraph agents, no direct adapter imports
- **Core Boundary:** `app/core/` — orchestration services that wire agents together

### Requirements to Structure Mapping

- Question answering → `app/core/rag_service.py` (legacy), `app/core/agentic_service.py` (agentic)
- Multi-step reasoning → `app/agents/supervisor_agent.py` orchestrates sub-agents
- Document ingestion → `app/core/ingest_service.py` + `app/factory.py` adapter wiring
- Citation verification → `app/agents/citation_checker_agent.py`
- Legal research → `app/agents/legal_research_agent.py`
- Semantic caching → `app/adapters/caches/`
- RAG evaluation → `scripts/eval_ragas.py`

### File Organization Rules

- Configuration files: `.env` root, `app/config.py` typed accessors, `docker-compose.yml` root
- One class per file for adapters and agents
- Static assets in `app/static/`
- DB schema in `init-db/init.sql` (initial), `alembic/` for incremental migrations
- External batch jobs in separate directories (`law-crawler/`)

---

## Architecture Validation Results

### Coherence Validation ✅

**Decision Compatibility:**
- FastAPI + LangChain + pgvector + Redis + Chroma are fully compatible; no version or integration conflicts
- Python 3.14 (current) aligns with all library requirements
- Docker/K8s deployment artifacts already exist

**Pattern Consistency:**
- Naming, structure, format, communication, and process patterns are internally consistent
- All patterns reference specific existing codebase conventions (snake_case, `%s`-style logging, `asyncio.wait_for`)
- One minor inconsistency: Structure Patterns say `tests/test_{module}.py` flat, while Enforcement says `tests/unit/`, `tests/integration/`, `tests/architecture/` — the structured approach supersedes

**Structure Alignment:**
- Directory tree maps 1:1 to actual codebase layout
- Architectural boundaries (ports, adapters, factory, core, agents) are respected by the existing code
- Tests verify isolation boundaries

### Requirements Coverage Validation ✅

**Functional Requirements Coverage:**
- Multi-turn agent conversation → `app/core/agentic_service.py` w/ `app/agents/supervisor_agent.py`
- Tool definitions → Adapter ports as tool abstractions via LangGraph
- Reasoning step tracking → Supervisor agent state machine
- Agent memory → Conveyed via session/user ID parameters
- Error recovery → Exponential backoff, graceful fallbacks, timeout handling

**Non-Functional Requirements Coverage:**
- Latency → `asyncio.wait_for()` timeouts, time-bounded agent loops
- Token efficiency → Redis semantic cache
- Observability → Logging patterns, LangSmith integration
- Reliability → Retry patterns, fallback on Chroma outage
- Scalability → Stateless API, scale via K8s; in-memory rate limit noted as single-instance limitation
- Coherence → Supervisor agent orchestrates and validates sub-agent outputs

### Implementation Readiness Validation ✅

**Decision Completeness:**
- All major decisions documented with trade-offs, failure modes, and acceptance tests
- Patterns include concrete code examples drawn from actual codebase

**Structure Completeness:**
- Complete directory tree with every file and folder documented
- Every module has a documented purpose and boundary

**Pattern Completeness:**
- Naming, structure, format, communication, process, config/DI, DB schema, and exception patterns all defined
- Enforcement guidelines with specific mandatory rules

### Gap Analysis Results

| Priority | Gap | Status |
|----------|-----|--------|
| Minor | Structure Patterns say flat `tests/test_*` — Enforcement says `tests/unit/`, `tests/integration/`, `tests/architecture/` | Accept structured approach |
| Minor | No ruff/CI configuration documented (which rules, pre-commit config) | Future enhancement |
| Minor | Frontmatter formatting inconsistent — `---` delimiters embedded mid-body | Cosmetic only |
| Medium | Chroma fallback to BM25 is documented but BM25 retriever must be verified wired and tested end-to-end | Documented in pre-mortem |
| Low | Sub-agent timeout budget (3 × 30s within 90s supervisor limit) fits but should be explicitly documented | Add timeout budget doc |
| Medium | In-memory rate limiting resets on pod restart — document that multi-instance requires Redis-backed rate limiter | Add to rate limiting guidance |
| Low | No monitoring/alerting for agent failure rates — recommend basic error rate alerts | Add to future enhancements |
| Low | Retry loops can burn tokens before timeout — recommend token budget monitoring | Add to future enhancements |

### Architecture Completeness Checklist

**Requirements Analysis**
- [x] Project context thoroughly analyzed
- [x] Scale and complexity assessed
- [x] Technical constraints identified
- [x] Cross-cutting concerns mapped

**Architectural Decisions**
- [x] Critical decisions documented with versions
- [x] Technology stack fully specified
- [x] Integration patterns defined
- [x] Performance considerations addressed

**Implementation Patterns**
- [x] Naming conventions established
- [x] Structure patterns defined
- [x] Communication patterns specified
- [x] Process patterns documented

**Project Structure**
- [x] Complete directory structure defined
- [x] Component boundaries established
- [x] Integration points mapped
- [x] Requirements to structure mapping complete

### Architecture Readiness Assessment

**Overall Status:** READY FOR IMPLEMENTATION
**Confidence Level:** High

**Key Strengths:**
- Architecture is grounded in an existing, working codebase — not theoretical
- All 16 checklist items confirmed complete
- No critical or important gaps identified
- Multiple validation rounds (code review gauntlet, abstraction laddering) applied

**Areas for Future Enhancement:**
- Publish ruff configuration and pre-commit hooks
- Implement Redis-backed rate limiting for multi-instance deployments
- Move `app/eval_ragas.py` → `scripts/eval_ragas.py` and root `shared.py` → `app/shared.py`
- Set up `alembic/` migration infrastructure

### Implementation Handoff

**AI Agent Guidelines:**
- Follow all architectural decisions exactly as documented
- Use implementation patterns consistently across all components
- Respect project structure and boundaries
- Refer to this document for all architectural questions

**First Implementation Priority:**
- The agentic conversion is already partially implemented. Focus on hardening: add Redis-backed rate limiting, set up Alembic migrations, and move evaluation script to `scripts/`

---

## MCP Decoupling: Knowledge Search Tool

### Decision

Decouple the `knowledge_search` tool from LangChain's in-process `@tool` API using the **Model Context Protocol (MCP)**. The tool logic becomes an independent MCP server accessible via stdio (or optionally SSE) transport. The application connects through an MCP client while keeping the `SupervisorAgent` interface unchanged.

### Context

The current `knowledge_search` tool at `app/agents/tools/knowledge_search.py` is a LangChain `@tool` that requires `RetrieverPort` injection at build time via Python import. It cannot be consumed by non-LangChain clients (e.g., direct MCP clients, future CLI tools, cross-language agents) without duplicating the logic. It also shares the application process's memory and failure domain.

### Architecture

```
Before:
SupervisorAgent → knowledge_search_tool (LangChain @tool) → RetrieverPort

After:
SupervisorAgent → MCPToolWrapper (LangChain-compatible) → MCP Client → [stdio]
                                                                          ↓
                                                           MCP Server (knowledge-search)
                                                                          ↓
                                                                   RetrieverPort
```

### Components

#### 1. MCP Server (`mcp-servers/knowledge-search/server.py`)

- Independent Python process using `mcp` SDK v1.x (`FastMCP`)
- Accepts retriever configuration via env vars (`RETRIEVER_TYPE`, `VECTOR_STORE_TYPE`, etc.)
- Imports the same adapter factory chain — reuses `app.ports` and `app.adapters`
- Exposes a single MCP tool:

```python
@server.tool()
async def knowledge_search(query: str, k: int = 5) -> list[dict]:
    """Search the knowledge base for documents relevant to the query.

    Args:
        query: Natural language search query.
        k: Number of results to return (default 5).
    """
```

- Transports: **stdio** (default, spawned as subprocess by parent), **SSE** (optional for sidecar)
- Runs its own lifecycle: startup → connection → tool invocations → shutdown

#### 2. MCP Client Adapter (`app/adapters/tools/mcp_tool_adapter.py`)

- Connects to the MCP server via `mcp.Client` with `StdioClientParameters`
- On connect: calls `client.list_tools()` for capability discovery
- Returns a callable matching the existing `@tool` signature (`ainvoke({"query": ...})`)
- Manages reconnection: on disconnect, respawns the subprocess (up to 3 retries)
- Timeout: wraps MCP calls in `asyncio.wait_for(..., timeout=30)`

```python
async def create_mcp_knowledge_search_tool(
    server_script: str,
    config: dict,
) -> Callable:
    """Returns an ainvoke-compatible tool backed by an MCP server subprocess."""
```

#### 3. Factory Wiring (`app/factory.py`)

New factory function replaces the previous ad-hoc tool creation:

```python
def create_knowledge_search_tool(retriever_port: RetrieverPort) -> Callable:
    if config.mcp_enabled:
        return create_mcp_knowledge_search_tool(...)
    # fallback: direct LangChain @tool (current behaviour)
    from app.agents.tools.knowledge_search import create_knowledge_search_tool
    return create_knowledge_search_tool(retriever_port, k=config.retrieval_k)
```

#### 4. SupervisorAgent — No Change

The `SupervisorAgent` still receives a callable and calls `tool.ainvoke({"query": ...})`. The MCP wrapper is a drop-in replacement.

### Transport Decision

| Transport | Use Case | Pros | Cons |
|-----------|----------|------|------|
| **stdio** | Dev, single-process deploy | Zero networking, no port conflicts, simplest setup | Process-coupled, no independent scaling |
| **SSE** | Production, K8s, multi-instance | Independent scaling, process isolation | Requires auth, port management, added latency |

**Recommendation**: stdio for the initial implementation; SSE as a future option when the server needs independent deployment.

### Failure Modes & Mitigations

1. **MCP server crashes on startup** → Log error, fall back to direct `@tool` (degraded but functional)
2. **MCP call times out** → Log warning, return `[]` (same as current tool's error path)
3. **Subprocess respawn loop** → After 3 consecutive failures, pin to direct `@tool` for the lifetime of the app
4. **Version mismatch** → Pin `mcp>=1.27,<2` to avoid v2 protocol breakage (scheduled stable: 2026-07-27)

### Implications

- **New dependency**: `mcp>=1.27,<2` in `requirements.txt`
- **New directory**: `mcp-servers/knowledge-search/` with its own `pyproject.toml` (can be developed/tested independently)
- **Docker**: If using stdio, no change. If using SSE, add the MCP server as a separate service in `docker-compose.yml`
- **Testability**: The MCP server can be integration-tested with the MCP Inspector or in-memory transport
- **Architecture boundary**: The MCP server lives outside `app/` — it is a consumer of `app.ports` and `app.adapters`, not part of the API/core boundary

### Acceptance Tests

| Test | What it verifies |
|------|-----------------|
| `test_mcp_server_startup` | Server starts, connects, lists tools |
| `test_mcp_knowledge_search` | Tool returns correct results matching direct `@tool` output |
| `test_mcp_fallback_on_crash` | When MCP server is killed, fallback to direct works |
| `test_mcp_timeout_returns_empty` | Slow MCP response yields empty list, not crash |

### New Config Variables (`app/config.py`)

```python
mcp_enabled: bool = os.getenv("MCP_ENABLED", "false").lower() == "true"
mcp_server_timeout: int = int(os.getenv("MCP_SERVER_TIMEOUT", "30"))
mcp_max_restarts: int = int(os.getenv("MCP_MAX_RESTARTS", "3"))

---

## A2A Decoupling: Agent-to-Agent Protocol for Sub-Agent Orchestration

### Decision

Decouple the three sub-agents (LegalResearchAgent, CitationCheckerAgent, ResponseSynthesizerAgent) from the SupervisorAgent's in-process LangGraph graph using the **Agent-to-Agent (A2A) Protocol** v1.0. Each sub-agent becomes an independent A2A server exposing its capabilities via an Agent Card. The SupervisorAgent becomes an A2A client that sends Tasks to sub-agents over **JSON-RPC 2.0** (the standard A2A transport binding) with **SSE streaming** for task lifecycle events.

This is complementary to the MCP decoupling: MCP decouples *tools* (agent-to-data), A2A decouples *agents* (agent-to-agent). Together they provide full process isolation for every component.

### Context

Currently, `SupervisorAgent` imports three sub-agents as Python classes and calls them in-process:

```python
from .legal_research_agent import LegalResearchAgent
from .citation_checker_agent import CitationCheckerAgent
from .response_synthesizer_agent import ResponseSynthesizerAgent
```

This means:
- All four agents share a single process, GIL, and memory space
- A crash in any sub-agent takes down the entire supervisor graph
- No independent scaling — the right-sizing is for the heaviest agent
- Cross-language agents are impossible (e.g., a Rust-based citation checker)
- No protocol boundary — sub-agent internals are exposed to the supervisor

With A2A, each sub-agent becomes a standalone service that the supervisor delegates work to via a standard protocol.

### Architecture

```
▲ JSON-RPC 2.0 over HTTP (A2A Tasks with SSE streaming)
│
┌────────────────────────────────────────────────────────────────────┐
│                    SupervisorAgent (A2A Client)                     │
│  LangGraph state machine — nodes now send A2A SendMessage requests │
│  instead of in-process await agent.run(...) calls                  │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  A2AClientRouter                                              │  │
│  │  • Resolves agent endpoints via env vars                      │  │
│  │  • Sends Tasks via JSON-RPC tasks/sendMessage                 │  │
│  │  • Receives SSE stream for task progress/completion           │  │
│  │  • Handles retries (3x backoff), timeouts, fallbacks          │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────┬──────────────┬──────────────────┬────────────────────────────┘
      │              │                  │
      │ A2A Task     │ A2A Task         │ A2A Task
      ▼              ▼                  ▼
┌────────────┐ ┌──────────────┐ ┌──────────────────┐
│ A2A Server │ │ A2A Server   │ │ A2A Server       │
│ Legal      │ │ Citation     │ │ Response          │
│ Research   │ │ Checker      │ │ Synthesizer       │
│ Agent      │ │ Agent        │ │ Agent             │
├────────────┤ ├──────────────┤ ├──────────────────┤
│ Port 8101  │ │ Port 8102    │ │ Port 8103        │
│ Agent Card │ │ Agent Card   │ │ Agent Card       │
│ /agent-card│ │ /agent-card  │ │ /agent-card      │
├────────────┤ ├──────────────┤ ├──────────────────┤
│ FastAPI    │ │ FastAPI      │ │ FastAPI          │
│ + A2AServer│ │ + A2AServer  │ │ + A2AServer      │
└─────┬──────┘ └──────┬───────┘ └────────┬─────────┘
      │               │                  │
      ▼               ▼                  ▼
 RetrieverPort   VectorStorePort      LLMPort
```

### Agent Card Design

Each sub-agent publishes an Agent Card at `/.well-known/agent-card` (or `/agent-card`) using the A2A v1.0 schema. The card advertises skills, capabilities, and the endpoint URL.

#### LegalResearchAgent Agent Card

```json
{
  "name": "legal-research-agent",
  "description": "Performs legal research: HyDE generation, sub-query decomposition, parallel retrieval, LLM relevance scoring, blended ranking.",
  "version": "1.0.0",
  "provider": {
    "organization": "company-knowledge-assistant",
    "url": ""
  },
  "capabilities": {
    "streaming": true,
    "pushNotifications": false,
    "stateful": true
  },
  "skills": [
    {
      "id": "legal_research",
      "name": "Legal Research",
      "description": "Given a legal query, returns ranked legal articles with relevance scores.",
      "tags": ["legal", "retrieval", "ranking"],
      "inputs": [
        {
          "name": "query",
          "type": "string",
          "description": "Natural language legal question"
        },
        {
          "name": "metadata",
          "type": "object",
          "description": "Optional filtering metadata (category, tenant_id)",
          "optional": true
        }
      ],
      "outputs": [
        {
          "name": "articles",
          "type": "array",
          "description": "Ranked list of Article objects with content, source, score"
        }
      ]
    }
  ],
  "defaultInterface": {
    "type": "rest",
    "url": "http://legal-research:8101/",
    "authentication": false
  },
  "interfaces": [
    {
      "type": "rest",
      "url": "http://legal-research:8101/",
      "authentication": false
    }
  ]
}
```

#### CitationCheckerAgent Agent Card

```json
{
  "name": "citation-checker-agent",
  "description": "Three-gate hallucination firewall: existence check, relevance scoring, LLM contradiction detection.",
  "version": "1.0.0",
  "skills": [
    {
      "id": "citation_check",
      "name": "Citation Verification",
      "description": "Verifies legal citations against the knowledge base through a 3-gate pipeline.",
      "inputs": [
        {
          "name": "articles",
          "type": "array",
          "description": "Articles to verify"
        },
        {
          "name": "query",
          "type": "string",
          "description": "Original query context"
        }
      ],
      "outputs": [
        {
          "name": "verified_citations",
          "type": "array",
          "description": "Citations that passed all three gates"
        },
        {
          "name": "invalid_citations",
          "type": "array",
          "description": "Citations that failed existence check"
        },
        {
          "name": "contradictions",
          "type": "array",
          "description": "Detected contradictions between citations"
        },
        {
          "name": "consistency_score",
          "type": "number",
          "description": "Overall consistency score (0.0–1.0)"
        }
      ]
    }
  ]
}
```

#### ResponseSynthesizerAgent Agent Card

```json
{
  "name": "response-synthesizer-agent",
  "description": "Generates grounded Vietnamese legal responses from verified citations.",
  "version": "1.0.0",
  "skills": [
    {
      "id": "response_synthesis",
      "name": "Response Synthesis",
      "description": "Synthesizes a natural-language legal response from verified citations.",
      "inputs": [
        {
          "name": "query",
          "type": "string",
          "description": "Original user question"
        },
        {
          "name": "citations",
          "type": "array",
          "description": "Verified Citation objects"
        }
      ],
      "outputs": [
        {
          "name": "response",
          "type": "string",
          "description": "Generated legal response in Vietnamese"
        },
        {
          "name": "metadata",
          "type": "object",
          "description": "Citation count, response length, optional error"
        }
      ]
    }
  ]
}
```

### Task/Response Schemas

Each sub-agent interaction follows the A2A `SendMessage` pattern over **JSON-RPC 2.0** with **SSE streaming**. The supervisor sends a JSON-RPC request to `POST /sendMessage` (or `POST /json-rpc` if using a single endpoint). The agent responds with an SSE stream of task status events (`working` → `completed`/`failed`), allowing the supervisor to observe progress without polling. For intermediate LangGraph nodes (HyDE generation, sub-query decomposition, retrieval), the agent emits `state: "working"` events with progress metadata.

#### Legal Research: SendMessage Request

```json
POST http://legal-research:8101/sendMessage
Content-Type: application/json
A2A-Version: 1.0.0

{
  "jsonrpc": "2.0",
  "method": "tasks/sendMessage",
  "params": {
    "id": "task-lr-<uuid>",
    "message": {
      "role": "user",
      "parts": [
        {
          "type": "text",
          "text": "What are the labor law requirements for overtime pay in Vietnam?"
        }
      ]
    },
    "metadata": {
      "category": "labor_law",
      "tenant_id": "default"
    }
  }
}
```

#### Legal Research: Response

```json
{
  "jsonrpc": "2.0",
  "result": {
    "id": "task-lr-<uuid>",
    "status": {
      "state": "completed",
      "timestamp": "2026-06-21T10:30:00Z"
    },
    "artifacts": [
      {
        "parts": [
          {
            "type": "data",
            "data": {
              "articles": [
                {
                  "id": "art-001",
                  "title": "Labor Code Article 98",
                  "content": "Overtime pay is calculated at 150% of normal wage on weekdays...",
                  "source": "labor_code_2024.pdf",
                  "score": 0.92
                }
              ],
              "reasoning_steps": [
                {"agent": "legal_research", "action": "hyde_generation", "status": "completed"},
                {"agent": "legal_research", "action": "retrieve_articles", "result_count": 5}
              ]
            }
          }
        ]
      }
    ]
  }
}
```

#### Citation Check: SendMessage Request

```json
POST http://citation-checker:8102/sendMessage
Content-Type: application/json
A2A-Version: 1.0.0

{
  "jsonrpc": "2.0",
  "method": "tasks/sendMessage",
  "params": {
    "id": "task-cc-<uuid>",
    "message": {
      "role": "user",
      "parts": [
        {
          "type": "data",
          "data": {
            "articles": [{"id": "art-001", "title": "...", "content": "...", "score": 0.92}],
            "query": "What are the labor law requirements for overtime pay in Vietnam?"
          }
        }
      ]
    }
  }
}
```

#### Response Synthesis: SendMessage Request

```json
POST http://response-synthesizer:8103/sendMessage
Content-Type: application/json
A2A-Version: 1.0.0

{
  "jsonrpc": "2.0",
  "method": "tasks/sendMessage",
  "params": {
    "id": "task-rs-<uuid>",
    "message": {
      "role": "user",
      "parts": [
        {
          "type": "data",
          "data": {
            "query": "What are the labor law requirements...",
            "citations": [
              {"article_id": "art-001", "content": "Overtime pay is calculated at 150%...", "relevance_score": 0.88}
            ]
          }
        }
      ]
    }
  }
}
```

### Transport Decision: JSON-RPC 2.0 with SSE Streaming

| Transport | Pros | Cons | Verdict |
|-----------|------|------|---------|
| **JSON-RPC 2.0 + SSE** | Standard A2A binding, `a2a-sdk` handles serialization, compact wire format | Requires SSE client on supervisor side | ✅ **Selected** |
| **HTTP+JSON/REST** | Familiar, no SDK dependency | No native streaming, diverges from standard | ❌ Lost to JSON-RPC |
| **gRPC** | Fast binary serialization, streaming | Protobuf toolchain, HTTP/2, overkill here | ❌ Premature optimization |

**Recommendation**: JSON-RPC 2.0 over HTTP with SSE streaming for task lifecycle events. Each sub-agent runs as a lightweight FastAPI app with:
- `POST /sendMessage` — initiate a task, returns an SSE stream of Task status events
- `POST /json-rpc` — alternative single-endpoint JSON-RPC handler (optional, for SDK compatibility)
- `GET /.well-known/agent-card` — serve Agent Card (optional `GET /agent-card` as alias)

Each sub-agent wraps its existing `run()` method inside an A2A task handler. The internal LangGraph state machine within each sub-agent remains unchanged — only the entry point changes. The `a2a-sdk` provides `A2AServer` and `A2AClient` classes that handle JSON-RPC serialization, SSE framing, and error codes natively.

**SSE Streaming Contract:**

```
→ POST /sendMessage  (JSON-RPC request body)
← SSE stream:
  event: task_status
  data: {"id": "task-lr-<uuid>", "status": {"state": "working", "timestamp": "...", "metadata": {"node": "hyde_generation"}}}

  event: task_status
  data: {"id": "task-lr-<uuid>", "status": {"state": "working", "timestamp": "...", "metadata": {"node": "subquery_decomposition", "subqueries": 3}}}

  event: task_status
  data: {"id": "task-lr-<uuid>", "status": {"state": "working", "timestamp": "...", "metadata": {"node": "parallel_retrieval", "results": 5}}}

  event: task_status
  data: {"id": "task-lr-<uuid>", "status": {"state": "completed", "timestamp": "..."}}
  event: task_artifact
  data: {"id": "task-lr-<uuid>", "artifact": {...}}
```

The supervisor receives the stream via `a2a-sdk`'s `A2AClient.send_task()` which returns an async iterator of status events, ending with the final artifact.

### Failure Handling

| Failure | Behavior | Supervisor Action |
|---------|----------|-------------------|
| **A2A agent unreachable** (connection refused, DNS failure) | Log error, increment retry counter | Up to 3 retries with exponential backoff (1s, 2s, 4s); then proceed with degraded state (empty results) |
| **Timeout** (no response within 30s) | Log warning, return timeout error | Treat same as unreachable; existing `asyncio.wait_for` pattern |
| **Bad response** (missing required fields, invalid JSON) | Log error, discard response | Return empty result set; supervisor quality gate handles it downstream |
| **Non-2xx HTTP** (500, 503, 429) | Log status code, backoff on 429 | Retry with backoff; after exhaustion, treat as degraded |
| **Agent returns terminal 'failed' state** | Task.state = "failed", error message in artifact | Log error, proceed with empty results (same as current agent exception path) |
| **Multiple sub-agents down** | Each fails independently | Supervisor continues with whatever results it has; quality gate at the end may produce a best-effort response |
| **Stale Agent Card** (endpoint changed) | HTTP 404 on POST; supervisor re-fetches Agent Card | One retry with fresh Agent Card fetch; then degrade |

**Critical invariant**: The supervisor *never* blocks waiting for a sub-agent. Every A2A SSE stream is consumed with `asyncio.wait_for` on the final artifact event, using a configurable timeout (default 60s). The supervisor can observe intermediate `working` events for logging/monitoring but does not require them to form the final response. The existing quality gate (`validate_quality`) at the end of the LangGraph handles partial or empty results gracefully.

### LangGraph Integration

The SupervisorAgent's LangGraph state machine stays in place. Only the node implementations change — they call A2A servers instead of in-process methods.

**Before (current `execute_legal_research`):**
```python
async def execute_legal_research(self, state):
    articles = await self.research_agent.run(state["query"])
    tool_results = await self.knowledge_search_tool.ainvoke({"query": state["query"]})
    return {"research_results": articles, ...}
```

**After (A2A-based with SSE streaming):**
```python
async def execute_legal_research(self, state):
    async for event in self._a2a_client.send_task_stream(
        agent="legal-research-agent",
        payload={"query": state["query"], "metadata": state.get("metadata", {})},
    ):
        if event.type == "task_status" and event.status.state == "working":
            logger.debug("Legal research progress: %s", event.status.metadata)
        elif event.type == "task_status" and event.status.state == "completed":
            articles = event.artifact.parts[0].data["articles"]
        elif event.type == "task_status" and event.status.state == "failed":
            logger.error("Legal research failed: %s", event.status.error)
            articles = []
    # knowledge_search_tool call unchanged (already MCP-decoupled)
    return {"research_results": articles, ...}
```

The `_a2a_client` is an `A2AClientRouter` injected at construction time:

```python
class SupervisorAgent:
    def __init__(self, ..., a2a_client: A2AClientRouter | None = None):
        self._a2a_client = a2a_client or InProcessFallbackClient(
            research_agent=research_agent,
            citation_agent=citation_agent,
            synthesis_agent=synthesis_agent,
        )
```

The `InProcessFallbackClient` implements the same interface as `A2AClientRouter` but calls existing agents in-process — essential for the phased migration.

### Deployment Topology

#### Phase 1 — Same-Process (In-Process Fallback)

```
Single container: FastAPI app + SupervisorAgent + all sub-agents (current)
Sub-agents remain in-process. A2A client layer wraps them via InProcessFallbackClient.
No networking, no new containers.
```

#### Phase 2 — Separate Processes (Docker Compose)

```
container: app (FastAPI, Supervisor, A2A Client)
container: legal-research-agent (FastAPI, Port 8101)
container: citation-checker-agent (FastAPI, Port 8102)
container: response-synthesizer-agent (FastAPI, Port 8103)
```

Each sub-agent container is a lightweight FastAPI app with:
- Its own `Dockerfile` (or single multi-stage build)
- Its own `requirements.txt` (minimal — just the A2A SDK and its specific dependencies)
- Independent scaling via `docker-compose up --scale legal-research-agent=3`

#### Phase 3 — Production (K8s)

```
Deployment: app (2+ replicas)
Deployment: legal-research-agent (3+ replicas)
Deployment: citation-checker-agent (2+ replicas)
Deployment: response-synthesizer-agent (2+ replicas)
Service: legal-research (ClusterIP, port 8101)
Service: citation-checker (ClusterIP, port 8102)
Service: response-synthesizer (ClusterIP, port 8103)
```

The A2A client in the supervisor resolves agent endpoints via **environment variables** (e.g., `A2A_LEGAL_RESEARCH_URL=http://legal-research:8101`). No service registry — each agent URL is configured at deployment time.

### Phasing Plan

| Phase | Scope | What changes | Risk |
|-------|-------|-------------|------|
| **0** | No A2A (current) | Nothing | — |
| **1** | A2A client interface + `InProcessFallbackClient` | New `A2AClientRouter` abstraction; all existing code unchanged | Low — new abstraction only, no behavioral change |
| **2** | LegalResearchAgent as A2A server | Extract to standalone FastAPI app + A2A agent card; supervisor points to it | Medium — first extraction, validates the pattern |
| **3** | CitationCheckerAgent as A2A server | Same pattern as phase 2 | Low — proven pattern |
| **4** | ResponseSynthesizerAgent as A2A server | Same pattern as phase 2 | Low — proven pattern |
| **5** | Docker Compose update | Add three new services, wiring, health checks | Medium — multi-container orchestration |
| **6** | Remove in-process code | Delete old imports, remove `InProcessFallbackClient` | Low — only after all agents verified in production |

Each phase is independently ship-able. The system remains fully operational at every step.

### New Dependencies

```txt
# requirements.txt (main app)
a2a-sdk>=1.1,<2

# Each sub-agent's requirements.txt
a2a-sdk>=1.1,<2
fastapi>=0.115.0,<0.116.0
uvicorn>=0.30.0,<0.31.0
```

### New Config Variables (`app/config.py`)

```python
# A2A agent URLs (empty = use InProcessFallbackClient)
a2a_legal_research_url: str = _str_env("A2A_LEGAL_RESEARCH_URL", "")
a2a_citation_checker_url: str = _str_env("A2A_CITATION_CHECKER_URL", "")
a2a_response_synthesizer_url: str = _str_env("A2A_RESPONSE_SYNTHESIZER_URL", "")
a2a_task_timeout: int = _int_env("A2A_TASK_TIMEOUT", 60)
a2a_max_retries: int = _int_env("A2A_MAX_RETRIES", 3)
```

### A2A vs MCP: Where Each Applies

```
                MCP                             A2A
   ┌─────────────────────────┐     ┌──────────────────────────┐
   │  Agent → Tool           │     │  Agent → Agent           │
   │  "Find this data"       │     │  "Do this task"          │
   │                         │     │                          │
   │  knowledge_search       │     │  legal_research          │
   │  (RetrieverPort)        │     │  citation_check          │
   │                         │     │  response_synthesis      │
   │  stdio (subprocess)     │     │  JSON-RPC 2.0 + SSE     │
   │  Single tool call       │     │  Complex task lifecycle  │
   │  LangChain compatible   │     │  Framework-agnostic      │
   └─────────────────────────┘     └──────────────────────────┘
```

Both are governed by the Linux Foundation Agentic AI Foundation. Both are production-stable. Both are used here.

### Acceptance Tests

| Test | What it verifies |
|------|-----------------|
| `test_a2a_client_router` | A2AClientRouter sends correct HTTP requests and parses responses |
| `test_a2a_inprocess_fallback` | InProcessFallbackClient returns same results as direct agent calls |
| `test_a2a_unreachable_agent` | When A2A agent is down, supervisor degrades gracefully (empty results) |
| `test_a2a_timeout` | Slow A2A agent causes timeout, supervisor continues |
| `test_a2a_bad_response` | Malformed A2A response yields empty results, not crash |
| `test_a2a_legal_research_endpoint` | Legal research agent serves valid Agent Card and responds to SendMessage |
| `test_a2a_citation_check_endpoint` | Citation checker agent serves valid Agent Card |
| `test_a2a_response_synthesis_endpoint` | Response synthesizer agent serves valid Agent Card |
| `test_a2a_supervisor_end_to_end` | Full chain with A2A agents produces same answer format as in-process |
```

---

## Dual-Store Reconciliation: pgvector + ChromaDB

Decision: Keep both pgvector and ChromaDB with clear role separation.

**Rationale:**
- pgvector: Existing adapters (`PGVectorStoreAdapter`, `DenseRetrieverAdapter`) already work. Canonical metadata lives in PostgreSQL. pgvector handles structured vector queries with SQL filtering.
- ChromaDB: Better for fast prototyping, local development, and embeddings-first workflows. Handles embedding storage/retrieval without Postgres overhead.

**Role Separation:**

| Store | Role | When Used |
|-------|------|-----------|
| PostgreSQL + pgvector | Production canonical store. Metadata + vectors in one DB. Supports complex SQL filters, joins, multi-tenant isolation. | Production, staging |
| ChromaDB | Development/prototyping store. Fast local setup, no Postgres dependency. Embeddings-first workflow. | Local dev, testing |

**Adapter Layer:**

```
app/ports/retriever_port.py          # Abstract interface
app/adapters/retrievers/
    pgvector_retriever.py            # Production: PGVectorStoreAdapter → DenseRetrieverAdapter
    chromadb_retriever.py            # Dev: ChromaDBAdapter → DenseRetrieverAdapter (NEW)
```

**Factory Toggle:**

```python
# app/factory.py
def create_retriever(config: Config) -> RetrieverPort:
    if config.vector_store == "chromadb":
        return ChromaDBAdapter(chroma_path=config.chromadb_path)
    return PGVectorStoreAdapter(connection=config.postgres_dsn)
```

Config variable: `VECTOR_STORE=pgvector|chromadb` (default: `pgvector`)

**Migration Path:**
1. Phase 1: Both adapters exist, `VECTOR_STORE` toggles between them
2. Phase 2: Add `chromadb_to_pgvector` sync script for data migration
3. Phase 3: Deprecate ChromaDB in production, keep for local dev only

**Boundary Tests Update:**
- Add `test_vector_store_toggle`: Factory returns correct adapter based on `VECTOR_STORE` config
- Add `test_chromadb_adapter_implements_retriever_port`: ChromaDB adapter satisfies `RetrieverPort`

---

## Architecture Validation Results

### Coherence Assessment

**Decision Compatibility:** All decisions work together. MCP (agent→tool) and A2A (agent→agent) are cleanly separated. LangGraph state machine is preserved inside A2A wrappers. Redis handles caching, rate limiting, and sessions consistently.

**Pattern Consistency:** Naming (snake_case), async (asyncio.wait_for with timeouts), error handling (retry + fallback), and logging (%s-style) are consistent across all sections.

**Structure Alignment:** Directory structure supports all architectural decisions. A2A servers are separate FastAPI apps. MCP server is outside `app/`. Ports remain pure interfaces.

### Requirements Coverage

**FR1-FR7:** All functional requirements covered. Session store (FR1), MCP tools (FR2), LangGraph reasoning (FR3), memory compression (FR4), retry+fallback (FR5), OAuth2/OIDC (FR6), pgvector+ChromaDB (FR7).

**NFR1-NFR7:** All non-functional requirements addressed. A2A timeout budget documented below. Token efficiency via Redis cache. Observability via LangSmith + SSE. Error handling via exponential backoff. Scalability via A2A K8s topology. Consistency via quality gates. Rate limiting via Redis.

### Gap Analysis

**Resolved:**
- ChromaDB vs pgvector conflict → Documented dual-store strategy with role separation

**Documented (non-blocking):**
- A2A timeout budget: 60s per A2A call (30s timeout + 3 retries), 180s total, capped by supervisor at 90s with partial results
- A2A inter-service auth: mTLS in Phase 3 (K8s), JWT introspection for local dev
- `app/auth/` added to project structure

### Architecture Completeness Checklist

- [x] Project context thoroughly analyzed
- [x] Scale and complexity assessed
- [x] Technical constraints identified
- [x] Cross-cutting concerns mapped
- [x] Critical decisions documented with versions
- [x] Technology stack fully specified
- [x] Integration patterns defined
- [x] Performance considerations addressed
- [x] Naming conventions established
- [x] Structure patterns defined
- [x] Communication patterns specified
- [x] Process patterns documented
- [x] Complete directory structure defined
- [x] Component boundaries established
- [x] Integration points mapped
- [x] Requirements to structure mapping complete

**Overall Status:** READY FOR IMPLEMENTATION

**Confidence:** High
