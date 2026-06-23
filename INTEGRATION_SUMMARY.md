# Agentic RAG Integration Summary

## Completed Integration Tasks

### 1. Configuration Management ✅
**File**: [app/config.py](app/config.py)

Added `RAG_MODE` environment variable support:
- `rag_mode: str = os.getenv("RAG_MODE", "legacy")`
- Accepts: `legacy` (default) or `agentic`
- Enables runtime toggling between traditional RAG (RAGService) and agentic RAG (AgenticService)
- Defaults to `legacy` for backward compatibility

### 2. Agent Factory Functions ✅
**File**: [app/factory.py](app/factory.py)

Added four new factory functions following existing port-based pattern:

```python
def create_legal_research_agent(retriever: RetrieverPort, llm: LLMPort) -> LegalResearchAgent
def create_citation_checker_agent(vector_store: VectorStorePort, llm: LLMPort) -> CitationCheckerAgent
def create_response_synthesizer_agent(llm: LLMPort) -> ResponseSynthesizerAgent
def create_agentic_service(vector_store, llm, retriever, query_transformer) -> AgenticService
```

**Key Design**:
- Follows existing factory pattern with imports deferred to function body
- Uses `llm.get_chat_model()` to extract LangChain BaseChatModel from LLMPort
- Agents receive injected dependencies (ports/models) instead of hardcoded backends
- Maintains consistency with other port-based adapters

### 3. FastAPI API Integration ✅
**File**: [app/api.py](app/api.py)

**Imports**:
- Added `from app.config import config`
- Added `create_agentic_service` to factory imports

**Bootstrap**:
- Added conditional AgenticService initialization based on `config.rag_mode`
```python
_agentic_service = None
if config.rag_mode.lower() == "agentic":
    _agentic_service = create_agentic_service(
        vector_store=_vector_store,
        llm=_llm,
        retriever=_retriever,
        query_transformer=_query_transformer,
    )
```

**Endpoint Update**:
- Updated `/ask` endpoint to use appropriate service:
```python
if _agentic_service:
    answer, sources, contexts = await _agentic_service.answer(...)
else:
    answer, sources, contexts = await _rag_service.answer(...)
```

### 4. Code Cleanup ✅
- Removed backup file: `app/agents/citation_checker_agent_v2.py`
- All refactored agent files consolidated to main versions

## Validation Results ✅

**CI Pipeline**: All checks pass on every push
- ✓ Lint: `ruff check .` (0 errors)
- ✓ Architecture boundary: `tests/test_architecture.py` (9/9 passing)
- ✓ Unit tests: `tests/unit/` (config, factory, agents, services, auth, exceptions, token tracking)
- ✓ Integration tests: `tests/integration/test_api.py` (auth flow, /health, /ask)
- ✓ A2A tests: `tests/test_a2a_client.py`, `tests/test_a2a_phase2.py`
- ✓ MCP tests: `tests/test_mcp_tool.py`
- ✓ Docker build: `docker-compose build` (main app + A2A agents)
- ✓ Docker run: All services start and pass health checks

### Files verified
- ✓ `app/api.py` — FastAPI bootstrap, conditional agentic service, JWT auth, rate limiting
- ✓ `app/config.py` — 60+ env vars across 12 categories (A2A, MCP, LangSmith, auth, session, retry)
- ✓ `app/factory.py` — Registry pattern with `@_register` decorators
- ✓ `app/core/agentic_service.py` — LangGraph multi-agent orchestration
- ✓ `app/core/rag_service.py` — Traditional RAG service
- ✓ `app/core/token_tracker.py` — Token usage tracking
- ✓ `app/core/retry.py` — Exponential backoff with jitter
- ✓ `app/agents/legal_research_agent.py` — HyDE + decomposition + rerank pipeline
- ✓ `app/agents/citation_checker_agent.py` — 3-gate hallucination firewall
- ✓ `app/agents/response_synthesizer_agent.py` — Vietnamese legal response generation
- ✓ `app/agents/supervisor_agent.py` — LangGraph state machine (6 nodes)
- ✓ `app/agents/a2a_servers/` — A2A agent servers (legal-research, citation-checker, response-synthesizer)
- ✓ `app/adapters/agents/a2a_remote_client.py` — Remote A2A HTTP client
- ✓ `app/adapters/agents/a2a_fallback_client.py` — In-process fallback client
- ✓ `app/auth/` — JWT auth, /auth/token, /auth/refresh
- ✓ `app/adapters/tools/mcp_tool_adapter.py` — MCP-backed knowledge search tool

## Dependency Injection Architecture

### Traditional RAG Path (Default)
```
FastAPI.ask() → RAGService
  ├── VectorStore (Postgres/Chroma/Pinecone)
  ├── LLM (OpenAI/Ollama)
  ├── Reranker (Cohere/local)
  ├── Retriever (LangChain)
  └── QueryTransformer (Decomposition/HyDE/None)
```

### Agentic RAG Path (Optional)
```
FastAPI.ask() → AgenticService
  ├── LegalResearchAgent
  │   ├── RetrieverPort
  │   └── BaseChatModel
  ├── CitationCheckerAgent
  │   ├── VectorStorePort
  │   └── BaseChatModel
  ├── ResponseSynthesizerAgent
  │   └── BaseChatModel
  └── SupervisorAgent
      ├── All three agents above
      └── LangGraph state machine
```

## Runtime Configuration

### Enable Agentic Mode (in-process agents)
```bash
export RAG_MODE=agentic
python -m uvicorn app.api:app --reload
```

### Enable Agentic Mode (A2A remote agents)
```bash
export RAG_MODE=agentic
# Point to running A2A agent servers:
export A2A_LEGAL_RESEARCH_URL=http://localhost:8101
export A2A_CITATION_CHECKER_URL=http://localhost:8102
export A2A_RESPONSE_SYNTHESIZER_URL=http://localhost:8103
docker compose up -d  # starts app + a2a agents + postgres + redis
```

### Enable Legacy Mode (Default)
```bash
export RAG_MODE=legacy
# or simply omit RAG_MODE
python -m uvicorn app.api:app --reload
```

## Key Design Decisions

1. **Port-Based Dependencies**: All agents accept abstract ports (RetrieverPort, VectorStorePort, LLMPort) instead of concrete implementations, enabling provider swapping.

2. **Backward Compatibility**: RAGService remains unchanged and is used by default, preserving existing behavior.

3. **Lazy Initialization**: AgenticService is only created if `RAG_MODE=agentic`, minimizing overhead for users not using agentic features.

4. **LangGraph Integration**: Agents use LangGraph StateGraph for workflow orchestration with conditional routing and parallel execution support.

5. **Unified Response Format**: Both services return `(answer_str, sources_list, contexts_list)` tuples, allowing drop-in service replacement.

6. **A2A Protocol Support**: Agents can run as standalone A2A servers (remote) or in-process (fallback). The factory selects the mode based on configured `A2A_*_URL` env vars.

7. **Factory Registry Pattern**: Adapter registration uses `@_register("kind", "key")` decorators — adding a new provider requires no switch/match changes, just a new adapter class and registration.

8. **MCP Tool Support**: The knowledge search tool supports both direct (LangChain `@tool`) and MCP-backed modes, selectable via `MCP_ENABLED` env var.

## Files Modified
1. `app/config.py` - Added rag_mode, A2A, MCP, LangSmith, session, retry config
2. `app/factory.py` - Refactored to registry pattern; added agent, A2A, MCP, document loader factories
3. `app/api.py` - Added agentic service bootstrap, conditional routing, /metrics endpoint, rate limiting
4. `app/core/token_tracker.py` - Token usage tracking
5. `app/core/retry.py` - Retry logic with exponential backoff
6. `app/core/agentic_service.py` - LangGraph orchestration with reply-in-thread
7. `tests/test_architecture.py` - Exemption for `app.core.a2a_client` in adapter isolation
8. Deleted `app/agents/citation_checker_agent_v2.py` - Removed backup file

## Files Created
- `app/core/agentic_service.py` - LangGraph multi-agent orchestration
- `app/core/a2a_client.py` - Abstract A2A client interface
- `app/exceptions.py` - AppError, ConfigurationError classes
- `app/agents/legal_research_agent.py` - HyDE + decomposition + rerank agent
- `app/agents/citation_checker_agent.py` - 3-gate hallucination firewall agent
- `app/agents/response_synthesizer_agent.py` - Vietnamese response generation agent
- `app/agents/supervisor_agent.py` - LangGraph state machine orchestrator
- `app/agents/tools/knowledge_search.py` - LangChain @tool wrapper
- `app/agents/a2a_servers/` - A2A server modules (3 agents)
- `app/adapters/agents/a2a_remote_client.py` - Remote A2A HTTP client
- `app/adapters/agents/a2a_fallback_client.py` - In-process fallback client
- `app/adapters/tools/mcp_tool_adapter.py` - MCP-backed knowledge search tool
- `app/auth/` - JWT auth module (router, jwt, dependencies)
- `app/adapters/caches/` - RedisCacheAdapter, NoneCacheAdapter
- `app/adapters/session_stores/` - RedisSessionStore, MemorySessionStore
- `app/adapters/rate_limiters/` - RedisRateLimiterAdapter, MemoryRateLimiterAdapter
- `Dockerfile.a2a` - A2A agent server image
- `scripts/a2a-entrypoint.sh` - A2A container entrypoint
- `tests/unit/` - 8 unit test modules
- `tests/integration/test_api.py` - FastAPI integration tests
- `tests/test_a2a_client.py` - A2A client tests
- `tests/test_a2a_phase2.py` - A2A phase 2 tests
- `tests/test_mcp_tool.py` - MCP tool adapter tests
- `tests/conftest.py` - Shared test fixtures
- Updated `app/ports/vector_store.py` - Added get_documents_by_ids()
- Updated `app/adapters/vector_stores/pgvector_store.py` - Implemented get_documents_by_ids()

## Completed Follow-Up Items ✅

The following items from the original "Next Steps" are now implemented:

| Item | Status | Details |
|------|--------|---------|
| Testing | ✅ | Unit tests (`tests/unit/`), integration tests, A2A tests, MCP tests, CI pipeline |
| Documentation | ✅ | Architecture docs, contributing guide, this integration summary updated |
| Monitoring | ✅ | `/metrics` endpoint exposing token usage, LLM call count, HTTP error count |
| Extended Features | ✅ | A2A protocol (remote agent servers), MCP-backed tools, LangSmith tracing, session management, rate limiting |

## Future Considerations

1. **Benchmarking**: Compare agentic vs. traditional RAG on legal query benchmarks (use `scripts/eval_ragas.py`)
2. **Fine-grained agent config**: Per-agent model selection, timeout overrides via env vars
3. **Alembic auto-migration**: Run migrations automatically on container startup
