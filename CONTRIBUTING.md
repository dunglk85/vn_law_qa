# Contributing

## Architecture Overview

This project uses a **Port/Adapter** (Hexagonal) architecture:

- **Ports** (`app/ports/`) — Abstract interfaces defining the contract (e.g., `EmbeddingsPort`, `VectorStorePort`)
- **Adapters** (`app/adapters/`) — Concrete implementations of port interfaces (e.g., `embeddings/openai_embeddings.py`)
- **Factory** (`app/factory.py`) — Config-driven dependency injection that selects adapters at runtime
- **Core** (`app/core/`) — Business logic that depends only on port abstractions
- **API** (`app/api.py`) — FastAPI routes, bootstrap wiring

## Adding a New Adapter Provider

The factory uses a **registry pattern** with `@_register` decorators. Follow this pattern to add a new provider for any port interface.

### Step 1: Create the adapter file

```
app/adapters/{service}/{provider}_{service}.py
```

Example — adding a new embedding provider:

```python
# app/adapters/embeddings/anthropic_embeddings.py
from __future__ import annotations

from langchain_core.embeddings import Embeddings

from app.ports.embeddings import EmbeddingsPort


class AnthropicEmbeddingsAdapter(EmbeddingsPort):
    def __init__(self) -> None:
        self._client = ...

    def get_embeddings(self) -> Embeddings:
        return self._client
```

Naming conventions:
- File: `{provider}_{service}.py`
- Class: `{Provider}{Service}Adapter`
- Must inherit from the corresponding `{Service}Port`

### Step 2: Register the adapter in the factory

In `app/factory.py`, use the `@_register` decorator:

```python
@_register("embeddings", "anthropic")
def _create_anthropic_embeddings(model: str, api_key: str | None) -> EmbeddingsPort:
    from app.adapters.embeddings.anthropic_embeddings import AnthropicEmbeddingsAdapter
    return AnthropicEmbeddingsAdapter(model=model, api_key=api_key)
```

The first argument is the **kind** (matches config key prefix), the second is the **key** (matches `*_TYPE` env var value). The import is lazy — it happens only when the adapter is actually resolved.

### Step 3: Set the environment variable

```bash
# .env
EMBEDDINGS_TYPE=anthropic
```

The public factory function already delegates to `_resolve("embeddings", config.embeddings_type, ...)` — no function body changes needed.

## Isolation Boundaries

Adapters **must not** import from:
- `app.core` — core business logic
- `app.api` — API layer

This is enforced by architecture tests in `tests/test_architecture.py`:

```bash
python -m pytest tests/test_architecture.py -v
```

The CI pipeline (`.github/workflows/ci.yml`) runs linting and tests on every push and pull request.

## Factory Error Handling

If a config value references an unknown provider, the factory's `_resolve()` function raises `ValueError` with all supported options listed:

```python
raise ValueError(f"Unknown EMBEDDINGS_TYPE='{key}'. Supported: openai, anthropic")
```

This ensures misconfigured deployments fail fast at startup.

## Adding an A2A Agent Server

Each agent (legal-research, citation-checker, response-synthesizer) can run as a standalone A2A server:

1. Create a server module in `app/agents/a2a_servers/{agent}_server.py`
2. It exposes a FastAPI app with A2A-compatible endpoints
3. The `Dockerfile.a2a` entrypoint maps `A2A_AGENT` env var to the module name
4. Ports: 8101 (legal-research), 8102 (citation-checker), 8103 (response-synthesizer)

```bash
# Run locally
A2A_AGENT=legal-research uvicorn app.agents.a2a_servers.legal_research_server:app --port 8101
```

## Adding an A2A Remote Client

To add an A2A remote adapter:

1. Create the adapter class in `app/adapters/agents/` implementing the agent interface
2. Wire it into `create_a2a_client() in `app/factory.py`
3. Set the corresponding `A2A_*_URL` env var to point to the remote server

## MCP (Model Context Protocol) Tool

The knowledge search tool can run as an MCP server when `MCP_ENABLED=true`:

- Direct mode (default): Uses `app/agents/tools/knowledge_search.py` LangChain `@tool`
- MCP mode: Uses `app/adapters/tools/mcp_tool_adapter.py` — lazy-connects on first invocation

## Testing

Run tests locally:

```bash
# All tests (requires Redis for rate_limiter tests)
python -m pytest tests/ -v

# Architecture boundary tests only
python -m pytest tests/test_architecture.py -v

# Unit tests only
python -m pytest tests/unit/ -v

# Integration tests
python -m pytest tests/integration/ -v

# Exclude Redis-dependent tests
python -m pytest tests/ --ignore=tests/test_rate_limiter.py -v
```

## Code Quality

- Linting: `ruff check .` (configured in `pyproject.toml` — line-length 120, py311 target)
- CI enforces: `ruff check .` + `python -m pytest tests/ --ignore=tests/test_rate_limiter.py -v`
- Follow existing conventions: `from __future__ import annotations` at the top of every file, `%s`-style logging, `logging.getLogger(__name__)` for new modules
