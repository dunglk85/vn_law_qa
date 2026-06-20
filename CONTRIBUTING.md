# Contributing

## Architecture Overview

This project uses a **Port/Adapter** (Hexagonal) architecture:

- **Ports** (`app/ports/`) — Abstract interfaces defining the contract (e.g., `EmbeddingsPort`, `VectorStorePort`)
- **Adapters** (`app/adapters/`) — Concrete implementations of port interfaces (e.g., `embeddings/openai_embeddings.py`)
- **Factory** (`app/factory.py`) — Config-driven dependency injection that selects adapters at runtime
- **Core** (`app/core/`) — Business logic that depends only on port abstractions
- **API** (`app/api.py`) — FastAPI routes, bootstrap wiring

## Adding a New Adapter Provider

Follow this pattern to add a new provider for any port interface.

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

### Step 2: Add a factory case

In `app/factory.py`, add a new `case` block to the relevant `create_*` function:

```python
def create_embeddings() -> EmbeddingsPort:
    match config.embeddings_type:
        case "openai":
            from app.adapters.embeddings.openai_embeddings import OpenAIEmbeddingsAdapter
            return OpenAIEmbeddingsAdapter()
        case "anthropic":
            from app.adapters.embeddings.anthropic_embeddings import AnthropicEmbeddingsAdapter
            return AnthropicEmbeddingsAdapter()
        case _:
            raise ValueError(f"Unknown embeddings provider: {config.embeddings_type}")
```

Import the adapter **inside** the `case` block (lazy import) to keep the factory fast and avoid circular imports.

### Step 3: Set the environment variable

```bash
# .env
EMBEDDINGS_TYPE=anthropic
```

Each `create_*` function reads its corresponding config value (e.g., `config.embeddings_type` → `EMBEDDINGS_TYPE`).

## Isolation Boundaries

Adapters **must not** import from:
- `app.core` — core business logic
- `app.api` — API layer

This is enforced by architecture tests in `tests/test_architecture.py`:

```bash
python -m pytest tests/test_architecture.py -v
```

The CI pipeline (`.github/workflows/ci.yml`) runs these tests on every push and pull request.

## Factory Error Handling

If a config value references an unknown provider, the factory raises `ValueError` with a clear message:

```python
raise ValueError(f"Unknown embeddings provider: {config.embeddings_type}")
```

This ensures misconfigured deployments fail fast at startup.

## Code Quality

- Linting: `ruff check .` (configured in `pyproject.toml`)
- CI enforces linting + architecture tests
- Follow existing conventions: `from __future__ import annotations` at the top of every file, `%s`-style logging, `logging.getLogger(__name__)` for new modules
