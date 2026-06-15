from __future__ import annotations
from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings


class EmbeddingsPort(ABC):
    """Abstract interface for any embeddings provider.

    Concrete adapters (OpenAI, HuggingFace, ...) must implement this.
    The rest of the system depends only on this interface, never on a
    specific provider.
    """

    @abstractmethod
    def get_embeddings(self) -> Embeddings:
        """Return a LangChain-compatible Embeddings instance."""
        ...
