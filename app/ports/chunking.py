from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class ChunkingPort(ABC):
    """Abstract interface for any document chunking strategy.

    Concrete adapters (Recursive, Semantic, ...) must implement this.
    The rest of the system depends only on this interface, never on a
    specific chunking strategy.
    """

    @abstractmethod
    def chunk(self, documents: list[Document]) -> list[Document]:
        """Split documents into smaller chunks."""
        ...
