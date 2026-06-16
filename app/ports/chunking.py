from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class ChunkingPort(ABC):
    """Abstract interface for any document chunking strategy.

    Concrete adapters (Recursive, Semantic, ...) must implement this.
    The rest of the system depends only on this interface, never on a
    specific chunking strategy.
    """

    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        """Split documents into smaller chunks."""
        ...
