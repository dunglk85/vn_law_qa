from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class RetrieverPort(ABC):
    """Abstract interface for any retrieval strategy.

    Concrete adapters (BM25, Hybrid Interleaving, Hybrid RRF, ...) must
    implement this. The rest of the system depends only on this interface.
    """

    @abstractmethod
    def get_retriever(self) -> BaseRetriever:
        """Return a LangChain-compatible BaseRetriever instance."""
        ...

    @abstractmethod
    def build_index(self, documents: List[Document]) -> None:
        """Build any required index from the document corpus."""
        ...
