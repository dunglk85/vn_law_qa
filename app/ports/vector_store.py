from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever


class VectorStorePort(ABC):
    """Abstract interface for any vector store backend.

    Swap PGVector → Chroma/Qdrant/Pinecone by creating a new adapter
    that implements this interface, then setting VECTOR_STORE_TYPE in .env.
    Business logic never touches a concrete store class.
    """

    @abstractmethod
    async def add_documents(self, documents: List[Document]) -> None:
        """Embed and persist a list of documents."""
        ...

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        """Return the top-k most similar documents for *query*."""
        ...

    @abstractmethod
    def as_retriever(self, search_kwargs: Optional[dict] = None) -> VectorStoreRetriever:
        """Return a LangChain VectorStoreRetriever for chain composition."""
        ...

    @abstractmethod
    async def create_index(self) -> None:
        """Create or apply a backend-specific ANN index for fast retrieval."""
        ...
