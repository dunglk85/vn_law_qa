from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class VectorStorePort(ABC):
    """Abstract interface for any vector store backend.

    Swap PGVector → Chroma/Qdrant/Pinecone by creating a new adapter
    that implements this interface, then setting VECTOR_STORE_TYPE in .env.
    Business logic never touches a concrete store class.
    """

    @abstractmethod
    async def add_documents(self, documents: list[Document]) -> None:
        """Embed and persist a list of documents."""
        ...

    @abstractmethod
    async def similarity_search(
        self,
        query: str,
        k: int,
        filter: dict | None = None,
    ) -> list[Document]:
        """Return the top-k most similar documents for *query*."""
        ...

    @abstractmethod
    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        """Return a list of documents by their unique IDs."""
        ...

    @abstractmethod
    def as_retriever(self, search_kwargs: dict | None = None) -> BaseRetriever:
        """Return a LangChain BaseRetriever for chain composition."""
        ...

    @abstractmethod
    async def create_index(self) -> None:
        """Create or apply a backend-specific ANN index for fast retrieval."""
        ...
