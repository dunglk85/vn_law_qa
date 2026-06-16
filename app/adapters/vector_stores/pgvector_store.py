from __future__ import annotations
from typing import List, Optional

from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from app.ports.vector_store import VectorStorePort
from app.ports.embeddings import EmbeddingsPort


class PGVectorStoreAdapter(VectorStorePort):
    """Concrete adapter for pgvector via langchain-postgres.

    To swap to another vector store:
      1. Create a new adapter implementing VectorStorePort
      2. Set VECTOR_STORE_TYPE=<new_type> in .env
      3. Register it in app/factory.py
    """

    COLLECTION_NAME = "langchain"

    def __init__(self, connection_string: str, embeddings_port: EmbeddingsPort) -> None:
        self._connection_string = connection_string
        self._embeddings = embeddings_port.get_embeddings()
        self._store: PGVector | None = None

    def _get_store(self) -> PGVector:
        """Lazy-init the underlying PGVector store."""
        if self._store is None:
            self._store = PGVector(
                embeddings=self._embeddings,
                connection=self._connection_string,
                collection_name=self.COLLECTION_NAME,
            )
        return self._store

    async def add_documents(self, documents: List[Document]) -> None:
        store = self._get_store()
        await store.aadd_documents(documents)

    async def similarity_search(
        self,
        query: str,
        k: int,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        store = self._get_store()
        return await store.asimilarity_search(query, k=k, filter=filter)

    def as_retriever(self, search_kwargs: Optional[dict] = None) -> VectorStoreRetriever:
        """Synchronous retriever — requires the store to have been initialised."""
        if self._store is None:
            raise RuntimeError(
                "PGVectorStoreAdapter: store not initialised. "
                "Call `await add_documents(...)` or `await similarity_search(...)` first, "
                "or call `_get_store()` explicitly before calling as_retriever()."
            )
        return self._store.as_retriever(search_kwargs=search_kwargs or {})

    async def create_index(self) -> None:
        print("PGVECTOR: Index creation skipped (using default index).")
