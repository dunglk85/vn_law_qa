from __future__ import annotations
from typing import List, Optional
import asyncio

from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from app.ports.vector_store import VectorStorePort
from app.ports.embeddings import EmbeddingsPort


class _SyncRetriever(BaseRetriever):
    """Wrapper that makes sync similarity_search work as async retriever."""
    _store: PGVector
    _search_kwargs: dict

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self._store.similarity_search(query, **self._search_kwargs)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return await asyncio.to_thread(self._get_relevant_documents, query, run_manager=run_manager)


class PGVectorStoreAdapter(VectorStorePort):
    """Concrete adapter for pgvector via langchain-postgres.

    To swap to another vector store:
      1. Create a new adapter implementing VectorStorePort
      2. Set VECTOR_STORE_TYPE=<new_type> in .env
      3. Register it in app/factory.py
    """

    COLLECTION_NAME = "langchain"

    def __init__(self, connection_string: str, embeddings_port: EmbeddingsPort) -> None:
        # Use psycopg (sync) driver - asyncpg has issues with multi-command statements
        self._connection_string = connection_string.replace("postgresql+asyncpg://", "postgresql+psycopg://")
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
        store.add_documents(documents)

    async def similarity_search(
        self,
        query: str,
        k: int,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        store = self._get_store()
        return store.similarity_search(query, k=k, filter=filter)

    def as_retriever(self, search_kwargs: Optional[dict] = None) -> VectorStoreRetriever:
        """Return a retriever that uses sync similarity_search via thread pool."""
        if self._store is None:
            raise RuntimeError(
                "PGVectorStoreAdapter: store not initialised. "
                "Call `await add_documents(...)` or `await similarity_search(...)` first."
            )
        kwargs = search_kwargs or {}
        wrapper = _SyncRetriever()
        wrapper._store = self._store
        wrapper._search_kwargs = kwargs
        return wrapper  # type: ignore

    async def create_index(self) -> None:
        print("PGVECTOR: Index creation skipped (using default index).")
