from __future__ import annotations
from typing import List, Optional

from langchain_postgres.v2.engine import PGEngine
from langchain_postgres.v2.async_vectorstore import AsyncPGVectorStore
from langchain_postgres.v2.indexes import HNSWIndex, DistanceStrategy
from langchain_classic.docstore.document import Document
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

    TABLE_NAME = "langchain_pg_embedding"
    METADATA_JSON_COL = "langchain_metadata"
    METADATA_COLUMNS = ["category"]

    def __init__(self, connection_string: str, embeddings_port: EmbeddingsPort) -> None:
        self._engine = PGEngine.from_connection_string(connection_string)
        self._embeddings = embeddings_port.get_embeddings()
        self._store: AsyncPGVectorStore | None = None

    async def _get_store(self) -> AsyncPGVectorStore:
        """Lazy-init the underlying AsyncPGVectorStore (async ctor)."""
        if self._store is None:
            self._store = await AsyncPGVectorStore.create(
                engine=self._engine,
                embedding_service=self._embeddings,
                table_name=self.TABLE_NAME,
                metadata_json_column=self.METADATA_JSON_COL,
                metadata_columns=self.METADATA_COLUMNS,
            )
        return self._store

    async def add_documents(self, documents: List[Document]) -> None:
        store = await self._get_store()
        await store.aadd_documents(documents)

    async def similarity_search(
        self,
        query: str,
        k: int,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        store = await self._get_store()
        return await store.asimilarity_search(query, k=k, filter=filter)

    def as_retriever(self, search_kwargs: Optional[dict] = None) -> VectorStoreRetriever:
        """Synchronous retriever — requires the store to have been initialised."""
        if self._store is None:
            raise RuntimeError(
                "PGVectorStoreAdapter: store not initialised. "
                "Call `await add_documents(...)` or `await similarity_search(...)` first, "
                "or await `_get_store()` explicitly before calling as_retriever()."
            )
        return self._store.as_retriever(search_kwargs=search_kwargs or {})

    async def create_index(self) -> None:
        store = await self._get_store()
        index = HNSWIndex(
            name="hnsw_idx",
            distance_strategy=DistanceStrategy.COSINE_DISTANCE,
            m=16,
            ef_construction=64,
        )
        await store.aapply_vector_index(index, concurrently=True)
        print("PGVECTOR: HNSW index created successfully.")
