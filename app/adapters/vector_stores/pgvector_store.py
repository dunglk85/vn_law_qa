from __future__ import annotations
from typing import List, Optional
import asyncio

import psycopg
from langchain_postgres import PGVector
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from app.ports.vector_store import VectorStorePort
from app.ports.embeddings import EmbeddingsPort
from app.config import config


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


def _create_hnsw_index_sync(connection_string: str, collection_name: str) -> None:
    conn_str = connection_string.replace("postgresql+psycopg://", "postgresql://")
    index_name = f"hnsw_{collection_name}_embedding_idx"

    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'langchain_pg_embedding' AND indexname = %s",
                (index_name,),
            )
            if cur.fetchone():
                print(f"PGVECTOR: HNSW index '{index_name}' already exists.")
                return

            cur.execute(
                f"CREATE INDEX {index_name} "
                "ON langchain_pg_embedding USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = %s, ef_construction = %s)",
                (config.hnsw_m, config.hnsw_ef_construction),
            )
            conn.commit()
            print(f"PGVECTOR: HNSW index '{index_name}' created successfully.")


def _create_ivfflat_index_sync(connection_string: str, collection_name: str) -> None:
    conn_str = connection_string.replace("postgresql+psycopg://", "postgresql://")
    index_name = f"ivfflat_{collection_name}_embedding_idx"

    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename = 'langchain_pg_embedding' AND indexname = %s",
                (index_name,),
            )
            if cur.fetchone():
                print(f"PGVECTOR: IVFFlat index '{index_name}' already exists.")
                return

            cur.execute(
                f"CREATE INDEX {index_name} "
                "ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops) "
                "WITH (lists = %s)",
                (config.ivfflat_lists,),
            )
            conn.commit()

            cur.execute("SET ivfflat.probes = %s", (config.ivfflat_probes,))
            conn.commit()
            print(f"PGVECTOR: IVFFlat index '{index_name}' created successfully (lists={config.ivfflat_lists}, probes={config.ivfflat_probes}).")


class PGVectorStoreAdapter(VectorStorePort):
    """Concrete adapter for pgvector via langchain-postgres.

    To swap to another vector store:
      1. Create a new adapter implementing VectorStorePort
      2. Set VECTOR_STORE_TYPE=<new_type> in .env
      3. Register it in app/factory.py
    """

    COLLECTION_NAME = "langchain"

    def __init__(self, connection_string: str, embeddings_port: EmbeddingsPort) -> None:
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
        await asyncio.to_thread(store.add_documents, documents)

    async def similarity_search(
        self,
        query: str,
        k: int,
        filter: Optional[dict] = None,
    ) -> List[Document]:
        store = self._get_store()
        return await asyncio.to_thread(store.similarity_search, query, k, filter)

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
        match config.index_type:
            case "hnsw":
                try:
                    await asyncio.to_thread(
                        _create_hnsw_index_sync, self._connection_string, self.COLLECTION_NAME
                    )
                except Exception as exc:
                    print(f"PGVECTOR: HNSW index creation failed: {exc}")
            case "ivfflat":
                try:
                    await asyncio.to_thread(
                        _create_ivfflat_index_sync, self._connection_string, self.COLLECTION_NAME
                    )
                except Exception as exc:
                    print(f"PGVECTOR: IVFFlat index creation failed: {exc}")
            case _:
                raise ValueError(
                    f"Unknown INDEX_TYPE='{config.index_type}'. "
                    "Supported: hnsw, ivfflat"
                )
