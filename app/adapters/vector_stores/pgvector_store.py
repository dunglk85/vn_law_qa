from __future__ import annotations

import asyncio
import logging

import psycopg
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_postgres import PGVector

from app.ports.embeddings import EmbeddingsPort
from app.ports.vector_store import VectorStorePort

logger = logging.getLogger(__name__)


class _SyncRetriever(BaseRetriever):
    """Wrapper that makes sync similarity_search work as async retriever."""
    _store: PGVector
    _search_kwargs: dict

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self._store.similarity_search(query, **self._search_kwargs)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return await asyncio.to_thread(self._get_relevant_documents, query, run_manager=run_manager)


def _create_hnsw_index_sync(
    connection_string: str, collection_name: str, hnsw_m: int, hnsw_ef_construction: int
) -> None:
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
                logger.info("PGVECTOR: HNSW index '%s' already exists.", index_name)
                return

            cur.execute(
                f'CREATE INDEX "{index_name}" '
                "ON langchain_pg_embedding USING hnsw (embedding vector_cosine_ops) "
                "WITH (m = %s, ef_construction = %s)",
                (hnsw_m, hnsw_ef_construction),
            )
            conn.commit()
            logger.info("PGVECTOR: HNSW index '%s' created successfully.", index_name)


def _create_ivfflat_index_sync(
    connection_string: str, collection_name: str, ivfflat_lists: int, ivfflat_probes: int
) -> None:
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
                logger.info("PGVECTOR: IVFFlat index '%s' already exists.", index_name)
                return

            cur.execute(
                f'CREATE INDEX "{index_name}" '
                "ON langchain_pg_embedding USING ivfflat (embedding vector_cosine_ops) "
                "WITH (lists = %s)",
                (ivfflat_lists,),
            )
            conn.commit()

            cur.execute("SET ivfflat.probes = %s", (ivfflat_probes,))
            conn.commit()
            logger.info(
                "PGVECTOR: IVFFlat index '%s' created successfully (lists=%d, probes=%d).",
                index_name, ivfflat_lists, ivfflat_probes
            )


class PGVectorStoreAdapter(VectorStorePort):
    """Concrete adapter for pgvector via langchain-postgres.

    To swap to another vector store:
      1. Create a new adapter implementing VectorStorePort
      2. Set VECTOR_STORE_TYPE=<new_type> in .env
      3. Register it in app/factory.py
    """

    COLLECTION_NAME = "langchain"

    def __init__(
        self,
        connection_string: str,
        embeddings_port: EmbeddingsPort,
        index_type: str = "hnsw",
        hnsw_m: int = 16,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 50,
        ivfflat_lists: int = 100,
        ivfflat_probes: int = 10,
    ) -> None:
        self._connection_string = connection_string.replace("postgresql+asyncpg://", "postgresql+psycopg://")
        self._embeddings = embeddings_port.get_embeddings()
        self._store: PGVector | None = None
        self._index_type = index_type
        self._hnsw_m = hnsw_m
        self._hnsw_ef_construction = hnsw_ef_construction
        self._hnsw_ef_search = hnsw_ef_search
        self._ivfflat_lists = ivfflat_lists
        self._ivfflat_probes = ivfflat_probes

    def _get_store(self) -> PGVector:
        """Lazy-init the underlying PGVector store."""
        if self._store is None:
            self._store = PGVector(
                embeddings=self._embeddings,
                connection=self._connection_string,
                collection_name=self.COLLECTION_NAME,
            )
        return self._store

    async def add_documents(self, documents: list[Document]) -> None:
        store = self._get_store()
        await asyncio.to_thread(store.add_documents, documents)

    async def similarity_search(
        self,
        query: str,
        k: int,
        filter: dict | None = None,
    ) -> list[Document]:
        store = self._get_store()
        if self._index_type == "hnsw":
            conn_str = self._connection_string.replace("postgresql+psycopg://", "postgresql://")
            try:
                with psycopg.connect(conn_str) as conn:
                    with conn.cursor() as cur:
                        cur.execute("SET hnsw.ef_search = %s", (self._hnsw_ef_search,))
                        conn.commit()
            except Exception as exc:
                logger.warning("PGVECTOR: Failed to set hnsw.ef_search: %s", exc)
        return await asyncio.to_thread(store.similarity_search, query, k, filter)

    async def get_documents_by_ids(self, ids: list[str]) -> list[Document]:
        store = self._get_store()
        results = await asyncio.to_thread(store.mget, ids)
        return [doc for doc in results if doc is not None]

    def as_retriever(self, search_kwargs: dict | None = None) -> BaseRetriever:
        """Return a retriever that uses sync similarity_search via thread pool."""
        self._get_store()
        kwargs = search_kwargs or {}
        wrapper = _SyncRetriever.model_construct(_store=self._store, _search_kwargs=kwargs)
        return wrapper

    async def create_index(self) -> None:
        match self._index_type:
            case "hnsw":
                try:
                    await asyncio.to_thread(
                        _create_hnsw_index_sync, self._connection_string, self.COLLECTION_NAME,
                        self._hnsw_m, self._hnsw_ef_construction,
                    )
                except Exception as exc:
                    logger.error("PGVECTOR: HNSW index creation failed: %s", exc)
            case "ivfflat":
                try:
                    await asyncio.to_thread(
                        _create_ivfflat_index_sync, self._connection_string, self.COLLECTION_NAME,
                        self._ivfflat_lists, self._ivfflat_probes,
                    )
                except Exception as exc:
                    logger.error("PGVECTOR: IVFFlat index creation failed: %s", exc)
            case _:
                raise ValueError(
                    f"Unknown INDEX_TYPE='{self._index_type}'. "
                    "Supported: hnsw, ivfflat"
                )
