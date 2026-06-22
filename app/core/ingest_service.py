"""app/core/ingest_service.py

Ingestion business logic — loads pre-chunked law documents from Parquet
files produced by the law-crawler gold layer, embeds them, and stores
in the vector store.

Chunking and metadata enrichment are handled by the law-crawler pipeline.
This module only handles loading, embedding, and storage.
"""
from __future__ import annotations

import asyncio
import logging

from langchain_core.documents import Document

from app.config import config
from app.ports.document_loader import DocumentLoaderPort
from app.ports.retriever import RetrieverPort
from app.ports.vector_store import VectorStorePort

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS_WARN = 4000


def _load_docs(
    loader: DocumentLoaderPort,
    data_dir: str = config.data_dir,
    tenant_id: str | None = None,
) -> list[Document]:
    """Load pre-chunked documents via the injected loader.

    Args:
        loader: DocumentLoaderPort implementation.
        data_dir: Path to directory containing gold chunk Parquet files.
        tenant_id: Optional tenant identifier for data isolation.

    Returns:
        List of documents with metadata.
    """
    docs = loader.load(data_dir)

    if tenant_id:
        for doc in docs:
            doc.metadata["tenant_id"] = tenant_id

    return docs


async def run_ingest(
    vector_store: VectorStorePort,
    retriever: RetrieverPort,
    loader: DocumentLoaderPort,
    data_dir: str = config.data_dir,
    tenant_id: str | None = None,
) -> dict:
    """Full ingestion pipeline.

    Loads pre-chunked documents from Parquet files, embeds them,
    and stores in the vector store.

    Args:
        vector_store: Any VectorStorePort implementation injected by the caller.
        retriever: Any RetrieverPort implementation injected by the caller.
        loader: Any DocumentLoaderPort implementation injected by the caller.
        data_dir: Path to directory containing gold chunk Parquet files.
        tenant_id: Optional tenant identifier for data isolation.

    Returns:
        dict with 'documents' and 'chunks' counts.
    """
    loop = asyncio.get_running_loop()
    docs = await loop.run_in_executor(
        None, _load_docs, loader, data_dir, tenant_id,
    )

    if not docs:
        logger.error("INGEST ERROR: no documents loaded from %s", data_dir)
        return {"documents": 0, "chunks": 0}

    oversized = sum(1 for d in docs if len(d.page_content) > MAX_CHUNK_CHARS_WARN)
    if oversized:
        logger.warning(
            "INGEST WARNING: %d chunks exceed %d chars — may exceed embedding model context window",
            oversized, MAX_CHUNK_CHARS_WARN,
        )

    try:
        await vector_store.add_documents(docs)
    except Exception:
        logger.exception("INGEST ERROR: embedding/storage failed")
        raise

    logger.info("INGEST: %d documents stored.", len(docs))

    await vector_store.create_index()
    retriever.build_index(docs)

    return {"documents": len(docs), "chunks": len(docs)}
