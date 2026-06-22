"""app/core/ingest_service.py

Ingestion business logic — loads, enriches, chunks, and stores documents.
This module knows NOTHING about PGVector, OpenAI, or any concrete provider.
It depends only on VectorStorePort, ChunkingPort, RetrieverPort, and
MetadataEnrichmentPort (the abstract interfaces).
"""
from __future__ import annotations

import glob
import logging
import os
import traceback

logger = logging.getLogger(__name__)

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document

# Patch unstructured to skip NLTK download check (we pre-bundle the data)
try:
    import unstructured.nlp.tokenize
    unstructured.nlp.tokenize.download_nltk_packages = lambda: None
except ImportError:
    pass

from app.config import config
from app.ports.chunking import ChunkingPort
from app.ports.metadata_enrichment import MetadataEnrichmentPort
from app.ports.retriever import RetrieverPort
from app.ports.vector_store import VectorStorePort

# --------------------------------------------------------------------------- #
# File loading                                                                 #
# --------------------------------------------------------------------------- #

async def _load_docs(
    base: str = config.data_dir,
    enricher: MetadataEnrichmentPort | None = None,
    tenant_id: str | None = None,
) -> list[Document]:
    """Recursively load all supported files under *base* into Documents.

    Args:
        base: Root directory to scan for documents.
        enricher: Optional metadata enrichment strategy to apply after loading.
        tenant_id: Optional tenant identifier for data isolation.

    Returns:
        List of documents with metadata (category, source, tenant_id, and any enrichment).
    """
    docs: list[Document] = []

    for path in glob.glob(os.path.join(base, "**", "*"), recursive=True):
        if os.path.isdir(path) or os.path.basename(path).startswith("."):
            continue

        ext = os.path.splitext(path)[1].lower()
        relative = os.path.relpath(path, base)
        category = relative.split(os.sep)[0] if os.sep in relative else "general"

        try:
            loaded: list[Document] = []
            match ext:
                case ".md":
                    loaded = UnstructuredMarkdownLoader(path).load()
                case ".pdf":
                    loaded = PyMuPDFLoader(path).load()
                case ".docx":
                    loaded = Docx2txtLoader(path).load()
                case ".txt":
                    loaded = TextLoader(path, encoding="utf-8").load()
                case _:
                    continue  # unsupported extension — skip silently

            for d in loaded:
                d.metadata["category"] = category
                d.metadata["source"] = path
                if tenant_id:
                    d.metadata["tenant_id"] = tenant_id
                docs.append(d)

        except Exception:
            logger.error("INGEST ERROR: failed to load %s", path)
            traceback.print_exc()

    if enricher is not None:
        docs = await enricher.enrich(docs)

    return docs


# --------------------------------------------------------------------------- #
# Public service function                                                      #
# --------------------------------------------------------------------------- #

async def run_ingest(
    vector_store: VectorStorePort,
    chunker: ChunkingPort,
    retriever: RetrieverPort,
    enricher: MetadataEnrichmentPort | None = None,
    tenant_id: str | None = None,
) -> dict:
    """Full ingestion pipeline.

    Args:
        vector_store: Any VectorStorePort implementation injected by the caller.
        chunker: Any ChunkingPort implementation injected by the caller.
        retriever: Any RetrieverPort implementation injected by the caller.
        enricher: Optional MetadataEnrichmentPort for metadata enrichment.
        tenant_id: Optional tenant identifier for data isolation.

    Returns:
        dict with 'documents' and 'chunks' counts.
    """
    docs = await _load_docs(enricher=enricher, tenant_id=tenant_id)

    if not docs:
        logger.error("INGEST ERROR: no documents loaded.")
        return {"documents": 0, "chunks": 0}

    try:
        chunks = chunker.chunk(docs)
    except Exception:
        logger.error("INGEST ERROR: chunking failed")
        traceback.print_exc()
        raise

    if not chunks:
        logger.error("INGEST ERROR: chunking produced no chunks.")
        return {"documents": len(docs), "chunks": 0}

    await vector_store.add_documents(chunks)
    logger.info("INGEST: %d docs → %d chunks stored.", len(docs), len(chunks))

    await vector_store.create_index()

    retriever.build_index(chunks)

    return {"documents": len(docs), "chunks": len(chunks)}
