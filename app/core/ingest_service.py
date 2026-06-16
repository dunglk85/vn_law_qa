"""app/core/ingest_service.py

Ingestion business logic — loads, chunks, and stores documents.
This module knows NOTHING about PGVector, OpenAI, or any concrete provider.
It depends only on VectorStorePort and ChunkingPort (the abstract interfaces).
"""
from __future__ import annotations
import glob
import os
import traceback
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    PyMuPDFLoader,
    Docx2txtLoader,
    TextLoader,
)

# Patch unstructured to skip NLTK download check (we pre-bundle the data)
try:
    import unstructured.nlp.tokenize
    unstructured.nlp.tokenize.download_nltk_packages = lambda: None
except ImportError:
    pass

from app.config import config
from app.ports.vector_store import VectorStorePort
from app.ports.chunking import ChunkingPort


# --------------------------------------------------------------------------- #
# File loading                                                                 #
# --------------------------------------------------------------------------- #

def _load_docs(base: str = config.data_dir) -> List[Document]:
    """Recursively load all supported files under *base* into Documents."""
    docs: List[Document] = []

    for path in glob.glob(os.path.join(base, "**", "*"), recursive=True):
        if os.path.isdir(path) or os.path.basename(path).startswith("."):
            continue

        ext = os.path.splitext(path)[1].lower()
        relative = os.path.relpath(path, base)
        category = relative.split(os.sep)[0] if os.sep in relative else "general"

        try:
            loaded: List[Document] = []
            match ext:
                case ".md":
                    loaded = UnstructuredMarkdownLoader(path).load()
                case ".pdf":
                    loaded = PyMuPDFLoader(path).load()
                case ".docx":
                    loaded = Docx2txtLoader(path).load()
                case ".txt":
                    loaded = TextLoader(path).load()
                case _:
                    continue  # unsupported extension — skip silently

            for d in loaded:
                d.metadata["category"] = category
                docs.append(d)

        except Exception:
            print(f"INGEST ERROR: failed to load {path}")
            traceback.print_exc()

    return docs


# --------------------------------------------------------------------------- #
# Public service function                                                      #
# --------------------------------------------------------------------------- #

async def run_ingest(vector_store: VectorStorePort, chunker: ChunkingPort) -> dict:
    """Full ingestion pipeline.

    Args:
        vector_store: Any VectorStorePort implementation injected by the caller.
        chunker: Any ChunkingPort implementation injected by the caller.

    Returns:
        dict with 'documents' and 'chunks' counts.
    """
    docs = _load_docs()

    try:
        chunks = chunker.chunk(docs)
    except Exception:
        print("INGEST ERROR: chunking failed")
        traceback.print_exc()
        raise

    await vector_store.add_documents(chunks)
    print(f"INGEST: {len(docs)} docs → {len(chunks)} chunks stored.")

    await vector_store.create_index()

    return {"documents": len(docs), "chunks": len(chunks)}
