"""app/core/ingest_service.py

Ingestion business logic — loads, chunks, and stores documents.
This module knows NOTHING about PGVector, OpenAI, or any concrete provider.
It depends only on VectorStorePort (the abstract interface).
"""
from __future__ import annotations
import glob
import os
import traceback
from typing import List

from langchain_classic.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredMarkdownLoader,
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)

from app.config import config
from app.ports.vector_store import VectorStorePort


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
                    loaded = UnstructuredWordDocumentLoader(path).load()
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
# Chunking                                                                     #
# --------------------------------------------------------------------------- #

def _chunk(docs: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )
    try:
        return splitter.split_documents(docs)
    except Exception:
        print("INGEST ERROR: chunking failed")
        traceback.print_exc()
        raise


# --------------------------------------------------------------------------- #
# Public service function                                                      #
# --------------------------------------------------------------------------- #

async def run_ingest(vector_store: VectorStorePort) -> dict:
    """Full ingestion pipeline.

    Args:
        vector_store: Any VectorStorePort implementation injected by the caller.

    Returns:
        dict with 'documents' and 'chunks' counts.
    """
    docs = _load_docs()
    chunks = _chunk(docs)

    await vector_store.add_documents(chunks)
    print(f"INGEST: {len(docs)} docs → {len(chunks)} chunks stored.")

    await vector_store.create_index()

    return {"documents": len(docs), "chunks": len(chunks)}
