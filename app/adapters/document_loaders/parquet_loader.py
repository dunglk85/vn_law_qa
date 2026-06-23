"""Parquet document loader adapter.

Reads pre-chunked law documents from Parquet files produced by the
law-crawler gold layer and converts them to LangChain Document objects.

Schema contract:
    The canonical schema definitions live in ``law-crawler/src/schema.py``
    (``LawDocumentChunk`` and ``VBQPPLChunk``). When the crawler and app
    are installed in the same Python environment, this module imports those
    models directly. Otherwise it falls back to inline dataclass definitions
    that must be kept in sync.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from app.ports.document_loader import DocumentLoaderPort

logger = logging.getLogger(__name__)

try:
    from src.schema import LawDocumentChunk, VBQPPLChunk
    HAS_SHARED_SCHEMA = True
except ImportError:
    HAS_SHARED_SCHEMA = False

    class LawDocumentChunk:
        __slots__ = ("chunk_id", "article_id", "title", "chude", "demuc",
                     "chuong", "chunk_index", "total_chunks", "text", "schema_version")
        def __init__(self, chunk_id="", article_id="", title="", chude="",
                     demuc="", chuong="", chunk_index=0, total_chunks=1, text="", schema_version="1.0.0"):
            self.chunk_id = chunk_id
            self.article_id = article_id
            self.title = title
            self.chude = chude
            self.demuc = demuc
            self.chuong = chuong
            self.chunk_index = chunk_index
            self.total_chunks = total_chunks
            self.text = text
            self.schema_version = schema_version

    class VBQPPLChunk:
        __slots__ = ("chunk_id", "source_id", "source_type", "parent_id",
                     "chunk_index", "total_chunks", "text", "schema_version")
        def __init__(self, chunk_id="", source_id="", source_type="vbqppl",
                     parent_id=None, chunk_index=0, total_chunks=1, text="", schema_version="1.0.0"):
            self.chunk_id = chunk_id
            self.source_id = source_id
            self.source_type = source_type
            self.parent_id = parent_id
            self.chunk_index = chunk_index
            self.total_chunks = total_chunks
            self.text = text
            self.schema_version = schema_version


_SLOT_KEYS_LAW = ("chunk_id", "article_id", "title", "chude",
                  "demuc", "chuong", "chunk_index", "total_chunks", "text", "schema_version")
_SLOT_KEYS_VB = ("chunk_id", "source_id", "source_type", "parent_id",
                 "chunk_index", "total_chunks", "text", "schema_version")


def _row_to_law_chunk(row: dict) -> LawDocumentChunk:
    kwargs = {k: row.get(k) for k in _SLOT_KEYS_LAW}
    kwargs["chunk_index"] = _safe_int(kwargs.get("chunk_index"), 0)
    kwargs["total_chunks"] = _safe_int(kwargs.get("total_chunks"), 1)
    for k in ("chunk_id", "article_id", "title", "chude", "demuc", "chuong"):
        kwargs[k] = _safe_str(kwargs.get(k))
    kwargs["text"] = _safe_str(kwargs.get("text"))
    kwargs["schema_version"] = _safe_str(kwargs.get("schema_version")) or "1.0.0"
    return LawDocumentChunk(**kwargs)


def _row_to_vb_chunk(row: dict) -> VBQPPLChunk:
    kwargs = {k: row.get(k) for k in _SLOT_KEYS_VB}
    kwargs["chunk_index"] = _safe_int(kwargs.get("chunk_index"), 0)
    kwargs["total_chunks"] = _safe_int(kwargs.get("total_chunks"), 1)
    for k in ("chunk_id", "source_id", "source_type"):
        kwargs[k] = _safe_str(kwargs.get(k))
    kwargs["text"] = _safe_str(kwargs.get("text"))
    parent = kwargs.get("parent_id")
    kwargs["parent_id"] = str(parent) if pd.notna(parent) and parent is not None else None
    kwargs["schema_version"] = _safe_str(kwargs.get("schema_version")) or "1.0.0"
    return VBQPPLChunk(**kwargs)


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def _safe_str(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value)
    return "" if s == "<NA>" else s


class ParquetLoaderAdapter(DocumentLoaderPort):
    """Loads pre-chunked documents from Parquet files.

    Expects gold layer output from law-crawler with the schemas
    defined in ``law-crawler/src/schema.py``:
        - law_document_chunks.parquet  → LawDocumentChunk
        - vbqppl_chunks.parquet       → VBQPPLChunk
    """

    def load(self, data_dir: str) -> list[Document]:
        data_path = Path(data_dir)
        if not data_path.exists() or not data_path.is_dir():
            logger.warning("Data directory does not exist or is not a directory: %s", data_dir)
            return []

        docs: list[Document] = []
        files_found = 0

        law_chunks_path = data_path / "law_document_chunks.parquet"
        if law_chunks_path.exists():
            files_found += 1
            law_docs = self._load_law_chunks(law_chunks_path)
            if not law_docs:
                logger.warning(
                    "law_document_chunks.parquet exists but contains no valid documents"
                )
            docs.extend(law_docs)
        else:
            logger.info("No law_document_chunks.parquet found in %s", data_dir)

        vbqppl_chunks_path = data_path / "vbqppl_chunks.parquet"
        if vbqppl_chunks_path.exists():
            files_found += 1
            vb_docs = self._load_vbqppl_chunks(vbqppl_chunks_path)
            if not vb_docs:
                logger.warning(
                    "vbqppl_chunks.parquet exists but contains no valid documents"
                )
            docs.extend(vb_docs)
        else:
            logger.info("No vbqppl_chunks.parquet found in %s", data_dir)

        if files_found == 0:
            logger.warning("No Parquet files found in %s", data_dir)
        else:
            logger.info("Loaded %d documents from Parquet files", len(docs))
        return docs

    def _load_law_chunks(self, path: Path) -> list[Document]:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.error("Failed to read Parquet file %s: %s", path, exc)
            return []

        logger.info("Loading %d law chunks from %s", len(df), path)

        docs: list[Document] = []
        for _, row in df.iterrows():
            chunk = _row_to_law_chunk(row.to_dict())
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "source": "law-crawler",
                    "chunk_id": chunk.chunk_id,
                    "article_id": chunk.article_id,
                    "title": chunk.title,
                    "chude": chunk.chude,
                    "demuc": chunk.demuc,
                    "chuong": chunk.chuong,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "category": "law",
                },
            )
            docs.append(doc)

        return docs

    def _load_vbqppl_chunks(self, path: Path) -> list[Document]:
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.error("Failed to read Parquet file %s: %s", path, exc)
            return []

        logger.info("Loading %d VBQPPL chunks from %s", len(df), path)

        docs: list[Document] = []
        for _, row in df.iterrows():
            chunk = _row_to_vb_chunk(row.to_dict())
            doc = Document(
                page_content=chunk.text,
                metadata={
                    "source": "law-crawler",
                    "chunk_id": chunk.chunk_id,
                    "source_id": chunk.source_id,
                    "source_type": chunk.source_type,
                    "parent_id": chunk.parent_id or "",
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "category": "vbqppl",
                },
            )
            docs.append(doc)

        return docs
