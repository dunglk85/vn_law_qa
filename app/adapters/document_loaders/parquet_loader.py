"""Parquet document loader adapter.

Reads pre-chunked law documents from Parquet files produced by the
law-crawler gold layer and converts them to LangChain Document objects.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from app.ports.document_loader import DocumentLoaderPort

logger = logging.getLogger(__name__)


class ParquetLoaderAdapter(DocumentLoaderPort):
    """Loads pre-chunked documents from Parquet files.
    
    Expects gold layer output from law-crawler with the following schemas:
    
    law_document_chunks.parquet:
        - chunk_id: str
        - article_id: str
        - title: str
        - chude: str (subject)
        - demuc: str (section)
        - chuong: str (chapter)
        - chunk_index: int
        - total_chunks: int
        - text: str
    
    vbqppl_chunks.parquet:
        - chunk_id: str
        - source_id: str
        - source_type: str
        - parent_id: str (nullable)
        - chunk_index: int
        - total_chunks: int
        - text: str
    """

    def load(self, data_dir: str) -> list[Document]:
        """Load all Parquet files from the given directory.
        
        Args:
            data_dir: Path to directory containing gold chunk Parquet files.
        
        Returns:
            List of LangChain Document objects with metadata.
        """
        data_path = Path(data_dir)
        if not data_path.exists():
            logger.warning("Data directory does not exist: %s", data_dir)
            return []

        docs: list[Document] = []

        # Load law document chunks
        law_chunks_path = data_path / "law_document_chunks.parquet"
        if law_chunks_path.exists():
            docs.extend(self._load_law_chunks(law_chunks_path))
        else:
            logger.info("No law_document_chunks.parquet found in %s", data_dir)

        # Load VBQPPL chunks
        vbqppl_chunks_path = data_path / "vbqppl_chunks.parquet"
        if vbqppl_chunks_path.exists():
            docs.extend(self._load_vbqppl_chunks(vbqppl_chunks_path))
        else:
            logger.info("No vbqppl_chunks.parquet found in %s", data_dir)

        logger.info("Loaded %d documents from Parquet files", len(docs))
        return docs

    def _load_law_chunks(self, path: Path) -> list[Document]:
        """Load law document chunks from Parquet."""
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.error("Failed to read Parquet file %s: %s", path, exc)
            return []

        logger.info("Loading %d law chunks from %s", len(df), path)

        docs: list[Document] = []
        for _, row in df.iterrows():
            doc = Document(
                page_content=str(row.get("text", "") or ""),
                metadata={
                    "source": "law-crawler",
                    "chunk_id": str(row.get("chunk_id", "") or ""),
                    "article_id": str(row.get("article_id", "") or ""),
                    "title": str(row.get("title", "") or ""),
                    "chude": str(row.get("chude", "") or ""),
                    "demuc": str(row.get("demuc", "") or ""),
                    "chuong": str(row.get("chuong", "") or ""),
                    "chunk_index": int(row.get("chunk_index", 0) or 0),
                    "total_chunks": int(row.get("total_chunks", 1) or 1),
                    "category": "law",
                },
            )
            docs.append(doc)

        return docs

    def _load_vbqppl_chunks(self, path: Path) -> list[Document]:
        """Load VBQPPL document chunks from Parquet."""
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            logger.error("Failed to read Parquet file %s: %s", path, exc)
            return []

        logger.info("Loading %d VBQPPL chunks from %s", len(df), path)

        docs: list[Document] = []
        for _, row in df.iterrows():
            parent_val = row.get("parent_id")
            parent_str = str(parent_val) if pd.notna(parent_val) and parent_val is not None else ""
            doc = Document(
                page_content=str(row.get("text", "") or ""),
                metadata={
                    "source": "law-crawler",
                    "chunk_id": str(row.get("chunk_id", "") or ""),
                    "source_id": str(row.get("source_id", "") or ""),
                    "source_type": str(row.get("source_type", "vbqppl") or "vbqppl"),
                    "parent_id": parent_str,
                    "chunk_index": int(row.get("chunk_index", 0) or 0),
                    "total_chunks": int(row.get("total_chunks", 1) or 1),
                    "category": "vbqppl",
                },
            )
            docs.append(doc)

        return docs
