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
        if not data_path.exists() or not data_path.is_dir():
            logger.warning("Data directory does not exist or is not a directory: %s", data_dir)
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
            text_val = row.get("text")
            text_str = str(text_val) if pd.notna(text_val) and text_val is not None else ""
            doc = Document(
                page_content=text_str,
                metadata={
                    "source": "law-crawler",
                    "chunk_id": str(row.get("chunk_id")) if pd.notna(row.get("chunk_id")) else "",
                    "article_id": str(row.get("article_id")) if pd.notna(row.get("article_id")) else "",
                    "title": str(row.get("title")) if pd.notna(row.get("title")) else "",
                    "chude": str(row.get("chude")) if pd.notna(row.get("chude")) else "",
                    "demuc": str(row.get("demuc")) if pd.notna(row.get("demuc")) else "",
                    "chuong": str(row.get("chuong")) if pd.notna(row.get("chuong")) else "",
                    "chunk_index": int(row.get("chunk_index")) if pd.notna(row.get("chunk_index")) else 0,
                    "total_chunks": int(row.get("total_chunks")) if pd.notna(row.get("total_chunks")) else 1,
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
            text_val = row.get("text")
            text_str = str(text_val) if pd.notna(text_val) and text_val is not None else ""
            parent_val = row.get("parent_id")
            parent_str = str(parent_val) if pd.notna(parent_val) and parent_val is not None else ""
            doc = Document(
                page_content=text_str,
                metadata={
                    "source": "law-crawler",
                    "chunk_id": str(row.get("chunk_id")) if pd.notna(row.get("chunk_id")) else "",
                    "source_id": str(row.get("source_id")) if pd.notna(row.get("source_id")) else "",
                    "source_type": str(row.get("source_type")) if pd.notna(row.get("source_type")) else "vbqppl",
                    "parent_id": parent_str,
                    "chunk_index": int(row.get("chunk_index")) if pd.notna(row.get("chunk_index")) else 0,
                    "total_chunks": int(row.get("total_chunks")) if pd.notna(row.get("total_chunks")) else 1,
                    "category": "vbqppl",
                },
            )
            docs.append(doc)

        return docs
