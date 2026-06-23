"""Shared schema models for law-crawler gold layer output.

These models define the canonical contract between the crawler pipeline
and the app's Parquet loader adapter. Both sides share these types so
that column renames or type changes are caught at build time, not silently
at runtime.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class LawDocumentChunk(BaseModel):
    """A single chunk of a Pháp Điển law document."""

    chunk_id: str = Field(description="Unique chunk identifier")
    article_id: str = Field(description="Article (điều) identifier")
    title: str = Field(default="", description="Article title")
    chude: str = Field(default="", description="Subject (chủ đề)")
    demuc: str = Field(default="", description="Section (đề mục)")
    chuong: str = Field(default="", description="Chapter (chương)")
    chunk_index: int = Field(default=0, description="Index within the article")
    total_chunks: int = Field(default=1, description="Total chunks for the article")
    text: str = Field(description="Chunk text content")


class VBQPPLChunk(BaseModel):
    """A single chunk of a VBQPPL legal document."""

    chunk_id: str = Field(description="Unique chunk identifier")
    source_id: str = Field(description="Source document ID")
    source_type: str = Field(default="vbqppl", description="Source type")
    parent_id: str | None = Field(default=None, description="Parent chapter ID")
    chunk_index: int = Field(default=0, description="Index within the document")
    total_chunks: int = Field(default=1, description="Total chunks for the document")
    text: str = Field(description="Chunk text content")
