from __future__ import annotations
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.ports.chunking import ChunkingPort


class RecursiveChunkerAdapter(ChunkingPort):
    """Splits documents using recursive character-based splitting.

    Uses LangChain's RecursiveCharacterTextSplitter with configurable
    chunk_size and chunk_overlap.
    """

    def __init__(self, chunk_size: int = 900, chunk_overlap: int = 120) -> None:
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, documents: List[Document]) -> List[Document]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        return splitter.split_documents(documents)
