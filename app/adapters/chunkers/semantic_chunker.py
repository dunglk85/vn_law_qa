from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_experimental.text_splitter import SemanticChunker

from app.ports.chunking import ChunkingPort


class SemanticChunkerAdapter(ChunkingPort):
    """Splits documents using semantic similarity between sentences.

    Uses LangChain's SemanticChunker which leverages embeddings to
    determine natural break points in the text.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        breakpoint_threshold_type: str = "percentile",
        breakpoint_threshold_amount: float = 95.0,
    ) -> None:
        self._embeddings = embeddings
        self._breakpoint_threshold_type = breakpoint_threshold_type
        self._breakpoint_threshold_amount = breakpoint_threshold_amount

    def chunk(self, documents: list[Document]) -> list[Document]:
        splitter = SemanticChunker(
            embeddings=self._embeddings,
            breakpoint_threshold_type=self._breakpoint_threshold_type,
            breakpoint_threshold_amount=self._breakpoint_threshold_amount,
        )
        return splitter.split_documents(documents)
