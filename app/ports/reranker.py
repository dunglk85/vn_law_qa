from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.documents.compressor import BaseDocumentCompressor


class RerankerPort(ABC):
    """Abstract interface for any reranker / compressor.

    Swap Cohere → FlashRank/Jina by creating a new adapter that implements
    this interface, then setting RERANKER_TYPE in .env.
    Setting RERANKER_TYPE=none uses the NoneRerankerAdapter (pass-through).
    """

    @abstractmethod
    def get_compressor(self) -> Optional[BaseDocumentCompressor]:
        """Return a LangChain BaseDocumentCompressor, or None to skip reranking."""
        ...
