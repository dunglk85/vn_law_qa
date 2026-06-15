from __future__ import annotations
from typing import Optional

from langchain_cohere import CohereRerank
from langchain_core.documents.compressor import BaseDocumentCompressor

from app.ports.reranker import RerankerPort


class CohereRerankerAdapter(RerankerPort):
    """Concrete adapter for Cohere Rerank models."""

    def __init__(
        self,
        model: str = "rerank-multilingual-v3.0",
        top_n: int = 3,
    ) -> None:
        self._model = model
        self._top_n = top_n

    def get_compressor(self) -> Optional[BaseDocumentCompressor]:
        return CohereRerank(model=self._model, top_n=self._top_n)
