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
        api_key: str | None = None,
    ) -> None:
        self._model = model
        self._top_n = top_n
        self._api_key = api_key

    def get_compressor(self) -> Optional[BaseDocumentCompressor]:
        return CohereRerank(
            model=self._model,
            top_n=self._top_n,
            cohere_api_key=self._api_key,
        )
