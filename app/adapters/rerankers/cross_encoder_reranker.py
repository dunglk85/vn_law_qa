from __future__ import annotations

from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.documents.compressor import BaseDocumentCompressor

from app.ports.reranker import RerankerPort


class CrossEncoderRerankerAdapter(RerankerPort):
    """Reranker using a cross-encoder model (e.g., ms-marco-MiniLM).

    Cross-encoders provide more accurate relevance scores than bi-encoders
    by jointly encoding query-document pairs, at the cost of higher latency.
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 3,
    ) -> None:
        self._model = model
        self._top_n = top_n
        self._compressor: BaseDocumentCompressor | None = None

    def get_compressor(self) -> BaseDocumentCompressor | None:
        if self._compressor is None:
            cross_encoder = HuggingFaceCrossEncoder(model_name=self._model)
            self._compressor = CrossEncoderReranker(model=cross_encoder, top_n=self._top_n)
        return self._compressor
