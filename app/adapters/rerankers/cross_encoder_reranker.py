from __future__ import annotations
from typing import List, Optional, Sequence

from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

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

    def get_compressor(self) -> Optional[BaseDocumentCompressor]:
        cross_encoder = HuggingFaceCrossEncoder(model_name=self._model)
        return CrossEncoderReranker(model=cross_encoder, top_n=self._top_n)
