from __future__ import annotations

from langchain_core.documents.compressor import BaseDocumentCompressor

from app.ports.reranker import RerankerPort


class NoneRerankerAdapter(RerankerPort):
    """Pass-through adapter — disables reranking entirely.

    Use by setting RERANKER_TYPE=none in .env.
    Useful for development, cost saving, or when reranking is not needed.
    """

    def get_compressor(self) -> BaseDocumentCompressor | None:
        return None
