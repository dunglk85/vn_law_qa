from __future__ import annotations

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.ports.retriever import RetrieverPort


class _BM25LangChainRetriever(BaseRetriever):
    """Wrapper around BM25Retriever to support async."""
    _bm25: BM25Retriever
    _k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self._bm25.invoke(query)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


class BM25RetrieverAdapter(RetrieverPort):
    """Sparse retrieval using BM25 (Best Match 25) algorithm.

    Uses term frequency-inverse document frequency (TF-IDF) scoring
    for keyword-based retrieval without semantic understanding.
    """

    def __init__(self, k: int = 5) -> None:
        self._k = k
        self._bm25: BM25Retriever | None = None

    def build_index(self, documents: list[Document]) -> None:
        self._bm25 = BM25Retriever.from_documents(documents)

    def get_retriever(self, search_kwargs: dict | None = None) -> BaseRetriever:
        if self._bm25 is None:
            raise RuntimeError("BM25RetrieverAdapter: index not built. Call build_index() first.")
        retriever = _BM25LangChainRetriever(_bm25=self._bm25, _k=self._k)
        return retriever
