from __future__ import annotations

import hashlib

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.ports.retriever import RetrieverPort
from app.ports.vector_store import VectorStorePort


class _HybridRRFRetriever(BaseRetriever):
    """Combines dense and sparse retrieval using Reciprocal Rank Fusion."""
    _dense_retriever: BaseRetriever
    _sparse_retriever: BM25Retriever
    _k: int
    _rrf_k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        sparse_k = self._k * 2
        original_k = self._sparse_retriever.k
        try:
            self._sparse_retriever.k = sparse_k
            dense_docs = self._dense_retriever.invoke(query)
            sparse_docs = self._sparse_retriever.invoke(query)
        finally:
            self._sparse_retriever.k = original_k

        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs):
            content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
            doc_map[content_hash] = doc
            scores[content_hash] = scores.get(content_hash, 0.0) + 1.0 / (self._rrf_k + rank + 1)

        for rank, doc in enumerate(sparse_docs):
            content_hash = hashlib.sha256(doc.page_content.encode()).hexdigest()
            if content_hash not in doc_map:
                doc_map[content_hash] = doc
            scores[content_hash] = scores.get(content_hash, 0.0) + 1.0 / (self._rrf_k + rank + 1)

        sorted_hashes = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[h] for h in sorted_hashes[:self._k]]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


class HybridRRFRetrieverAdapter(RetrieverPort):
    """Hybrid retrieval combining dense (vector) and sparse (BM25) via RRF.

    Reciprocal Rank Fusion (RRF) scores each document based on its rank
    in both retrieval lists, providing robust combination that handles
    different score scales between dense and sparse methods.
    """

    def __init__(self, vector_store: VectorStorePort, k: int = 5, rrf_k: int = 60) -> None:
        self._vector_store = vector_store
        self._k = k
        self._rrf_k = rrf_k
        self._bm25: BM25Retriever | None = None

    def build_index(self, documents: list[Document]) -> None:
        self._bm25 = BM25Retriever.from_documents(documents)

    def get_retriever(self, search_kwargs: dict | None = None) -> BaseRetriever:
        if self._bm25 is None:
            raise RuntimeError("HybridRRFRetrieverAdapter: index not built. Call build_index() first.")
        kwargs = search_kwargs or {"k": self._k}
        vector_store_retriever = self._vector_store.as_retriever(search_kwargs=kwargs)
        wrapper = _HybridRRFRetriever()
        wrapper._dense_retriever = vector_store_retriever
        wrapper._sparse_retriever = self._bm25
        wrapper._k = self._k
        wrapper._rrf_k = self._rrf_k
        return wrapper
