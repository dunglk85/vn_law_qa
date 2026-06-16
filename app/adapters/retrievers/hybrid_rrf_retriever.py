from __future__ import annotations
from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.retrievers import BM25Retriever

from app.ports.retriever import RetrieverPort


class _HybridRRFRetriever(BaseRetriever):
    """Combines dense and sparse retrieval using Reciprocal Rank Fusion."""
    _dense_retriever: VectorStoreRetriever
    _sparse_retriever: BM25Retriever
    _k: int
    _rrf_k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        self._sparse_retriever.k = self._k * 2
        dense_docs = self._dense_retriever.invoke(query)
        sparse_docs = self._sparse_retriever.invoke(query)

        scores: dict[int, float] = {}
        doc_map: dict[int, Document] = {}

        for rank, doc in enumerate(dense_docs):
            doc_id = id(doc)
            doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)

        for rank, doc in enumerate(sparse_docs):
            doc_id = id(doc)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (self._rrf_k + rank + 1)

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        return [doc_map[doc_id] for doc_id in sorted_ids[:self._k]]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


class HybridRRFRetrieverAdapter(RetrieverPort):
    """Hybrid retrieval combining dense (vector) and sparse (BM25) via RRF.

    Reciprocal Rank Fusion (RRF) scores each document based on its rank
    in both retrieval lists, providing robust combination that handles
    different score scales between dense and sparse methods.
    """

    def __init__(self, vector_store_retriever: VectorStoreRetriever, k: int = 5, rrf_k: int = 60) -> None:
        self._vector_store_retriever = vector_store_retriever
        self._k = k
        self._rrf_k = rrf_k
        self._bm25: Optional[BM25Retriever] = None

    def build_index(self, documents: List[Document]) -> None:
        self._bm25 = BM25Retriever.from_documents(documents)

    def get_retriever(self) -> BaseRetriever:
        if self._bm25 is None:
            raise RuntimeError("HybridRRFRetrieverAdapter: index not built. Call build_index() first.")
        wrapper = _HybridRRFRetriever()
        wrapper._dense_retriever = self._vector_store_retriever
        wrapper._sparse_retriever = self._bm25
        wrapper._k = self._k
        wrapper._rrf_k = self._rrf_k
        return wrapper
