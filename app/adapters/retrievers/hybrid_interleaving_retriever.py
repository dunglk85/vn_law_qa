from __future__ import annotations
from typing import List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.retrievers import BM25Retriever

from app.ports.retriever import RetrieverPort


class _HybridInterleavingRetriever(BaseRetriever):
    """Combines dense and sparse retrieval by interleaving results."""
    _dense_retriever: VectorStoreRetriever
    _sparse_retriever: BM25Retriever
    _k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        self._sparse_retriever.k = self._k
        dense_docs = self._dense_retriever.invoke(query)
        sparse_docs = self._sparse_retriever.invoke(query)

        interleaved: List[Document] = []
        seen_ids = set()

        for d, s in zip(dense_docs, sparse_docs):
            d_id = id(d)
            if d_id not in seen_ids:
                interleaved.append(d)
                seen_ids.add(d_id)
            if len(interleaved) >= self._k:
                break

            s_id = id(s)
            if s_id not in seen_ids:
                interleaved.append(s)
                seen_ids.add(s_id)
            if len(interleaved) >= self._k:
                break

        for d in dense_docs[len(interleaved):]:
            if len(interleaved) >= self._k:
                break
            d_id = id(d)
            if d_id not in seen_ids:
                interleaved.append(d)
                seen_ids.add(d_id)

        for s in sparse_docs[len(interleaved):]:
            if len(interleaved) >= self._k:
                break
            s_id = id(s)
            if s_id not in seen_ids:
                interleaved.append(s)
                seen_ids.add(s_id)

        return interleaved[:self._k]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


class HybridInterleavingRetrieverAdapter(RetrieverPort):
    """Hybrid retrieval combining dense (vector) and sparse (BM25) via interleaving.

    Alternates between dense and sparse results, providing diversity
    in retrieval while maintaining relevance from both approaches.
    """

    def __init__(self, vector_store_retriever: VectorStoreRetriever, k: int = 5) -> None:
        self._vector_store_retriever = vector_store_retriever
        self._k = k
        self._bm25: Optional[BM25Retriever] = None

    def build_index(self, documents: List[Document]) -> None:
        self._bm25 = BM25Retriever.from_documents(documents)

    def get_retriever(self) -> BaseRetriever:
        if self._bm25 is None:
            raise RuntimeError("HybridInterleavingRetrieverAdapter: index not built. Call build_index() first.")
        wrapper = _HybridInterleavingRetriever()
        wrapper._dense_retriever = self._vector_store_retriever
        wrapper._sparse_retriever = self._bm25
        wrapper._k = self._k
        return wrapper
