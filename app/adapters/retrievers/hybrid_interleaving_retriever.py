from __future__ import annotations

import hashlib

from langchain_community.retrievers import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.ports.retriever import RetrieverPort
from app.ports.vector_store import VectorStorePort


class _HybridInterleavingRetriever(BaseRetriever):
    """Combines dense and sparse retrieval by interleaving results."""
    _dense_retriever: BaseRetriever
    _sparse_retriever: BM25Retriever
    _k: int

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        sparse_k = self._k * 2
        original_k = getattr(self._sparse_retriever, "k", None)
        self._sparse_retriever.k = sparse_k
        try:
            dense_docs = self._dense_retriever.invoke(query)
            sparse_docs = self._sparse_retriever.invoke(query)
        finally:
            if original_k is not None:
                self._sparse_retriever.k = original_k

        interleaved: list[Document] = []
        seen_hashes: set[str] = set()

        def _try_add(doc: Document) -> bool:
            h = hashlib.sha256(doc.page_content.encode()).hexdigest()
            if h not in seen_hashes:
                interleaved.append(doc)
                seen_hashes.add(h)
                return True
            return False

        for d, s in zip(dense_docs, sparse_docs):
            _try_add(d)
            if len(interleaved) >= self._k:
                break
            _try_add(s)
            if len(interleaved) >= self._k:
                break

        for d in dense_docs:
            if len(interleaved) >= self._k:
                break
            _try_add(d)

        for s in sparse_docs:
            if len(interleaved) >= self._k:
                break
            _try_add(s)

        return interleaved[:self._k]

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self._get_relevant_documents(query, run_manager=run_manager)


class HybridInterleavingRetrieverAdapter(RetrieverPort):
    """Hybrid retrieval combining dense (vector) and sparse (BM25) via interleaving.

    Alternates between dense and sparse results, providing diversity
    in retrieval while maintaining relevance from both approaches.
    """

    def __init__(self, vector_store: VectorStorePort, k: int = 5) -> None:
        self._vector_store = vector_store
        self._k = k
        self._bm25: BM25Retriever | None = None

    def build_index(self, documents: list[Document]) -> None:
        self._bm25 = BM25Retriever.from_documents(documents)

    def get_retriever(self, search_kwargs: dict | None = None) -> BaseRetriever:
        if self._bm25 is None:
            raise RuntimeError("HybridInterleavingRetrieverAdapter: index not built. Call build_index() first.")
        kwargs = search_kwargs or {"k": self._k}
        vector_store_retriever = self._vector_store.as_retriever(search_kwargs=kwargs)
        wrapper = _HybridInterleavingRetriever()
        wrapper._dense_retriever = vector_store_retriever
        wrapper._sparse_retriever = self._bm25
        wrapper._k = self._k
        return wrapper
