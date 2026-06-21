from __future__ import annotations
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from app.ports.retriever import RetrieverPort
from app.ports.vector_store import VectorStorePort


class DenseRetrieverAdapter(RetrieverPort):
    """Dense retrieval using vector similarity search.

    Wraps the vector store's retriever directly.
    """

    def __init__(self, vector_store: VectorStorePort, k: int = 5) -> None:
        self._vector_store = vector_store
        self._k = k

    def build_index(self, documents: List[Document]) -> None:
        pass

    def get_retriever(self, search_kwargs: Optional[dict] = None) -> BaseRetriever:
        kwargs = search_kwargs or {"k": self._k}
        return self._vector_store.as_retriever(search_kwargs=kwargs)
