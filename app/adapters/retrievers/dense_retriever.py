from __future__ import annotations
from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever

from app.ports.retriever import RetrieverPort


class DenseRetrieverAdapter(RetrieverPort):
    """Dense retrieval using vector similarity search.

    Wraps the vector store's retriever directly.
    """

    def __init__(self, vector_store_retriever: VectorStoreRetriever) -> None:
        self._vector_store_retriever = vector_store_retriever

    def build_index(self, documents: List[Document]) -> None:
        pass

    def get_retriever(self) -> BaseRetriever:
        return self._vector_store_retriever
