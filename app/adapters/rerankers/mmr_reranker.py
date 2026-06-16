from __future__ import annotations
from typing import List, Optional, Sequence

import numpy as np
from langchain_core.callbacks import Callbacks
from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.embeddings import Embeddings
from langchain.vectorstores.utils import maximal_marginal_relevance

from app.ports.reranker import RerankerPort


class _MMRCompressor(BaseDocumentCompressor):
    """Compressor that applies Maximal Marginal Relevance to documents."""

    embeddings: Embeddings
    k: int
    lambda_mult: float

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if len(documents) <= self.k:
            return documents

        query_embedding = self.embeddings.embed_query(query)
        doc_embeddings = self.embeddings.embed_documents([d.page_content for d in documents])

        doc_embeddings_array = np.array(doc_embeddings)
        query_embedding_array = np.array(query_embedding)

        mmr_indices = maximal_marginal_relevance(
            query_embedding_array,
            doc_embeddings_array,
            k=self.k,
            lambda_mult=self.lambda_mult,
        )

        return [documents[i] for i in mmr_indices]


class MMRRerankerAdapter(RerankerPort):
    """Reranker using Maximal Marginal Relevance (MMR).

    MMR balances relevance and diversity by selecting documents that are
    both relevant to the query and dissimilar to already-selected documents.
    The lambda_mult parameter controls the trade-off:
    - 1.0 = pure relevance (no diversity)
    - 0.0 = pure diversity (no relevance)
    """

    def __init__(
        self,
        embeddings: Embeddings,
        top_n: int = 3,
        lambda_mult: float = 0.5,
    ) -> None:
        self._embeddings = embeddings
        self._top_n = top_n
        self._lambda_mult = lambda_mult

    def get_compressor(self) -> Optional[BaseDocumentCompressor]:
        return _MMRCompressor(
            embeddings=self._embeddings,
            k=self._top_n,
            lambda_mult=self._lambda_mult,
        )
