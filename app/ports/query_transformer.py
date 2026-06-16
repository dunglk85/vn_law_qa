from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class QueryTransformerPort(ABC):
    """Abstract interface for query transformation strategies.

    Query transformers modify the user's question before retrieval to
    improve relevance. Examples: HyDE, decomposition, step-back prompting.
    """

    @abstractmethod
    async def transform(self, query: str) -> List[str]:
        """Transform the original query into one or more retrieval queries.

        Args:
            query: The user's original question.

        Returns:
            List of transformed queries to use for retrieval.
        """
        ...
