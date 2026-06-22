from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class MetadataEnrichmentPort(ABC):
    """Abstract interface for metadata enrichment strategies.

    Enrichment adapters add metadata fields to documents (e.g., category,
    title, summary, keywords). The rest of the system depends only on this
    interface.
    """

    @abstractmethod
    async def enrich(self, documents: list[Document]) -> list[Document]:
        """Enrich documents with additional metadata.

        Args:
            documents: List of documents to enrich.

        Returns:
            List of documents with enriched metadata.
        """
        ...
