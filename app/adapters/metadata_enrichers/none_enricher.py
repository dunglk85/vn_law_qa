from __future__ import annotations

from langchain_core.documents import Document

from app.ports.metadata_enrichment import MetadataEnrichmentPort


class NoneEnricherAdapter(MetadataEnrichmentPort):
    """No-op metadata enrichment — returns documents unchanged.

    Use by setting METADATA_ENRICHER_TYPE=none in .env.
    """

    async def enrich(self, documents: list[Document]) -> list[Document]:
        return documents
