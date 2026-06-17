from __future__ import annotations
from typing import List

from langchain_core.documents import Document

from app.ports.metadata_enrichment import MetadataEnrichmentPort


class NoneEnricherAdapter(MetadataEnrichmentPort):
    """No-op metadata enrichment — returns documents unchanged.

    Use by setting METADATA_ENRICHER_TYPE=none in .env.
    """

    def enrich(self, documents: List[Document]) -> List[Document]:
        return documents
