from __future__ import annotations
import os
from typing import List

from langchain_core.documents import Document

from app.ports.metadata_enrichment import MetadataEnrichmentPort


class BasicEnricherAdapter(MetadataEnrichmentPort):
    """Basic metadata enrichment — adds category, source, filename, and file_type.

    Extracts metadata from file paths without external dependencies.
    """

    def enrich(self, documents: List[Document]) -> List[Document]:
        for doc in documents:
            source = doc.metadata.get("source", "")
            if source:
                filename = os.path.basename(source)
                file_type = os.path.splitext(filename)[1].lower().lstrip(".")
                doc.metadata["filename"] = filename
                doc.metadata["file_type"] = file_type
        return documents
