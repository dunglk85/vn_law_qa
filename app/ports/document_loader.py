"""Document loader port — abstract interface for loading documents.

Implementations load documents from various sources and return
LangChain Document objects with metadata.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.documents import Document


class DocumentLoaderPort(ABC):
    """Abstract interface for document loading.

    Implementations read documents from a data source (e.g., Parquet files,
    databases, file systems) and return them as LangChain Document objects.
    """

    @abstractmethod
    def load(self, data_dir: str) -> list[Document]:
        """Load documents from the given data directory.

        Args:
            data_dir: Path to directory containing document data.

        Returns:
            List of LangChain Document objects with metadata.
        """
        ...
