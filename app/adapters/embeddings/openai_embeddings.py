from __future__ import annotations
from langchain_openai import OpenAIEmbeddings

from app.config import config
from app.ports.embeddings import EmbeddingsPort


class OpenAIEmbeddingsAdapter(EmbeddingsPort):
    """Concrete adapter that wraps OpenAI text-embedding models."""

    def __init__(self, model: str = "text-embedding-3-small") -> None:
        self._model = model
        self._instance: OpenAIEmbeddings | None = None

    def get_embeddings(self) -> OpenAIEmbeddings:
        """Return a cached OpenAIEmbeddings instance (lazy init)."""
        if self._instance is None:
            self._instance = OpenAIEmbeddings(
                model=self._model,
                openai_api_key=config.openai_api_key,
            )
        return self._instance
