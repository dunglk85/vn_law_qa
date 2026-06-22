from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel


class LLMPort(ABC):
    """Abstract interface for any chat LLM provider.

    Swap OpenAI → Gemini/Claude/Ollama by creating a new adapter that
    implements this interface, then setting LLM_TYPE in .env.
    """

    @abstractmethod
    def get_chat_model(self) -> BaseChatModel:
        """Return a LangChain-compatible BaseChatModel instance."""
        ...
