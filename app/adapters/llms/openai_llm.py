from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.ports.llm import LLMPort


class OpenAILLMAdapter(LLMPort):
    """Concrete adapter for OpenAI Chat models (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None) -> None:
        self._model = model
        self._api_key = api_key
        self._instance: BaseChatModel | None = None

    def get_chat_model(self) -> BaseChatModel:
        if self._instance is None:
            self._instance = ChatOpenAI(model=self._model, openai_api_key=self._api_key)
        return self._instance
