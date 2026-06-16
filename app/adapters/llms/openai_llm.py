from __future__ import annotations
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from app.config import config
from app.ports.llm import LLMPort


class OpenAILLMAdapter(LLMPort):
    """Concrete adapter for OpenAI Chat models (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model

    def get_chat_model(self) -> BaseChatModel:
        return ChatOpenAI(model=self._model, openai_api_key=config.openai_api_key)
