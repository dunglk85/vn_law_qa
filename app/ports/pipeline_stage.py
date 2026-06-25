from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from langchain_core.language_models import BaseChatModel


@dataclass
class StageInput:
    prompt: str
    context: dict[str, Any] = field(default_factory=dict)
    previous_output: str | None = None


@dataclass
class StageOutput:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class PipelineStagePort(ABC):
    def __init__(self, chat_model: BaseChatModel, config: dict | None = None) -> None:
        self._model = chat_model
        self._config = config or {}

    @abstractmethod
    async def run(self, input: StageInput) -> StageOutput:
        ...

    @abstractmethod
    async def stream(self, input: StageInput) -> AsyncIterator[str]:
        ...
