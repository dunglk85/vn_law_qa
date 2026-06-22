from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass
class A2AEvent:
    type: str
    status: dict | None = None
    artifact: Any = None


class A2AClientRouter(ABC):
    @abstractmethod
    async def send_task_stream(
        self, agent: str, payload: dict
    ) -> AsyncIterator[A2AEvent]:
        ...
