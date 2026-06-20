from __future__ import annotations

from abc import ABC, abstractmethod


class SessionStorePort(ABC):
    @abstractmethod
    async def load(self, session_id: str) -> list[dict]:
        ...

    @abstractmethod
    async def save(self, session_id: str, history: list[dict]) -> None:
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        ...

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        ...
