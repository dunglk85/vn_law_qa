from __future__ import annotations

from abc import ABC, abstractmethod

_SESSION_DATA_DEFAULT: dict = {"history": [], "summary": ""}


class SessionStorePort(ABC):
    @abstractmethod
    async def load(self, session_id: str) -> dict:
        ...

    @abstractmethod
    async def save(self, session_id: str, session_data: dict) -> None:
        ...

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        ...

    @abstractmethod
    async def exists(self, session_id: str) -> bool:
        ...
