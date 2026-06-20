from __future__ import annotations

import copy
import time

from app.ports.session_store import SessionStorePort


class MemorySessionStore(SessionStorePort):
    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl = ttl_seconds
        self._store: dict[str, tuple[list[dict], float]] = {}

    def _is_expired(self, timestamp: float) -> bool:
        return time.time() - timestamp > self._ttl

    def _evict_expired(self) -> None:
        expired = [sid for sid, (_, ts) in self._store.items() if self._is_expired(ts)]
        for sid in expired:
            del self._store[sid]

    async def load(self, session_id: str) -> list[dict]:
        self._evict_expired()
        entry = self._store.get(session_id)
        if entry is None:
            return []
        return copy.deepcopy(entry[0])

    async def save(self, session_id: str, history: list[dict]) -> None:
        self._store[session_id] = (copy.deepcopy(history), time.time())

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)

    async def exists(self, session_id: str) -> bool:
        self._evict_expired()
        return session_id in self._store
