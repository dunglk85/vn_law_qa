from __future__ import annotations

from abc import ABC, abstractmethod


class RateLimiterPort(ABC):
    @abstractmethod
    async def check(self, client_ip: str) -> bool:
        ...
