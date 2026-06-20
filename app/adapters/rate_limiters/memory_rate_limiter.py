from __future__ import annotations

import time


class MemoryRateLimiter:
    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._store: dict[str, list[float]] = {}

    async def check(self, client_ip: str) -> bool:
        now = time.time()
        window_start = now - self._window_seconds
        if client_ip not in self._store:
            self._store[client_ip] = []
        self._store[client_ip] = [t for t in self._store[client_ip] if t > window_start]
        if not self._store[client_ip]:
            stale_ips = [ip for ip, times in self._store.items() if not times or times[-1] <= window_start]
            for ip in stale_ips:
                del self._store[ip]
            if client_ip not in self._store:
                self._store[client_ip] = []
        if len(self._store[client_ip]) >= self._max_requests:
            return False
        self._store[client_ip].append(now)
        return True
