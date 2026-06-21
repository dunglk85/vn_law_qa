"""A2A remote client — sends JSON-RPC tasks to A2A servers via HTTP with SSE streaming."""
from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx
from httpx_sse import aconnect_sse

from app.core.a2a_client import A2AClientRouter, A2AEvent

logger = logging.getLogger(__name__)


class A2ARemoteClient(A2AClientRouter):
    """Routes A2A tasks to remote servers by agent name."""

    def __init__(
        self,
        agent_map: dict[str, str],
        timeout: float = 60.0,
    ) -> None:
        self._agent_map = agent_map
        self._timeout = timeout

    def _resolve_url(self, agent: str) -> str:
        url = self._agent_map.get(agent)
        if url is None:
            raise ValueError(f"No A2A endpoint configured for agent '{agent}'")
        return url.rstrip("/")

    async def send_task_stream(
        self, agent: str, payload: dict
    ) -> AsyncIterator[A2AEvent]:
        url = self._resolve_url(agent)

        rpc_body = {
            "jsonrpc": "2.0",
            "method": "tasks/sendMessage",
            "params": {
                "id": f"task-{agent}-{id(payload)}",
                "message": {
                    "role": "user",
                    "parts": [{"type": "data", "data": payload}],
                },
            },
            "id": 1,
        }

        query = payload.get("query", "")
        logger.info("A2A remote: sending task to %s (query=%.60s)", url, query)

        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            async with aconnect_sse(
                client, "POST", f"{url}/",
                json=rpc_body,
            ) as event_source:
                async for event in event_source.aiter_sse():
                    if event.event == "task_status":
                        data = json.loads(event.data)
                        yield A2AEvent(type="task_status", status=data.get("status"))
                    elif event.event == "task_artifact":
                        data = json.loads(event.data)
                        yield A2AEvent(type="task_artifact", artifact=data.get("artifact"))
                    elif event.event == "error":
                        data = json.loads(event.data)
                        logger.error("A2A remote error: %s", data)
                        yield A2AEvent(type="task_status", status={"state": "failed", "error": str(data)})
