"""A2A remote client — sends JSON-RPC tasks to A2A servers via HTTP with SSE streaming."""
from __future__ import annotations

import asyncio
import json
import logging
import random
import uuid
from collections.abc import AsyncIterator

import httpx
from httpx_sse import aconnect_sse

from app.config import config
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
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _resolve_url(self, agent: str) -> str:
        url = self._agent_map.get(agent)
        if url is None:
            raise ValueError(f"No A2A endpoint configured for agent '{agent}'")
        return url.rstrip("/")

    async def send_task_stream(
        self, agent: str, payload: dict
    ) -> AsyncIterator[A2AEvent]:
        url = self._resolve_url(agent)
        task_id = f"task-{agent}-{uuid.uuid4()}"

        rpc_body = {
            "jsonrpc": "2.0",
            "method": "tasks/sendMessage",
            "params": {
                "id": task_id,
                "message": {
                    "role": "user",
                    "parts": [{"type": "data", "data": payload}],
                },
            },
            "id": 1,
        }

        query = payload.get("query", "")
        logger.info("A2A remote: sending task %s to %s (query=%.60s)", task_id, url, query)

        max_attempts = config.a2a_max_retries + 1
        last_exc: Exception | None = None

        for attempt in range(max_attempts):
            try:
                deadline = asyncio.get_running_loop().time() + self._timeout
                async for event in self._stream_sse_events(url, rpc_body, deadline):
                    yield event
                return
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    delay = min(1.0 * (2 ** attempt), 10.0)
                    jitter = random.uniform(0, delay * 0.1)
                    logger.warning(
                        "a2a_%s attempt %d/%d failed: %s. Retrying in %.2fs",
                        agent, attempt + 1, max_attempts, exc, delay + jitter,
                    )
                    await asyncio.sleep(delay + jitter)

        logger.error("A2A remote: task %s failed after %d attempts: %s", task_id, max_attempts, last_exc)
        yield A2AEvent(type="task_status", status={"state": "failed", "error": str(last_exc)})

    async def _stream_sse_events(self, url: str, rpc_body: dict, deadline: float) -> AsyncIterator[A2AEvent]:
        async with aconnect_sse(
            self._client, "POST", f"{url}/",
            json=rpc_body,
        ) as event_source:
            async for event in event_source.aiter_sse():
                try:
                    if asyncio.get_running_loop().time() > deadline:
                        logger.warning("A2A SSE stream: deadline exceeded")
                        break
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
                except json.JSONDecodeError as exc:
                    logger.warning("A2A remote: malformed SSE event data, skipping: %s", exc)
