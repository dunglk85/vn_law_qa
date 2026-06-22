"""MCP client adapter — wraps an MCP server subprocess as a LangChain-compatible tool.

Usage:
    tool = await create_mcp_knowledge_search_tool()
    result = await tool.ainvoke({"query": "..."})
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

_MCP_SERVER_SCRIPT = str(Path(__file__).parents[3] / "mcp-servers" / "knowledge_search_server.py")


class MCPKnowledgeSearchTool:
    """LangChain-compatible tool backed by an MCP server subprocess.

    Duck-types the LangChain @tool interface (ainvoke with dict arg)
    so it can be a drop-in replacement for the existing knowledge_search_tool.
    """

    def __init__(
        self,
        server_script: str = _MCP_SERVER_SCRIPT,
        server_timeout: int = 30,
        max_restarts: int = 3,
    ) -> None:
        self._server_script = server_script
        self._server_timeout = server_timeout
        self._max_restarts = max_restarts
        self._session: ClientSession | None = None
        self._read: Any = None
        self._write: Any = None
        self._exit_stack: asyncio.AsyncExitStack | None = None
        self._restart_count = 0
        self._connected = False

    async def _connect(self) -> None:
        from contextlib import AsyncExitStack

        if self._connected:
            return

        self._exit_stack = AsyncExitStack()
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[self._server_script],
        )
        try:
            streams = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self._read, self._write = streams
            self._session = await self._exit_stack.enter_async_context(
                ClientSession(self._read, self._write)
            )
            await self._session.initialize()
            self._connected = True
            self._restart_count = 0
            logger.info("MCP knowledge_search server connected")
        except Exception as exc:
            await self._exit_stack.aclose()
            self._exit_stack = None
            raise ConnectionError(f"Failed to connect to MCP server: {exc}") from exc

    async def _reconnect(self) -> bool:
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
            self._exit_stack = None
        self._session = None
        self._connected = False
        self._restart_count += 1

        if self._restart_count > self._max_restarts:
            logger.error("MCP server exceeded max restarts (%d)", self._max_restarts)
            return False

        logger.info("Reconnecting MCP server (attempt %d/%d)", self._restart_count, self._max_restarts)
        try:
            await self._connect()
            return True
        except Exception as exc:
            logger.warning("Reconnect attempt %d failed: %s", self._restart_count, exc)
            return False

    async def ainvoke(self, input_data: dict[str, Any]) -> list[dict]:
        """Invoke the knowledge_search MCP tool.

        Args:
            input_data: dict with 'query' (str) and optional 'k' (int).

        Returns:
            list of result dicts, each with 'content', 'source', 'score'.
        """
        query = input_data.get("query", "")
        k = input_data.get("k")

        for attempt in range(self._max_restarts + 1):
            try:
                if not self._connected:
                    await self._connect()

                result = await asyncio.wait_for(
                    self._session.call_tool("knowledge_search", {"query": query, "k": k}),
                    timeout=self._server_timeout,
                )
                return [self._decode_content(item) for item in result.content]
            except (ConnectionError, OSError) as exc:
                logger.warning("MCP connection lost (attempt %d): %s", attempt + 1, exc)
                ok = await self._reconnect()
                if not ok:
                    logger.error("MCP server unavailable after %d attempts", self._max_restarts)
                    return []
            except TimeoutError:
                logger.error("MCP knowledge_search timed out after %ds", self._server_timeout)
                return []
            except Exception as exc:
                logger.error("MCP knowledge_search failed: %s", exc)
                return []

        return []

    @staticmethod
    def _decode_content(item: Any) -> dict:
        """Decode an MCP content item to a dict."""
        if hasattr(item, "text") and item.text:
            import json
            try:
                return json.loads(item.text)
            except (json.JSONDecodeError, TypeError):
                return {"content": item.text, "source": "unknown", "score": None}
        return {"content": str(item), "source": "unknown", "score": None}

    async def close(self) -> None:
        if self._exit_stack is not None:
            try:
                await self._exit_stack.aclose()
            except Exception:
                pass
        self._session = None
        self._connected = False


async def create_mcp_knowledge_search_tool(
    server_timeout: int = 30,
    max_restarts: int = 3,
) -> MCPKnowledgeSearchTool:
    """Factory: create and connect an MCP-backed knowledge search tool."""
    tool = MCPKnowledgeSearchTool(
        server_timeout=server_timeout,
        max_restarts=max_restarts,
    )
    try:
        await tool._connect()
    except ConnectionError as exc:
        logger.warning("MCP tool created but not connected: %s", exc)
    return tool


def create_mcp_knowledge_search_tool_lazy(
    server_timeout: int = 30,
    max_restarts: int = 3,
) -> MCPKnowledgeSearchTool:
    """Factory: create an MCP-backed knowledge search tool with lazy initialization.

    The tool connects on first ainvoke() call — safe to call from sync code
    (e.g., FastAPI startup) without asyncio.run().
    """
    return MCPKnowledgeSearchTool(
        server_timeout=server_timeout,
        max_restarts=max_restarts,
    )
