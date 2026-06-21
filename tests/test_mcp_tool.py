from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.adapters.tools.mcp_tool_adapter import MCPKnowledgeSearchTool


class TestMCPKnowledgeSearchTool:
    @pytest.mark.asyncio
    async def test_ainvoke_returns_results(self):
        tool = MCPKnowledgeSearchTool(server_timeout=30, max_restarts=1)
        tool._connected = True
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_item = MagicMock()
        mock_item.text = '{"content": "test doc", "source": "test.pdf", "score": 0.95}'
        mock_result.content = [mock_item]
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        tool._session = mock_session

        results = await tool.ainvoke({"query": "test query"})

        assert len(results) == 1
        assert results[0]["content"] == "test doc"
        assert results[0]["source"] == "test.pdf"
        assert results[0]["score"] == 0.95
        mock_session.call_tool.assert_awaited_once_with(
            "knowledge_search", {"query": "test query", "k": None}
        )

    @pytest.mark.asyncio
    async def test_ainvoke_passes_k_parameter(self):
        tool = MCPKnowledgeSearchTool(server_timeout=30, max_restarts=1)
        tool._connected = True
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.content = []
        mock_session.call_tool = AsyncMock(return_value=mock_result)
        tool._session = mock_session

        await tool.ainvoke({"query": "test", "k": 10})

        mock_session.call_tool.assert_awaited_once_with(
            "knowledge_search", {"query": "test", "k": 10}
        )

    @pytest.mark.asyncio
    async def test_returns_empty_on_timeout(self):
        tool = MCPKnowledgeSearchTool(server_timeout=0.01, max_restarts=0)
        tool._connected = True
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=TimeoutError("timed out"))
        tool._session = mock_session

        results = await tool.ainvoke({"query": "test"})

        assert results == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self):
        tool = MCPKnowledgeSearchTool(server_timeout=30, max_restarts=0)
        tool._connected = True
        mock_session = AsyncMock()
        mock_session.call_tool = AsyncMock(side_effect=Exception("server error"))
        tool._session = mock_session

        results = await tool.ainvoke({"query": "test"})

        assert results == []

    @pytest.mark.asyncio
    async def test_connects_on_first_invoke(self):
        tool = MCPKnowledgeSearchTool(server_timeout=30, max_restarts=1)
        tool._connected = False

        with patch.object(tool, "_connect", AsyncMock()) as mock_connect:
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.content = []
            mock_session.call_tool = AsyncMock(return_value=mock_result)
            tool._session = mock_session

            await tool.ainvoke({"query": "test"})

            mock_connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reconnects_on_connection_error(self):
        tool = MCPKnowledgeSearchTool(server_timeout=30, max_restarts=2)
        tool._connected = False

        with (
            patch.object(tool, "_connect", AsyncMock()) as mock_connect,
            patch.object(tool, "_reconnect", AsyncMock(return_value=True)),
        ):
            mock_connect.side_effect = [ConnectionError("fail"), None]
            mock_session = AsyncMock()
            mock_result = MagicMock()
            mock_result.content = []
            mock_session.call_tool = AsyncMock(return_value=mock_result)
            tool._session = mock_session
            tool._connected = False

            tool._connect = AsyncMock(side_effect=[ConnectionError("fail"), None])
            with patch.object(tool, "_reconnect", AsyncMock(return_value=True)):
                results = await tool.ainvoke({"query": "test"})

                assert results == []


class TestFactoryFallback:
    @pytest.mark.skip(reason="Requires full dependency tree (langchain-core)")
    @pytest.mark.asyncio
    async def test_direct_tool_when_mcp_disabled(self):
        from app.factory import create_knowledge_search_tool

        with (
            patch("app.factory.config") as mock_config,
            patch("app.agents.tools.knowledge_search.create_knowledge_search_tool") as mock_direct,
        ):
            mock_config.mcp_enabled = False
            mock_retriever = MagicMock()
            mock_direct.return_value = "direct-tool"

            result = create_knowledge_search_tool(retriever_port=mock_retriever)

            assert result == "direct-tool"
            mock_direct.assert_called_once_with(mock_retriever, k=mock_config.retrieval_k)
