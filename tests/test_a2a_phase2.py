"""Tests for A2A remote client and server."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator
from unittest.mock import MagicMock, patch

import pytest

from app.core.a2a_client import A2AEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Remote Client Tests
# ---------------------------------------------------------------------------


def _mock_aconnect_sse(events):
    """Return a mock aconnect_sse that yields given SSE events as tuples of (event_type, data_dict)."""
    class _MockSSEEvent:
        def __init__(self, event: str, data: dict):
            self.event = event
            self.data = json.dumps(data)

    sse_events = [_MockSSEEvent(e, d) for e, d in events]

    @asynccontextmanager
    async def mock_connect(*args, **kwargs):
        class MockEventSource:
            async def aiter_sse(self) -> AsyncIterator:
                for ev in sse_events:
                    yield ev

        yield MockEventSource()

    return mock_connect
    """Return a mock aconnect_sse that yields given SSE events."""
    @asynccontextmanager
    async def mock_connect(*args, **kwargs):
        class MockEventSource:
            async def aiter_sse(self) -> AsyncIterator:
                for ev in events:
                    yield ev

        yield MockEventSource()

    return mock_connect


class TestA2ARemoteClient:
    @pytest.mark.asyncio
    async def test_send_task_stream_yields_events(self):
        from app.adapters.agents.a2a_remote_client import A2ARemoteClient

        client = A2ARemoteClient(
            agent_map={"legal-research-agent": "http://test:8101"},
            timeout=10,
        )

        mock_connect = _mock_aconnect_sse([
            ("task_status", {"id": "t1", "status": {"state": "working"}}),
            ("task_status", {"id": "t1", "status": {"state": "completed"}}),
            ("task_artifact", {"id": "t1", "artifact": {"articles": []}}),
        ])

        with patch("app.adapters.agents.a2a_remote_client.aconnect_sse", mock_connect):
            events = []
            async for ev in client.send_task_stream(
                agent="legal-research-agent",
                payload={"query": "test query"},
            ):
                events.append(ev)

        assert len(events) == 3
        assert events[0].type == "task_status"
        assert events[0].status == {"state": "working"}
        assert events[1].status == {"state": "completed"}
        assert events[2].type == "task_artifact"
        assert events[2].artifact == {"articles": []}

    @pytest.mark.asyncio
    async def test_error_event(self):
        from app.adapters.agents.a2a_remote_client import A2ARemoteClient

        client = A2ARemoteClient(
            agent_map={"legal-research-agent": "http://test:8101"},
            timeout=10,
        )

        mock_connect = _mock_aconnect_sse([
            ("error", {"message": "Internal error"}),
        ])

        with patch("app.adapters.agents.a2a_remote_client.aconnect_sse", mock_connect):
            events = []
            async for ev in client.send_task_stream(
                agent="legal-research-agent",
                payload={"query": "test"},
            ):
                events.append(ev)

        assert len(events) == 1
        assert events[0].type == "task_status"
        assert events[0].status["state"] == "failed"

    @pytest.mark.asyncio
    async def test_inherits_abc(self):
        from app.adapters.agents.a2a_remote_client import A2ARemoteClient

        client = A2ARemoteClient(agent_map={"test": "http://test:8101"})
        assert isinstance(client, A2ARemoteClient)
        from app.core.a2a_client import A2AClientRouter

        assert isinstance(client, A2AClientRouter)

    @pytest.mark.asyncio
    async def test_unknown_agent_raises(self):
        from app.adapters.agents.a2a_remote_client import A2ARemoteClient

        client = A2ARemoteClient(agent_map={})
        with pytest.raises(ValueError, match="No A2A endpoint configured for agent 'unknown'"):
            async for _ in client.send_task_stream(agent="unknown", payload={}):
                pass

    def test_resolve_url(self):
        from app.adapters.agents.a2a_remote_client import A2ARemoteClient

        client = A2ARemoteClient(agent_map={
            "legal-research-agent": "http://legal:8101",
            "citation-checker-agent": "http://citation:8102",
        })
        assert client._resolve_url("legal-research-agent") == "http://legal:8101"
        assert client._resolve_url("citation-checker-agent") == "http://citation:8102"

    def test_resolve_url_missing_raises(self):
        from app.adapters.agents.a2a_remote_client import A2ARemoteClient

        client = A2ARemoteClient(agent_map={"legal-research-agent": "http://legal:8101"})
        with pytest.raises(ValueError, match="No A2A endpoint configured for agent 'missing'"):
            client._resolve_url("missing")


# ---------------------------------------------------------------------------
# Server Tests (skip if langchain not available)
# ---------------------------------------------------------------------------


class TestA2ALegalResearchServer:
    def test_agent_card_content(self):
        pytest.importorskip("langchain_core")
        from app.agents.a2a_servers.legal_research_server import AGENT_CARD

        assert AGENT_CARD["name"] == "legal-research-agent"
        assert AGENT_CARD["capabilities"]["streaming"] is True
        assert len(AGENT_CARD["skills"]) == 1
        assert AGENT_CARD["skills"][0]["id"] == "legal_research"

    def test_article_to_dict(self):
        pytest.importorskip("langchain_core")
        from app.agents.a2a_servers.legal_research_server import _article_to_dict
        from app.core.models import Article

        a = Article(id="a1", content="test", metadata={"source": "test.pdf"}, relevance_score=0.95)
        d = _article_to_dict(a)
        assert d["id"] == "a1"
        assert d["content"] == "test"
        assert d["relevance_score"] == 0.95

    def test_article_to_dict_from_dict(self):
        pytest.importorskip("langchain_core")
        from app.agents.a2a_servers.legal_research_server import _article_to_dict

        d = _article_to_dict({"id": "a1", "content": "test"})
        assert d["id"] == "a1"

    def test_error_stream(self):
        pytest.importorskip("langchain_core")
        from app.agents.a2a_servers.legal_research_server import _error_stream
        import asyncio

        async def collect():
            events = []
            async for ev in _error_stream("task-1", "oops"):
                events.append(ev)
            return events

        events = asyncio.run(collect())
        assert len(events) == 1
        assert events[0]["event"] == "task_status"
        data = json.loads(events[0]["data"])
        assert data["status"]["state"] == "failed"
        assert data["status"]["error"] == "oops"


# ---------------------------------------------------------------------------
# Factory Tests
# ---------------------------------------------------------------------------


class TestFactoryA2AClient:
    def test_factory_creates_fallback_when_no_url(self):
        pytest.importorskip("langchain_core")
        from app.factory import create_a2a_client

        fake = MagicMock()
        client = create_a2a_client(research_agent=fake, citation_agent=fake, synthesis_agent=fake)
        from app.adapters.agents.a2a_fallback_client import InProcessFallbackClient

        assert isinstance(client, InProcessFallbackClient)

    def test_factory_raises_on_fallback_without_agents(self):
        pytest.importorskip("langchain_core")
        from app.factory import create_a2a_client

        with pytest.raises(ValueError, match="All agents required"):
            create_a2a_client()

    def test_remote_behavior(self):
        from app.adapters.agents.a2a_remote_client import A2ARemoteClient

        client = A2ARemoteClient(agent_map={"legal-research-agent": "http://legal:8101"})
        assert client._agent_map["legal-research-agent"] == "http://legal:8101"
        assert client._timeout == 60.0
