"""Tests for A2A client interface and InProcessFallbackClient."""
from __future__ import annotations

from typing import Any

import pytest

from app.core.a2a_client import A2AClientRouter, A2AEvent


class FakeResearchAgent:
    async def run(self, query: str) -> list[dict]:
        return [{"id": "art-1", "content": query, "metadata": {}, "relevance_score": 0.9}]


class FakeCitationAgent:
    async def run(self, articles: list, query: str) -> list[dict]:
        return [
            {"article_id": a["id"], "content": a["content"], "relevance": 0.9, "verified": True}
            for a in articles
        ]


class FakeSynthesisAgent:
    async def synthesize(self, query: str, citations: list) -> dict:
        return {"response": f"Answer for: {query}", "citations": citations, "metadata": {}}


@pytest.fixture
def fallback_client():
    from app.adapters.agents.a2a_fallback_client import InProcessFallbackClient

    return InProcessFallbackClient(
        research_agent=FakeResearchAgent(),
        citation_agent=FakeCitationAgent(),
        synthesis_agent=FakeSynthesisAgent(),
    )


class TestA2AEvent:
    def test_event_creation(self):
        ev = A2AEvent(type="task_status", status={"state": "completed"})
        assert ev.type == "task_status"
        assert ev.status == {"state": "completed"}
        assert ev.artifact is None


class TestA2AClientRouter:
    def test_interface_abstract(self):
        with pytest.raises(TypeError):
            A2AClientRouter()  # type: ignore[abstract]


class TestInProcessFallbackClient:
    @pytest.mark.asyncio
    async def test_legal_research(self, fallback_client):
        events = []
        async for ev in fallback_client.send_task_stream(
            agent="legal-research-agent",
            payload={"query": "test query", "metadata": {}},
        ):
            events.append(ev)
        assert len(events) == 2
        assert events[0].type == "task_status"
        assert events[0].status == {"state": "completed"}
        assert events[1].type == "task_artifact"
        articles = events[1].artifact["articles"]
        assert len(articles) == 1
        assert articles[0]["id"] == "art-1"

    @pytest.mark.asyncio
    async def test_citation_check(self, fallback_client):
        events = []
        async for ev in fallback_client.send_task_stream(
            agent="citation-checker-agent",
            payload={"articles": [{"id": "art-1", "content": "test"}], "query": "test"},
        ):
            events.append(ev)
        assert len(events) == 2
        assert events[1].type == "task_artifact"
        citations = events[1].artifact["citations"]
        assert len(citations) == 1
        assert citations[0]["article_id"] == "art-1"

    @pytest.mark.asyncio
    async def test_response_synthesis(self, fallback_client):
        events = []
        async for ev in fallback_client.send_task_stream(
            agent="response-synthesizer-agent",
            payload={"query": "test", "citations": []},
        ):
            events.append(ev)
        assert len(events) == 2
        assert events[1].type == "task_artifact"
        assert events[1].artifact["response"] == "Answer for: test"

    @pytest.mark.asyncio
    async def test_unknown_agent(self, fallback_client):
        with pytest.raises(ValueError, match="Unknown A2A agent"):
            async for _ in fallback_client.send_task_stream(
                agent="unknown-agent",
                payload={},
            ):
                pass

    def test_inherits_abc(self, fallback_client):
        assert isinstance(fallback_client, A2AClientRouter)


class TestA2AConfig:
    def test_config_has_a2a_fields(self):
        from app.config import config

        assert hasattr(config, "a2a_legal_research_url")
        assert hasattr(config, "a2a_citation_checker_url")
        assert hasattr(config, "a2a_response_synthesizer_url")
        assert hasattr(config, "a2a_task_timeout")
        assert hasattr(config, "a2a_max_retries")
