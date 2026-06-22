from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from app.core.a2a_client import A2AClientRouter, A2AEvent

logger = logging.getLogger(__name__)


class InProcessFallbackClient(A2AClientRouter):
    def __init__(self, research_agent: Any, citation_agent: Any, synthesis_agent: Any) -> None:
        self._research_agent = research_agent
        self._citation_agent = citation_agent
        self._synthesis_agent = synthesis_agent

    async def send_task_stream(
        self, agent: str, payload: dict
    ) -> AsyncIterator[A2AEvent]:
        if agent == "legal-research-agent":
            query = payload["query"]
            articles = await self._research_agent.run(query)
            yield A2AEvent(type="task_status", status={"state": "completed"})
            yield A2AEvent(type="task_artifact", artifact={"articles": articles})

        elif agent == "citation-checker-agent":
            articles = payload["articles"]
            query = payload["query"]
            citations = await self._citation_agent.run(articles, query)
            yield A2AEvent(type="task_status", status={"state": "completed"})
            yield A2AEvent(type="task_artifact", artifact={"citations": citations})

        elif agent == "response-synthesizer-agent":
            query = payload["query"]
            citations = payload["citations"]
            result = await self._synthesis_agent.synthesize(query, citations)
            yield A2AEvent(type="task_status", status={"state": "completed"})
            yield A2AEvent(type="task_artifact", artifact=result)

        else:
            raise ValueError(f"Unknown A2A agent: {agent}")
