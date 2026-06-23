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
        try:
            if agent == "legal-research-agent":
                query = payload.get("query")
                if not query:
                    raise ValueError("Missing 'query' in payload for legal-research-agent")
                articles = await self._research_agent.run(query)
                yield A2AEvent(type="task_status", status={"state": "completed"})
                yield A2AEvent(type="task_artifact", artifact={"articles": articles})

            elif agent == "citation-checker-agent":
                articles = payload.get("articles", [])
                query = payload.get("query", "")
                if not query:
                    raise ValueError("Missing 'query' in payload for citation-checker-agent")
                citations = await self._citation_agent.run(articles, query)
                yield A2AEvent(type="task_status", status={"state": "completed"})
                yield A2AEvent(type="task_artifact", artifact={"citations": citations})

            elif agent == "response-synthesizer-agent":
                query = payload.get("query", "")
                citations = payload.get("citations", [])
                if not query:
                    raise ValueError("Missing 'query' in payload for response-synthesizer-agent")
                result = await self._synthesis_agent.synthesize(query, citations)
                yield A2AEvent(type="task_status", status={"state": "completed"})
                yield A2AEvent(type="task_artifact", artifact=result)

            else:
                raise ValueError(f"Unknown A2A agent: {agent}")
        except Exception as exc:
            logger.exception("InProcessFallbackClient: agent '%s' failed", agent)
            yield A2AEvent(type="task_status", status={"state": "failed", "error": str(exc)})
