"""Standalone A2A JSON-RPC server wrapping the LegalResearchAgent.

Usage:
    uvicorn app.agents.a2a_servers.legal_research_server:app --port 8101

Requires env vars: VECTOR_STORE_TYPE, DATABASE_URL, LLM_TYPE, LLM_MODEL, etc.
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.core.models import Article
from app.factory import create_embeddings, create_llm, create_retriever, create_vector_store

logger = logging.getLogger(__name__)

app = FastAPI(title="Legal Research A2A Agent")


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "legal-research-agent",
    "description": "Performs legal research: HyDE generation, sub-query decomposition, parallel retrieval, LLM relevance scoring, blended ranking.",
    "version": "1.0.0",
    "capabilities": {
        "streaming": True,
        "pushNotifications": False,
        "stateful": True,
    },
    "skills": [
        {
            "id": "legal_research",
            "name": "Legal Research",
            "description": "Given a legal query, returns ranked legal articles with relevance scores.",
            "tags": ["legal", "retrieval", "ranking"],
            "inputs": [
                {"name": "query", "type": "string", "description": "Natural language legal question"},
                {"name": "metadata", "type": "object", "description": "Optional filtering metadata"},
            ],
            "outputs": [
                {"name": "articles", "type": "array", "description": "Ranked list of Article objects"}
            ],
        }
    ],
}


@app.get("/.well-known/agent-card")
@app.get("/agent-card")
async def get_agent_card():
    return JSONResponse(AGENT_CARD)


@app.get("/health")
async def health():
    return {"status": "ok", "agent": "legal-research-agent", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Agent instantiation
# ---------------------------------------------------------------------------

_embeddings = create_embeddings()
_vector_store = create_vector_store(embeddings=_embeddings)
_retriever = create_retriever(vector_store=_vector_store)
_llm = create_llm()

from datetime import UTC

from app.agents.legal_research_agent import LegalResearchAgent

_research_agent = LegalResearchAgent(_retriever, _llm.get_chat_model())


# ---------------------------------------------------------------------------
# JSON-RPC Methods
# ---------------------------------------------------------------------------


async def _handle_send_message(params: dict) -> EventSourceResponse:
    task_id = params.get("id", f"task-{uuid.uuid4().hex[:12]}")
    message = params.get("message", {})
    parts = message.get("parts", [])
    query = ""
    for part in parts:
        if part.get("type") == "text":
            query = part["text"]
            break
        if part.get("type") == "data" and isinstance(part.get("data"), dict):
            query = part["data"].get("query", "")

    if not query:
        return EventSourceResponse(
            _error_stream(task_id, "No query provided in message")
        )

    async def event_stream():
        yield {"event": "task_status", "data": json.dumps({
            "id": task_id,
            "status": {"state": "working", "timestamp": _now()},
        })}

        try:
            articles = await _research_agent.run(query)
            articles_dicts = [a.to_dict() if hasattr(a, "to_dict") else _article_to_dict(a) for a in articles]

            yield {"event": "task_status", "data": json.dumps({
                "id": task_id,
                "status": {"state": "completed", "timestamp": _now()},
            })}
            yield {"event": "task_artifact", "data": json.dumps({
                "id": task_id,
                "artifact": {"articles": articles_dicts},
            })}
        except Exception as exc:
            logger.exception("Legal research failed")
            yield {"event": "task_status", "data": json.dumps({
                "id": task_id,
                "status": {"state": "failed", "timestamp": _now(), "error": str(exc)},
            })}

    return EventSourceResponse(event_stream())


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.UTC if hasattr(timezone, "UTC") else UTC).isoformat()


async def _error_stream(task_id: str, message: str):
    yield {"event": "task_status", "data": json.dumps({
        "id": task_id,
        "status": {"state": "failed", "timestamp": _now(), "error": message},
    })}


def _article_to_dict(a: Any) -> dict:
    if isinstance(a, Article):
        return a.to_dict()
    if isinstance(a, dict):
        return a
    return {"id": getattr(a, "id", ""), "content": getattr(a, "content", ""),
            "metadata": getattr(a, "metadata", {}), "relevance_score": getattr(a, "relevance_score", 0.0)}


# ---------------------------------------------------------------------------
# JSON-RPC endpoint
# ---------------------------------------------------------------------------


@app.post("/", include_in_schema=False)
async def jsonrpc_handler(request: Request):
    body = await request.json()
    jsonrpc = body.get("jsonrpc", "2.0")
    method = body.get("method", "")
    params = body.get("params", {})
    rpc_id = body.get("id")

    if method == "tasks/sendMessage":
        response = await _handle_send_message(params)
        return response

    return JSONResponse(
        {
            "jsonrpc": jsonrpc,
            "id": rpc_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        },
        status_code=404,
    )


@app.post("/sendMessage", include_in_schema=False)
async def send_message_direct(request: Request):
    """Convenience endpoint for non-JSON-RPC callers."""
    body = await request.json()
    params = {"id": f"task-{uuid.uuid4().hex[:12]}", "message": body.get("message", {})}
    return await _handle_send_message(params)
