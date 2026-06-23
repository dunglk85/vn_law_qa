"""Standalone A2A JSON-RPC server wrapping the CitationCheckerAgent.

Usage:
    uvicorn app.agents.a2a_servers.citation_checker_server:app --port 8102

Requires env vars: VECTOR_STORE_TYPE, DATABASE_URL, LLM_TYPE, LLM_MODEL, etc.
"""
from __future__ import annotations

import json
import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.core.models import Article
from app.factory import create_embeddings, create_llm, create_vector_store

# Module-level lazy cache
_embed = None
_vstore = None
_llm_inst = None
_citation = None

logger = logging.getLogger(__name__)

app = FastAPI(title="Citation Checker A2A Agent")


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "citation-checker-agent",
    "description": "Three-gate hallucination firewall: existence verification, relevance scoring, LLM contradiction detection.",
    "version": "1.0.0",
    "capabilities": {
        "streaming": True,
        "pushNotifications": False,
        "stateful": True,
    },
    "skills": [
        {
            "id": "citation_check",
            "name": "Citation Verification",
            "description": "Verifies legal citations against the vector store for existence, relevance, and consistency.",
            "tags": ["legal", "citation", "verification", "hallucination"],
            "inputs": [
                {"name": "articles", "type": "array", "description": "List of Article objects to verify"},
                {"name": "query", "type": "string", "description": "Original query for cross-reference context"},
            ],
            "outputs": [
                {"name": "citations", "type": "array", "description": "Verified Citation objects that passed all gates"}
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
    return {"status": "ok", "agent": "citation-checker-agent", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Agent instantiation (lazy)
# ---------------------------------------------------------------------------


def _get_agent():
    global _embed, _vstore, _llm_inst, _citation
    if _citation is None:
        _embed = create_embeddings()
        _vstore = create_vector_store(embeddings=_embed)
        _llm_inst = create_llm()
        from app.agents.citation_checker_agent import CitationCheckerAgent

        _citation = CitationCheckerAgent(_vstore, _llm_inst.get_chat_model())
    return _citation


# ---------------------------------------------------------------------------
# JSON-RPC Methods
# ---------------------------------------------------------------------------


async def _handle_send_message(params: dict) -> EventSourceResponse:
    task_id = params.get("id", f"task-{uuid.uuid4().hex[:12]}")
    message = params.get("message", {})
    parts = message.get("parts", [])

    articles: list[Article] = []
    query = ""

    for part in parts:
        if part.get("type") == "data" and isinstance(part.get("data"), dict):
            data = part["data"]
            query = data.get("query", "")
            raw_articles = data.get("articles", [])
            for raw in raw_articles:
                if isinstance(raw, dict):
                    articles.append(Article(
                        id=raw.get("id", ""),
                        content=raw.get("content", ""),
                        metadata=raw.get("metadata", {}),
                        relevance_score=raw.get("relevance_score", 0.0),
                    ))
            break
        if part.get("type") == "text":
            try:
                data = json.loads(part["text"])
                query = data.get("query", "")
                raw_articles = data.get("articles", [])
                for raw in raw_articles:
                    if isinstance(raw, dict):
                        articles.append(Article(
                            id=raw.get("id", ""),
                            content=raw.get("content", ""),
                            metadata=raw.get("metadata", {}),
                            relevance_score=raw.get("relevance_score", 0.0),
                        ))
            except (json.JSONDecodeError, TypeError):
                query = part["text"]

    if not articles:
        return EventSourceResponse(
            _error_stream(task_id, "No articles provided in message")
        )

    async def event_stream():
        yield {"event": "task_status", "data": json.dumps({
            "id": task_id,
            "status": {"state": "working", "timestamp": _now()},
        })}

        try:
            citations = await _get_agent().run(articles, query)
            citations_dicts = [c.to_dict() for c in citations]

            yield {"event": "task_status", "data": json.dumps({
                "id": task_id,
                "status": {"state": "completed", "timestamp": _now()},
            })}
            yield {"event": "task_artifact", "data": json.dumps({
                "id": task_id,
                "artifact": {"citations": citations_dicts},
            })}
        except Exception as exc:
            logger.exception("Citation check failed")
            yield {"event": "task_status", "data": json.dumps({
                "id": task_id,
                "status": {"state": "failed", "timestamp": _now(), "error": str(exc)},
            })}

    return EventSourceResponse(event_stream())


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


async def _error_stream(task_id: str, message: str):
    yield {"event": "task_status", "data": json.dumps({
        "id": task_id,
        "status": {"state": "failed", "timestamp": _now(), "error": message},
    })}


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
