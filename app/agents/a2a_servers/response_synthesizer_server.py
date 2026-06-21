"""Standalone A2A JSON-RPC server wrapping the ResponseSynthesizerAgent.

Usage:
    uvicorn app.agents.a2a_servers.response_synthesizer_server:app --port 8103

Requires env vars: LLM_TYPE, LLM_MODEL, etc.
"""
from __future__ import annotations

import json
import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from app.core.models import Citation
from app.factory import create_llm

logger = logging.getLogger(__name__)

app = FastAPI(title="Response Synthesizer A2A Agent")


# ---------------------------------------------------------------------------
# Agent Card
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "response-synthesizer-agent",
    "description": "Generates grounded Vietnamese legal responses from verified citations.",
    "version": "1.0.0",
    "capabilities": {
        "streaming": True,
        "pushNotifications": False,
        "stateful": True,
    },
    "skills": [
        {
            "id": "response_synthesis",
            "name": "Response Synthesis",
            "description": "Synthesizes a natural-language legal response from verified citations.",
            "tags": ["legal", "synthesis", "generation"],
            "inputs": [
                {"name": "query", "type": "string", "description": "Original user question"},
                {"name": "citations", "type": "array", "description": "Verified Citation objects to ground the response"},
            ],
            "outputs": [
                {"name": "response", "type": "string", "description": "Generated legal response in Vietnamese"},
                {"name": "citations", "type": "array", "description": "Citations included in the response"},
                {"name": "metadata", "type": "object", "description": "Metadata about the synthesis (citation count, length)"},
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
    return {"status": "ok", "agent": "response-synthesizer-agent", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Agent instantiation
# ---------------------------------------------------------------------------

_llm = create_llm()

from app.agents.response_synthesizer_agent import ResponseSynthesizerAgent

_synthesis_agent = ResponseSynthesizerAgent(_llm.get_chat_model())


# ---------------------------------------------------------------------------
# JSON-RPC Methods
# ---------------------------------------------------------------------------


async def _handle_send_message(params: dict) -> EventSourceResponse:
    task_id = params.get("id", f"task-{uuid.uuid4().hex[:12]}")
    message = params.get("message", {})
    parts = message.get("parts", [])

    query = ""
    citations: list[Citation] = []

    for part in parts:
        if part.get("type") == "data" and isinstance(part.get("data"), dict):
            data = part["data"]
            query = data.get("query", "")
            raw_citations = data.get("citations", [])
            for raw in raw_citations:
                if isinstance(raw, dict):
                    citations.append(Citation(
                        article_id=raw.get("article_id", ""),
                        content=raw.get("content", ""),
                        relevance=raw.get("relevance", 0.0),
                        verified=raw.get("verified", False),
                    ))
            break
        if part.get("type") == "text":
            try:
                data = json.loads(part["text"])
                query = data.get("query", "")
                raw_citations = data.get("citations", [])
                for raw in raw_citations:
                    if isinstance(raw, dict):
                        citations.append(Citation(
                            article_id=raw.get("article_id", ""),
                            content=raw.get("content", ""),
                            relevance=raw.get("relevance", 0.0),
                            verified=raw.get("verified", False),
                        ))
            except (json.JSONDecodeError, TypeError):
                query = part["text"]

    async def event_stream():
        yield {"event": "task_status", "data": json.dumps({
            "id": task_id,
            "status": {"state": "working", "timestamp": _now()},
        })}

        try:
            result = await _synthesis_agent.synthesize(query, citations)

            yield {"event": "task_status", "data": json.dumps({
                "id": task_id,
                "status": {"state": "completed", "timestamp": _now()},
            })}
            yield {"event": "task_artifact", "data": json.dumps({
                "id": task_id,
                "artifact": result,
            })}
        except Exception as exc:
            logger.exception("Response synthesis failed")
            yield {"event": "task_status", "data": json.dumps({
                "id": task_id,
                "status": {"state": "failed", "timestamp": _now(), "error": str(exc)},
            })}

    return EventSourceResponse(event_stream())


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.UTC if hasattr(timezone, "UTC") else timezone.utc).isoformat()


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
