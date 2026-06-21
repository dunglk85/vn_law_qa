# app/api.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.auth.dependencies import require_role
from app.auth.router import router as auth_router
from app.config import config
from app.core.agentic_service import create_agentic_service
from app.core.ingest_service import run_ingest
from app.core.rag_service import RAGService
from app.factory import (
    create_cache,
    create_chunker,
    create_embeddings,
    create_llm,
    create_metadata_enricher,
    create_query_transformer,
    create_rate_limiter,
    create_reranker,
    create_retriever,
    create_session_store,
    create_vector_store,
)

logger = logging.getLogger(__name__)

_INGEST_API_KEY = os.getenv("INGEST_API_KEY", "")

# --------------------------------------------------------------------------- #
# App bootstrap                                                                #
# --------------------------------------------------------------------------- #

app = FastAPI(title="Company Knowledge Assistant")
app.include_router(auth_router)

# Static frontend
static_dir = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# --------------------------------------------------------------------------- #
# Dependency wiring (done once at startup via factory)                         #
# --------------------------------------------------------------------------- #

# Shared embeddings instance — reused by both vector_store and cache
# so embeddings are never created twice with different configs.
_embeddings = create_embeddings()
_vector_store = create_vector_store(embeddings=_embeddings)
_llm = create_llm()
_reranker = create_reranker(embeddings=_embeddings)
_cache = create_cache(embeddings=_embeddings)
_chunker = create_chunker(embeddings=_embeddings)
_retriever = create_retriever(vector_store=_vector_store)
_query_transformer = create_query_transformer(llm=_llm)
_enricher = create_metadata_enricher(llm=_llm)

# Activate LLM cache (e.g. Redis semantic cache, or no-op)
try:
    _cache.apply()
except Exception as e:
    logger.warning("CACHE: Failed to activate cache (%s). Running without cache.", e)

# Rate limiter (Redis-backed with in-memory fallback)
_rate_limiter = create_rate_limiter()

# Session store (Redis-backed with in-memory fallback)
_session_store = create_session_store()

# RAGService: business logic with injected dependencies
_rag_service = RAGService(
    vector_store=_vector_store,
    llm=_llm,
    reranker=_reranker,
    retriever=_retriever,
    query_transformer=_query_transformer,
)

# AgenticService: optional alternative using LangGraph agents
_agentic_service = None
_VALID_RAG_MODES = {"legacy", "agentic"}
if config.rag_mode.lower() not in _VALID_RAG_MODES:
    logger.warning("Unknown RAG_MODE='%s'. Falling back to legacy. Valid: %s", config.rag_mode, _VALID_RAG_MODES)
if config.rag_mode.lower() == "agentic":
    _agentic_service = create_agentic_service(
        vector_store=_vector_store,
        llm=_llm,
        retriever=_retriever,
        query_transformer=_query_transformer,
        session_store=_session_store,
    )
    logger.info("RAG_MODE=agentic: AgenticService initialized")
else:
    logger.info("RAG_MODE=legacy: RAGService initialized (default)")

# --------------------------------------------------------------------------- #
# Ingestion state                                                              #
# --------------------------------------------------------------------------- #

_ingest_lock = asyncio.Lock()
_ingest_task: asyncio.Task | None = None
_ingest_last: dict = {
    "status": "idle",       # idle | running | succeeded | failed
    "started_at": None,
    "finished_at": None,
    "stats": None,          # {"documents": ..., "chunks": ...}
    "error": None,
}


async def _ingest_job() -> None:
    _ingest_last.update({
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "stats": None,
        "error": None,
    })
    try:
        stats = await run_ingest(_vector_store, _chunker, _retriever, _enricher)
        _ingest_last.update({"status": "succeeded", "finished_at": time.time(), "stats": stats})
    except Exception as e:
        _ingest_last.update({"status": "failed", "finished_at": time.time(), "error": str(e)})


# --------------------------------------------------------------------------- #
# Request models                                                               #
# --------------------------------------------------------------------------- #

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    category: str | None = None
    session_id: str | None = None


# --------------------------------------------------------------------------- #
# Endpoints                                                                    #
# --------------------------------------------------------------------------- #

@app.get("/")
async def root_page() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest", response_model=None)
async def kick_off_ingest(
    _user: dict = Depends(require_role("admin")),
    x_api_key: str | None = Header(None),
):
    if _INGEST_API_KEY and x_api_key != _INGEST_API_KEY:
        return JSONResponse(
            {"ok": False, "message": "Invalid or missing API key"},
            status_code=401,
        )
    global _ingest_task
    async with _ingest_lock:
        if _ingest_task and not _ingest_task.done():
            return JSONResponse(
                {"ok": False, "message": "Ingestion already running"},
                status_code=409,
            )
        _ingest_task = asyncio.create_task(_ingest_job())
    return {"ok": True, "message": "Ingestion started"}


@app.get("/ingest/status")
async def ingest_status() -> dict:
    return {"ok": True, **_ingest_last}


def _get_client_ip(request: Request) -> str:
    """Extract client IP, respecting X-Forwarded-For behind trusted proxies."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


@app.post("/ask")
async def ask(
    q: AskRequest,
    request: Request,
    _user: dict = Depends(require_role("admin", "viewer")),
) -> dict:
    client_ip = _get_client_ip(request)
    if not await _rate_limiter.check(client_ip):
        return JSONResponse(
            {"ok": False, "message": "Rate limit exceeded. Try again later."},
            status_code=429,
        )

    start = time.perf_counter()
    logger.info("POST /ask question=%.80s client=%s", q.question, client_ip)

    tenant_id = _user.get("tenant_id") if isinstance(_user, dict) else None
    session_id = q.session_id or _user.get("sub", "anonymous") if isinstance(_user, dict) else "anonymous"

    try:
        async with asyncio.timeout(config.ask_timeout):
            if _agentic_service:
                answer, sources, contexts, reasoning_steps = await _agentic_service.answer(
                    question=q.question,
                    category=q.category,
                    tenant_id=tenant_id,
                    session_id=session_id,
                )
            else:
                answer, sources, contexts = await _rag_service.answer(
                    question=q.question,
                    category=q.category,
                    tenant_id=tenant_id,
                )
                reasoning_steps = []
    except TimeoutError:
        logger.error("/ask timed out after %.0fs", config.ask_timeout)
        return JSONResponse(
            {"ok": False, "message": "Request timed out. Please try again."},
            status_code=504,
        )

    elapsed = time.perf_counter() - start
    logger.info("/ask completed in %.2fs client=%s", elapsed, client_ip)

    response: dict = {
        "answer": answer,
        "sources": sources,
        "contexts": contexts,
    }

    if reasoning_steps:
        trace_json = json.dumps(reasoning_steps, default=str)
        if len(trace_json) > 102400:
            trace_id = str(uuid.uuid4())
            logger.info("Reasoning trace truncated, full trace_id=%s", trace_id)
            response["reasoning_trace"] = reasoning_steps[:3]
            response["reasoning_trace_truncated"] = True
            response["trace_id"] = trace_id
        else:
            response["reasoning_trace"] = reasoning_steps

    return response
