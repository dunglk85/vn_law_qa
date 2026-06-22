# app/api.py
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
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
from app.exceptions import AppError
from app.factory import (
    create_a2a_client,
    create_cache,
    create_chunker,
    create_citation_checker_agent,
    create_embeddings,
    create_knowledge_search_tool,
    create_legal_research_agent,
    create_llm,
    create_metadata_enricher,
    create_query_transformer,
    create_rate_limiter,
    create_reranker,
    create_response_synthesizer_agent,
    create_retriever,
    create_session_store,
    create_supervisor_agent,
    create_vector_store,
)

logger = logging.getLogger(__name__)

_INGEST_API_KEY = os.getenv("INGEST_API_KEY", "")
_VALID_RAG_MODES = {"legacy", "agentic"}

# --------------------------------------------------------------------------- #
# Lifespan context manager (startup/shutdown)                                  #
# --------------------------------------------------------------------------- #


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle: startup wiring and shutdown cleanup."""
    logger.info("Starting up: initializing dependencies")

    try:
        embeddings = create_embeddings()
        app.state.vector_store = create_vector_store(embeddings=embeddings)
        app.state.llm = create_llm()
        app.state.reranker = create_reranker(embeddings=embeddings)
        app.state.cache = create_cache(embeddings=embeddings)
        app.state.chunker = create_chunker(embeddings=embeddings)
        app.state.retriever = create_retriever(vector_store=app.state.vector_store)
        app.state.query_transformer = create_query_transformer(llm=app.state.llm)
        app.state.enricher = create_metadata_enricher(llm=app.state.llm)

        try:
            app.state.cache.apply()
        except Exception as e:
            logger.warning("CACHE: Failed to activate cache (%s). Running without cache.", e)

        app.state.rate_limiter = create_rate_limiter()
        app.state.session_store = create_session_store()

        app.state.rag_service = RAGService(
            vector_store=app.state.vector_store,
            llm=app.state.llm,
            reranker=app.state.reranker,
            retriever=app.state.retriever,
            query_transformer=app.state.query_transformer,
        )

        if config.rag_mode.lower() not in _VALID_RAG_MODES:
            logger.warning(
                "Unknown RAG_MODE='%s'. Falling back to legacy. Valid: %s",
                config.rag_mode, _VALID_RAG_MODES
            )

        app.state.knowledge_search_tool = create_knowledge_search_tool(retriever_port=app.state.retriever)

        if config.rag_mode.lower() == "agentic":
            research_agent = create_legal_research_agent(retriever=app.state.retriever, llm=app.state.llm)
            citation_agent = create_citation_checker_agent(vector_store=app.state.vector_store, llm=app.state.llm)
            synthesis_agent = create_response_synthesizer_agent(llm=app.state.llm)
            a2a_client = create_a2a_client(
                research_agent=research_agent,
                citation_agent=citation_agent,
                synthesis_agent=synthesis_agent,
            )
            supervisor = create_supervisor_agent(
                research_agent=research_agent,
                citation_agent=citation_agent,
                synthesis_agent=synthesis_agent,
                llm=app.state.llm,
                knowledge_search_tool=app.state.knowledge_search_tool,
                a2a_client=a2a_client,
            )
            app.state.agentic_service = create_agentic_service(
                vector_store=app.state.vector_store,
                llm=app.state.llm,
                retriever=app.state.retriever,
                query_transformer=app.state.query_transformer,
                supervisor=supervisor,
                session_store=app.state.session_store,
            )
            logger.info("RAG_MODE=agentic: AgenticService initialized with A2A client")
        else:
            app.state.agentic_service = None
            logger.info("RAG_MODE=legacy: RAGService initialized (default)")

        logger.info("Startup complete")

    except Exception as exc:
        logger.error("Startup failed: %s", exc, exc_info=True)
        raise

    yield

    logger.info("Shutting down: cleaning up resources")
    if hasattr(app.state, "knowledge_search_tool") and hasattr(app.state.knowledge_search_tool, "close"):
        try:
            await app.state.knowledge_search_tool.close()
        except Exception as exc:
            logger.warning("Failed to close MCP tool: %s", exc)


# --------------------------------------------------------------------------- #
# App bootstrap                                                                #
# --------------------------------------------------------------------------- #

app = FastAPI(title="Company Knowledge Assistant", lifespan=lifespan)
app.include_router(auth_router)


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError):
    """Handle AppError exceptions with structured JSON responses."""
    logger.error(
        "AppError: %s (status=%d, details=%s)",
        exc.message,
        exc.status_code,
        exc.details,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "ok": False,
            "error": exc.message,
            "details": exc.details,
        },
    )


static_dir = Path(__file__).with_name("static")
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# --------------------------------------------------------------------------- #
# Ingestion state                                                              #
# --------------------------------------------------------------------------- #

_ingest_lock = asyncio.Lock()
_ingest_task: asyncio.Task | None = None
_ingest_last: dict = {
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "stats": None,
    "error": None,
}


async def _ingest_job(vector_store, chunker, retriever, enricher, tenant_id=None) -> None:
    _ingest_last.update({
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "stats": None,
        "error": None,
    })
    try:
        stats = await run_ingest(vector_store, chunker, retriever, enricher, tenant_id=tenant_id)
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
    request: Request,
    _user: dict = Depends(require_role("admin")),
    x_api_key: str | None = Header(None),
):
    if _INGEST_API_KEY and x_api_key != _INGEST_API_KEY:
        return JSONResponse(
            {"ok": False, "message": "Invalid or missing API key"},
            status_code=401,
        )
    global _ingest_task
    tenant_id = _user.get("tenant_id") if isinstance(_user, dict) else None
    async with _ingest_lock:
        if _ingest_task and not _ingest_task.done():
            return JSONResponse(
                {"ok": False, "message": "Ingestion already running"},
                status_code=409,
            )
        _ingest_task = asyncio.create_task(
            _ingest_job(
                request.app.state.vector_store,
                request.app.state.chunker,
                request.app.state.retriever,
                request.app.state.enricher,
                tenant_id=tenant_id,
            )
        )
    return {"ok": True, "message": "Ingestion started"}


@app.get("/ingest/status")
async def ingest_status() -> dict:
    return {"ok": True, **_ingest_last}


def _get_client_ip(request: Request) -> str:
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
    from app.core.token_tracker import reset_tracker

    client_ip = _get_client_ip(request)
    if not await request.app.state.rate_limiter.check(client_ip):
        return JSONResponse(
            {"ok": False, "message": "Rate limit exceeded. Try again later."},
            status_code=429,
        )

    trace_id = str(uuid.uuid4())
    start = time.perf_counter()
    logger.info("POST /ask trace_id=%s question=%.80s client=%s", trace_id, q.question, client_ip)

    tracker = reset_tracker()

    tenant_id = _user.get("tenant_id") if isinstance(_user, dict) else None
    session_id = q.session_id or _user.get("sub", "anonymous") if isinstance(_user, dict) else "anonymous"

    quality_score = None
    try:
        async with asyncio.timeout(config.ask_timeout):
            if request.app.state.agentic_service:
                answer, sources, contexts, reasoning_steps = await request.app.state.agentic_service.answer(
                    question=q.question,
                    category=q.category,
                    tenant_id=tenant_id,
                    session_id=session_id,
                )
                if reasoning_steps:
                    for step in reversed(reasoning_steps):
                        if step.get("action") == "validate_quality":
                            output = step.get("output", "")
                            if "quality_score=" in output:
                                try:
                                    quality_score = float(output.split("quality_score=")[1].split(",")[0])
                                except (ValueError, IndexError):
                                    pass
                            break
            else:
                answer, sources, contexts, tenant_sources = await request.app.state.rag_service.answer(
                    question=q.question,
                    category=q.category,
                    tenant_id=tenant_id,
                )
                reasoning_steps = []
    except TimeoutError:
        elapsed = time.perf_counter() - start
        logger.error(
            "/ask timed out trace_id=%s after %.0fs tokens=%d",
            trace_id,
            config.ask_timeout,
            tracker.total_tokens,
        )
        return JSONResponse(
            {"ok": False, "message": "Request timed out. Please try again."},
            status_code=504,
        )

    elapsed = time.perf_counter() - start
    logger.info(
        "/ask completed trace_id=%s latency=%.2fs tokens=%d llm_calls=%d quality=%.2f client=%s",
        trace_id,
        elapsed,
        tracker.total_tokens,
        tracker.llm_call_count,
        quality_score or 0.0,
        client_ip,
    )

    response: dict = {
        "answer": answer,
        "sources": sources,
        "contexts": contexts,
    }

    if tenant_sources:
        response["tenant_sources"] = tenant_sources

    if reasoning_steps:
        trace_json = json.dumps(reasoning_steps, default=str)
        if len(trace_json) > 102400:
            logger.info("Reasoning trace truncated trace_id=%s", trace_id)
            response["reasoning_trace"] = reasoning_steps[:3]
            response["reasoning_trace_truncated"] = True
            response["trace_id"] = trace_id
        else:
            response["reasoning_trace"] = reasoning_steps

    return response
