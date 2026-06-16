# app/api.py
from __future__ import annotations
import asyncio
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.factory import create_cache, create_embeddings, create_llm, create_reranker, create_vector_store
from app.core.ingest_service import run_ingest
from app.core.rag_service import RAGService

# --------------------------------------------------------------------------- #
# App bootstrap                                                                #
# --------------------------------------------------------------------------- #

app = FastAPI(title="Company Knowledge Assistant")

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
_reranker = create_reranker()
_cache = create_cache(embeddings=_embeddings)

# Activate LLM cache (e.g. Redis semantic cache, or no-op)
_cache.apply()

# RAGService: business logic with injected dependencies
_rag_service = RAGService(
    vector_store=_vector_store,
    llm=_llm,
    reranker=_reranker,
)

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
        stats = await run_ingest(_vector_store)
        _ingest_last.update({"status": "succeeded", "finished_at": time.time(), "stats": stats})
    except Exception as e:
        _ingest_last.update({"status": "failed", "finished_at": time.time(), "error": str(e)})


# --------------------------------------------------------------------------- #
# Request models                                                               #
# --------------------------------------------------------------------------- #

class AskRequest(BaseModel):
    question: str
    category: str | None = None


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
async def kick_off_ingest():
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


@app.post("/ask")
async def ask(q: AskRequest) -> dict:
    start = time.perf_counter()

    answer, sources, contexts = await _rag_service.answer(
        question=q.question,
        category=q.category,
    )

    elapsed = time.perf_counter() - start
    print(f"⏱️  /ask took {elapsed:.2f}s")

    return {
        "answer": answer,
        "sources": sources,
        "contexts": contexts,
    }