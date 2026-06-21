"""MCP server exposing knowledge_search as an MCP tool.

Runs as a subprocess (stdio transport) or standalone (SSE transport).
Configuration is read from the same env vars as the main app.
"""
from __future__ import annotations

import logging
import os
import sys

from mcp.server.fastmcp import FastMCP

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

server = FastMCP("knowledge-search")


def _build_retriever():
    """Lazy-import the retriever adapter from the main app package."""
    from app.config import config
    from app.factory import create_embeddings, create_retriever, create_vector_store

    embeddings = create_embeddings()
    vector_store = create_vector_store(embeddings=embeddings)
    retriever = create_retriever(vector_store=vector_store)
    return retriever, config.retrieval_k


_retriever = None
_k = 5


def _get_retriever():
    global _retriever, _k
    if _retriever is None:
        _retriever, _k = _build_retriever()
    return _retriever, _k


@server.tool()
async def knowledge_search(query: str, k: int | None = None) -> list[dict]:
    """Search the knowledge base for documents relevant to the query.

    Args:
        query: Natural language search query.
        k: Number of results to return (default: configured RETRIEVAL_K).
    """
    retriever_port, default_k = _get_retriever()
    top_k = k if k is not None else default_k
    retriever = retriever_port.get_retriever(search_kwargs={"k": top_k})
    try:
        from langchain_core.documents import Document

        docs: list[Document] = await retriever.ainvoke(query)
    except Exception as exc:
        logger.error("knowledge_search failed: %s", exc)
        return []

    results: list[dict] = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown") if doc.metadata else "unknown"
        results.append({
            "content": doc.page_content,
            "source": source,
            "score": doc.metadata.get("score", None) if doc.metadata else None,
        })
    return results


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "stdio":
        server.run(transport="stdio")
    else:
        server.run(transport="sse")
