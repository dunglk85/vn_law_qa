from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.tools import tool

from app.ports.retriever import RetrieverPort

logger = logging.getLogger(__name__)


def create_knowledge_search_tool(retriever_port: RetrieverPort, k: int = 5):
    """Factory: returns a LangChain @tool bound to a specific RetrieverPort."""
    _cached_retriever = None

    @tool
    async def knowledge_search(query: str) -> list[dict]:
        """Search the knowledge base for documents relevant to the query.

        Args:
            query: Natural language search query.
        """
        nonlocal _cached_retriever
        if _cached_retriever is None:
            _cached_retriever = retriever_port.get_retriever(search_kwargs={"k": k})
        retriever = _cached_retriever
        try:
            docs: list[Document] = await retriever.ainvoke(query)
        except Exception as exc:
            logger.error("knowledge_search tool failed: %s", exc)
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

    return knowledge_search
