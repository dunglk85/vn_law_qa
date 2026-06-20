"""Agentic RAG orchestration service.

This module wires the supervisor workflow and generic adapters together
without depending on any concrete backend implementation.
"""
from __future__ import annotations
import logging
from typing import List, Optional, Tuple

from app.config import config
from app.ports.llm import LLMPort
from app.ports.query_transformer import QueryTransformerPort
from app.ports.retriever import RetrieverPort
from app.ports.vector_store import VectorStorePort
from app.agents.legal_research_agent import LegalResearchAgent
from app.agents.citation_checker_agent import CitationCheckerAgent
from app.agents.response_synthesizer_agent import ResponseSynthesizerAgent
from app.agents.supervisor_agent import SupervisorAgent

logger = logging.getLogger(__name__)

_NO_CONTEXT_ANSWER = "I don't know. No relevant context was found to answer your question."


class AgenticService:
    """Orchestrates an agentic RAG workflow with injected ports."""

    def __init__(
        self,
        vector_store: VectorStorePort,
        llm: LLMPort,
        retriever: RetrieverPort,
        query_transformer: QueryTransformerPort,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm
        self._retriever = retriever
        self._query_transformer = query_transformer
        chat_model = llm.get_chat_model()

        self._supervisor = SupervisorAgent(
            research_agent=LegalResearchAgent(self._retriever, chat_model),
            citation_agent=CitationCheckerAgent(self._vector_store, chat_model),
            synthesis_agent=ResponseSynthesizerAgent(chat_model),
            llm=chat_model,
        )
        self._warmed_up = False

    async def _ensure_warmup(self) -> None:
        if not self._warmed_up:
            try:
                await self._vector_store.similarity_search("warmup", k=1)
            except Exception as exc:
                logger.warning("Warmup query failed (non-fatal): %s", exc)
            self._warmed_up = True

    async def answer(
        self,
        question: str,
        category: Optional[str] = None,
        user_id: str = "anonymous",
        session_id: str = "default-session",
    ) -> Tuple[str, List[str], List[str]]:
        await self._ensure_warmup()

        transformed_queries = await self._query_transformer.transform(question)
        question_for_agents = transformed_queries[0] if transformed_queries else question

        result = await self._supervisor.run(
            question_for_agents,
            user_id=user_id,
            session_id=session_id,
            metadata={"category": category} if category else {},
        )

        answer = result.get("final_response") or _NO_CONTEXT_ANSWER
        citations = result.get("verified_citations") or []
        sources = sorted({c.article_id for c in citations})
        contexts = [c.content for c in citations]

        return answer, sources, contexts
