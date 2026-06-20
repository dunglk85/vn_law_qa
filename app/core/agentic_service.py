"""Agentic RAG orchestration service.

This module wires the supervisor workflow and generic adapters together
without depending on any concrete backend implementation.
"""
from __future__ import annotations

import asyncio
import logging
import time

from app.config import config
from app.ports.llm import LLMPort
from app.ports.query_transformer import QueryTransformerPort
from app.ports.retriever import RetrieverPort
from app.ports.session_store import SessionStorePort
from app.ports.vector_store import VectorStorePort

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
        supervisor=None,
        session_store: SessionStorePort | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm
        self._retriever = retriever
        self._query_transformer = query_transformer
        self._session_store = session_store

        if supervisor is not None:
            self._supervisor = supervisor
        else:
            from app.agents.citation_checker_agent import CitationCheckerAgent
            from app.agents.legal_research_agent import LegalResearchAgent
            from app.agents.response_synthesizer_agent import ResponseSynthesizerAgent
            from app.agents.supervisor_agent import SupervisorAgent
            from app.agents.tools.knowledge_search import create_knowledge_search_tool

            chat_model = llm.get_chat_model()
            knowledge_search_tool = create_knowledge_search_tool(retriever, k=config.retrieval_k)
            self._supervisor = SupervisorAgent(
                research_agent=LegalResearchAgent(self._retriever, chat_model),
                citation_agent=CitationCheckerAgent(self._vector_store, chat_model),
                synthesis_agent=ResponseSynthesizerAgent(chat_model),
                llm=chat_model,
                knowledge_search_tool=knowledge_search_tool,
            )
        self._warmed_up = False

    async def _ensure_warmup(self) -> None:
        if not self._warmed_up:
            try:
                await self._vector_store.similarity_search("warmup", k=1)
            except Exception as exc:
                logger.warning("Warmup query failed (non-fatal): %s", exc)
            self._warmed_up = True

    async def _load_session(self, session_id: str) -> dict:
        if self._session_store is None:
            return {"history": [], "summary": ""}
        try:
            return await self._session_store.load(session_id)
        except Exception as exc:
            logger.warning("Session store unavailable, running stateless: %s", exc)
            return {"history": [], "summary": ""}

    async def _save_session(self, session_id: str, session_data: dict) -> None:
        if self._session_store is None:
            return
        try:
            await self._session_store.save(session_id, session_data)
        except Exception as exc:
            logger.warning("Failed to save session: %s", exc)

    def _estimate_tokens(self, texts: list[str]) -> int:
        total = 0
        for t in texts:
            total += len(t) // 4
        return total

    def _summarize_with_llm(self, texts: list[str]) -> str:
        chat_model = self._llm.get_chat_model()
        prompt = (
            "Summarize the following conversation, preserving key facts, "
            "decisions, user preferences, and any specific data mentioned. "
            "Keep the summary concise but informative.\n\n"
            "---\n"
        )
        for t in texts:
            prompt += f"{t}\n"
        prompt += "\n---\nSummary:"
        result = chat_model.invoke(prompt)
        return result.content.strip()

    def _compress_history(
        self,
        history: list[dict],
        existing_summary: str,
    ) -> tuple[list[dict], str]:
        recent_count = config.recent_turns_to_keep * 2

        if len(history) <= recent_count:
            return history, existing_summary

        older = history[:-recent_count]
        recent = history[-recent_count:]

        older_texts = []
        for turn in older:
            role = turn.get("role", "user")
            content = turn.get("content", "")
            older_texts.append(f"{role}: {content}")

        combined_text = existing_summary + "\n" + "\n".join(older_texts) if existing_summary else "\n".join(older_texts)
        token_count = self._estimate_tokens([combined_text])

        if token_count > config.max_history_tokens:
            try:
                new_summary = self._summarize_with_llm([combined_text])
                logger.info("History compressed: %d older turns summarized", len(older))
            except Exception as exc:
                logger.warning("Summarization failed, keeping raw history: %s", exc)
                return history, existing_summary
        else:
            new_summary = combined_text

        return recent, new_summary

    async def answer(
        self,
        question: str,
        category: str | None = None,
        tenant_id: str | None = None,
        user_id: str = "anonymous",
        session_id: str = "default-session",
    ) -> tuple[str, list[str], list[str]]:
        await self._ensure_warmup()

        session_data = await self._load_session(session_id)
        history = session_data.get("history", [])
        summary = session_data.get("summary", "")

        history, summary = self._compress_history(history, summary)

        transformed_queries = await self._query_transformer.transform(question)
        question_for_agents = transformed_queries[0] if transformed_queries else question

        metadata: dict = {}
        if category:
            metadata["category"] = category
        if tenant_id and tenant_id != "*":
            metadata["tenant_id"] = tenant_id

        summary_context = f"[Previous conversation summary]\n{summary}" if summary else ""

        try:
            result = await asyncio.wait_for(
                self._supervisor.run(
                    question_for_agents,
                    user_id=user_id,
                    session_id=session_id,
                    metadata=metadata,
                    conversation_history=history,
                    summary_context=summary_context,
                ),
                timeout=config.agent_timeout,
            )
        except TimeoutError:
            logger.error("Supervisor timed out after %.0fs", config.agent_timeout)
            return _NO_CONTEXT_ANSWER, [], []

        answer = result.get("final_response") or _NO_CONTEXT_ANSWER
        citations = result.get("verified_citations") or []
        sources = sorted({c.article_id for c in citations})
        contexts = [c.content for c in citations]

        history.append({"role": "user", "content": question, "timestamp": time.time()})
        history.append({"role": "assistant", "content": answer, "timestamp": time.time()})
        await self._save_session(session_id, {"history": history, "summary": summary})

        return answer, sources, contexts


def create_agentic_service(
    vector_store: VectorStorePort,
    llm: LLMPort,
    retriever: RetrieverPort,
    query_transformer: QueryTransformerPort,
    session_store: SessionStorePort | None = None,
) -> AgenticService:
    return AgenticService(vector_store, llm, retriever, query_transformer, session_store=session_store)
