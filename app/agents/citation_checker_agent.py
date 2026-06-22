"""
CitationCheckerAgent — three-gate hallucination firewall
────────────────────────────────────────────────────────
Gate 1  verify_existence   — article must exist in ChromaDB right now
Gate 2  check_relevance    — blended score ≥ RELEVANCE_THRESHOLD
Gate 3  cross_reference    — LLM contradiction detection between citations

Zero survivors → empty list → supervisor triggers retry_research.
"""
import asyncio
import logging
from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.core.models import (
    RELEVANCE_THRESHOLD,
    Article,
    Citation,
    format_citations,
    llm_ainvoke,
    parse_json,
)
from app.ports.vector_store import VectorStorePort

logger = logging.getLogger(__name__)


class CitationCheckState(TypedDict):
    articles: list[Article]
    query_context: str
    verified_citations: list[Citation]
    invalid_citations: list[str]
    irrelevant_citations: list[str]
    contradictions: list[str]
    consistency_score: float
    gate_log: list[str]


class CitationCheckerAgent:
    def __init__(self, vector_store: VectorStorePort, llm: BaseChatModel | None = None):
        if llm is None:
            raise ValueError("CitationCheckerAgent requires a chat model")
        self.vector_store = vector_store
        self.llm = llm
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        wf = StateGraph(CitationCheckState)
        wf.add_node("verify_existence", self.verify_existence)
        wf.add_node("check_relevance",  self.check_relevance)
        wf.add_node("cross_reference",  self.cross_reference)
        wf.add_edge(START,              "verify_existence")
        wf.add_edge("verify_existence", "check_relevance")
        wf.add_edge("check_relevance",  "cross_reference")
        wf.add_edge("cross_reference",  END)
        return wf.compile()

    # ── gate 1 ─────────────────────────────────────────────────────────────

    async def verify_existence(self, state: CitationCheckState) -> dict:
        verified : list[Citation] = []
        invalid  : list[str]      = []
        gate_log : list[str]      = []

        async def _check(a: Article):
            try:
                docs = await self.vector_store.get_documents_by_ids([a.id])
                if docs:
                    return Citation(article_id=a.id, content=a.content,
                                    relevance=a.relevance_score, verified=True), None
                logger.warning("Gate 1 FAIL — not found in vector store: %s", a.id)
                return None, a.id
            except Exception as exc:
                logger.error("Gate 1 ERROR %s: %s", a.id, exc)
                return None, a.id

        results = await asyncio.gather(*[_check(a) for a in state.get("articles", [])])
        for cit, inv in results:
            if cit:
                verified.append(cit)
            if inv:
                invalid.append(inv)

        gate_log.append(f"Gate 1: {len(verified)} passed, {len(invalid)} failed {invalid}")
        logger.info(gate_log[-1])
        return {"verified_citations": verified, "invalid_citations": invalid,
                "gate_log": gate_log}

    # ── gate 2 ─────────────────────────────────────────────────────────────

    async def check_relevance(self, state: CitationCheckState) -> dict:
        gate_log    = list(state.get("gate_log", []))
        relevant    : list[Citation] = []
        irrelevant  : list[str]      = []
        for c in state.get("verified_citations", []):
            if c.relevance >= RELEVANCE_THRESHOLD:
                relevant.append(c)
            else:
                irrelevant.append(c.article_id)
                logger.info("Gate 2 DROP %s score=%.3f", c.article_id, c.relevance)
        gate_log.append(
            f"Gate 2: {len(relevant)} passed, {len(irrelevant)} dropped {irrelevant}")
        logger.info(gate_log[-1])
        return {"verified_citations": relevant, "irrelevant_citations": irrelevant,
                "gate_log": gate_log}

    # ── gate 3 ─────────────────────────────────────────────────────────────

    async def cross_reference(self, state: CitationCheckState) -> dict:
        citations = state.get("verified_citations", [])
        gate_log  = list(state.get("gate_log", []))
        if len(citations) < 2:
            gate_log.append("Gate 3: skipped (<2 citations)")
            return {"verified_citations": citations, "contradictions": [],
                    "consistency_score": 1.0, "gate_log": gate_log}

        prompt = (
            "Tìm các cặp MÂU THUẪN giữa các điều luật sau "
            "(điều cũ bị thay thế, hoặc hai điều quy định ngược nhau).\n\n"
            f"Câu hỏi: {state.get('query_context','')}\n\n"
            f"{format_citations(citations)}\n\n"
            'JSON: {"has_contradictions": bool, "contradictory_pairs": '
            '[{"keep": "<id>", "remove": "<id>", "reason": "..."}]}'
        )
        try:
            r    = await llm_ainvoke(self.llm, prompt, call_name="cross_reference")
            data = parse_json(r.content, "cross_reference")
        except Exception as exc:
            logger.error("Gate 3 LLM failed: %s", exc)
            gate_log.append("Gate 3: LLM error, skipped")
            return {"verified_citations": citations, "contradictions": [],
                    "consistency_score": 1.0, "gate_log": gate_log}

        to_remove: set[str] = set()
        contradictions: list[str] = []
        if data.get("has_contradictions"):
            for pair in data.get("contradictory_pairs", []):
                rid = pair.get("remove")
                if rid:
                    to_remove.add(rid)
                    contradictions.append(rid)
                    logger.warning("Gate 3 REMOVE %s: %s", rid, pair.get("reason",""))

        survivors = [c for c in citations if c.article_id not in to_remove]
        score     = round(len(survivors) / len(citations), 3) if citations else 0.0
        gate_log.append(
            f"Gate 3: {len(survivors)}/{len(citations)} survived, removed {list(to_remove)}")
        logger.info(gate_log[-1])
        return {"verified_citations": survivors, "contradictions": contradictions,
                "consistency_score": score, "gate_log": gate_log}

    # ── public API ─────────────────────────────────────────────────────────

    async def run(self, articles: list[Article], query: str) -> list[Citation]:
        result = await self.workflow.ainvoke({
            "articles": articles, "query_context": query,
            "verified_citations": [], "invalid_citations": [],
            "irrelevant_citations": [], "contradictions": [],
            "consistency_score": 0.0, "gate_log": [],
        })
        logger.info("CitationChecker: %d passed. Log: %s",
                    len(result["verified_citations"]),
                    " | ".join(result["gate_log"]))
        return result["verified_citations"]
