"""
LegalResearchAgent
──────────────────
Pipeline:
  prepare_vectors  (parallel: HyDE + sub-query decomposition)
    → retrieve_articles  (parallel ChromaDB, deduplicated)
    → rank_results       (parallel LLM re-ranker, blended score)
"""
import asyncio
import hashlib
import logging
from typing import Optional, TypedDict, Any

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.ports.retriever import RetrieverPort
from app.core.models import (
    Article, llm_ainvoke, parse_json, parse_list,
    N_RESULTS_PER_VECTOR, TOP_K_RESEARCH, TOP_K_LLM_SCORE,
    HYDE_ENABLED, SUBQUERY_COUNT,
)

logger = logging.getLogger(__name__)


class LegalResearchState(TypedDict):
    query: str
    hyde_document: str
    decomposed_queries: list[str]
    query_vectors: list[str]
    retrieved_articles: list[Article]
    ranked_articles: list[Article]
    metadata: dict[str, Any]


class LegalResearchAgent:
    def __init__(self, retriever: RetrieverPort, llm: Optional[BaseChatModel] = None):
        if llm is None:
            raise ValueError("LegalResearchAgent requires a chat model")
        self.retriever = retriever
        self.llm = llm
        self.workflow = self._build_workflow()

    # ── graph ──────────────────────────────────────────────────────────────

    def _build_workflow(self) -> StateGraph:
        wf = StateGraph(LegalResearchState)
        wf.add_node("prepare_vectors",   self.prepare_vectors)
        wf.add_node("retrieve_articles", self.retrieve_articles)
        wf.add_node("rank_results",      self.rank_results)
        wf.add_edge(START,               "prepare_vectors")
        wf.add_edge("prepare_vectors",   "retrieve_articles")
        wf.add_edge("retrieve_articles", "rank_results")
        wf.add_edge("rank_results",      END)
        return wf.compile()

    # ── node 1: HyDE + decompose (parallel) ────────────────────────────────

    async def prepare_vectors(self, state: LegalResearchState) -> dict:
        query = state["query"]
        hyde_text, sub_queries = await asyncio.gather(
            self._generate_hyde(query),
            self._decompose_query(query),
        )
        vectors: list[str] = [query]
        if hyde_text and HYDE_ENABLED:
            vectors.append(hyde_text)
        vectors.extend(q for q in sub_queries if q and q != query)
        logger.info("prepare_vectors: %d vectors", len(vectors))
        return {"hyde_document": hyde_text, "decomposed_queries": sub_queries,
                "query_vectors": vectors}

    async def _generate_hyde(self, query: str) -> str:
        prompt = (
            "Bạn là chuyên gia pháp luật Việt Nam. Viết một đoạn điều luật ngắn "
            "(3-5 câu) trả lời CHÍNH XÁC câu hỏi sau. Chỉ viết nội dung điều luật.\n\n"
            f"Câu hỏi: {query}\n\nĐiều luật giả định:"
        )
        try:
            r = await llm_ainvoke(self.llm, prompt)
            return r.content.strip()
        except Exception as exc:
            logger.warning("HyDE failed: %s", exc)
            return ""

    async def _decompose_query(self, query: str) -> list[str]:
        prompt = (
            f"Phân tách câu hỏi pháp luật sau thành {SUBQUERY_COUNT} câu hỏi con "
            "độc lập, mỗi câu tập trung MỘT khía cạnh.\n\n"
            f"Câu hỏi: {query}\n\n"
            'JSON (mảng chuỗi): ["câu 1", "câu 2", ...]'
        )
        try:
            r = await llm_ainvoke(self.llm, prompt)
            qs = parse_list(r.content, "decompose_query")
            return [q.strip() for q in qs if q.strip()] or [query]
        except Exception as exc:
            logger.warning("Decompose failed: %s", exc)
            return [query]

    # ── node 2: parallel retrieval ─────────────────────────────────────────

    async def retrieve_articles(self, state: LegalResearchState) -> dict:
        vectors = state.get("query_vectors") or [state["query"]]
        retriever = self.retriever.get_retriever(search_kwargs={"k": N_RESULTS_PER_VECTOR})

        async def _query(v: str) -> list[Article]:
            try:
                docs = await retriever.ainvoke(v)
                results: list[Article] = []
                for doc in docs:
                    metadata = doc.metadata or {}
                    chunk_id = metadata.get("chunk_id") or f"{metadata.get('source','unknown')}::{hashlib.md5(doc.page_content.encode()).hexdigest()}"
                    relevance_score = float(metadata.get("score", -1.0) or -1.0)
                    results.append(
                        Article(
                            id=chunk_id,
                            content=doc.page_content,
                            metadata={**metadata, "chunk_id": chunk_id},
                            relevance_score=relevance_score,
                        )
                    )
                return results
            except Exception as exc:
                logger.error("Retrieve articles failed: %s", exc)
                return []

        batches = await asyncio.gather(*[_query(v) for v in vectors])
        best: dict[str, Article] = {}
        for batch in batches:
            for a in batch:
                if a.id not in best or a.relevance_score > best[a.id].relevance_score:
                    best[a.id] = a
        logger.info("retrieve_articles: %d unique", len(best))
        return {"retrieved_articles": list(best.values())}

    # ── node 3: parallel LLM re-ranker ────────────────────────────────────

    async def rank_results(self, state: LegalResearchState) -> dict:
        articles = state.get("retrieved_articles", [])
        query    = state["query"]
        if not articles:
            return {"ranked_articles": [], "metadata": state.get("metadata", {})}

        candidates = sorted(articles, key=lambda a: a.relevance_score, reverse=True)
        candidates = candidates[:TOP_K_LLM_SCORE]

        scored = await asyncio.gather(
            *[self._llm_score(a, query) for a in candidates]
        )

        def _blend(item: tuple[Article, float]) -> float:
            a, s = item
            return 0.4 * a.relevance_score + 0.6 * (s / 10.0)

        ranked = sorted(scored, key=_blend, reverse=True)
        final: list[Article] = []
        for a, s in ranked[:TOP_K_RESEARCH]:
            final.append(Article(
                id=a.id,
                content=a.content,
                metadata={**a.metadata, "llm_relevance_score": s, "blended_score": round(_blend((a, s)), 4)},
                relevance_score=a.relevance_score,
            ))

        return {"ranked_articles": final,
                "metadata": {**state.get("metadata", {}), "ranked_count": len(final)}}

    async def _llm_score(self, article: Article, query: str) -> tuple[Article, float]:
        prompt = (
            "Đánh giá mức độ liên quan 0-10:\n"
            f"Văn bản: {article.content[:400]}\nCâu hỏi: {query}\n"
            'JSON: {"score": <0-10>, "reason": "..."}'
        )
        try:
            r    = await llm_ainvoke(self.llm, prompt)
            data = parse_json(r.content, "_llm_score")
            return article, max(0.0, min(10.0, float(data.get("score", 5))))
        except Exception as exc:
            logger.warning("LLM score failed %s: %s (defaulting to 5.0)", article.id, exc)
            return article, 5.0

    # ── public API ─────────────────────────────────────────────────────────

    async def run(self, query: str) -> list[Article]:
        result = await self.workflow.ainvoke({
            "query": query, "hyde_document": "", "decomposed_queries": [],
            "query_vectors": [], "retrieved_articles": [], "ranked_articles": [],
            "metadata": {},
        })
        return result["ranked_articles"]
