"""
SupervisorAgent
───────────────
Orchestrates: analyze_query → plan_tasks → legal_research
              → citation_check → response_synthesis → validate_quality

retry_count is incremented inside execute_citation_check (a node) so
LangGraph persists it. Routers are pure functions — never mutate state.
"""
import logging
from typing import Annotated, Optional, TypedDict, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from shared import Article, Citation, Task, llm_ainvoke, parse_json
from shared import MAX_RETRIES, QUALITY_THRESHOLD

from .legal_research_agent import LegalResearchAgent
from .citation_checker_agent import CitationCheckerAgent
from .response_synthesizer_agent import ResponseSynthesizerAgent

logger = logging.getLogger(__name__)


class SupervisorState(TypedDict):
    messages:           Annotated[list[BaseMessage], add_messages]
    query:              str
    user_id:            str
    session_id:         str
    legal_domain:       Optional[str]
    task_plan:          list[Task]
    research_results:   list[Article]
    verified_citations: list[Citation]
    final_response:     Optional[str]
    quality_score:      Optional[float]
    error:              Optional[str]
    retry_count:        int
    metadata:           dict[str, Any]


class SupervisorAgent:
    def __init__(
        self,
        research_agent:  LegalResearchAgent,
        citation_agent:  CitationCheckerAgent,
        synthesis_agent: ResponseSynthesizerAgent,
        llm: Optional[BaseChatModel] = None,
    ):
        if llm is None:
            raise ValueError("SupervisorAgent requires a chat model")
        self.research_agent  = research_agent
        self.citation_agent  = citation_agent
        self.synthesis_agent = synthesis_agent
        self.llm             = llm
        self.workflow        = self._build_workflow()

    # ── graph ──────────────────────────────────────────────────────────────

    def _build_workflow(self) -> StateGraph:
        wf = StateGraph(SupervisorState)
        wf.add_node("analyze_query",              self.analyze_query)
        wf.add_node("plan_tasks",                 self.plan_tasks)
        wf.add_node("execute_legal_research",     self.execute_legal_research)
        wf.add_node("execute_citation_check",     self.execute_citation_check)
        wf.add_node("execute_response_synthesis", self.execute_response_synthesis)
        wf.add_node("validate_quality",           self.validate_quality)

        wf.add_edge(START,          "analyze_query")
        wf.add_edge("analyze_query","plan_tasks")
        wf.add_edge("plan_tasks",   "execute_legal_research")

        wf.add_conditional_edges(
            "execute_legal_research", self.route_after_research,
            {"citation_check": "execute_citation_check", "error": END},
        )
        wf.add_conditional_edges(
            "execute_citation_check", self.route_after_citation_check,
            {"synthesis": "execute_response_synthesis",
             "retry_research": "execute_legal_research", "error": END},
        )
        wf.add_edge("execute_response_synthesis", "validate_quality")
        wf.add_conditional_edges(
            "validate_quality", self.route_after_validation,
            {"complete": END, "retry_synthesis": "execute_response_synthesis", "error": END},
        )
        return wf.compile()

    # ── nodes ──────────────────────────────────────────────────────────────

    async def analyze_query(self, state: SupervisorState) -> dict:
        prompt = (
            "Phân tích câu hỏi pháp luật sau, trả lời chỉ JSON:\n"
            f"Câu hỏi: {state['query']}\n\n"
            '{"legal_domain":"civil_law|labor_law|...","intent":"question|advice",'
            '"complexity":"simple|moderate|complex","key_terms":[]}'
        )
        try:
            r        = await llm_ainvoke(self.llm, prompt)
            analysis = parse_json(r.content, "analyze_query")
        except Exception as exc:
            logger.error("analyze_query failed: %s", exc)
            analysis = {}
        return {"legal_domain": analysis.get("legal_domain"),
                "metadata": {**state.get("metadata", {}), "analysis": analysis}}

    async def plan_tasks(self, state: SupervisorState) -> dict:
        domain = state.get("legal_domain") or "general"
        return {"task_plan": [
            Task(task_type="legal_research",
                 description=f"Research {domain} for: {state['query']}"),
            Task(task_type="citation_check",   description="Verify citations"),
            Task(task_type="response_synthesis", description="Synthesise response"),
        ]}

    async def execute_legal_research(self, state: SupervisorState) -> dict:
        try:
            articles = await self.research_agent.run(state["query"])
            logger.info("research: %d articles (retry=%d)",
                        len(articles), state.get("retry_count", 0))
            return {"research_results": articles, "error": None,
                    "metadata": {**state.get("metadata", {}), "research_complete": True}}
        except Exception as exc:
            logger.error("LegalResearchAgent failed: %s", exc)
            return {"error": str(exc), "research_results": []}

    async def execute_citation_check(self, state: SupervisorState) -> dict:
        try:
            citations = await self.citation_agent.run(
                state.get("research_results", []), state["query"])
            logger.info("citation check: %d verified", len(citations))
            # Increment retry_count HERE (inside a node) so LangGraph persists it.
            new_retry = state.get("retry_count", 0) + (0 if citations else 1)
            return {"verified_citations": citations, "retry_count": new_retry, "error": None}
        except Exception as exc:
            logger.error("CitationCheckerAgent failed: %s", exc)
            return {"error": str(exc), "verified_citations": [],
                    "retry_count": state.get("retry_count", 0) + 1}

    async def execute_response_synthesis(self, state: SupervisorState) -> dict:
        new_retry = state.get("retry_count", 0) + 1
        try:
            result = await self.synthesis_agent.synthesize(
                state["query"], state.get("verified_citations", []))
            return {"final_response": result["response"], "error": None,
                    "retry_count": new_retry}
        except Exception as exc:
            logger.error("ResponseSynthesizerAgent failed: %s", exc)
            return {"error": str(exc), "final_response": None,
                    "retry_count": new_retry}

    async def validate_quality(self, state: SupervisorState) -> dict:
        response  = state.get("final_response") or ""
        citations = state.get("verified_citations", [])
        score = 0.0
        if response:                                           score += 0.4
        if citations:                                          score += 0.3
        if any(c.article_id in response for c in citations):  score += 0.2
        if len(response) > 100:                               score += 0.1
        logger.info("quality: %.2f", score)
        return {"quality_score": round(score, 2)}

    # ── routers ────────────────────────────────────────────────────────────

    def route_after_research(self, state: SupervisorState) -> str:
        return "error" if state.get("error") else "citation_check"

    def route_after_citation_check(self, state: SupervisorState) -> str:
        if state.get("error"):
            return "error"
        if not state.get("verified_citations"):
            if state.get("retry_count", 0) >= MAX_RETRIES:
                logger.warning("Max retries reached, proceeding to synthesis")
                return "synthesis"
            return "retry_research"
        return "synthesis"

    def route_after_validation(self, state: SupervisorState) -> str:
        if state.get("error"):                               return "error"
        if (state.get("quality_score") or 0) >= QUALITY_THRESHOLD: return "complete"
        if state.get("retry_count", 0) >= MAX_RETRIES:     return "error"
        return "retry_synthesis"

    # ── public API ─────────────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        user_id: str,
        session_id: str,
        metadata: dict[str, Any] | None = None,
        conversation_history: list[dict] | None = None,
        summary_context: str = "",
    ) -> dict:
        history_messages: list[BaseMessage] = []
        if conversation_history:
            for turn in conversation_history:
                role = turn.get("role", "user")
                content = turn.get("content", "")
                if role == "assistant":
                    history_messages.append(AIMessage(content=content))
                else:
                    history_messages.append(HumanMessage(content=content))

        if summary_context:
            summary_message = HumanMessage(
                content=f"[SYSTEM: The following is a summary of earlier conversation context]\n{summary_context}"
            )
            history_messages.insert(0, summary_message)

        return await self.workflow.ainvoke({
            "messages": [*history_messages, HumanMessage(content=query)],
            "query": query, "user_id": user_id, "session_id": session_id,
            "legal_domain": None, "task_plan": [], "research_results": [],
            "verified_citations": [], "final_response": None,
            "quality_score": None, "error": None, "retry_count": 0,
            "metadata": metadata or {},
        })
