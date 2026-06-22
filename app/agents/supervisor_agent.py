"""
SupervisorAgent
───────────────
Orchestrates: analyze_query → plan_tasks → legal_research
              → citation_check → response_synthesis → validate_quality

retry_count is incremented inside execute_citation_check (a node) so
LangGraph persists it. Routers are pure functions — never mutate state.
"""
from __future__ import annotations

import logging
import time
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app.config import config
from app.core.a2a_client import A2AClientRouter
from app.core.models import (
    MAX_RETRIES,
    QUALITY_THRESHOLD,
    Article,
    Citation,
    Task,
    format_citations,
    llm_ainvoke,
    parse_json,
)
from app.core.retry import retry_with_backoff

from .citation_checker_agent import CitationCheckerAgent
from .legal_research_agent import LegalResearchAgent
from .response_synthesizer_agent import ResponseSynthesizerAgent

logger = logging.getLogger(__name__)

_FALLBACK_RESPONSE = (
    "I was unable to process your request due to an internal error. "
    "Please try again later."
)


class SupervisorState(TypedDict):
    messages:           Annotated[list[BaseMessage], add_messages]
    query:              str
    user_id:            str
    session_id:         str
    legal_domain:       str | None
    task_plan:          list[Task]
    research_results:   list[Article]
    verified_citations: list[Citation]
    final_response:     str | None
    quality_score:      float | None
    error:              str | None
    retry_count:        int
    metadata:           dict[str, Any]
    reasoning_steps:    list[dict]


class SupervisorAgent:
    def __init__(
        self,
        research_agent:  LegalResearchAgent,
        citation_agent:  CitationCheckerAgent,
        synthesis_agent: ResponseSynthesizerAgent,
        llm: BaseChatModel | None = None,
        knowledge_search_tool=None,
        a2a_client: A2AClientRouter | None = None,
    ):
        if llm is None:
            raise ValueError("SupervisorAgent requires a chat model")
        self.research_agent  = research_agent
        self.citation_agent  = citation_agent
        self.synthesis_agent = synthesis_agent
        self.llm             = llm
        self.knowledge_search_tool = knowledge_search_tool
        self._a2a_client     = a2a_client
        self.workflow        = self._build_workflow()

    def _step(self, agent: str, action: str, **kwargs) -> dict:
        return {
            "agent": agent,
            "action": action,
            "status": kwargs.get("status", "completed"),
            "input": kwargs.get("input", ""),
            "output": kwargs.get("output", ""),
            "tool_calls": kwargs.get("tool_calls", []),
            "error": kwargs.get("error", ""),
            "timestamp": time.time(),
        }

    def _validate_subagent_response(self, response: dict, required_keys: list[str]) -> bool:
        for key in required_keys:
            if key not in response or response[key] is None:
                return False
        return True

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
        node_start = time.time()
        prompt = (
            "Phân tích câu hỏi pháp luật sau, trả lời chỉ JSON:\n"
            f"Câu hỏi: {state['query']}\n\n"
            '{"legal_domain":"civil_law|labor_law|...","intent":"question|advice",'
            '"complexity":"simple|moderate|complex","key_terms":[]}'
        )
        try:
            r = await retry_with_backoff(
                lambda: llm_ainvoke(self.llm, prompt, call_name="analyze_query"),
                max_attempts=config.tool_retry_max_attempts,
                base_delay=config.tool_retry_base_delay,
                desc="analyze_query",
            )
            analysis = parse_json(r.content, "analyze_query")
            step = self._step("supervisor", "analyze_query",
                input=state["query"][:200],
                output=str(analysis),
            )
            step["duration_ms"] = round((time.time() - node_start) * 1000)
        except Exception as exc:
            logger.error("analyze_query failed: %s", exc)
            analysis = {}
            step = self._step("supervisor", "analyze_query",
                input=state["query"][:200],
                status="failed",
                error=str(exc),
            )
            step["duration_ms"] = round((time.time() - node_start) * 1000)
        return {
            "legal_domain": analysis.get("legal_domain"),
            "metadata": {**state.get("metadata", {}), "analysis": analysis},
            "reasoning_steps": [*state.get("reasoning_steps", []), step],
        }

    async def plan_tasks(self, state: SupervisorState) -> dict:
        domain = state.get("legal_domain") or "general"
        tasks = [
            Task(task_type="legal_research",
                 description=f"Research {domain} for: {state['query']}"),
            Task(task_type="citation_check",   description="Verify citations"),
            Task(task_type="response_synthesis", description="Synthesise response"),
        ]
        step = self._step("supervisor", "plan_tasks",
            input=f"domain={domain}",
            output=f"tasks={[t.task_type for t in tasks]}",
        )
        return {
            "task_plan": tasks,
            "reasoning_steps": [*state.get("reasoning_steps", []), step],
        }

    async def _run_legal_research(self, state: SupervisorState) -> list[Article]:
        if self._a2a_client is not None:
            articles: list[Article] = []
            async for ev in self._a2a_client.send_task_stream(
                agent="legal-research-agent",
                payload={"query": state["query"], "metadata": state.get("metadata", {})},
            ):
                if ev.type == "task_artifact":
                    articles = ev.artifact.get("articles", [])
            return articles
        return await self.research_agent.run(state["query"])

    async def execute_legal_research(self, state: SupervisorState) -> dict:
        node_start = time.time()
        steps = list(state.get("reasoning_steps", []))
        articles: list[Article] = []
        try:
            articles = await retry_with_backoff(
                lambda: self._run_legal_research(state),
                max_attempts=config.tool_retry_max_attempts,
                base_delay=config.tool_retry_base_delay,
                desc="legal_research",
            )
            logger.info("research: %d articles", len(articles))
        except Exception as exc:
            logger.error("LegalResearchAgent failed: %s", exc)
            step = self._step("supervisor", "legal_research",
                input=state["query"][:200],
                status="failed",
                error=str(exc),
            )
            step["duration_ms"] = round((time.time() - node_start) * 1000)
            steps.append(step)
            return {"error": str(exc), "research_results": [], "reasoning_steps": steps}

        tool_calls: list[dict] = []
        if self.knowledge_search_tool is not None:
            try:
                tool_results = await retry_with_backoff(
                    lambda: self.knowledge_search_tool.ainvoke({"query": state["query"]}),
                    max_attempts=config.tool_retry_max_attempts,
                    base_delay=config.tool_retry_base_delay,
                    desc="knowledge_search_tool",
                )
                logger.info("knowledge_search_tool: %d results", len(tool_results))
                tool_calls.append({
                    "tool": "knowledge_search",
                    "input": state["query"][:200],
                    "result_count": len(tool_results),
                })
            except Exception as exc:
                logger.warning("knowledge_search_tool failed: %s", exc)
                tool_calls.append({
                    "tool": "knowledge_search",
                    "input": state["query"][:200],
                    "error": str(exc),
                })

        step = self._step("supervisor", "legal_research",
            input=state["query"][:200],
            output=f"articles={len(articles)}, tool_results={len(tool_calls)}",
            tool_calls=tool_calls,
        )
        step["duration_ms"] = round((time.time() - node_start) * 1000)
        steps.append(step)

        return {
            "research_results": articles,
            "error": None,
            "metadata": {
                **state.get("metadata", {}),
                "research_complete": True,
                "tool_invoked": self.knowledge_search_tool is not None,
                "tool_result_count": len(tool_calls),
            },
            "reasoning_steps": steps,
        }

    async def _run_citation_check(self, state: SupervisorState) -> list[Citation]:
        if self._a2a_client is not None:
            citations: list[Citation] = []
            async for ev in self._a2a_client.send_task_stream(
                agent="citation-checker-agent",
                payload={
                    "articles": state.get("research_results", []),
                    "query": state["query"],
                },
            ):
                if ev.type == "task_artifact":
                    citations = ev.artifact.get("citations", [])
            return citations
        return await self.citation_agent.run(
            state.get("research_results", []), state["query"])

    async def execute_citation_check(self, state: SupervisorState) -> dict:
        node_start = time.time()
        steps = list(state.get("reasoning_steps", []))
        try:
            citations = await retry_with_backoff(
                lambda: self._run_citation_check(state),
                max_attempts=config.tool_retry_max_attempts,
                base_delay=config.tool_retry_base_delay,
                desc="citation_check",
            )
            logger.info("citation check: %d verified", len(citations))
            new_retry = state.get("retry_count", 0) + (0 if citations else 1)

            step = self._step("supervisor", "citation_check",
                input=f"articles={len(state.get('research_results',[]))}",
                output=f"citations={len(citations)}",
            )
            step["duration_ms"] = round((time.time() - node_start) * 1000)
            steps.append(step)

            return {"verified_citations": citations, "retry_count": new_retry, "error": None,
                    "reasoning_steps": steps}
        except Exception as exc:
            logger.error("CitationCheckerAgent failed: %s", exc)
            step = self._step("supervisor", "citation_check",
                status="failed",
                error=str(exc),
            )
            step["duration_ms"] = round((time.time() - node_start) * 1000)
            steps.append(step)
            return {"error": str(exc), "verified_citations": [],
                    "retry_count": state.get("retry_count", 0) + 1,
                    "reasoning_steps": steps}

    async def _run_response_synthesis(self, state: SupervisorState) -> dict:
        if self._a2a_client is not None:
            result: dict = {}
            async for ev in self._a2a_client.send_task_stream(
                agent="response-synthesizer-agent",
                payload={
                    "query": state["query"],
                    "citations": state.get("verified_citations", []),
                },
            ):
                if ev.type == "task_artifact":
                    result = ev.artifact
            return result
        return await self.synthesis_agent.synthesize(
            state["query"], state.get("verified_citations", []))

    async def execute_response_synthesis(self, state: SupervisorState) -> dict:
        node_start = time.time()
        steps = list(state.get("reasoning_steps", []))
        try:
            result = await retry_with_backoff(
                lambda: self._run_response_synthesis(state),
                max_attempts=config.tool_retry_max_attempts,
                base_delay=config.tool_retry_base_delay,
                desc="response_synthesis",
            )

            if not self._validate_subagent_response(result, ["response"]):
                raise ValueError("Synthesis response missing required fields")

            step = self._step("supervisor", "response_synthesis",
                input=f"citations={len(state.get('verified_citations',[]))}",
                output=f"response_length={len(result['response'])}",
            )
            step["duration_ms"] = round((time.time() - node_start) * 1000)
            steps.append(step)
            return {"final_response": result["response"], "error": None,
                    "retry_count": state.get("retry_count", 0), "reasoning_steps": steps}
        except Exception as exc:
            logger.error("ResponseSynthesizerAgent failed: %s", exc)
            step = self._step("supervisor", "response_synthesis",
                status="failed",
                error=str(exc),
            )
            step["duration_ms"] = round((time.time() - node_start) * 1000)
            steps.append(step)
            return {"error": str(exc), "final_response": None,
                    "retry_count": state.get("retry_count", 0) + 1, "reasoning_steps": steps}

    def _heuristic_score(self, response: str, citations: list[Citation]) -> float:
        score = 0.0
        if response:
            score += 0.4
        if citations:
            score += 0.3
        if any(c.article_id in response for c in citations):
            score += 0.2
        if len(response) > 100:
            score += 0.1
        return score

    async def _llm_quality_check(self, query: str, response: str, citations: list[Citation]) -> dict:
        citation_context = format_citations(citations) if citations else "(none)"
        prompt = (
            "Evaluate the assistant response below for quality and accuracy.  "
            "Return ONLY valid JSON, no explanation.\n\n"
            f"User question: {query}\n\n"
            f"Verified source citations:\n{citation_context}\n\n"
            f"Assistant response:\n{response}\n\n"
            "JSON schema:\n"
            '{"query_relevance": 0.0-1.0,\n'
            ' "citation_grounding": 0.0-1.0,\n'
            ' "completeness": 0.0-1.0,\n'
            ' "hallucination_risk": 0.0-1.0,\n'
            ' "overall_score": 0.0-1.0,\n'
            ' "issues": ["issue1", "issue2"]}\n\n'
            "hallucination_risk is HIGH (closer to 1.0) when the response makes "
            "claims NOT supported by the citations.  overall_score should be the "
            "average of the first three dimensions minus hallucination_risk."
        )
        result = await retry_with_backoff(
            lambda: llm_ainvoke(self.llm, prompt, call_name="quality_check"),
            max_attempts=config.tool_retry_max_attempts,
            base_delay=config.tool_retry_base_delay,
            desc="llm_quality_check",
        )
        return parse_json(result.content, "llm_quality_check")

    async def validate_quality(self, state: SupervisorState) -> dict:
        node_start = time.time()
        response  = state.get("final_response") or ""
        citations = state.get("verified_citations", [])
        query     = state["query"]
        steps     = list(state.get("reasoning_steps", []))

        if not response:
            step = self._step("supervisor", "validate_quality",
                input="no_response", output="score=0.0")
            step["duration_ms"] = round((time.time() - node_start) * 1000)
            steps.append(step)
            return {"quality_score": 0.0, "reasoning_steps": steps}

        try:
            evaluation = await self._llm_quality_check(query, response, citations)
            if "overall_score" in evaluation:
                score = float(evaluation["overall_score"])
                issues = evaluation.get("issues", [])
                if issues:
                    logger.info("Quality issues: %s", issues)
            else:
                score = self._heuristic_score(response, citations)
            logger.info("Quality: LLM=%.2f", score)
        except Exception as exc:
            logger.warning("LLM quality check failed, falling back to heuristic: %s", exc)
            score = self._heuristic_score(response, citations)
            logger.info("Quality: heuristic=%.2f", score)

        step = self._step("supervisor", "validate_quality",
            input=f"response_len={len(response)}, citations={len(citations)}",
            output=f"quality_score={round(score,2)}",
        )
        step["duration_ms"] = round((time.time() - node_start) * 1000)
        steps.append(step)
        return {
            "quality_score": round(score, 2),
            "reasoning_steps": steps,
        }

    # ── routers ────────────────────────────────────────────────────────────

    def route_after_research(self, state: SupervisorState) -> str:
        return "error" if state.get("error") else "citation_check"

    def route_after_citation_check(self, state: SupervisorState) -> str:
        if state.get("error"):
            if state.get("retry_count", 0) >= MAX_RETRIES:
                logger.warning("Max retries on citation check, proceeding to synthesis")
                return "synthesis"
            return "error"
        if not state.get("verified_citations"):
            if state.get("retry_count", 0) >= MAX_RETRIES:
                logger.warning("Max retries reached, proceeding to synthesis with empty citations")
                return "synthesis"
            return "retry_research"
        return "synthesis"

    def route_after_validation(self, state: SupervisorState) -> str:
        if state.get("error"):
            return "error"
        if (state.get("quality_score") or 0) >= QUALITY_THRESHOLD:
            return "complete"
        if state.get("retry_count", 0) >= MAX_RETRIES:
            logger.warning("Max retries on quality, returning best-effort")
            return "complete"
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

        result = await self.workflow.ainvoke({
            "messages": [*history_messages, HumanMessage(content=query)],
            "query": query, "user_id": user_id, "session_id": session_id,
            "legal_domain": None, "task_plan": [], "research_results": [],
            "verified_citations": [], "final_response": None,
            "quality_score": None, "error": None, "retry_count": 0,
            "metadata": metadata or {},
            "reasoning_steps": [],
        })

        if result.get("error") and not result.get("final_response"):
            logger.error("All sub-agents failed: %s", result.get("error"))
            result["final_response"] = _FALLBACK_RESPONSE

        return result
