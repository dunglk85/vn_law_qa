"""Domain models, utilities, and re-exported configuration constants."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict, dataclass
from typing import Any

from app.config import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Article:
    id: str
    content: str
    metadata: dict[str, Any]
    relevance_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Citation:
    article_id: str
    content: str
    relevance: float = 0.0
    verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Task:
    task_type: str
    description: str


# ---------------------------------------------------------------------------
# Re-exported config constants (backward compatibility)
# New code should import from app.config directly.
# ---------------------------------------------------------------------------

MAX_RETRIES = config.max_retries
QUALITY_THRESHOLD = config.quality_threshold
N_RESULTS_PER_VECTOR = config.n_results_per_vector
TOP_K_RESEARCH = config.top_k_research
TOP_K_LLM_SCORE = config.top_k_llm_score
HYDE_ENABLED = config.hyde_enabled
SUBQUERY_COUNT = config.subquery_count
RELEVANCE_THRESHOLD = config.relevance_threshold


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

async def llm_ainvoke(llm, prompt: str, timeout: float | None = None, *, call_name: str = ""):
    from app.core.token_tracker import get_tracker

    if timeout is None:
        timeout = config.llm_timeout
    result = await asyncio.wait_for(llm.ainvoke(prompt), timeout=timeout)

    usage = {}
    if hasattr(result, "response_metadata") and isinstance(result.response_metadata, dict):
        usage = result.response_metadata.get("usage", {}) or {}
        if not usage:
            usage = result.response_metadata.get("token_usage", {}) or {}

    prompt_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0
    completion_tokens = usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0

    tracker = get_tracker()
    if tracker is not None:
        tracker.record(prompt_tokens, completion_tokens, call_name=call_name)

    logger.info(
        "LLM call%s: prompt=%d completion=%d total=%d",
        f" ({call_name})" if call_name else "",
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens,
    )

    return result


def parse_json(content: str, context: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.warning("parse_json failed for %s: non-JSON response (%.100r)", context, content)
        return {}


def parse_list(content: str, context: str) -> list[str]:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        logger.warning("parse_list failed for %s: non-JSON response (%.100r)", context, content)
    return []


def format_citations(citations: list[Citation]) -> str:
    if not citations:
        return ""
    lines = []
    for c in citations:
        lines.append(
            f"- id: {c.article_id}\n  relevance: {c.relevance}\n  verified: {c.verified}\n  content: {c.content}"
        )
    return "\n".join(lines)
