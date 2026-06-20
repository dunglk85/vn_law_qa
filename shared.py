from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Article:
    id: str
    content: str
    metadata: dict[str, Any]
    relevance_score: float = 0.0


@dataclass(frozen=True)
class Citation:
    article_id: str
    content: str
    relevance: float = 0.0
    verified: bool = False


@dataclass(frozen=True)
class Task:
    task_type: str
    description: str


MAX_RETRIES = 2
QUALITY_THRESHOLD = 0.75
N_RESULTS_PER_VECTOR = 5
TOP_K_RESEARCH = 5
TOP_K_LLM_SCORE = 8
HYDE_ENABLED = True
SUBQUERY_COUNT = 3
RELEVANCE_THRESHOLD = 0.5


def parse_json(content: str, context: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def parse_list(content: str, context: str) -> list[str]:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except json.JSONDecodeError:
        pass
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
