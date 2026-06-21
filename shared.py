"""Backward-compatible re-export from ``app.core.models``.

New code should import from ``app.core.models`` or ``app.config`` directly.
"""
from app.core.models import (  # noqa: F401
    Article,
    Citation,
    Task,
    format_citations,
    llm_ainvoke,
    parse_json,
    parse_list,
    MAX_RETRIES,
    QUALITY_THRESHOLD,
    N_RESULTS_PER_VECTOR,
    TOP_K_RESEARCH,
    TOP_K_LLM_SCORE,
    HYDE_ENABLED,
    SUBQUERY_COUNT,
    RELEVANCE_THRESHOLD,
)
