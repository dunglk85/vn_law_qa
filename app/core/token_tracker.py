"""Token tracking for LLM calls.

Uses contextvars to accumulate token usage per request without changing
function signatures. Every llm_ainvoke() call adds to the active tracker.
"""
from __future__ import annotations

import contextvars
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_current_tracker: contextvars.ContextVar[TokenTracker | None] = contextvars.ContextVar(
    "token_tracker", default=None
)


@dataclass
class TokenTracker:
    """Accumulates token usage for a single request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    llm_call_count: int = 0
    _calls: list[dict] = field(default_factory=list, repr=False)

    def record(self, prompt_tokens: int, completion_tokens: int, call_name: str = "") -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.total_tokens += prompt_tokens + completion_tokens
        self.llm_call_count += 1
        self._calls.append({
            "call": call_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        })

    def to_dict(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "llm_call_count": self.llm_call_count,
        }


def get_tracker() -> TokenTracker | None:
    return _current_tracker.get()


def set_tracker(tracker: TokenTracker | None) -> contextvars.Token:
    return _current_tracker.set(tracker)


def reset_tracker() -> TokenTracker:
    tracker = TokenTracker()
    _current_tracker.set(tracker)
    return tracker
