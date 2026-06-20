from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class ReasoningStep:
    agent: str
    action: str
    status: str = "completed"
    input: str = ""
    output: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    error: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "agent": self.agent,
            "action": self.action,
            "status": self.status,
            "input": self.input[:500],
            "output": self.output[:1000],
            "tool_calls": self.tool_calls,
            "error": self.error,
            "timestamp": self.timestamp,
        }
