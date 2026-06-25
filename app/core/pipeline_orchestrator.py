from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from app.ports.pipeline_stage import PipelineStagePort, StageInput

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Config-driven stage pipeline with inter-stage streaming."""

    def __init__(self, stages: dict[str, PipelineStagePort], stage_order: list[str]) -> None:
        self._stages = stages
        self._order = stage_order

    async def run(self, user_input: str, context: dict[str, Any] | None = None) -> str:
        ctx = context or {}
        current = StageInput(prompt=user_input, context=ctx)

        for name in self._order:
            stage = self._stages.get(name)
            if stage is None:
                logger.warning("Stage '%s' not configured, skipping", name)
                continue
            logger.info("Pipeline stage: %s", name)
            output = await stage.run(current)
            ctx[f"{name}_output"] = output.content
            current = StageInput(
                prompt=output.content,
                context=ctx,
                previous_output=output.content,
            )

        return current.prompt

    async def stream(
        self,
        user_input: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        ctx = context or {}
        current = StageInput(prompt=user_input, context=ctx)

        for name in self._order:
            stage = self._stages.get(name)
            if stage is None:
                continue

            yield {"type": "stage_start", "stage": name}

            tokens: list[str] = []
            async for token in stage.stream(current):
                tokens.append(token)
                yield {"type": "token", "stage": name, "token": token}

            full = "".join(tokens)
            ctx[f"{name}_output"] = full
            current = StageInput(prompt=full, context=ctx, previous_output=full)

            yield {"type": "stage_end", "stage": name, "output": full}

    @property
    def stage_order(self) -> list[str]:
        return list(self._order)

    @property
    def configured_stages(self) -> list[str]:
        return list(self._stages.keys())
