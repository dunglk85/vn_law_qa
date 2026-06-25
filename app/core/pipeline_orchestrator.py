from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from app.ports.pipeline_stage import PipelineStagePort, StageInput

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Config-driven stage pipeline with inter-stage streaming."""

    def __init__(self, stages: dict[str, PipelineStagePort], stage_order: list[str]) -> None:
        self._stages = stages
        self._order = stage_order

    async def run(self, user_input: str, context: dict[str, Any] | None = None) -> str:
        ctx = dict(context or {})
        current = StageInput(prompt=user_input, context=ctx)
        skipped: list[str] = []

        for name in self._order:
            stage = self._stages.get(name)
            if stage is None:
                logger.warning("Stage '%s' not configured, skipping", name)
                skipped.append(name)
                continue
            logger.info("Pipeline stage: %s", name)
            try:
                output = await stage.run(current)
            except Exception as exc:
                logger.error("Pipeline stage '%s' failed: %s", name, exc)
                break
            ctx[f"{name}_output"] = output.content
            current = StageInput(
                prompt=output.content,
                context=ctx,
                previous_output=output.content,
            )

        if skipped:
            logger.warning("Pipeline incomplete: %d stage(s) not configured (%s)",
                           len(skipped), ", ".join(skipped))
        return current.prompt

    async def stream(
        self,
        user_input: str,
        context: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        ctx = dict(context or {})
        current = StageInput(prompt=user_input, context=ctx)

        for name in self._order:
            stage = self._stages.get(name)
            if stage is None:
                yield {"type": "stage_error", "stage": name, "error": "Stage not configured"}
                return
            yield {"type": "stage_start", "stage": name}

            try:
                tokens: list[str] = []
                async for token in stage.stream(current):
                    tokens.append(token)
                    yield {"type": "token", "stage": name, "token": token}

                full = "".join(tokens)
                ctx[f"{name}_output"] = full
                current = StageInput(prompt=full, context=ctx, previous_output=full)

                yield {"type": "stage_end", "stage": name, "output": full}
            except Exception as exc:
                logger.error("Stage '%s' failed during streaming: %s", name, exc)
                yield {"type": "stage_error", "stage": name, "error": str(exc)}
                return

    @property
    def stage_order(self) -> list[str]:
        return list(self._order)

    @property
    def configured_stages(self) -> list[str]:
        return list(self._stages.keys())
