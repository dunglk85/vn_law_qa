"""Application-level wiring — composes factory-built adapters into core services.

This module bridges the Factory (concrete adapters) and Core (business logic)
layers. It is NOT imported by any checked layer, so it can freely depend on both.
"""
from __future__ import annotations

import logging

from app.config import config
from app.factory import create_llm_with_model, create_pipeline_stage
from app.ports.pipeline_stage import PipelineStagePort

logger = logging.getLogger(__name__)


def create_pipeline_orchestrator(default_llm):
    """Build the pipeline, resolving each stage's model independently.

    Env var ``PIPELINE_MODEL_<STAGE>`` overrides the model per stage.
    Falls back to the default LLM if not set.
    """
    from app.core.pipeline_orchestrator import PipelineOrchestrator

    stage_order = [
        s.strip() for s in config.pipeline_stage_order.split(",") if s.strip()
    ]
    stages: dict[str, PipelineStagePort] = {}

    for name in stage_order:
        model_override = getattr(config, f"pipeline_model_{name}", "")
        if model_override:
            chat_model = create_llm_with_model(model_override).get_chat_model()
        else:
            chat_model = default_llm.get_chat_model()

        stages[name] = create_pipeline_stage(name, chat_model)

    logger.info(
        "Pipeline orchestrator built: %d stages (%s)",
        len(stages),
        ", ".join(f"{n}={type(s).__name__}" for n, s in stages.items()),
    )
    return PipelineOrchestrator(stages, stage_order)
