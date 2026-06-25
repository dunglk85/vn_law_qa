"""Application-level wiring — composes factory-built adapters into core services.

This module bridges the Factory (concrete adapters) and Core (business logic)
layers. It is NOT imported by any checked layer, so it can freely depend on both.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import yaml

from app.config import config
from app.factory import create_llm_with_model, create_pipeline_stage
from app.ports.pipeline_stage import PipelineStagePort

logger = logging.getLogger(__name__)

_LOADED_PROMPTS: dict[str, str] | None = None
_LAST_LOADED: float = 0.0


def _load_prompts() -> dict[str, str]:
    global _LOADED_PROMPTS, _LAST_LOADED

    ttl = config.pipeline_prompts_cache_ttl
    now = time.monotonic()
    if _LOADED_PROMPTS is not None and (ttl > 0 and now - _LAST_LOADED < ttl):
        return _LOADED_PROMPTS

    path = Path(config.pipeline_prompts_path)
    if not path.is_file():
        if _LOADED_PROMPTS is None:
            logger.warning("Prompts file not found at %s — using stage defaults", path)
            _LOADED_PROMPTS = {}
        return _LOADED_PROMPTS

    try:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            logger.warning("Prompts file has invalid structure, expected a dict, got %s", type(data).__name__)
            _LOADED_PROMPTS = {}
            return _LOADED_PROMPTS
        _LOADED_PROMPTS = {k: v if isinstance(v, str) else str(v) for k, v in data.items() if v}
        _LAST_LOADED = now
        logger.info("Loaded %d prompts from %s", len(_LOADED_PROMPTS), path)
    except Exception as exc:
        logger.error("Failed to reload prompts from %s: %s", path, exc)

    return _LOADED_PROMPTS or {}


def _stage_config(name: str) -> dict:
    """Build a config dict for a pipeline stage from prompts file."""
    prompts = _load_prompts()
    cfg: dict = {}
    prompt = prompts.get(name)
    if prompt:
        cfg["system_prompt"] = prompt.strip()
    return cfg


def create_pipeline_orchestrator(default_llm):
    """Build the pipeline, resolving each stage's model + prompt independently.

    Models are set via ``PIPELINE_MODEL_<STAGE>`` env vars (empty = use default).
    Prompts are read from ``config/prompts.yaml`` (missing entry = use stage default).
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

        stages[name] = create_pipeline_stage(name, chat_model, config=_stage_config(name))

        logger.info(
            "Pipeline stage %s: model=%s, stage_type=%s",
            name,
            model_override or type(default_llm).__name__,
            type(stages[name]).__name__,
        )

    logger.info(
        "Pipeline orchestrator built: %d stages (%s)",
        len(stages),
        ", ".join(stages.keys()),
    )
    return PipelineOrchestrator(stages, stage_order)
