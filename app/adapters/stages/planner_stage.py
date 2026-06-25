from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from app.ports.pipeline_stage import PipelineStagePort, StageInput, StageOutput

DEFAULT_SYSTEM_PROMPT = (
    "You are a task planner. Decompose the user's request into a sequence of "
    "concrete, executable sub-tasks. Return a numbered list of tasks.\n"
    "Format:\n"
    "1. task description\n"
    "2. task description\n"
    "...")


class PlannerStage(PipelineStagePort):
    def __init__(self, chat_model, config=None):
        super().__init__(chat_model, config)
        self._system_prompt = (config or {}).get("system_prompt", DEFAULT_SYSTEM_PROMPT)

    async def run(self, input: StageInput) -> StageOutput:
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=input.prompt),
        ]
        response = await self._model.ainvoke(messages)
        return StageOutput(content=response.content.strip())

    async def stream(self, input: StageInput) -> AsyncIterator[str]:
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=input.prompt),
        ]
        async for chunk in self._model.astream(messages):
            if chunk.content:
                yield chunk.content
