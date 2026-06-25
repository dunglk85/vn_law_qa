from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from app.ports.pipeline_stage import PipelineStagePort, StageInput, StageOutput

SYSTEM_PROMPT = (
    "You are a task planner. Decompose the user's request into a sequence of "
    "concrete, executable sub-tasks. Return a numbered list of tasks.\n"
    "Format:\n"
    "1. task description\n"
    "2. task description\n"
    "...")


class PlannerStage(PipelineStagePort):
    async def run(self, input: StageInput) -> StageOutput:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=input.prompt),
        ]
        response = await self._model.ainvoke(messages)
        return StageOutput(content=response.content.strip())

    async def stream(self, input: StageInput) -> AsyncIterator[str]:
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=input.prompt),
        ]
        async for chunk in self._model.astream(messages):
            if chunk.content:
                yield chunk.content
