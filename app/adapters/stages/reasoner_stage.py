from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from app.ports.pipeline_stage import PipelineStagePort, StageInput, StageOutput

SYSTEM_PROMPT = (
    "You are a deep reasoner. Think step by step before answering. "
    "Analyze the question thoroughly, consider multiple perspectives, "
    "identify assumptions, and provide a well-reasoned response.\n"
    "Use chain-of-thought reasoning internally, then present your conclusion."
)


class ReasonerStage(PipelineStagePort):
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
