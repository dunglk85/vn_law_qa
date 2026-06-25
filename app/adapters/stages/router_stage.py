from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from app.ports.pipeline_stage import PipelineStagePort, StageInput, StageOutput

SYSTEM_PROMPT = (
    "You are a query router. Classify the user's question into exactly one category.\n"
    "Respond with ONLY the category label, nothing else.\n\n"
    "Categories: [general, technical, policy, legal, code]"
)


class RouterStage(PipelineStagePort):
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
