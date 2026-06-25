from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from app.ports.pipeline_stage import PipelineStagePort, StageInput, StageOutput

SYSTEM_PROMPT = (
    "You are a code generation specialist. Write clean, production-quality code "
    "in response to the user's request. Include type hints, handle errors, "
    "and add a brief usage example if applicable.\n"
    "Respond with the code in a language-appropriate code block."
)


class CodeWriterStage(PipelineStagePort):
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
