from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from app.ports.pipeline_stage import PipelineStagePort, StageInput, StageOutput

DEFAULT_SYSTEM_PROMPT = (
    "You are a quality evaluator. Judge the assistant's response on these criteria:\n"
    "- relevance: does it answer the user's question?\n"
    "- accuracy: is it factually correct?\n"
    "- completeness: does it cover all aspects?\n"
    "- clarity: is it well-written and easy to understand?\n\n"
    "Respond with a JSON object:\n"
    '{"score": 0.0-1.0, "issues": ["issue1", "issue2"], "suggestions": ["suggestion1"]}'
)


class EvaluatorStage(PipelineStagePort):
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
