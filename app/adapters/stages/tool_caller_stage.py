from __future__ import annotations

from typing import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage

from app.ports.pipeline_stage import PipelineStagePort, StageInput, StageOutput

SYSTEM_PROMPT = (
    "You are a tool selector. Given the user's request and available tools, "
    "decide which tools to invoke and with what arguments.\n\n"
    "Available tools:\n"
    "- knowledge_search(query): search company knowledge base\n"
    "- code_interpreter(code): execute Python code\n"
    "- web_search(query): search the web\n\n"
    "Respond with a JSON list of tool calls:\n"
    '[{"tool": "name", "args": {...}}]'
)


class ToolCallerStage(PipelineStagePort):
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
