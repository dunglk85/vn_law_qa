from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.ports.query_transformer import QueryTransformerPort

logger = logging.getLogger(__name__)

_DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Break down complex questions into 2-3 simpler sub-questions that can be answered independently. Return each sub-question on a new line, prefixed with a number and period (e.g., '1. ...')."),
    ("user", "Complex question: {question}\n\nSub-questions:"),
])


class DecompositionQueryTransformerAdapter(QueryTransformerPort):
    """Query decomposition transformer.

    Breaks down a complex question into multiple simpler sub-questions,
    retrieves for each, and combines the results. Useful for multi-part
    questions or questions requiring multiple pieces of information.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    async def transform(self, query: str) -> list[str]:
        try:
            chain = _DECOMPOSITION_PROMPT | self._llm
            result = await chain.ainvoke({"question": query})
            text = result.content.strip()
        except Exception as exc:
            logger.warning("Decomposition transformer failed, using original query: %s", exc)
            return [query]

        subqueries: list[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                subquery = line.split(".", 1)[1].strip()
                if subquery:
                    subqueries.append(subquery)

        if not subqueries:
            return [query]

        return [query] + subqueries
