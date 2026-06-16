from __future__ import annotations
import asyncio
from typing import List

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.ports.query_transformer import QueryTransformerPort


_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Generate a brief, factual answer to the question."),
    ("user", "Question: {question}\n\nAnswer:"),
])

_DECOMPOSITION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Break down complex questions into 2-3 simpler sub-questions that can be answered independently. Return each sub-question on a new line, prefixed with a number and period (e.g., '1. ...')."),
    ("user", "Complex question: {question}\n\nSub-questions:"),
])


class HyDEDecompositionQueryTransformerAdapter(QueryTransformerPort):
    """Combined HyDE + Decomposition transformer.

    Pipeline:
    1. Generate a hypothetical answer (HyDE) to improve semantic matching
    2. Decompose the original question into sub-questions
    3. Return all queries for retrieval: original + hypothetical + sub-queries

    This combines the strengths of both approaches: HyDE handles vocabulary
    mismatch while decomposition handles multi-part questions.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    async def transform(self, query: str) -> List[str]:
        hyde_chain = _HYDE_PROMPT | self._llm
        decomposition_chain = _DECOMPOSITION_PROMPT | self._llm

        hyde_result, decomposition_result = await asyncio.gather(
            hyde_chain.ainvoke({"question": query}),
            decomposition_chain.ainvoke({"question": query}),
        )

        hypothetical_answer = hyde_result.content.strip()
        decomposition_text = decomposition_result.content.strip()

        subqueries: List[str] = []
        for line in decomposition_text.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and "." in line:
                subquery = line.split(".", 1)[1].strip()
                if subquery:
                    subqueries.append(subquery)

        queries = [query, hypothetical_answer] + subqueries
        return queries
