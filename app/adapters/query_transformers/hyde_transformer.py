from __future__ import annotations

import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.ports.query_transformer import QueryTransformerPort

logger = logging.getLogger(__name__)

_HYDE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Generate a brief, factual answer to the question."),
    ("user", "Question: {question}\n\nAnswer:"),
])


class HyDEQueryTransformerAdapter(QueryTransformerPort):
    """Hypothetical Document Embedding (HyDE) transformer.

    Uses the LLM to generate a hypothetical answer to the question,
    then uses that answer as the retrieval query. This can improve
    retrieval when the question's wording differs from relevant documents.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        self._llm = llm

    async def transform(self, query: str) -> list[str]:
        try:
            chain = _HYDE_PROMPT | self._llm
            result = await chain.ainvoke({"question": query})
            hypothetical_answer = result.content.strip()
            if hypothetical_answer:
                return [hypothetical_answer]
        except Exception as exc:
            logger.warning("HyDE transformer failed, using original query: %s", exc)
        return [query]
