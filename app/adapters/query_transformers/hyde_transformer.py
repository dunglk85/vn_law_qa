from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.ports.query_transformer import QueryTransformerPort

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
        chain = _HYDE_PROMPT | self._llm
        result = await chain.ainvoke({"question": query})
        hypothetical_answer = result.content.strip()
        return [hypothetical_answer]
