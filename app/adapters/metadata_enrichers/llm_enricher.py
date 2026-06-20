from __future__ import annotations
import asyncio
from typing import List

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate

from app.ports.metadata_enrichment import MetadataEnrichmentPort


_ENRICH_TIMEOUT = 30.0

_SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Generate a concise one-sentence summary of the document content."),
    ("user", "Document content:\n{content}\n\nSummary:"),
])

_KEYWORDS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Extract 3-5 key topics or keywords from the document content. Return them as a comma-separated list."),
    ("user", "Document content:\n{content}\n\nKeywords:"),
])


class LLMEnricherAdapter(MetadataEnrichmentPort):
    """LLM-based metadata enrichment — adds summary and keywords.

    Uses the LLM to generate a summary and extract keywords from each
    document. Useful for improving retrieval quality and providing
    document-level context.
    """

    def __init__(self, llm: BaseChatModel, max_content_length: int = 2000) -> None:
        self._llm = llm
        self._max_content_length = max_content_length

    def _truncate(self, content: str) -> str:
        if len(content) > self._max_content_length:
            return content[:self._max_content_length] + "..."
        return content

    async def enrich(self, documents: List[Document]) -> List[Document]:
        async def _enrich_doc(doc: Document) -> None:
            content = self._truncate(doc.page_content)
            try:
                summary_chain = _SUMMARY_PROMPT | self._llm
                summary_result = await asyncio.wait_for(
                    summary_chain.ainvoke({"content": content}), timeout=_ENRICH_TIMEOUT)
                doc.metadata["summary"] = summary_result.content.strip()

                keywords_chain = _KEYWORDS_PROMPT | self._llm
                keywords_result = await asyncio.wait_for(
                    keywords_chain.ainvoke({"content": content}), timeout=_ENRICH_TIMEOUT)
                keywords_text = keywords_result.content.strip()
                doc.metadata["keywords"] = [k.strip() for k in keywords_text.split(",") if k.strip()]
            except Exception as exc:
                print(f"ENRICH ERROR: failed to enrich document: {exc}")

        tasks = [_enrich_doc(doc) for doc in documents]
        await asyncio.gather(*tasks, return_exceptions=True)
        return documents
