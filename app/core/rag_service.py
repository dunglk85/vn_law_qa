"""app/core/rag_service.py

RAG pipeline business logic.
This module knows NOTHING about PGVector, OpenAI, Cohere, or Redis.
It depends only on Port interfaces injected at construction time.
"""
from __future__ import annotations
from typing import List, Optional, Tuple

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.prompts import ChatPromptTemplate

from app.config import config
from app.ports.vector_store import VectorStorePort
from app.ports.llm import LLMPort
from app.ports.reranker import RerankerPort
from app.ports.retriever import RetrieverPort


# --------------------------------------------------------------------------- #
# Prompt (business concern — belongs in core, not in adapters)                #
# --------------------------------------------------------------------------- #

_SYSTEM = """You are a grounded company knowledge assistant.
Always base answers strictly on the provided context.
If the answer isn't present, reply with "I don't know."
Respond concisely and clearly.
"""

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    (
        "user",
        "Question:\n{input}\n\n"
        "Context:\n{context}\n\n"
        "Rule: Prefer the most recent policy by effective date.",
    ),
])


# --------------------------------------------------------------------------- #
# RAGService                                                                   #
# --------------------------------------------------------------------------- #

class RAGService:
    """Orchestrates retrieval-augmented generation.

    All dependencies are injected via constructor — swap any provider by
    passing a different Port implementation from app/factory.py.
    """

    def __init__(
        self,
        vector_store: VectorStorePort,
        llm: LLMPort,
        reranker: RerankerPort,
        retriever: RetrieverPort,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm
        self._reranker = reranker
        self._retriever = retriever

    def _build_retriever(self, category: Optional[str]):
        base_retriever = self._retriever.get_retriever()

        compressor = self._reranker.get_compressor()
        if compressor is not None:
            return ContextualCompressionRetriever(
                base_retriever=base_retriever,
                base_compressor=compressor,
            )
        return base_retriever

    async def answer(
        self,
        question: str,
        category: Optional[str] = None,
    ) -> Tuple[str, List[str], List[str]]:
        """Run the full RAG pipeline and return (answer, sources, contexts).

        Args:
            question: User's natural-language question.
            category: Optional metadata filter (e.g. 'guides', 'policies').

        Returns:
            answer    — LLM-generated answer string
            sources   — sorted list of unique source file paths
            contexts  — list of retrieved chunk texts (for evaluation)
        """
        # Ensure the underlying store is initialised before calling as_retriever()
        await self._vector_store.similarity_search("warmup", k=1)

        retriever = self._build_retriever(category)
        chat_model = self._llm.get_chat_model()

        doc_chain = create_stuff_documents_chain(chat_model, _PROMPT)
        rag_chain = create_retrieval_chain(retriever, doc_chain)

        result = await rag_chain.ainvoke({"input": question})

        answer: str = result["answer"]
        docs: List[Document] = result["context"]

        sources = sorted({
            d.metadata.get("source")
            for d in docs
            if d.metadata.get("source")
        })
        contexts = [d.page_content for d in docs]

        return answer, sources, contexts
