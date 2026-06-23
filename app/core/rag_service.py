"""app/core/rag_service.py

RAG pipeline business logic.
This module knows NOTHING about PGVector, OpenAI, Cohere, or Redis.
It depends only on Port interfaces injected at construction time.
"""
from __future__ import annotations

import hashlib

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever

from app.config import config
from app.ports.llm import LLMPort
from app.ports.query_transformer import QueryTransformerPort
from app.ports.reranker import RerankerPort
from app.ports.retriever import RetrieverPort
from app.ports.vector_store import VectorStorePort

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

_NO_CONTEXT_ANSWER = "I don't know. No relevant context was found to answer your question."


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
        query_transformer: QueryTransformerPort,
    ) -> None:
        self._vector_store = vector_store
        self._llm = llm
        self._reranker = reranker
        self._retriever = retriever
        self._query_transformer = query_transformer
        self._warmed_up = False

    async def _ensure_warmup(self) -> None:
        if not self._warmed_up:
            try:
                await self._vector_store.similarity_search("warmup", k=1)
            except Exception:
                pass
            self._warmed_up = True

    def _build_retriever(self, category: str | None = None, tenant_id: str | None = None) -> BaseRetriever:
        search_kwargs: dict = {"k": config.retrieval_k}
        filter_dict: dict = {}
        if category:
            filter_dict["category"] = category
        if tenant_id and tenant_id != "*":
            filter_dict["tenant_id"] = tenant_id
        if filter_dict:
            search_kwargs["filter"] = filter_dict

        base_retriever = self._retriever.get_retriever(search_kwargs=search_kwargs)

        compressor = self._reranker.get_compressor()
        if compressor is not None:
            return ContextualCompressionRetriever(
                base_retriever=base_retriever,
                base_compressor=compressor,
            )
        return base_retriever

    async def _retrieve_with_transformed_queries(
        self,
        retriever: BaseRetriever,
        queries: list[str],
    ) -> list[Document]:
        """Retrieve documents for multiple queries and deduplicate."""
        all_docs: list[Document] = []
        seen_content: set[str] = set()

        for query in queries:
            docs = await retriever.ainvoke(query)
            for doc in docs:
                content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
                if content_hash not in seen_content:
                    all_docs.append(doc)
                    seen_content.add(content_hash)

        return all_docs[:config.retrieval_k]

    async def answer(
        self,
        question: str,
        category: str | None = None,
        tenant_id: str | None = None,
    ) -> tuple[str, list[str], list[str], dict[str, str] | None]:
        """Run the full RAG pipeline and return (answer, sources, contexts, tenant_sources).

        Args:
            question: User's natural-language question.
            category: Optional metadata filter (e.g. 'guides', 'policies').
            tenant_id: Optional tenant filter for data isolation. Use "*" for admin cross-tenant.

        Returns:
            answer          — LLM-generated answer string
            sources         — sorted list of unique source file paths
            contexts        — list of retrieved chunk texts (for evaluation)
            tenant_sources  — dict mapping source → tenant_id (only for admin cross-tenant queries)
        """
        await self._ensure_warmup()

        transformed_queries = await self._query_transformer.transform(question)
        if not transformed_queries:
            transformed_queries = [question]

        retriever = self._build_retriever(category=category, tenant_id=tenant_id)

        if len(transformed_queries) > 1:
            docs = await self._retrieve_with_transformed_queries(retriever, transformed_queries)

            if not docs:
                return _NO_CONTEXT_ANSWER, [], [], None

            chat_model = self._llm.get_chat_model()
            doc_chain = create_stuff_documents_chain(chat_model, _PROMPT)
            result = await doc_chain.ainvoke({"input": question, "context": docs})

            answer: str = result
        else:
            chat_model = self._llm.get_chat_model()
            doc_chain = create_stuff_documents_chain(chat_model, _PROMPT)
            rag_chain = create_retrieval_chain(retriever, doc_chain)

            result = await rag_chain.ainvoke({"input": question})
            answer = result["answer"]
            docs = result["context"]

        sources = sorted({
            d.metadata.get("source")
            for d in docs
            if d.metadata.get("source")
        })
        contexts = [d.page_content for d in docs]

        if tenant_id == "*":
            tenant_sources = {}
            for d in docs:
                src = d.metadata.get("source")
                if src:
                    tid = d.metadata.get("tenant_id", "unassigned")
                    if src not in tenant_sources:
                        tenant_sources[src] = tid
            return answer, sources, contexts, tenant_sources

        return answer, sources, contexts, None
