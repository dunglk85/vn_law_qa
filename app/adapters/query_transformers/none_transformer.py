from __future__ import annotations

from app.ports.query_transformer import QueryTransformerPort


class NoneQueryTransformerAdapter(QueryTransformerPort):
    """Pass-through transformer — returns the original query unchanged.

    Use by setting QUERY_TRANSFORMER_TYPE=none in .env.
    """

    async def transform(self, query: str) -> list[str]:
        return [query]
