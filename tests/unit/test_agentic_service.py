"""Unit tests for AgenticService"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.agentic_service import AgenticService


class TestAgenticService:
    @pytest.fixture
    def mock_ports(self):
        return {
            "vector_store": MagicMock(),
            "llm": MagicMock(),
            "retriever": MagicMock(),
            "query_transformer": MagicMock(),
            "supervisor": MagicMock(),
            "session_store": MagicMock(),
        }

    @pytest.fixture
    def service(self, mock_ports):
        return AgenticService(**mock_ports)

    @pytest.mark.asyncio
    async def test_init_creates_supervisor(self, mock_ports):
        service = AgenticService(**mock_ports)
        assert service._supervisor is not None

    @pytest.mark.asyncio
    async def test_ensure_warmup_calls_vector_store(self, service):
        service._vector_store.similarity_search = AsyncMock()

        await service._ensure_warmup()

        service._vector_store.similarity_search.assert_called_once_with("warmup", k=1)
        assert service._warmed_up is True

    @pytest.mark.asyncio
    async def test_ensure_warmup_only_runs_once(self, service):
        service._vector_store.similarity_search = AsyncMock()

        await service._ensure_warmup()
        await service._ensure_warmup()

        assert service._vector_store.similarity_search.call_count == 1

    @pytest.mark.asyncio
    async def test_ensure_warmup_handles_error(self, service):
        service._vector_store.similarity_search = AsyncMock(side_effect=Exception("DB error"))

        await service._ensure_warmup()

        assert service._warmed_up is True

    @pytest.mark.asyncio
    async def test_load_session_no_store(self, service):
        service._session_store = None

        result = await service._load_session("session1")

        assert result == {"history": [], "summary": ""}

    @pytest.mark.asyncio
    async def test_load_session_with_store(self, service):
        session_data = {"history": [{"role": "user", "content": "hi"}], "summary": "test"}
        service._session_store.load = AsyncMock(return_value=session_data)

        result = await service._load_session("session1")

        assert result == session_data

    @pytest.mark.asyncio
    async def test_load_session_handles_error(self, service):
        service._session_store.load = AsyncMock(side_effect=Exception("Redis down"))

        result = await service._load_session("session1")

        assert result == {"history": [], "summary": ""}

    @pytest.mark.asyncio
    async def test_save_session_no_store(self, service):
        service._session_store = None

        await service._save_session("session1", {"history": [], "summary": ""})

    @pytest.mark.asyncio
    async def test_save_session_with_store(self, service):
        service._session_store.save = AsyncMock()

        await service._save_session("session1", {"history": [], "summary": "test"})

        service._session_store.save.assert_called_once_with("session1", {"history": [], "summary": "test"})

    def test_estimate_tokens(self, service):
        texts = ["hello world", "test"]
        result = service._estimate_tokens(texts)
        assert result == 3  # (11 // 4) + (4 // 4) = 2 + 1

    @pytest.mark.asyncio
    async def test_compress_history_short(self, service):
        history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]

        compressed, summary = await service._compress_history(history, "")

        assert compressed == history
        assert summary == ""

    @pytest.mark.asyncio
    async def test_compress_history_long(self, service):
        service._llm.get_chat_model = MagicMock(return_value=MagicMock())
        service._llm.get_chat_model().ainvoke = AsyncMock(return_value=MagicMock(content="summary"))

        history = [{"role": f"user{i}", "content": f"content{i}"} for i in range(20)]

        compressed, summary = await service._compress_history(history, "")

        assert len(compressed) < len(history)
