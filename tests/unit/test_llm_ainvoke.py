"""Unit tests for llm_ainvoke token tracking in models.py"""
import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.models import llm_ainvoke
from app.core.token_tracker import TokenTracker, set_tracker


class MockAIMessage:
    def __init__(self, content="", response_metadata=None):
        self.content = content
        self.response_metadata = response_metadata or {}


class TestLlmAinvoke:
    @pytest.fixture(autouse=True)
    def setup_tracker(self):
        self.tracker = TokenTracker()
        set_tracker(self.tracker)
        yield
        set_tracker(None)

    @pytest.mark.asyncio
    async def test_llm_ainvoke_tracks_tokens(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MockAIMessage(
                content="test response",
                response_metadata={"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
            )
        )

        result = await llm_ainvoke(mock_llm, "test prompt", call_name="test_call")

        assert result.content == "test response"
        assert self.tracker.prompt_tokens == 100
        assert self.tracker.completion_tokens == 50
        assert self.tracker.total_tokens == 150
        assert self.tracker.llm_call_count == 1

    @pytest.mark.asyncio
    async def test_llm_ainvoke_multiple_calls(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            side_effect=[
                MockAIMessage("resp1", {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}),
                MockAIMessage("resp2", {"usage": {"prompt_tokens": 200, "completion_tokens": 100}}),
            ]
        )

        await llm_ainvoke(mock_llm, "prompt1", call_name="call1")
        await llm_ainvoke(mock_llm, "prompt2", call_name="call2")

        assert self.tracker.prompt_tokens == 300
        assert self.tracker.completion_tokens == 150
        assert self.tracker.total_tokens == 450
        assert self.tracker.llm_call_count == 2

    @pytest.mark.asyncio
    async def test_llm_ainvoke_no_tracker(self):
        set_tracker(None)

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MockAIMessage("response", {"usage": {"prompt_tokens": 100}})
        )

        result = await llm_ainvoke(mock_llm, "prompt")
        assert result.content == "response"

    @pytest.mark.asyncio
    async def test_llm_ainvoke_missing_usage(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MockAIMessage("response", {})
        )

        await llm_ainvoke(mock_llm, "prompt", call_name="test")

        assert self.tracker.prompt_tokens == 0
        assert self.tracker.completion_tokens == 0
        assert self.tracker.total_tokens == 0
        assert self.tracker.llm_call_count == 1

    @pytest.mark.asyncio
    async def test_llm_ainvoke_timeout(self):
        mock_llm = MagicMock()

        async def slow_ainvoke(*args, **kwargs):
            await asyncio.sleep(10)
            return MockAIMessage("response")

        mock_llm.ainvoke = slow_ainvoke

        with pytest.raises(asyncio.TimeoutError):
            await llm_ainvoke(mock_llm, "prompt", timeout=0.1)

    @pytest.mark.asyncio
    async def test_llm_ainvoke_default_timeout(self):
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(
            return_value=MockAIMessage("response", {"usage": {"prompt_tokens": 10}})
        )

        result = await llm_ainvoke(mock_llm, "prompt")
        assert result.content == "response"
