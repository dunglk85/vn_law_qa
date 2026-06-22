"""Unit tests for factory.py registry resolution"""
from unittest.mock import MagicMock, patch

import pytest

from app.factory import _register, _registry, _resolve


class TestFactoryRegistry:
    def test_register_and_resolve(self):
        @_register("test_kind", "test_key")
        def test_factory():
            return "test_result"

        result = _resolve("test_kind", "test_key")
        assert result == "test_result"

        del _registry[("test_kind", "test_key")]

    def test_resolve_unknown_kind(self):
        with pytest.raises(ValueError, match="Unknown TEST_KIND"):
            _resolve("test_kind", "nonexistent")

    def test_resolve_unknown_key(self):
        @_register("test_kind2", "key1")
        def factory1():
            return "result1"

        with pytest.raises(ValueError, match="Unknown TEST_KIND2"):
            _resolve("test_kind2", "nonexistent")

        del _registry[("test_kind2", "key1")]

    def test_register_with_kwargs(self):
        @_register("test_kind3", "test_key3")
        def test_factory3(param1=None, param2=None):
            return f"{param1}_{param2}"

        result = _resolve("test_kind3", "test_key3", param1="a", param2="b")
        assert result == "a_b"

        del _registry[("test_kind3", "test_key3")]

    def test_register_multiple_keys_same_kind(self):
        @_register("multi_kind", "key_a")
        def factory_a():
            return "A"

        @_register("multi_kind", "key_b")
        def factory_b():
            return "B"

        assert _resolve("multi_kind", "key_a") == "A"
        assert _resolve("multi_kind", "key_b") == "B"

        del _registry[("multi_kind", "key_a")]
        del _registry[("multi_kind", "key_b")]

    def test_register_overwrites_existing(self):
        @_register("overwrite_kind", "key1")
        def factory_v1():
            return "v1"

        @_register("overwrite_kind", "key1")
        def factory_v2():
            return "v2"

        result = _resolve("overwrite_kind", "key1")
        assert result == "v2"

        del _registry[("overwrite_kind", "key1")]


class TestFactoryFunctions:
    def test_create_embeddings_openai(self):
        from app.factory import create_embeddings

        with patch("app.factory.config") as mock_config:
            mock_config.embeddings_type = "openai"
            mock_config.embeddings_model = "text-embedding-3-small"
            mock_config.openai_api_key = "test-key"

            with patch("app.adapters.embeddings.openai_embeddings.OpenAIEmbeddingsAdapter") as mock_adapter:
                mock_adapter.return_value = MagicMock()
                create_embeddings()
                mock_adapter.assert_called_once_with(model="text-embedding-3-small", api_key="test-key")

    def test_create_llm_openai(self):
        from app.factory import create_llm

        with patch("app.factory.config") as mock_config:
            mock_config.llm_type = "openai"
            mock_config.llm_model = "gpt-4o-mini"
            mock_config.openai_api_key = "test-key"

            with patch("app.adapters.llms.openai_llm.OpenAILLMAdapter") as mock_adapter:
                mock_adapter.return_value = MagicMock()
                create_llm()
                mock_adapter.assert_called_once_with(model="gpt-4o-mini", api_key="test-key")

    def test_create_rate_limiter_redis(self):
        from app.factory import create_rate_limiter

        with patch("app.factory.config") as mock_config:
            mock_config.redis_url = "redis://localhost:6379"
            mock_config.rate_limit_max = 100
            mock_config.rate_limit_window = 60

            with patch("app.adapters.rate_limiters.redis_rate_limiter.RedisRateLimiterAdapter") as mock_adapter:
                mock_adapter.return_value = MagicMock()
                create_rate_limiter()
                mock_adapter.assert_called_once_with(max_requests=100, window_seconds=60, redis_url="redis://localhost:6379")

    def test_create_rate_limiter_memory_fallback(self):
        from app.factory import create_rate_limiter

        with patch("app.factory.config") as mock_config:
            mock_config.redis_url = ""
            mock_config.rate_limit_max = 50
            mock_config.rate_limit_window = 30

            with patch("app.adapters.rate_limiters.memory_rate_limiter.MemoryRateLimiterAdapter") as mock_adapter:
                mock_adapter.return_value = MagicMock()
                create_rate_limiter()
                mock_adapter.assert_called_once_with(max_requests=50, window_seconds=30)
