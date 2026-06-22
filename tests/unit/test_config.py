"""Unit tests for config.py env var parsing"""
import os
from unittest.mock import patch


class TestConfigParsing:
    def test_int_env_valid_value(self):
        from app.config import _int_env

        with patch.dict(os.environ, {"TEST_INT": "42"}):
            result = _int_env("TEST_INT", 0)
            assert result == 42

    def test_int_env_invalid_value(self):
        from app.config import _int_env

        with patch.dict(os.environ, {"TEST_INT": "not_a_number"}):
            result = _int_env("TEST_INT", 99)
            assert result == 99

    def test_int_env_missing_key(self):
        from app.config import _int_env

        os.environ.pop("MISSING_INT", None)
        result = _int_env("MISSING_INT", 77)
        assert result == 77

    def test_float_env_valid_value(self):
        from app.config import _float_env

        with patch.dict(os.environ, {"TEST_FLOAT": "3.14"}):
            result = _float_env("TEST_FLOAT", 0.0)
            assert result == 3.14

    def test_float_env_invalid_value(self):
        from app.config import _float_env

        with patch.dict(os.environ, {"TEST_FLOAT": "not_a_float"}):
            result = _float_env("TEST_FLOAT", 2.71)
            assert result == 2.71

    def test_float_env_missing_key(self):
        from app.config import _float_env

        os.environ.pop("MISSING_FLOAT", None)
        result = _float_env("MISSING_FLOAT", 1.5)
        assert result == 1.5

    def test_str_env_valid_value(self):
        from app.config import _str_env

        with patch.dict(os.environ, {"TEST_STR": "hello"}):
            result = _str_env("TEST_STR", "default")
            assert result == "hello"

    def test_str_env_missing_key(self):
        from app.config import _str_env

        os.environ.pop("MISSING_STR", None)
        result = _str_env("MISSING_STR", "default_value")
        assert result == "default_value"

    def test_str_env_empty_string(self):
        from app.config import _str_env

        with patch.dict(os.environ, {"TEST_STR": ""}):
            result = _str_env("TEST_STR", "default")
            assert result == ""


class TestAppConfigDefaults:
    def test_config_singleton_exists(self):
        from app.config import config

        assert config is not None

    def test_config_has_required_fields(self):
        from app.config import config

        assert hasattr(config, "vector_store_type")
        assert hasattr(config, "llm_type")
        assert hasattr(config, "embeddings_type")
        assert hasattr(config, "reranker_type")
        assert hasattr(config, "cache_type")
        assert hasattr(config, "chunker_type")
        assert hasattr(config, "retriever_type")
        assert hasattr(config, "rag_mode")

    def test_config_vector_store_default(self):
        from app.config import config

        assert config.vector_store_type == "pgvector"

    def test_config_llm_default(self):
        from app.config import config

        assert config.llm_type == "openai"

    def test_config_embeddings_default(self):
        from app.config import config

        assert config.embeddings_type == "openai"

    def test_config_hnsw_m_default(self):
        from app.config import config

        assert config.hnsw_m == 16

    def test_config_hnsw_ef_search_exists(self):
        from app.config import config

        assert hasattr(config, "hnsw_ef_search")
        assert config.hnsw_ef_search == 50

    def test_config_timeout_defaults(self):
        from app.config import config

        assert config.llm_timeout == 30.0
        assert config.agent_timeout == 90.0
        assert config.ask_timeout == 120.0

    def test_config_langsmith_fields_exist(self):
        from app.config import config

        assert hasattr(config, "langsmith_tracing")
        assert hasattr(config, "langsmith_project")
        assert isinstance(config.langsmith_project, str)
