"""Integration tests for API endpoints"""
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.api import app


@pytest.fixture
def client():
    return TestClient(app)


class TestParquetLoader:
    def test_loads_parquet_chunks(self):
        pd = pytest.importorskip("pandas", reason="pandas not installed")
        from app.adapters.document_loaders.parquet_loader import ParquetLoaderAdapter

        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame([{
                "chunk_id": "a_c0", "article_id": "a", "title": "Article 1",
                "chude": "Civil", "demuc": "Section 1", "chuong": "Chapter 1",
                "chunk_index": 0, "total_chunks": 1, "text": "Content here",
            }])
            df.to_parquet(Path(tmp) / "law_document_chunks.parquet", index=False)

            loader = ParquetLoaderAdapter()
            docs = loader.load(tmp)
            assert len(docs) == 1
            assert docs[0].page_content == "Content here"
            assert docs[0].metadata["chunk_id"] == "a_c0"

    def test_loads_empty_directory_gracefully(self):
        from app.adapters.document_loaders.parquet_loader import ParquetLoaderAdapter

        with tempfile.TemporaryDirectory() as tmp:
            loader = ParquetLoaderAdapter()
            docs = loader.load(tmp)
            assert docs == []

    def test_loads_missing_directory(self):
        from app.adapters.document_loaders.parquet_loader import ParquetLoaderAdapter

        loader = ParquetLoaderAdapter()
        docs = loader.load("/nonexistent/path/12345")
        assert docs == []

    def test_handles_pd_na_text(self):
        pd = pytest.importorskip("pandas", reason="pandas not installed")
        from app.adapters.document_loaders.parquet_loader import ParquetLoaderAdapter

        with tempfile.TemporaryDirectory() as tmp:
            df = pd.DataFrame([{
                "chunk_id": "x_c0", "article_id": "x", "title": "T",
                "chude": "C", "demuc": "D", "chuong": "Ch",
                "chunk_index": 0, "total_chunks": 1, "text": pd.NA,
            }])
            df.to_parquet(Path(tmp) / "law_document_chunks.parquet", index=False)

            loader = ParquetLoaderAdapter()
            docs = loader.load(tmp)
            assert len(docs) == 1
            assert docs[0].page_content == ""


@pytest.fixture
def mock_deps():
    with patch("app.api.create_embeddings"), \
         patch("app.api.create_vector_store") as mock_vs, \
         patch("app.api.create_llm") as mock_llm, \
         patch("app.api.create_reranker") as mock_rerank, \
         patch("app.api.create_cache") as mock_cache, \
         patch("app.api.create_retriever") as mock_ret, \
         patch("app.api.create_query_transformer") as mock_qt, \
         patch("app.api.create_rate_limiter") as mock_rl, \
         patch("app.api.create_session_store") as mock_ss, \
         patch("app.api.create_knowledge_search_tool") as mock_kst, \
         patch("app.api.create_supervisor_agent") as mock_sup, \
         patch("app.api.create_agentic_service") as mock_ag, \
         patch("app.api.create_a2a_client") as mock_a2a:

        mock_vs.return_value = MagicMock()
        mock_llm.return_value = MagicMock()
        mock_rerank.return_value = MagicMock()
        mock_cache.return_value = MagicMock()
        mock_ret.return_value = MagicMock()
        mock_qt.return_value = MagicMock()
        mock_rl.return_value = AsyncMock()
        mock_ss.return_value = MagicMock()
        mock_kst.return_value = MagicMock()
        mock_sup.return_value = MagicMock()
        mock_ag.return_value = MagicMock()
        mock_a2a.return_value = MagicMock()

        app.state.rate_limiter = AsyncMock()
        app.state.agentic_service = None
        rag_service = AsyncMock()
        rag_service.answer = AsyncMock(return_value=("answer", ["source"], ["context"], None))
        app.state.rag_service = rag_service

        yield {
            "vector_store": mock_vs,
            "llm": mock_llm,
            "reranker": mock_rerank,
            "cache": mock_cache,
            "retriever": mock_ret,
            "query_transformer": mock_qt,
            "rate_limiter": mock_rl,
            "session_store": mock_ss,
            "knowledge_search_tool": mock_kst,
            "supervisor": mock_sup,
            "agentic_service": mock_ag,
            "a2a_client": mock_a2a,
        }


class TestHealthEndpoint:
    def test_health_check(self, client, mock_deps):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestAuthFlow:
    def test_login_success(self, client, mock_deps):
        with patch("app.auth.router.config") as mock_config:
            mock_config.admin_username = "admin"
            mock_config.admin_password = "admin"
            mock_config.jwt_secret = "test-secret"
            mock_config.jwt_algorithm = "HS256"
            mock_config.access_token_expire_minutes = 30
            mock_config.refresh_token_expire_days = 7

            response = client.post("/auth/token", json={"username": "admin", "password": "admin"})
            assert response.status_code == 200
            data = response.json()
            assert "access_token" in data
            assert "refresh_token" in data
            assert data["token_type"] == "Bearer"

    def test_login_invalid_credentials(self, client, mock_deps):
        with patch("app.auth.router.config") as mock_config:
            mock_config.admin_username = "admin"
            mock_config.admin_password = "admin"

            response = client.post("/auth/token", json={"username": "admin", "password": "wrong"})
            assert response.status_code == 401

    def test_protected_endpoint_requires_auth(self, client, mock_deps):
        with patch("app.auth.dependencies.config") as mock_config:
            mock_config.app_api_key = "test-api-key"
            response = client.post("/ask", json={"question": "test"})
            assert response.status_code == 401

    def test_protected_endpoint_with_valid_api_key(self, client, mock_deps):
        with patch("app.auth.dependencies.config") as mock_config:
            mock_config.app_api_key = "test-api-key"
            headers = {"X-API-Key": "test-api-key"}
            response = client.post("/ask", json={"question": "test"}, headers=headers)
            assert response.status_code in [200, 504]


class TestAskEndpoint:
    def test_ask_without_auth(self, client, mock_deps):
        with patch("app.auth.dependencies.config") as mock_config:
            mock_config.app_api_key = "test-api-key"
            response = client.post("/ask", json={"question": "test"})
            assert response.status_code == 401

    def test_ask_with_api_key(self, client, mock_deps):
        with patch("app.auth.dependencies.config") as mock_config:
            mock_config.app_api_key = "test-api-key"
            headers = {"X-API-Key": "test-api-key"}
            response = client.post("/ask", json={"question": "test"}, headers=headers)
            assert response.status_code in [200, 504]
