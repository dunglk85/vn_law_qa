"""Unit tests for auth/jwt.py token lifecycle"""
from datetime import UTC, datetime

import pytest

from app.auth.jwt import create_access_token, create_refresh_token, decode_token, validate_claims


class TestAccessToken:
    def test_create_access_token(self):
        token = create_access_token(user_id="user1", tenant_id="tenant1", roles=["admin"])
        assert isinstance(token, str)
        assert len(token) > 0

    def test_decode_access_token(self):
        token = create_access_token(user_id="user1", tenant_id="tenant1", roles=["admin"])
        payload = decode_token(token)

        assert payload["sub"] == "user1"
        assert payload["tenant_id"] == "tenant1"
        assert payload["roles"] == ["admin"]

    def test_decode_invalid_token(self):

        with pytest.raises(ValueError, match="Invalid token"):
            decode_token("invalid.token.here")

    def test_access_token_has_expiry(self):
        token = create_access_token(user_id="user1")
        payload = decode_token(token)

        assert "exp" in payload
        assert payload["exp"] > datetime.now(UTC).timestamp()

    def test_access_token_default_roles(self):
        token = create_access_token(user_id="user1")
        payload = decode_token(token)

        assert payload["roles"] == []

    def test_access_token_default_tenant(self):
        token = create_access_token(user_id="user1")
        payload = decode_token(token)

        assert payload["tenant_id"] == ""


class TestRefreshToken:
    def test_create_refresh_token(self):
        token = create_refresh_token()
        assert isinstance(token, str)
        assert len(token) > 0

    def test_refresh_tokens_are_unique(self):
        token1 = create_refresh_token()
        token2 = create_refresh_token()
        assert token1 != token2


class TestValidateClaims:
    def test_validate_claims_success(self):
        payload = {
            "sub": "user1",
            "tenant_id": "tenant1",
            "roles": ["admin"],
        }
        validate_claims(payload)

    def test_validate_claims_missing_sub(self):
        payload = {
            "tenant_id": "tenant1",
            "roles": ["admin"],
        }
        with pytest.raises(ValueError, match="missing required claim: sub"):
            validate_claims(payload)

    def test_validate_claims_missing_tenant_id(self):
        payload = {
            "sub": "user1",
            "roles": ["admin"],
        }
        with pytest.raises(ValueError, match="missing required claim: tenant_id"):
            validate_claims(payload)

    def test_validate_claims_missing_roles(self):
        payload = {
            "sub": "user1",
            "tenant_id": "tenant1",
        }
        with pytest.raises(ValueError, match="missing required claim: roles"):
            validate_claims(payload)

    def test_validate_claims_empty_roles(self):
        payload = {
            "sub": "user1",
            "tenant_id": "tenant1",
            "roles": [],
        }
        validate_claims(payload)
