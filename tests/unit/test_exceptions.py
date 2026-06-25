"""Unit tests for exceptions.py"""
from app.exceptions import (
    AppError,
    AuthenticationError,
    AuthorizationError,
    ConfigurationError,
    ExternalServiceError,
    NotFoundError,
    RateLimitError,
    RequestTimeoutError,
    ValidationError,
)


class TestAppError:
    def test_init_defaults(self):
        error = AppError("Test error")
        assert error.message == "Test error"
        assert error.status_code == 500
        assert error.details == {}

    def test_init_with_status_code(self):
        error = AppError("Test error", status_code=404)
        assert error.status_code == 404

    def test_init_with_details(self):
        details = {"field": "value"}
        error = AppError("Test error", details=details)
        assert error.details == details

    def test_str_representation(self):
        error = AppError("Test error")
        assert str(error) == "Test error"


class TestSpecificExceptions:
    def test_configuration_error(self):
        error = ConfigurationError("Invalid config")
        assert error.status_code == 500
        assert error.message == "Invalid config"

    def test_authentication_error(self):
        error = AuthenticationError()
        assert error.status_code == 401
        assert error.message == "Authentication failed"

    def test_authentication_error_custom_message(self):
        error = AuthenticationError("Invalid credentials")
        assert error.message == "Invalid credentials"

    def test_authorization_error(self):
        error = AuthorizationError()
        assert error.status_code == 403
        assert error.message == "Insufficient permissions"

    def test_not_found_error(self):
        error = NotFoundError()
        assert error.status_code == 404
        assert error.message == "Resource not found"

    def test_validation_error(self):
        error = ValidationError("Invalid input")
        assert error.status_code == 400
        assert error.message == "Invalid input"

    def test_external_service_error(self):
        error = ExternalServiceError("Service unavailable")
        assert error.status_code == 502
        assert error.message == "Service unavailable"

    def test_timeout_error(self):
        error = RequestTimeoutError()
        assert error.status_code == 504
        assert error.message == "Operation timed out"

    def test_rate_limit_error(self):
        error = RateLimitError()
        assert error.status_code == 429
        assert error.message == "Rate limit exceeded"
