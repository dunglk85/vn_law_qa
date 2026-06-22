"""Application exception hierarchy.

All custom exceptions inherit from AppError, which provides consistent
error handling and HTTP status code mapping.
"""
from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base exception for all application errors.

    Attributes:
        message: Human-readable error message
        status_code: HTTP status code to return
        details: Optional additional error details
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ConfigurationError(AppError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=500, details=details)


class AuthenticationError(AppError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=401, details=details)


class AuthorizationError(AppError):
    """Raised when user lacks required permissions."""

    def __init__(self, message: str = "Insufficient permissions", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=403, details=details)


class NotFoundError(AppError):
    """Raised when a requested resource is not found."""

    def __init__(self, message: str = "Resource not found", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=404, details=details)


class ValidationError(AppError):
    """Raised when input validation fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=400, details=details)


class ExternalServiceError(AppError):
    """Raised when an external service call fails."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=502, details=details)


class TimeoutError(AppError):
    """Raised when an operation times out."""

    def __init__(self, message: str = "Operation timed out", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=504, details=details)


class RateLimitError(AppError):
    """Raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded", details: dict[str, Any] | None = None) -> None:
        super().__init__(message, status_code=429, details=details)
