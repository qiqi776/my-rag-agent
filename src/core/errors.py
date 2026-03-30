"""Project-specific exception types."""

from __future__ import annotations


class MinimalRAGError(Exception):
    """Base class for project-specific errors."""


class ConfigError(MinimalRAGError):
    """Raised when configuration loading or validation fails."""


class UnsupportedFileTypeError(MinimalRAGError):
    """Raised when a requested file type is not supported."""


class EmptyQueryError(MinimalRAGError):
    """Raised when a query is empty or only whitespace."""

