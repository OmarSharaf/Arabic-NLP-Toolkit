"""
Custom exceptions for arabic-nlp-toolkit.

All library exceptions inherit from ArabicNLPError so callers
can catch them with a single except clause if needed.
"""

from __future__ import annotations


class ArabicNLPError(Exception):
    """Base exception for all arabic-nlp-toolkit errors."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        if self.code:
            return f"[{self.code}] {self.message}"
        return self.message


class ModelNotFoundError(ArabicNLPError):
    """Raised when a required model file is not found."""

    def __init__(self, model_name: str, path: str | None = None) -> None:
        msg = f"Model '{model_name}' not found"
        if path:
            msg += f" at path: {path}"
        msg += ". Run `arabic-nlp download` to fetch required models."
        super().__init__(msg, code="MODEL_NOT_FOUND")
        self.model_name = model_name
        self.path = path


class InvalidInputError(ArabicNLPError):
    """Raised when the input text is invalid or empty."""

    def __init__(self, reason: str = "Input text must be a non-empty string") -> None:
        super().__init__(reason, code="INVALID_INPUT")


class UnsupportedDialectError(ArabicNLPError):
    """Raised when an unsupported dialect is specified."""

    def __init__(self, dialect: str, supported: list[str] | None = None) -> None:
        msg = f"Dialect '{dialect}' is not supported"
        if supported:
            msg += f". Supported dialects: {', '.join(supported)}"
        super().__init__(msg, code="UNSUPPORTED_DIALECT")
        self.dialect = dialect


class UnsupportedLanguageError(ArabicNLPError):
    """Raised when non-Arabic text is passed to an Arabic-only function."""

    def __init__(self, detected_language: str | None = None) -> None:
        msg = "Input does not appear to be Arabic text"
        if detected_language:
            msg += f" (detected: {detected_language})"
        super().__init__(msg, code="UNSUPPORTED_LANGUAGE")
        self.detected_language = detected_language


class ConfigurationError(ArabicNLPError):
    """Raised when the library is misconfigured."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"Configuration error: {reason}", code="CONFIG_ERROR")


class BatchProcessingError(ArabicNLPError):
    """Raised when a batch operation partially or fully fails."""

    def __init__(self, failed: int, total: int, details: str | None = None) -> None:
        msg = f"Batch processing failed: {failed}/{total} items failed"
        if details:
            msg += f". Details: {details}"
        super().__init__(msg, code="BATCH_ERROR")
        self.failed = failed
        self.total = total
