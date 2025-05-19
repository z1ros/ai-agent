"""Exception hierarchy.

A single base class lets callers catch everything this package raises without
also swallowing unrelated runtime errors. Each subclass maps cleanly onto an
HTTP status in the REST layer.
"""

from __future__ import annotations


class EnrichmentError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(EnrichmentError):
    """Settings are missing or invalid. Not recoverable at runtime."""


class InvalidEmailError(EnrichmentError):
    """The supplied address could not be parsed. Maps to HTTP 422."""


class ProviderError(EnrichmentError):
    """A provider failed.

    Carries the provider name so partial failures can be reported per-provider
    instead of collapsing into one opaque message.
    """

    def __init__(self, provider: str, message: str) -> None:
        self.provider = provider
        self.message = message
        super().__init__(f"[{provider}] {message}")


class ProviderTimeoutError(ProviderError):
    """A provider exceeded its deadline. Retryable."""


class RateLimitError(ProviderError):
    """A provider rejected the call for rate limiting. Retryable with backoff."""


class AllProvidersFailedError(EnrichmentError):
    """Every configured provider failed. Maps to HTTP 502."""

    def __init__(self, errors: dict[str, str]) -> None:
        self.errors = errors
        detail = "; ".join(f"{name}: {msg}" for name, msg in errors.items())
        super().__init__(f"all providers failed ({detail})")
