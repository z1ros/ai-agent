"""Application configuration.

Settings are loaded from environment variables (optionally via a .env file) and
validated once at import time by :func:`get_settings`. Invalid configuration
raises immediately at startup rather than surfacing as a confusing failure on
the first request.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError, field_validator

load_dotenv()


class Settings(BaseModel):
    """Validated runtime configuration."""

    # --- Model / LLM -----------------------------------------------------
    google_api_key: str | None = Field(
        default=None,
        description="Google AI Studio key. Only required when the LLM-backed "
        "signature parser is enabled.",
    )
    model_name: str = Field(
        default="gemini-2.0-flash",
        description="Model used for unstructured signature parsing.",
    )

    # --- Providers -------------------------------------------------------
    enabled_providers: list[str] = Field(
        default_factory=lambda: ["inference"],
        description="Ordered list of enrichment providers to query.",
    )
    provider_timeout_seconds: float = Field(default=10.0, gt=0, le=120)

    # --- Resilience ------------------------------------------------------
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_base_delay_seconds: float = Field(default=0.5, gt=0, le=30)
    rate_limit_per_second: float = Field(default=10.0, gt=0)
    rate_limit_burst: int = Field(default=20, ge=1)

    # --- Cache -----------------------------------------------------------
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, ge=0)
    cache_max_entries: int = Field(default=10_000, ge=1)

    # --- Server ----------------------------------------------------------
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"

    @field_validator("enabled_providers", mode="before")
    @classmethod
    def _split_providers(cls, value: object) -> object:
        """Accept a comma-separated string from the environment."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("enabled_providers")
    @classmethod
    def _require_at_least_one(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("at least one provider must be enabled")
        return value


def _env(name: str) -> str | None:
    """Read an environment variable, treating blank strings as unset."""
    raw = os.getenv(name)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def _build_settings() -> Settings:
    """Assemble settings from the environment.

    Only keys that are actually present are passed through, so unset variables
    fall back to the model defaults rather than overriding them with ``None``.
    """
    candidates = {
        "google_api_key": _env("GOOGLE_API_KEY"),
        "model_name": _env("MODEL_NAME"),
        "enabled_providers": _env("ENABLED_PROVIDERS"),
        "provider_timeout_seconds": _env("PROVIDER_TIMEOUT_SECONDS"),
        "max_retries": _env("MAX_RETRIES"),
        "retry_base_delay_seconds": _env("RETRY_BASE_DELAY_SECONDS"),
        "rate_limit_per_second": _env("RATE_LIMIT_PER_SECOND"),
        "rate_limit_burst": _env("RATE_LIMIT_BURST"),
        "cache_enabled": _env("CACHE_ENABLED"),
        "cache_ttl_seconds": _env("CACHE_TTL_SECONDS"),
        "cache_max_entries": _env("CACHE_MAX_ENTRIES"),
        "host": _env("HOST"),
        "port": _env("PORT"),
        "log_level": _env("LOG_LEVEL"),
        "log_format": _env("LOG_FORMAT"),
    }
    provided = {key: value for key, value in candidates.items() if value is not None}
    # Values arrive as strings because that is all an environment can hold;
    # pydantic coerces each one to its declared field type during validation.
    # model_validate takes the mapping as data rather than typed kwargs, which
    # is both the correct entry point for this and type-checkable.
    return Settings.model_validate(provided)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Raises:
        RuntimeError: if the environment holds invalid configuration. The
            underlying validation errors are included so the operator can see
            every problem at once instead of fixing them one restart at a time.
    """
    try:
        return _build_settings()
    except ValidationError as exc:
        raise RuntimeError(f"Invalid configuration:\n{exc}") from exc
