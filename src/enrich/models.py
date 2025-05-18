"""Typed domain models.

Every boundary in the system (API request, provider response, MCP tool result)
is described here. Validation happens at the edge so malformed data fails with
a precise error instead of propagating as ``None`` into downstream code.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Confidence(str, Enum):
    """How much to trust a given field.

    Providers disagree, and a caller needs to know whether a value was read
    directly out of a signature block or guessed from a domain name.
    """

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EmailKind(str, Enum):
    """Classification of an address, used to skip pointless enrichment."""

    PERSONAL = "personal"
    CORPORATE = "corporate"
    ROLE = "role"
    NO_REPLY = "no_reply"
    UNKNOWN = "unknown"


class ParsedEmail(BaseModel):
    """The deterministic breakdown of an email address."""

    model_config = ConfigDict(frozen=True)

    address: EmailStr
    local_part: str
    domain: str
    kind: EmailKind = EmailKind.UNKNOWN
    is_free_provider: bool = False

    @field_validator("domain")
    @classmethod
    def _normalize_domain(cls, value: str) -> str:
        return value.strip().lower().rstrip(".")


class Attribute(BaseModel):
    """A single enriched fact, carrying its provenance.

    Provenance is the point. A bare ``{"title": "CTO"}`` is unusable in a
    pipeline because the caller cannot tell whether it was scraped, inferred,
    or hallucinated.
    """

    value: str
    confidence: Confidence
    source: str = Field(description="Provider name that produced this value.")

    @field_validator("value")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("attribute value must not be blank")
        return stripped


class PersonProfile(BaseModel):
    """What we managed to learn about a person.

    Every field is optional. Enrichment is best-effort by nature, and a partial
    profile is a normal, useful result rather than a failure.
    """

    full_name: Attribute | None = None
    first_name: Attribute | None = None
    last_name: Attribute | None = None
    title: Attribute | None = None
    company: Attribute | None = None
    company_domain: Attribute | None = None
    location: Attribute | None = None
    phone: Attribute | None = None
    linkedin_url: Attribute | None = None

    def is_empty(self) -> bool:
        """True when no provider contributed anything."""
        return all(value is None for value in self.model_dump().values())

    def filled_fields(self) -> list[str]:
        """Names of the fields that actually carry a value."""
        return [name for name, value in self if value is not None]


class EnrichmentRequest(BaseModel):
    """Inbound request to enrich a single address."""

    email: EmailStr
    signature_block: str | None = Field(
        default=None,
        max_length=4000,
        description="Raw signature text, if the caller has it. Greatly "
        "improves result quality.",
    )
    skip_cache: bool = False


class EnrichmentResult(BaseModel):
    """The response returned to callers."""

    email: EmailStr
    parsed: ParsedEmail
    profile: PersonProfile
    providers_queried: list[str] = Field(default_factory=list)
    providers_succeeded: list[str] = Field(default_factory=list)
    errors: dict[str, str] = Field(
        default_factory=dict,
        description="Provider name -> error message, for providers that failed. "
        "A partial failure does not fail the request.",
    )
    cached: bool = False
    duration_ms: float = 0.0
    retrieved_at: datetime = Field(default_factory=_utcnow)


class BatchEnrichmentRequest(BaseModel):
    """Enrich many addresses in one call."""

    emails: Annotated[list[EmailStr], Field(min_length=1, max_length=100)]
    skip_cache: bool = False


class BatchEnrichmentResult(BaseModel):
    """Results for a batch, plus a rollup so callers can alert on failure rate."""

    results: list[EnrichmentResult]
    total: int
    succeeded: int
    failed: int


class ProviderResponse(BaseModel):
    """Raw provider output, before it is merged into a profile."""

    model_config = ConfigDict(extra="allow")

    provider: str
    profile: PersonProfile
    raw: dict[str, Any] = Field(default_factory=dict)
