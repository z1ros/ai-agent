"""Tests for the enrichment service.

No live provider calls: every provider here is a stub, so the suite is
deterministic and runs offline in CI.
"""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from enrich.config import Settings
from enrich.errors import InvalidEmailError, ProviderError, ProviderTimeoutError
from enrich.models import (
    Attribute,
    Confidence,
    EnrichmentRequest,
    ParsedEmail,
    PersonProfile,
    ProviderResponse,
)
from enrich.providers.base import EnrichmentProvider
from enrich.providers.inference import InferenceProvider
from enrich.service import EnrichmentService

SIGNATURE = """Jane Doe
Chief Technology Officer
Acme Robotics
+1 (415) 555-2671
https://linkedin.com/in/janedoe
"""


class StubProvider(EnrichmentProvider):
    """Returns a fixed profile and counts how often it was called."""

    requires_network = False

    def __init__(self, name: str, profile: PersonProfile | None = None) -> None:
        self.name = name
        self._profile = profile or PersonProfile()
        self.calls = 0

    async def enrich(self, parsed, *, signature_block=None) -> ProviderResponse:
        self.calls += 1
        return ProviderResponse(provider=self.name, profile=self._profile)


class FailingProvider(EnrichmentProvider):
    """Always raises, to exercise partial-failure handling."""

    requires_network = False

    def __init__(self, name: str = "failing", exc: Exception | None = None) -> None:
        self.name = name
        self._exc = exc or ProviderError(name, "upstream exploded")
        self.calls = 0

    async def enrich(self, parsed, *, signature_block=None) -> ProviderResponse:
        self.calls += 1
        raise self._exc


class SlowProvider(EnrichmentProvider):
    """Sleeps past the configured timeout."""

    name = "slow"
    requires_network = False

    async def enrich(self, parsed, *, signature_block=None) -> ProviderResponse:
        await asyncio.sleep(5)
        return ProviderResponse(provider=self.name, profile=PersonProfile())


def make_settings(**overrides) -> Settings:
    base = {
        "enabled_providers": ["inference"],
        "cache_enabled": False,
        "max_retries": 0,
        "provider_timeout_seconds": 1.0,
    }
    return Settings(**{**base, **overrides})


@pytest.fixture
def service() -> EnrichmentService:
    return EnrichmentService(
        settings=make_settings(), providers=[InferenceProvider()]
    )


class TestEnrich:
    async def test_corporate_address(self, service: EnrichmentService) -> None:
        result = await service.enrich(
            EnrichmentRequest(email="jane.doe@acmerobotics.com")
        )
        assert result.profile.first_name is not None
        assert result.profile.first_name.value == "Jane"
        assert result.profile.company is not None
        assert result.profile.company.value == "Acmerobotics"
        assert result.providers_succeeded == ["inference"]
        assert result.errors == {}

    async def test_signature_upgrades_confidence(
        self, service: EnrichmentService
    ) -> None:
        without = await service.enrich(
            EnrichmentRequest(email="jane.doe@acmerobotics.com")
        )
        with_sig = await service.enrich(
            EnrichmentRequest(
                email="jane.doe@acmerobotics.com", signature_block=SIGNATURE
            )
        )

        assert without.profile.full_name.confidence is Confidence.LOW
        assert with_sig.profile.full_name.confidence is Confidence.HIGH
        assert with_sig.profile.title is not None
        assert with_sig.profile.phone is not None
        assert with_sig.profile.linkedin_url is not None

    async def test_role_account_yields_company_but_no_name(
        self, service: EnrichmentService
    ) -> None:
        """sales@ is a mailbox, not a person. Inventing a name poisons a CRM."""
        result = await service.enrich(EnrichmentRequest(email="sales@acme.com"))
        assert result.profile.first_name is None
        assert result.profile.full_name is None
        assert result.profile.company is not None

    async def test_no_reply_is_skipped_entirely(
        self, service: EnrichmentService
    ) -> None:
        result = await service.enrich(EnrichmentRequest(email="noreply@acme.com"))
        assert result.providers_queried == []
        assert result.profile.is_empty()

    async def test_free_provider_yields_no_company(
        self, service: EnrichmentService
    ) -> None:
        result = await service.enrich(EnrichmentRequest(email="jane.doe@gmail.com"))
        assert result.profile.company is None
        assert result.profile.first_name.value == "Jane"

    async def test_malformed_address_rejected_at_the_model_boundary(self) -> None:
        """EmailStr catches obvious garbage before the service is involved."""
        with pytest.raises(ValidationError):
            EnrichmentRequest(email="jane@localhost")

    async def test_service_rejects_addresses_that_pass_the_model(
        self, service: EnrichmentService
    ) -> None:
        """Our parser is stricter than EmailStr, so it needs its own guard.

        Constructed via model_construct to bypass validation, which is how a
        caller reaching the service directly (rather than through a transport)
        could supply an unvalidated address.
        """
        request = EnrichmentRequest.model_construct(
            email="not-an-email", signature_block=None, skip_cache=False
        )
        with pytest.raises(InvalidEmailError):
            await service.enrich(request)

    async def test_duration_is_recorded(self, service: EnrichmentService) -> None:
        result = await service.enrich(EnrichmentRequest(email="jane@acme.com"))
        assert result.duration_ms >= 0


class TestPartialFailure:
    async def test_one_provider_failing_does_not_fail_request(self) -> None:
        good = StubProvider(
            "good",
            PersonProfile(
                company=Attribute(
                    value="Acme", confidence=Confidence.HIGH, source="good"
                )
            ),
        )
        service = EnrichmentService(
            settings=make_settings(), providers=[good, FailingProvider()]
        )

        result = await service.enrich(EnrichmentRequest(email="jane@acme.com"))

        assert result.profile.company.value == "Acme"
        assert result.providers_succeeded == ["good"]
        assert "failing" in result.errors
        assert "upstream exploded" in result.errors["failing"]

    async def test_all_providers_failing_returns_empty_profile(self) -> None:
        service = EnrichmentService(
            settings=make_settings(),
            providers=[FailingProvider("a"), FailingProvider("b")],
        )
        result = await service.enrich(EnrichmentRequest(email="jane@acme.com"))

        assert result.profile.is_empty()
        assert set(result.errors) == {"a", "b"}
        assert result.providers_succeeded == []

    async def test_provider_timeout_is_captured(self) -> None:
        service = EnrichmentService(
            settings=make_settings(provider_timeout_seconds=0.05),
            providers=[SlowProvider()],
        )
        result = await service.enrich(EnrichmentRequest(email="jane@acme.com"))
        assert "slow" in result.errors


class TestRetry:
    async def test_retries_transient_failures(self) -> None:
        provider = FailingProvider(
            "flaky", exc=ProviderTimeoutError("flaky", "timed out")
        )
        service = EnrichmentService(
            settings=make_settings(max_retries=2, retry_base_delay_seconds=0.001),
            providers=[provider],
        )
        await service.enrich(EnrichmentRequest(email="jane@acme.com"))
        # Initial attempt plus two retries.
        assert provider.calls == 3

    async def test_does_not_retry_permanent_failures(self) -> None:
        provider = FailingProvider("broken", exc=ProviderError("broken", "bad key"))
        service = EnrichmentService(
            settings=make_settings(max_retries=3, retry_base_delay_seconds=0.001),
            providers=[provider],
        )
        await service.enrich(EnrichmentRequest(email="jane@acme.com"))
        assert provider.calls == 1


class TestCaching:
    async def test_second_call_is_served_from_cache(self) -> None:
        provider = StubProvider("stub")
        service = EnrichmentService(
            settings=make_settings(cache_enabled=True, cache_ttl_seconds=60),
            providers=[provider],
        )

        first = await service.enrich(EnrichmentRequest(email="jane@acme.com"))
        second = await service.enrich(EnrichmentRequest(email="jane@acme.com"))

        assert first.cached is False
        assert second.cached is True
        assert provider.calls == 1

    async def test_skip_cache_forces_a_fresh_call(self) -> None:
        provider = StubProvider("stub")
        service = EnrichmentService(
            settings=make_settings(cache_enabled=True, cache_ttl_seconds=60),
            providers=[provider],
        )

        await service.enrich(EnrichmentRequest(email="jane@acme.com"))
        result = await service.enrich(
            EnrichmentRequest(email="jane@acme.com", skip_cache=True)
        )

        assert result.cached is False
        assert provider.calls == 2

    async def test_signature_is_part_of_the_cache_key(self) -> None:
        """The same address with a signature is a materially different result."""
        provider = StubProvider("stub")
        service = EnrichmentService(
            settings=make_settings(cache_enabled=True, cache_ttl_seconds=60),
            providers=[provider],
        )

        await service.enrich(EnrichmentRequest(email="jane@acme.com"))
        await service.enrich(
            EnrichmentRequest(email="jane@acme.com", signature_block=SIGNATURE)
        )

        assert provider.calls == 2


class TestBatch:
    async def test_enriches_many(self, service: EnrichmentService) -> None:
        result = await service.enrich_batch(
            ["jane.doe@acme.com", "ivan.petrov@corp.io"]
        )
        assert result.total == 2
        assert result.succeeded == 2
        assert result.failed == 0

    async def test_isolates_a_bad_address(self, service: EnrichmentService) -> None:
        """Regression: a malformed address previously aborted the whole batch."""
        result = await service.enrich_batch(
            ["jane.doe@acme.com", "not-an-email", "ivan@corp.io"]
        )
        assert result.total == 3
        assert result.succeeded == 2
        assert result.failed == 1

    async def test_all_bad_addresses(self, service: EnrichmentService) -> None:
        result = await service.enrich_batch(["nope", "also-nope"])
        assert result.succeeded == 0
        assert result.failed == 2


class TestProviderSelection:
    async def test_unsupported_providers_are_not_called(self) -> None:
        class PickyProvider(StubProvider):
            def supports(self, parsed: ParsedEmail) -> bool:
                return not parsed.is_free_provider

        picky = PickyProvider("picky")
        service = EnrichmentService(settings=make_settings(), providers=[picky])

        await service.enrich(EnrichmentRequest(email="jane@gmail.com"))
        assert picky.calls == 0

        await service.enrich(EnrichmentRequest(email="jane@acme.com"))
        assert picky.calls == 1


class TestLifecycle:
    async def test_context_manager_closes_providers(self) -> None:
        closed: list[str] = []

        class ClosableProvider(StubProvider):
            async def close(self) -> None:
                closed.append(self.name)

        provider = ClosableProvider("closable")
        async with EnrichmentService(
            settings=make_settings(), providers=[provider]
        ) as svc:
            await svc.enrich(EnrichmentRequest(email="jane@acme.com"))

        assert closed == ["closable"]
