"""The enrichment service.

Composes parsing, providers, resilience, caching, and merging into one
entry point. This is the only object the transport layers (MCP, REST) talk to,
which keeps them thin and keeps the business logic testable without a server.
"""

from __future__ import annotations

import asyncio
import time
from types import TracebackType

from .cache import CacheBackend, NullCache, TTLCache
from .config import Settings, get_settings
from .errors import InvalidEmailError, ProviderError
from .logging import correlation_context, get_logger
from .merge import merge_profiles
from .models import (
    BatchEnrichmentResult,
    EnrichmentRequest,
    EnrichmentResult,
    ParsedEmail,
    ProviderResponse,
)
from .parsing import parse_email

# Importing the module registers the provider as a side effect.
from .providers import inference as _inference  # noqa: F401
from .providers.base import EnrichmentProvider, build_provider
from .resilience import TokenBucket, retry_async

logger = get_logger(__name__)


class EnrichmentService:
    """Enriches email addresses using the configured provider chain.

    Safe for concurrent use. Instantiate once per process and share it.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        providers: list[EnrichmentProvider] | None = None,
        cache: CacheBackend[EnrichmentResult] | None = None,
    ) -> None:
        self.settings = settings or get_settings()

        # Injectable for tests, built from config otherwise.
        self.providers = (
            providers
            if providers is not None
            else [build_provider(name) for name in self.settings.enabled_providers]
        )

        if cache is not None:
            self.cache = cache
        elif self.settings.cache_enabled:
            self.cache = TTLCache[EnrichmentResult](
                ttl_seconds=self.settings.cache_ttl_seconds,
                max_entries=self.settings.cache_max_entries,
            )
        else:
            self.cache = NullCache[EnrichmentResult]()

        self._limiter = TokenBucket(
            rate=self.settings.rate_limit_per_second,
            burst=self.settings.rate_limit_burst,
        )

    # -- lifecycle -------------------------------------------------------

    async def __aenter__(self) -> EnrichmentService:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Release provider resources. Idempotent."""
        await asyncio.gather(
            *(provider.close() for provider in self.providers),
            return_exceptions=True,
        )

    # -- public API ------------------------------------------------------

    async def enrich(self, request: EnrichmentRequest) -> EnrichmentResult:
        """Enrich a single address.

        Partial provider failure is not request failure: whatever succeeded is
        returned, and the failures are reported in ``errors``.

        Raises:
            InvalidEmailError: the address could not be parsed.
        """
        started = time.perf_counter()

        with correlation_context() as correlation_id:
            try:
                parsed = parse_email(str(request.email))
            except ValueError as exc:
                raise InvalidEmailError(str(exc)) from exc

            cache_key = self._cache_key(parsed.address, request.signature_block)

            if not request.skip_cache and (hit := await self.cache.get(cache_key)):
                logger.info("cache hit", extra={"email_domain": parsed.domain})
                return hit.model_copy(update={"cached": True})

            logger.info(
                "enrichment started",
                extra={
                    "email_domain": parsed.domain,
                    "kind": parsed.kind.value,
                    "correlation_id": correlation_id,
                },
            )

            responses, errors, queried = await self._query_providers(
                parsed, request.signature_block
            )

            result = EnrichmentResult(
                email=parsed.address,
                parsed=parsed,
                profile=merge_profiles(responses),
                providers_queried=queried,
                providers_succeeded=[r.provider for r in responses],
                errors=errors,
                cached=False,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )

            if not request.skip_cache and responses:
                await self.cache.set(cache_key, result)

            logger.info(
                "enrichment finished",
                extra={
                    "email_domain": parsed.domain,
                    "fields_found": len(result.profile.filled_fields()),
                    "providers_succeeded": len(responses),
                    "providers_failed": len(errors),
                    "duration_ms": result.duration_ms,
                },
            )
            return result

    async def enrich_batch(
        self, emails: list[str], *, skip_cache: bool = False
    ) -> BatchEnrichmentResult:
        """Enrich many addresses concurrently.

        One bad address does not fail the batch. The rate limiter bounds actual
        provider concurrency regardless of batch size.
        """

        async def run_one(raw: str) -> EnrichmentResult:
            # Build the request inside the task. Constructing it eagerly in a
            # comprehension would let a malformed address raise synchronously
            # and abort the whole batch before gather() ever runs.
            return await self.enrich(EnrichmentRequest(email=raw, skip_cache=skip_cache))

        settled = await asyncio.gather(
            *(run_one(email) for email in emails), return_exceptions=True
        )

        results: list[EnrichmentResult] = []
        failed = 0
        for email, outcome in zip(emails, settled, strict=True):
            if isinstance(outcome, EnrichmentResult):
                results.append(outcome)
            else:
                failed += 1
                logger.warning(
                    "batch item failed",
                    extra={"email": email, "error": str(outcome)},
                )

        return BatchEnrichmentResult(
            results=results,
            total=len(emails),
            succeeded=len(results),
            failed=failed,
        )

    # -- internals -------------------------------------------------------

    async def _query_providers(
        self, parsed: ParsedEmail, signature_block: str | None
    ) -> tuple[list[ProviderResponse], dict[str, str], list[str]]:
        """Run every applicable provider concurrently, collecting failures.

        Returns responses in configured provider order, which is what
        :func:`~.merge.merge_profiles` relies on for tie-breaking.
        """
        applicable = [p for p in self.providers if p.supports(parsed)]
        queried = [p.name for p in applicable]

        if not applicable:
            return [], {}, []

        settled = await asyncio.gather(
            *(self._call_provider(p, parsed, signature_block) for p in applicable),
            return_exceptions=True,
        )

        responses: list[ProviderResponse] = []
        errors: dict[str, str] = {}

        for provider, outcome in zip(applicable, settled, strict=True):
            if isinstance(outcome, ProviderResponse):
                responses.append(outcome)
            elif isinstance(outcome, BaseException):
                message = (
                    outcome.message
                    if isinstance(outcome, ProviderError)
                    else str(outcome)
                )
                errors[provider.name] = message
                logger.warning(
                    "provider failed",
                    extra={"provider": provider.name, "error": message},
                )

        return responses, errors, queried

    async def _call_provider(
        self,
        provider: EnrichmentProvider,
        parsed: ParsedEmail,
        signature_block: str | None,
    ) -> ProviderResponse:
        """Invoke one provider under the rate limiter, timeout, and retry policy."""
        if provider.requires_network:
            await self._limiter.acquire()

        async def attempt() -> ProviderResponse:
            return await asyncio.wait_for(
                provider.enrich(parsed, signature_block=signature_block),
                timeout=self.settings.provider_timeout_seconds,
            )

        return await retry_async(
            attempt,
            max_retries=self.settings.max_retries,
            base_delay=self.settings.retry_base_delay_seconds,
            provider=provider.name,
        )

    @staticmethod
    def _cache_key(address: str, signature_block: str | None) -> str:
        """Cache key.

        The signature block is part of the key: the same address with a
        signature yields a materially richer profile than without one, so they
        must not share an entry.
        """
        if not signature_block:
            return address
        return f"{address}:{hash(signature_block)}"
