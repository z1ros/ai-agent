"""Tests for profile merging, caching, and the resilience primitives."""

from __future__ import annotations

import asyncio

import pytest

from enrich.cache import NullCache, TTLCache
from enrich.errors import ProviderError, ProviderTimeoutError
from enrich.merge import merge_profiles
from enrich.models import Attribute, Confidence, PersonProfile, ProviderResponse
from enrich.resilience import TokenBucket, retry_async


def attr(value: str, confidence: Confidence, source: str = "test") -> Attribute:
    return Attribute(value=value, confidence=confidence, source=source)


def response(provider: str, **fields) -> ProviderResponse:
    return ProviderResponse(provider=provider, profile=PersonProfile(**fields))


class TestMerge:
    def test_higher_confidence_wins(self) -> None:
        merged = merge_profiles(
            [
                response("low", company=attr("Guess", Confidence.LOW)),
                response("high", company=attr("Acme Inc", Confidence.HIGH)),
            ]
        )
        assert merged.company.value == "Acme Inc"

    def test_ties_break_by_provider_order(self) -> None:
        """Equal confidence keeps the earlier provider, so config expresses trust."""
        merged = merge_profiles(
            [
                response("first", company=attr("First", Confidence.MEDIUM)),
                response("second", company=attr("Second", Confidence.MEDIUM)),
            ]
        )
        assert merged.company.value == "First"

    def test_fields_combine_across_providers(self) -> None:
        merged = merge_profiles(
            [
                response("a", company=attr("Acme", Confidence.MEDIUM)),
                response("b", title=attr("CTO", Confidence.HIGH)),
            ]
        )
        assert merged.company.value == "Acme"
        assert merged.title.value == "CTO"

    def test_empty_input_yields_empty_profile(self) -> None:
        assert merge_profiles([]).is_empty()

    def test_provenance_survives_merging(self) -> None:
        merged = merge_profiles([response("clearbit", title=attr("CTO", Confidence.HIGH, "clearbit"))])
        assert merged.title.source == "clearbit"


class TestNameReconciliation:
    def test_full_name_derives_missing_parts(self) -> None:
        merged = merge_profiles(
            [response("sig", full_name=attr("Jane Doe", Confidence.HIGH))]
        )
        assert merged.first_name.value == "Jane"
        assert merged.last_name.value == "Doe"
        assert merged.first_name.confidence is Confidence.HIGH

    def test_high_confidence_full_name_overrides_low_parts(self) -> None:
        """A signature-read name must beat a guess from the address.

        Otherwise a consumer reading only first_name gets the guess while
        full_name says something different.
        """
        merged = merge_profiles(
            [
                response("sig", full_name=attr("Jane Doe", Confidence.HIGH)),
                response(
                    "guess",
                    first_name=attr("Jdoe", Confidence.LOW),
                    last_name=attr("Wrong", Confidence.LOW),
                ),
            ]
        )
        assert merged.first_name.value == "Jane"
        assert merged.last_name.value == "Doe"

    def test_parts_compose_a_missing_full_name(self) -> None:
        merged = merge_profiles(
            [
                response(
                    "a",
                    first_name=attr("Jane", Confidence.MEDIUM),
                    last_name=attr("Doe", Confidence.MEDIUM),
                )
            ]
        )
        assert merged.full_name.value == "Jane Doe"

    def test_composed_full_name_takes_the_weaker_confidence(self) -> None:
        merged = merge_profiles(
            [
                response(
                    "a",
                    first_name=attr("Jane", Confidence.HIGH),
                    last_name=attr("Doe", Confidence.LOW),
                )
            ]
        )
        assert merged.full_name.confidence is Confidence.LOW

    def test_single_word_full_name_is_left_alone(self) -> None:
        merged = merge_profiles(
            [response("a", full_name=attr("Cher", Confidence.HIGH))]
        )
        assert merged.first_name is None


class TestTTLCache:
    async def test_stores_and_retrieves(self) -> None:
        cache: TTLCache[str] = TTLCache(ttl_seconds=60, max_entries=10)
        await cache.set("k", "v")
        assert await cache.get("k") == "v"

    async def test_missing_key_returns_none(self) -> None:
        cache: TTLCache[str] = TTLCache(ttl_seconds=60, max_entries=10)
        assert await cache.get("absent") is None

    async def test_entries_expire(self) -> None:
        cache: TTLCache[str] = TTLCache(ttl_seconds=0, max_entries=10)
        await cache.set("k", "v")
        assert await cache.get("k") is None

    async def test_evicts_least_recently_used(self) -> None:
        cache: TTLCache[str] = TTLCache(ttl_seconds=60, max_entries=2)
        await cache.set("a", "1")
        await cache.set("b", "2")
        await cache.get("a")  # 'a' becomes most recently used
        await cache.set("c", "3")

        assert await cache.get("b") is None
        assert await cache.get("a") == "1"
        assert await cache.get("c") == "3"

    async def test_reports_hit_rate(self) -> None:
        cache: TTLCache[str] = TTLCache(ttl_seconds=60, max_entries=10)
        await cache.set("k", "v")
        await cache.get("k")
        await cache.get("missing")

        stats = await cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    async def test_clear_empties_the_cache(self) -> None:
        cache: TTLCache[str] = TTLCache(ttl_seconds=60, max_entries=10)
        await cache.set("k", "v")
        await cache.clear()
        assert await cache.get("k") is None

    @pytest.mark.parametrize(
        ("ttl", "max_entries"), [(-1, 10), (60, 0)]
    )
    def test_rejects_invalid_construction(self, ttl: int, max_entries: int) -> None:
        with pytest.raises(ValueError):
            TTLCache(ttl_seconds=ttl, max_entries=max_entries)


class TestNullCache:
    async def test_never_stores(self) -> None:
        cache: NullCache[str] = NullCache()
        await cache.set("k", "v")
        assert await cache.get("k") is None


class TestTokenBucket:
    async def test_burst_is_served_immediately(self) -> None:
        bucket = TokenBucket(rate=1.0, burst=5)
        loop = asyncio.get_running_loop()
        started = loop.time()

        for _ in range(5):
            await bucket.acquire()

        assert loop.time() - started < 0.1

    async def test_throttles_beyond_burst(self) -> None:
        bucket = TokenBucket(rate=20.0, burst=1)
        loop = asyncio.get_running_loop()
        started = loop.time()

        await bucket.acquire()
        await bucket.acquire()

        # The second call must wait roughly 1/rate seconds.
        assert loop.time() - started >= 0.03

    async def test_rejects_request_larger_than_capacity(self) -> None:
        bucket = TokenBucket(rate=1.0, burst=2)
        with pytest.raises(ValueError):
            await bucket.acquire(tokens=5)

    @pytest.mark.parametrize(("rate", "burst"), [(0, 5), (-1, 5), (1, 0)])
    def test_rejects_invalid_construction(self, rate: float, burst: int) -> None:
        with pytest.raises(ValueError):
            TokenBucket(rate=rate, burst=burst)


class TestRetryAsync:
    async def test_returns_on_first_success(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        result = await retry_async(
            operation, max_retries=3, base_delay=0.001, provider="p"
        )
        assert result == "ok"
        assert calls == 1

    async def test_retries_then_succeeds(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise ProviderTimeoutError("p", "timeout")
            return "ok"

        result = await retry_async(
            operation, max_retries=3, base_delay=0.001, provider="p"
        )
        assert result == "ok"
        assert calls == 3

    async def test_raises_after_exhausting_retries(self) -> None:
        async def operation() -> str:
            raise ProviderTimeoutError("p", "timeout")

        with pytest.raises(ProviderError):
            await retry_async(
                operation, max_retries=2, base_delay=0.001, provider="p"
            )

    async def test_does_not_retry_permanent_errors(self) -> None:
        calls = 0

        async def operation() -> str:
            nonlocal calls
            calls += 1
            raise ProviderError("p", "bad credentials")

        with pytest.raises(ProviderError):
            await retry_async(
                operation, max_retries=5, base_delay=0.001, provider="p"
            )
        assert calls == 1
