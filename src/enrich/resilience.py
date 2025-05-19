"""Retry, backoff, and rate limiting.

Both primitives are async and dependency-free. They are deliberately small:
the goal is predictable behaviour under load, not a general-purpose scheduler.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .errors import ProviderError, ProviderTimeoutError, RateLimitError
from .logging import get_logger

T = TypeVar("T")

logger = get_logger(__name__)

# Only these are worth retrying. A malformed-request or auth failure will fail
# identically every time, and retrying it just multiplies the latency.
RETRYABLE = (ProviderTimeoutError, RateLimitError, asyncio.TimeoutError, ConnectionError)


class TokenBucket:
    """Async token bucket rate limiter.

    Tokens refill continuously at ``rate`` per second up to ``burst``. Callers
    await :meth:`acquire`, which sleeps only as long as needed for a token to
    become available.
    """

    def __init__(self, rate: float, burst: int) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        if burst < 1:
            raise ValueError("burst must be at least 1")

        self._rate = rate
        self._capacity = float(burst)
        self._tokens = float(burst)
        self._lock = asyncio.Lock()
        self._updated_at: float | None = None

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until ``tokens`` are available, then consume them."""
        if tokens > self._capacity:
            raise ValueError(
                f"requested {tokens} tokens exceeds bucket capacity {self._capacity}"
            )

        while True:
            async with self._lock:
                now = self._now()
                if self._updated_at is None:
                    self._updated_at = now

                elapsed = now - self._updated_at
                self._updated_at = now
                self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                deficit = tokens - self._tokens
                wait_for = deficit / self._rate

            # Sleep outside the lock so other callers can still refill.
            await asyncio.sleep(wait_for)


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay: float,
    provider: str,
    max_delay: float = 30.0,
) -> T:
    """Run ``operation``, retrying transient failures with exponential backoff.

    Backoff is ``base_delay * 2**attempt`` with full jitter, capped at
    ``max_delay``. Jitter matters when several workers fail against the same
    provider at once: without it they retry in lockstep and re-create the
    spike that caused the failure.

    Raises:
        ProviderError: the last failure, once retries are exhausted.
    """
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return await operation()
        except RETRYABLE as exc:
            last_error = exc
            if attempt == max_retries:
                break

            delay = min(base_delay * (2**attempt), max_delay)
            delay = random.uniform(0, delay)

            logger.warning(
                "provider call failed, retrying",
                extra={
                    "provider": provider,
                    "attempt": attempt + 1,
                    "max_retries": max_retries,
                    "delay_seconds": round(delay, 3),
                    "error": str(exc),
                },
            )
            await asyncio.sleep(delay)
        except ProviderError:
            # Non-retryable provider errors propagate immediately.
            raise

    assert last_error is not None
    raise ProviderError(provider, f"failed after {max_retries} retries: {last_error}")
