"""In-process TTL cache.

Enrichment results are stable over minutes-to-hours and providers charge per
call, so caching is a direct cost saving rather than only a latency one.

Deliberately in-process: it has no operational dependency and is correct for a
single instance. :class:`CacheBackend` is the seam to implement against when a
deployment needs Redis for cross-instance sharing.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


class CacheBackend(ABC, Generic[T]):
    """Interface for a cache implementation."""

    @abstractmethod
    async def get(self, key: str) -> T | None: ...

    @abstractmethod
    async def set(self, key: str, value: T) -> None: ...

    @abstractmethod
    async def clear(self) -> None: ...


@dataclass(slots=True)
class _Entry(Generic[T]):
    value: T
    expires_at: float


class TTLCache(CacheBackend[T]):
    """LRU cache with per-entry expiry.

    Bounded by ``max_entries`` so a long-running process cannot grow without
    limit on unique lookups, which is the usual way a naive dict cache turns
    into a memory leak.
    """

    def __init__(self, ttl_seconds: int, max_entries: int) -> None:
        if ttl_seconds < 0:
            raise ValueError("ttl_seconds must not be negative")
        if max_entries < 1:
            raise ValueError("max_entries must be at least 1")

        self._ttl = float(ttl_seconds)
        self._max_entries = max_entries
        self._entries: OrderedDict[str, _Entry[T]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits = 0
        self._misses = 0

    def _now(self) -> float:
        return asyncio.get_running_loop().time()

    async def get(self, key: str) -> T | None:
        async with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None

            if entry.expires_at <= self._now():
                del self._entries[key]
                self._misses += 1
                return None

            self._entries.move_to_end(key)
            self._hits += 1
            return entry.value

    async def set(self, key: str, value: T) -> None:
        if self._ttl == 0:
            return

        async with self._lock:
            self._entries[key] = _Entry(value=value, expires_at=self._now() + self._ttl)
            self._entries.move_to_end(key)

            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    async def clear(self) -> None:
        async with self._lock:
            self._entries.clear()

    async def stats(self) -> dict[str, float | int]:
        """Hit-rate metrics, suitable for a health or metrics endpoint."""
        async with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._entries),
                "max_entries": self._max_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
            }


class NullCache(CacheBackend[T]):
    """No-op cache used when caching is disabled.

    Lets the service call the cache unconditionally instead of branching on a
    config flag at every call site.
    """

    async def get(self, key: str) -> T | None:
        return None

    async def set(self, key: str, value: T) -> None:
        return None

    async def clear(self) -> None:
        return None
