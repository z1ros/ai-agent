"""Provider interface and registry.

A provider turns a parsed email address into whatever it can learn about the
person behind it. Adding a commercial source (Clearbit, Apollo, People Data
Labs) means implementing :class:`EnrichmentProvider` and registering it. No
other module changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

from ..models import ParsedEmail, ProviderResponse


class EnrichmentProvider(ABC):
    """Base class for every enrichment source.

    Implementations must be safe to call concurrently: one instance is shared
    across all in-flight requests.
    """

    #: Stable identifier, used in config, logs, and attribute provenance.
    name: str = "base"

    #: Whether this provider needs network access. Used to skip it in offline
    #: test runs and to decide whether the rate limiter applies.
    requires_network: bool = True

    @abstractmethod
    async def enrich(
        self,
        parsed: ParsedEmail,
        *,
        signature_block: str | None = None,
    ) -> ProviderResponse:
        """Look up a person.

        Should return an empty profile rather than raising when it simply finds
        nothing. Raise :class:`~..errors.ProviderError` only for genuine
        failures (auth, timeout, malformed upstream response), so that "no data"
        and "provider broken" stay distinguishable.
        """

    def supports(self, parsed: ParsedEmail) -> bool:
        """Whether this provider is worth calling for the given address.

        Lets a provider opt out cheaply. A B2B source, for example, has nothing
        to say about a gmail.com address, and skipping it saves a paid call.
        """
        return True

    async def close(self) -> None:
        """Release any held resources. Called once on shutdown."""
        return None


ProviderFactory = Callable[[], EnrichmentProvider]

_REGISTRY: dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a provider factory under ``name``."""
    if name in _REGISTRY:
        raise ValueError(f"provider {name!r} is already registered")
    _REGISTRY[name] = factory


def build_provider(name: str) -> EnrichmentProvider:
    """Instantiate a registered provider by name."""
    try:
        factory = _REGISTRY[name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise ValueError(
            f"unknown provider {name!r} (registered: {available})"
        ) from None
    return factory()


def available_providers() -> list[str]:
    """Names of every registered provider."""
    return sorted(_REGISTRY)
