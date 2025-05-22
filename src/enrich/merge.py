"""Profile merging.

Providers disagree. This module decides which value wins, using a single rule:
higher confidence beats lower, and ties are broken by provider order (the order
in which they were configured, so an operator can express "trust Clearbit over
inference" purely through configuration).
"""

from __future__ import annotations

from .models import Attribute, Confidence, PersonProfile, ProviderResponse

_CONFIDENCE_RANK: dict[Confidence, int] = {
    Confidence.LOW: 0,
    Confidence.MEDIUM: 1,
    Confidence.HIGH: 2,
}

_FIELDS: tuple[str, ...] = tuple(PersonProfile.model_fields)


def _beats(candidate: Attribute, incumbent: Attribute | None) -> bool:
    """Whether ``candidate`` should replace ``incumbent``.

    Strictly greater, so an equal-confidence value from a later provider does
    not displace an earlier one. That keeps provider order meaningful.
    """
    if incumbent is None:
        return True
    return _CONFIDENCE_RANK[candidate.confidence] > _CONFIDENCE_RANK[incumbent.confidence]


def merge_profiles(responses: list[ProviderResponse]) -> PersonProfile:
    """Combine provider responses into one profile.

    ``responses`` must be in provider-priority order (highest priority first).
    """
    merged = PersonProfile()

    for response in responses:
        for field in _FIELDS:
            candidate = getattr(response.profile, field)
            if candidate is None:
                continue
            if _beats(candidate, getattr(merged, field)):
                setattr(merged, field, candidate)

    _reconcile_names(merged)
    return merged


def _reconcile_names(profile: PersonProfile) -> None:
    """Keep the name fields internally consistent.

    Without this, a HIGH-confidence ``full_name`` from a signature can coexist
    with LOW-confidence first/last names guessed from the address, which then
    disagree. Downstream consumers reading only ``first_name`` would get the
    guess instead of the fact.
    """
    if profile.full_name is None:
        if profile.first_name and profile.last_name:
            confidence = min(
                profile.first_name.confidence,
                profile.last_name.confidence,
                key=lambda value: _CONFIDENCE_RANK[value],
            )
            profile.full_name = Attribute(
                value=f"{profile.first_name.value} {profile.last_name.value}",
                confidence=confidence,
                source=profile.first_name.source,
            )
        return

    parts = profile.full_name.value.split()
    if len(parts) < 2:
        return

    full_rank = _CONFIDENCE_RANK[profile.full_name.confidence]

    if (
        profile.first_name is None
        or _CONFIDENCE_RANK[profile.first_name.confidence] < full_rank
    ):
        profile.first_name = Attribute(
            value=parts[0],
            confidence=profile.full_name.confidence,
            source=profile.full_name.source,
        )
    if (
        profile.last_name is None
        or _CONFIDENCE_RANK[profile.last_name.confidence] < full_rank
    ):
        profile.last_name = Attribute(
            value=parts[-1],
            confidence=profile.full_name.confidence,
            source=profile.full_name.source,
        )
