"""Zero-dependency enrichment from the address itself.

Everything here is derived from the address and (when supplied) the signature
block. No network, no API key, no cost. It is the default provider so the
service is useful out of the box, and it establishes the baseline that paid
providers are merged on top of.

Confidence is assigned honestly: values read out of a signature block are
HIGH, values derived from a domain are MEDIUM, and values guessed from a local
part are LOW.
"""

from __future__ import annotations

import re

from ..models import Attribute, Confidence, EmailKind, ParsedEmail, PersonProfile, ProviderResponse
from ..parsing import company_from_domain, guess_name_parts
from .base import EnrichmentProvider, register_provider

# Matches most international formats without being so loose that it captures
# order numbers and zip codes.
_PHONE_RE = re.compile(
    r"(?:(?<=\s)|^)(\+?\d{1,3}[\s.\-]?)?(\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{2,4}\b"
)
_LINKEDIN_RE = re.compile(
    r"(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/in/([A-Za-z0-9\-_%]{3,100})",
    re.IGNORECASE,
)
_TITLE_KEYWORDS = (
    "ceo", "cto", "cfo", "coo", "cmo", "founder", "co-founder", "cofounder",
    "president", "director", "manager", "head of", "lead", "engineer",
    "developer", "designer", "analyst", "consultant", "specialist",
    "coordinator", "associate", "partner", "principal", "architect",
    "scientist", "researcher", "intern", "vp", "vice president", "chief",
)


def _clean_line(line: str) -> str:
    return line.strip().strip("|·—–-").strip()


class InferenceProvider(EnrichmentProvider):
    """Derives what it can from the address and signature text alone."""

    name = "inference"
    requires_network = False

    def supports(self, parsed: ParsedEmail) -> bool:
        # Automated senders have no person behind them.
        return parsed.kind is not EmailKind.NO_REPLY

    async def enrich(
        self,
        parsed: ParsedEmail,
        *,
        signature_block: str | None = None,
    ) -> ProviderResponse:
        profile = PersonProfile()

        if parsed.kind is EmailKind.ROLE:
            # A shared mailbox has a company but no individual. Inferring a
            # name from "sales@acme.com" would invent a person.
            self._apply_company(profile, parsed)
            return ProviderResponse(
                provider=self.name,
                profile=profile,
                raw={"kind": parsed.kind.value, "reason": "role account"},
            )

        self._apply_name(profile, parsed)
        self._apply_company(profile, parsed)

        if signature_block:
            self._apply_signature(profile, signature_block)

        return ProviderResponse(
            provider=self.name,
            profile=profile,
            raw={
                "kind": parsed.kind.value,
                "domain": parsed.domain,
                "had_signature": bool(signature_block),
            },
        )

    # -- component extractors -------------------------------------------

    def _apply_name(self, profile: PersonProfile, parsed: ParsedEmail) -> None:
        first, last = guess_name_parts(parsed.local_part)
        if first:
            profile.first_name = Attribute(
                value=first, confidence=Confidence.LOW, source=self.name
            )
        if last:
            profile.last_name = Attribute(
                value=last, confidence=Confidence.LOW, source=self.name
            )
        if first and last:
            profile.full_name = Attribute(
                value=f"{first} {last}", confidence=Confidence.LOW, source=self.name
            )

    def _apply_company(self, profile: PersonProfile, parsed: ParsedEmail) -> None:
        if parsed.is_free_provider:
            return

        company = company_from_domain(parsed.domain)
        if company:
            profile.company = Attribute(
                value=company, confidence=Confidence.MEDIUM, source=self.name
            )
            profile.company_domain = Attribute(
                value=parsed.domain, confidence=Confidence.HIGH, source=self.name
            )

    def _apply_signature(self, profile: PersonProfile, block: str) -> None:
        """Read fields out of a signature block.

        Signature values override inferred ones: a person writing their own
        name is far better evidence than a guess from an address.
        """
        lines = [_clean_line(line) for line in block.splitlines()]
        lines = [line for line in lines if line]

        if match := _LINKEDIN_RE.search(block):
            profile.linkedin_url = Attribute(
                value=f"https://linkedin.com/in/{match.group(1)}",
                confidence=Confidence.HIGH,
                source=self.name,
            )

        if match := _PHONE_RE.search(block):
            phone = match.group(0).strip()
            if sum(char.isdigit() for char in phone) >= 7:
                profile.phone = Attribute(
                    value=phone, confidence=Confidence.HIGH, source=self.name
                )

        for line in lines:
            lowered = line.lower()
            if any(keyword in lowered for keyword in _TITLE_KEYWORDS) and len(line) < 80:
                profile.title = Attribute(
                    value=line, confidence=Confidence.HIGH, source=self.name
                )
                break

        # The first short, alphabetic, multi-word line is conventionally the
        # sender's name.
        for line in lines[:3]:
            words = line.split()
            if (
                2 <= len(words) <= 4
                and all(word.replace(".", "").replace("-", "").isalpha() for word in words)
                and line is not profile.title
            ):
                profile.full_name = Attribute(
                    value=line, confidence=Confidence.HIGH, source=self.name
                )
                profile.first_name = Attribute(
                    value=words[0], confidence=Confidence.HIGH, source=self.name
                )
                profile.last_name = Attribute(
                    value=words[-1], confidence=Confidence.HIGH, source=self.name
                )
                break


register_provider("inference", InferenceProvider)
