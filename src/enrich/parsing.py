"""Deterministic email address parsing.

No model calls here by design. Splitting an address, classifying it, and
recovering a name from a local part are rule-based problems with correct
answers, so they are plain functions that can be unit tested exhaustively.
"""

from __future__ import annotations

import re
from email.utils import parseaddr

from .models import EmailKind, ParsedEmail

# Consumer mailbox providers. An address here tells us nothing about an
# employer, so domain-based company inference must be skipped.
FREE_EMAIL_DOMAINS: frozenset[str] = frozenset(
    {
        "gmail.com", "googlemail.com", "yahoo.com", "ymail.com", "hotmail.com",
        "outlook.com", "live.com", "msn.com", "aol.com", "icloud.com", "me.com",
        "mac.com", "proton.me", "protonmail.com", "pm.me", "gmx.com", "gmx.net",
        "mail.com", "zoho.com", "yandex.com", "yandex.ru", "tutanota.com",
        "fastmail.com", "hey.com", "qq.com", "163.com", "126.com", "naver.com",
        "web.de", "t-online.de", "orange.fr", "free.fr", "libero.it",
        "ukr.net", "i.ua", "meta.ua",
    }
)

# Shared mailboxes that belong to a function, not a person.
ROLE_LOCAL_PARTS: frozenset[str] = frozenset(
    {
        "info", "support", "help", "helpdesk", "contact", "hello", "hi",
        "sales", "marketing", "admin", "administrator", "webmaster",
        "postmaster", "hostmaster", "abuse", "security", "privacy", "legal",
        "billing", "accounts", "accounting", "finance", "invoices", "ap", "ar",
        "hr", "jobs", "careers", "recruiting", "press", "media", "pr",
        "team", "office", "mail", "email", "enquiries", "inquiries",
        "service", "services", "customerservice", "feedback", "orders",
        "shop", "store", "newsletter", "subscribe", "unsubscribe",
    }
)

# Substrings that mark an unattended sender.
NO_REPLY_MARKERS: tuple[str, ...] = (
    "noreply", "no-reply", "no_reply", "donotreply", "do-not-reply",
    "do_not_reply", "notifications", "notification", "automated", "mailer-daemon",
    "bounce", "bounces",
)

# Tokens that appear in local parts but are never part of a person's name.
_NOISE_TOKENS: frozenset[str] = frozenset(
    {"mail", "email", "contact", "me", "the", "real", "official", "dev", "work"}
)

_LOCAL_SPLIT_RE = re.compile(r"[._\-+]+")
_TRAILING_DIGITS_RE = re.compile(r"\d+$")
_CAMEL_SPLIT_RE = re.compile(r"(?<=[a-z])(?=[A-Z])")


def normalize_address(raw: str) -> str:
    """Reduce a raw address (possibly ``"Name" <a@b.com>``) to ``a@b.com``.

    Raises:
        ValueError: if no parsable address is present.
    """
    if not raw or not raw.strip():
        raise ValueError("email address must not be blank")

    _, addr = parseaddr(raw.strip())
    if not addr or "@" not in addr:
        raise ValueError(f"could not parse an email address from {raw!r}")

    local, _, domain = addr.rpartition("@")
    domain = domain.strip().lower().rstrip(".")
    local = local.strip()

    if not local or not domain or "." not in domain:
        raise ValueError(f"malformed email address: {raw!r}")

    return f"{local}@{domain}"


def strip_subaddress(local_part: str) -> str:
    """Remove a ``+tag`` suffix, so ``yura+jobs`` becomes ``yura``."""
    return local_part.split("+", 1)[0]


def classify(local_part: str, domain: str) -> EmailKind:
    """Classify an address so callers can skip pointless enrichment."""
    local_lower = strip_subaddress(local_part).lower()
    bare = local_lower.replace(".", "").replace("-", "").replace("_", "")

    if any(marker in bare for marker in NO_REPLY_MARKERS):
        return EmailKind.NO_REPLY
    if local_lower in ROLE_LOCAL_PARTS or bare in ROLE_LOCAL_PARTS:
        return EmailKind.ROLE
    if domain in FREE_EMAIL_DOMAINS:
        return EmailKind.PERSONAL
    return EmailKind.CORPORATE


def guess_name_parts(local_part: str) -> tuple[str | None, str | None]:
    """Best-effort first/last name recovery from a local part.

    Handles the common separator and camelCase conventions
    (``jane.doe``, ``jane_doe``, ``jdoe``, ``janeDoe``). Returns ``(None, None)``
    when the local part carries no usable signal, which is the honest answer
    for something like ``xX_dragon_Xx``.

    This is intentionally conservative: a wrong name is worse than no name,
    because downstream systems will happily address a stranger by it.
    """
    cleaned = _TRAILING_DIGITS_RE.sub("", strip_subaddress(local_part).strip())
    if not cleaned:
        return None, None

    tokens = [t for t in _LOCAL_SPLIT_RE.split(cleaned) if t]
    if len(tokens) == 1 and not tokens[0].islower():
        tokens = [t for t in _CAMEL_SPLIT_RE.split(tokens[0]) if t]

    tokens = [
        t for t in tokens
        if t.isalpha() and len(t) > 1 and t.lower() not in _NOISE_TOKENS
    ]

    if not tokens:
        return None, None
    if len(tokens) == 1:
        # A lone token is ambiguous: "jane" is a first name, "jdoe" is a
        # squashed full name we cannot split reliably. Treat it as a first
        # name only, and never invent a surname.
        return tokens[0].capitalize(), None

    return tokens[0].capitalize(), tokens[-1].capitalize()


def company_from_domain(domain: str) -> str | None:
    """Derive a display company name from a corporate domain.

    Returns ``None`` for free providers, where the domain says nothing about
    where the person works.
    """
    if domain in FREE_EMAIL_DOMAINS:
        return None

    labels = domain.split(".")
    if len(labels) < 2:
        return None

    # Drop the public suffix. Handles both "acme.com" and "acme.co.uk".
    if len(labels) >= 3 and len(labels[-2]) <= 3 and len(labels[-1]) <= 3:
        root = labels[-3]
    else:
        root = labels[-2]

    if not root or root in {"www", "mail", "email"}:
        return None

    return root.replace("-", " ").title()


def parse_email(raw: str) -> ParsedEmail:
    """Parse and classify an address. The entry point for this module."""
    address = normalize_address(raw)
    local_part, _, domain = address.rpartition("@")

    return ParsedEmail(
        address=address,
        local_part=local_part,
        domain=domain,
        kind=classify(local_part, domain),
        is_free_provider=domain in FREE_EMAIL_DOMAINS,
    )
