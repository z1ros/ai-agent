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
        # A lone token is ambiguous. "jane" is a plausible first name, but
        # "jdoe" is an initial glued to a surname and "jsmith2" is the same.
        # Emitting "Jdoe" as a first name puts a non-existent human into the
        # caller's CRM, which is worse than returning nothing, so a lone token
        # must actually look like a name before we use it.
        token = tokens[0]
        if not _is_name_like(token):
            return None, None
        return token.capitalize(), None

    first, last = tokens[0], tokens[-1]
    if not (_is_name_like(first) and _is_name_like(last)):
        return None, None

    return first.capitalize(), last.capitalize()


_VOWELS: frozenset[str] = frozenset("aeiouy")

# Common surnames that appear glued to a leading initial ("jdoe", "msmith").
# Kept deliberately small: it only needs to cover the high-frequency cases,
# because the fallback rule below is what does the general work.
_COMMON_SURNAMES: frozenset[str] = frozenset(
    {
        "doe", "smith", "jones", "brown", "davis", "miller", "wilson", "moore",
        "taylor", "anderson", "thomas", "jackson", "white", "harris", "martin",
        "thompson", "garcia", "martinez", "robinson", "clark", "rodriguez",
        "lewis", "lee", "walker", "hall", "allen", "young", "king", "wright",
        "scott", "green", "baker", "adams", "nelson", "hill", "campbell",
        "mitchell", "roberts", "carter", "phillips", "evans", "turner",
        "parker", "collins", "edwards", "stewart", "morris", "murphy",
        "cook", "rogers", "morgan", "peterson", "cooper", "reed", "bailey",
        "bell", "gomez", "kelly", "howard", "ward", "cox", "diaz", "richardson",
        "wood", "watson", "brooks", "bennett", "gray", "james", "reyes",
        "cruz", "hughes", "price", "myers", "long", "foster", "sanders",
        "ross", "morales", "powell", "sullivan", "russell", "ortiz", "jenkins",
        "gutierrez", "perry", "butler", "barnes", "fisher", "johnson",
        "williams", "jonson", "gonzalez", "hernandez", "lopez", "perez",
    }
)


def _is_name_like(token: str) -> bool:
    """Whether a token plausibly is a human name component.

    Rejects two shapes. First, an initial glued to a surname (``jdoe``,
    ``msmith``). Second, keyboard noise (``xx``, ``bcdfg``), which would
    otherwise sail through as a capitalized word.

    The rule is deliberately narrow rather than an allow-list of valid letter
    clusters. An allow-list silently drops any name whose onset was not
    anticipated, and those misses fall hardest on non-English names, which is
    a much worse failure than occasionally accepting a token we should not.
    """
    lowered = token.lower()
    # Two characters is the floor: "Li", "Wu", and "Bo" are real given names
    # and surnames, and excluding them would drop a large population.
    if len(lowered) < 2 or not lowered.isalpha():
        return False

    # Every real name contains a vowel. Catches "bcdfg", "ttt", "xx".
    if not any(char in _VOWELS for char in lowered):
        return False

    # A single leading consonant followed by a recognised surname is an
    # initial-plus-surname, not a given name.
    if lowered[0] not in _VOWELS and lowered[1:] in _COMMON_SURNAMES:
        return False

    # Reject a long consonant run, which mashed-together identifiers have and
    # names do not. The threshold is five rather than four so that genuine
    # names such as "Schmidt" and "Dvorzhak" survive.
    run = 0
    for char in lowered:
        run = 0 if char in _VOWELS else run + 1
        if run >= 5:
            return False

    return True


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
