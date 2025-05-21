"""Tests for deterministic address parsing."""

from __future__ import annotations

import pytest

from enrich.models import EmailKind
from enrich.parsing import (
    classify,
    company_from_domain,
    guess_name_parts,
    normalize_address,
    parse_email,
    strip_subaddress,
)


class TestNormalizeAddress:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("jane@acme.com", "jane@acme.com"),
            ("  jane@acme.com  ", "jane@acme.com"),
            ("Jane Doe <jane@acme.com>", "jane@acme.com"),
            ('"Doe, Jane" <jane@ACME.com>', "jane@acme.com"),
            ("jane@ACME.COM", "jane@acme.com"),
            ("jane@acme.com.", "jane@acme.com"),
        ],
    )
    def test_normalizes(self, raw: str, expected: str) -> None:
        assert normalize_address(raw) == expected

    def test_preserves_local_part_case(self) -> None:
        # Local parts are case-sensitive per RFC 5321; only the domain is not.
        assert normalize_address("Jane.Doe@acme.com") == "Jane.Doe@acme.com"

    @pytest.mark.parametrize(
        "raw", ["", "   ", "not-an-email", "@acme.com", "jane@", "jane@localhost"]
    )
    def test_rejects_malformed(self, raw: str) -> None:
        with pytest.raises(ValueError):
            normalize_address(raw)


class TestStripSubaddress:
    def test_removes_plus_tag(self) -> None:
        assert strip_subaddress("yura+jobs") == "yura"

    def test_leaves_plain_local_part(self) -> None:
        assert strip_subaddress("yura") == "yura"


class TestClassify:
    @pytest.mark.parametrize(
        "local", ["noreply", "no-reply", "donotreply", "notifications", "bounces"]
    )
    def test_detects_no_reply(self, local: str) -> None:
        assert classify(local, "example.com") is EmailKind.NO_REPLY

    @pytest.mark.parametrize("local", ["sales", "info", "support", "hr", "billing"])
    def test_detects_role(self, local: str) -> None:
        assert classify(local, "acme.com") is EmailKind.ROLE

    def test_free_domain_is_personal(self) -> None:
        assert classify("jane", "gmail.com") is EmailKind.PERSONAL

    def test_custom_domain_is_corporate(self) -> None:
        assert classify("jane", "acme.com") is EmailKind.CORPORATE

    def test_no_reply_beats_role(self) -> None:
        # An unattended sender should be filtered before role handling.
        assert classify("noreply-support", "acme.com") is EmailKind.NO_REPLY


class TestGuessNameParts:
    @pytest.mark.parametrize(
        ("local", "expected"),
        [
            ("jane.doe", ("Jane", "Doe")),
            ("jane_doe", ("Jane", "Doe")),
            ("jane-doe", ("Jane", "Doe")),
            ("janeDoe", ("Jane", "Doe")),
            ("jane.doe99", ("Jane", "Doe")),
            ("jane", ("Jane", None)),
            ("ivan_petrov", ("Ivan", "Petrov")),
            ("dmytro.kovalenko", ("Dmytro", "Kovalenko")),
            ("xiu.li", ("Xiu", "Li")),
        ],
    )
    def test_extracts_names(self, local: str, expected: tuple[str | None, str | None]) -> None:
        assert guess_name_parts(local) == expected

    @pytest.mark.parametrize("local", ["jdoe", "msmith", "jjones", "bwilliams"])
    def test_rejects_initial_plus_surname(self, local: str) -> None:
        """An initial glued to a surname must not become a first name.

        Regression test: 'jdoe' previously produced first_name='Jdoe',
        inventing a person that downstream systems would address by name.
        """
        assert guess_name_parts(local) == (None, None)

    @pytest.mark.parametrize("local", ["xx", "bcdfg", "ttt", "n", "xX_dragon_Xx"])
    def test_rejects_noise(self, local: str) -> None:
        assert guess_name_parts(local) == (None, None)

    @pytest.mark.parametrize(
        "local", ["chris", "schmidt", "wright", "bradley", "svetlana", "przemyslaw"]
    )
    def test_accepts_consonant_heavy_real_names(self, local: str) -> None:
        """Regression test for an earlier onset allow-list that dropped these."""
        first, _ = guess_name_parts(local)
        assert first is not None

    def test_strips_subaddress_before_guessing(self) -> None:
        assert guess_name_parts("jane.doe+newsletter") == ("Jane", "Doe")

    def test_filters_noise_tokens(self) -> None:
        assert guess_name_parts("jane.doe.work") == ("Jane", "Doe")


class TestCompanyFromDomain:
    @pytest.mark.parametrize(
        ("domain", "expected"),
        [
            ("acme.com", "Acme"),
            ("acme-robotics.com", "Acme Robotics"),
            ("acme.co.uk", "Acme"),
            ("startup.io", "Startup"),
        ],
    )
    def test_derives_company(self, domain: str, expected: str) -> None:
        assert company_from_domain(domain) == expected

    def test_returns_none_for_free_providers(self) -> None:
        assert company_from_domain("gmail.com") is None


class TestParseEmail:
    def test_full_parse(self) -> None:
        parsed = parse_email("Jane Doe <Jane.Doe@Acme.COM>")
        assert parsed.address == "Jane.Doe@acme.com"
        assert parsed.local_part == "Jane.Doe"
        assert parsed.domain == "acme.com"
        assert parsed.kind is EmailKind.CORPORATE
        assert parsed.is_free_provider is False

    def test_flags_free_provider(self) -> None:
        assert parse_email("jane@gmail.com").is_free_provider is True

    def test_result_is_immutable(self) -> None:
        parsed = parse_email("jane@acme.com")
        with pytest.raises(Exception):
            parsed.domain = "evil.com"  # type: ignore[misc]
