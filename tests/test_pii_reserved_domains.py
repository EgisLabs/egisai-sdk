"""Reserved / placeholder email domains are skipped by the PII scanner.

Covers the RFC 2606 / RFC 6761 reserved sets plus the small set of
widely-used placeholder domains we suppress for noise reduction.
"""

from __future__ import annotations

import pytest

from egisai.policy._pii_helpers import is_reserved_email_domain
from egisai.policy.pii import sanitize, scan


@pytest.mark.parametrize(
    "domain",
    [
        "example.com",
        "example.org",
        "example.net",
        "EXAMPLE.NET",
        "api.example.invalid",
        "deeper.sub.example.test",
        "service.example",
        "localhost",
        "anything.localhost",
        "test.com",
    ],
)
def test_reserved_domain_recognised(domain: str) -> None:
    assert is_reserved_email_domain(domain) is True


@pytest.mark.parametrize(
    "domain",
    [
        "acmecorp.com",
        "egisai.co",
        "exam.ple",
        "examples.com",
        "test-corp.io",
        "",
    ],
)
def test_real_domain_not_treated_as_reserved(domain: str) -> None:
    assert is_reserved_email_domain(domain) is False


@pytest.mark.parametrize(
    "address",
    [
        "alice@example.com",
        "alice@example.org",
        "alice@example.net",
        "alice@test.com",
        "alice@api.example.invalid",
    ],
)
def test_scan_skips_reserved_email_addresses(address: str) -> None:
    findings = scan(address)
    assert "email" not in {f.type for f in findings}, (
        f"reserved address {address!r} should not be flagged as PII"
    )


def test_scan_detects_real_email() -> None:
    findings = scan("contact alice@acmecorp.com for details")
    assert "email" in {f.type for f in findings}


def test_sanitize_leaves_reserved_address_intact() -> None:
    text = "see alice@example.net for an example"
    redacted, records = sanitize(text, types=["email"])
    assert "alice@example.net" in redacted
    assert records == []


def test_sanitize_redacts_real_address_but_keeps_reserved() -> None:
    text = "real bob@acme.com vs placeholder alice@example.net"
    redacted, _records = sanitize(text, types=["email"])
    assert "bob@acme.com" not in redacted
    assert "alice@example.net" in redacted
