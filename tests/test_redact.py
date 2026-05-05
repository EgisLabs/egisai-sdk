"""API key redaction helper. The full key never appears in our
own log/print output — even at DEBUG levels, even in error
messages, even when a customer pastes a stack trace into a
support ticket."""

from __future__ import annotations

from egisai._redact import redact_api_key


def test_redacts_long_egis_key_to_prefix_plus_ellipsis() -> None:
    """A real ``egis_live_…`` key shows enough prefix to
    disambiguate from other keys in logs (test vs prod) without
    leaking the secret part. The synthetic key below is shape-
    identical to a production key but is otherwise meaningless —
    using a real key here would put it in source control."""
    redacted = redact_api_key("egis_live_AAAABBBBCCCCDDDDEEEEFFFFGGGGHHHH-IJK")
    assert redacted == "egis_live_AA…"  # first 12 chars + ellipsis
    # Critically — the random suffix is GONE.
    assert "IJK" not in redacted
    assert "EEEEFFFF" not in redacted


def test_redacts_short_or_test_key_completely() -> None:
    """Short test placeholders / dev keys get fully redacted —
    a 12-char window on a 12-char key would reveal the whole thing."""
    assert redact_api_key("egis_test_x") == "(redacted)"


def test_empty_or_none_renders_as_unset() -> None:
    """Distinguish "no key configured" from "redacted" — the two
    failure modes have different remediations."""
    assert redact_api_key(None) == "(unset)"
    assert redact_api_key("") == "(unset)"


def test_openai_keys_get_same_redaction_treatment() -> None:
    """Same helper handles vendor-style keys (e.g. ``sk-…``) when they
    appear in logs or error paths. The string below is a low-entropy
    placeholder of the right *shape* — never a real key."""
    placeholder = "sk-proj-" + ("A" * 16) + "-" + ("B" * 16) + "-" + ("C" * 16)
    redacted = redact_api_key(placeholder)
    assert redacted == "sk-proj-AAAA…"
    assert "BBBB" not in redacted
    assert "CCCC" not in redacted
