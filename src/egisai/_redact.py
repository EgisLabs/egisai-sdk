"""Helpers for scrubbing sensitive values out of logs and errors."""

from __future__ import annotations


def redact_api_key(key: str | None) -> str:
    """Return a log-safe preview of an API key.

    Shows enough of the prefix to disambiguate keys without
    revealing the secret half. Returns ``"(unset)"`` for empty
    values and ``"(redacted)"`` for keys too short to abbreviate.
    """
    if not key:
        return "(unset)"
    if len(key) <= 14:
        return "(redacted)"
    return f"{key[:12]}…"
