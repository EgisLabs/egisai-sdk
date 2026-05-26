"""Pure-Python PII helpers shared by the regex fallback and Presidio recognizers.

Lives in its own module so importing :mod:`egisai.policy.pii` doesn't
require Presidio to be installed — the regex fallback uses only these
helpers and stays available even when the Presidio analyzer fails to
load (e.g. air-gapped environments where ``en_core_web_lg`` can't be
downloaded). The Presidio-backed recognizer classes import from here
too, so we don't fork validation logic across two modules.
"""
from __future__ import annotations

import math

# ── Reserved / placeholder email domains ────────────────────────────
#
# RFC 2606 §3 / RFC 6761 §6 reserved TLDs and SLDs guaranteed never
# to resolve, plus a small placeholder allow-list (``test.com``)
# that shows up in docs and auto-fill data so often that flagging
# it is more noise than signal.
_RFC_RESERVED_EMAIL_TLDS: frozenset[str] = frozenset(
    {".test", ".example", ".invalid", ".localhost"}
)
_RFC_RESERVED_EMAIL_SLDS: frozenset[str] = frozenset(
    {"example.com", "example.net", "example.org"}
)
_PLACEHOLDER_EMAIL_DOMAINS: frozenset[str] = frozenset({"test.com"})


def is_reserved_email_domain(domain: str) -> bool:
    """Return True if ``domain`` is RFC-reserved or a known placeholder.

    Case-insensitive. ``endswith`` covers any subdomain beneath a
    reserved TLD (``api.example.invalid`` etc.). Used as a post-
    filter on Presidio's ``EmailRecognizer`` results AND inside the
    regex fallback's email matcher so Behavior matches in both modes.
    """
    if not domain:
        return False
    d = domain.lower().strip()
    if d == "localhost":
        return True
    if d in _RFC_RESERVED_EMAIL_SLDS or d in _PLACEHOLDER_EMAIL_DOMAINS:
        return True
    return any(d.endswith(tld) for tld in _RFC_RESERVED_EMAIL_TLDS)


# ── Math validators ────────────────────────────────────────────────


def shannon_entropy(s: str) -> float:
    """Shannon entropy in bits per character. 0 on empty input."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def luhn_check(digits: str) -> bool:
    """ISO/IEC 7812 Luhn algorithm — credit cards, NPI, several IDs."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


# ── English word-form digit decoding ────────────────────────────────
#
# Token map for the obfuscation attack where the user spells a number
# out (``"one two three four five six seven eight nine"``) instead of
# typing it as digits, hoping to slip past regex-based detectors.

_WORD_DIGIT_TOKEN_MAP: dict[str, str] = {
    "zero": "0", "oh": "0", "naught": "0", "nought": "0",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}


def word_run_to_digits(run: str) -> str:
    """Turn a matched word-digit run into the digits it spells out."""
    import re as _re

    token_re = _re.compile(
        r"(?:zero|oh|naught|nought|one|two|three|four|"
        r"five|six|seven|eight|nine)",
        _re.IGNORECASE,
    )
    return "".join(
        _WORD_DIGIT_TOKEN_MAP[w.lower()]
        for w in token_re.findall(run)
    )
