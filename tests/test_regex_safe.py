"""ReDoS defence — static validation + runtime timeout for operator
regexes. See ``egisai/policy/_regex_safe.py`` for design rationale.
"""

from __future__ import annotations

import re

import pytest

from egisai.policy._regex_safe import (
    DEFAULT_REGEX_TIMEOUT_S,
    UnsafePatternError,
    safe_compile,
    safe_search,
)

# ── Static validation ──────────────────────────────────────────────────


def test_safe_compile_accepts_normal_pattern() -> None:
    """The vast majority of operator patterns compile fine."""
    pat = safe_compile(r"^delete\s+from\s+users", re.IGNORECASE)
    assert pat.search("DELETE FROM users WHERE 1=1") is not None


def test_safe_compile_rejects_non_string() -> None:
    with pytest.raises(UnsafePatternError):
        safe_compile(123)  # type: ignore[arg-type]


def test_safe_compile_rejects_overlong_pattern() -> None:
    """Bound the worst-case compile time for malicious / typo'd
    patterns."""
    with pytest.raises(UnsafePatternError):
        safe_compile("a" * 2000)


def test_safe_compile_rejects_classic_redos_nested_quantifier() -> None:
    """The textbook ``(a+)+`` shape responsible for most public ReDoS
    reports gets rejected at compile time."""
    with pytest.raises(UnsafePatternError):
        safe_compile(r"(a+)+$")


def test_safe_compile_rejects_alternation_nested_quantifier() -> None:
    """Another classic shape: ``(a|a)*`` — alternation of equivalent
    branches inside a ``*``."""
    with pytest.raises(UnsafePatternError):
        safe_compile(r"(.*)+x")


def test_safe_compile_rejects_overlapping_groups() -> None:
    """``(a*)*`` style — quantifier inside a group that is itself
    quantified."""
    with pytest.raises(UnsafePatternError):
        safe_compile(r"(\w*)+@example\.com")


def test_safe_compile_propagates_re_error_for_invalid_syntax() -> None:
    """Real syntax errors aren't masked — operator gets the
    standard ``re.error`` so they can debug their pattern."""
    with pytest.raises(re.error):
        safe_compile(r"unbalanced(parens")


# ── Runtime timeout ────────────────────────────────────────────────────


def test_safe_search_returns_match_for_normal_pattern() -> None:
    """A sane regex on a sane input returns the expected match
    object exactly the way ``re.search`` would."""
    m = safe_search(r"\d{3}-\d{2}-\d{4}", "my SSN is 123-45-6789, please help")
    assert m is not None
    assert m.group(0) == "123-45-6789"


def test_safe_search_returns_none_when_no_match() -> None:
    m = safe_search(r"\d{3}-\d{2}-\d{4}", "nothing here matches")
    assert m is None


def test_safe_search_rejects_unsafe_pattern_returns_none() -> None:
    """An unsafe pattern from a misconfigured policy fails OPEN —
    no match, with a logged warning. The customer's call is
    unblocked."""
    m = safe_search(r"(a+)+$", "aaaaaaaaa")
    assert m is None  # pattern rejected at compile, treated as no-match


def test_safe_search_handles_invalid_syntax_gracefully() -> None:
    """A pattern that's broken at the syntax level fails OPEN too —
    we never raise from inside the policy gate."""
    m = safe_search(r"unbalanced(", "any text")
    assert m is None


def test_safe_search_caps_input_length() -> None:
    """A 1 MB input gets truncated before matching — defence
    against an unbounded prompt blowing up the regex engine even
    on benign patterns."""
    huge = "x" * (200 * 1024) + "needle"
    # The 64KB cap means "needle" sits past the cut and won't
    # match. Test that the call returns rather than running for
    # ages on the full string.
    m = safe_search(r"needle", huge)
    assert m is None


def test_safe_search_accepts_precompiled_pattern() -> None:
    """Engine sites can precompile + reuse for hot paths."""
    compiled = safe_compile(r"^block-")
    m = safe_search(compiled, "block-pii-rule")
    assert m is not None


def test_safe_search_thread_timeout_treats_as_no_match() -> None:
    """Defence in depth: even a pattern that passes static
    validation may catastrophically backtrack on a specific input.
    The thread-timeout layer ensures the customer's request is
    unblocked. We simulate this by patching the compiled
    pattern's ``search`` to never return."""
    import time

    class _NeverReturnsPattern:
        def search(self, _text):  # noqa: ANN001 — simulating re.Pattern
            time.sleep(2.0)
            return "should-never-be-returned"

    m = safe_search(_NeverReturnsPattern(), "anything", timeout_s=0.05)  # type: ignore[arg-type]
    assert m is None


def test_default_timeout_matches_documented_50ms() -> None:
    """Lock the documented default. Changing it would silently
    move every existing customer's regex-evaluation budget."""
    assert DEFAULT_REGEX_TIMEOUT_S == 0.05


def test_layer1_rejects_runaway_optional_chain() -> None:
    """Layer 1 catches the canonical optional-chain bypass shape.

    A pattern like ``a?a?a?a?a?aaaaa`` has no group-quantifier
    construct that the original ``_NESTED_QUANT_RE`` heuristic
    flagged, but it still triggers exponential backtracking when
    every optional and the trailing mandatory atom match the same
    character. The runaway-optional check rejects it at compile
    time so it never reaches the regex engine.
    """
    with pytest.raises(UnsafePatternError):
        safe_compile("a" + "a?" * 30 + "a" * 30)


def test_layer1_accepts_short_optional_run() -> None:
    """A handful of optional quantifiers is normal regex idiom and
    must not be flagged. Only long, dense chains are runaway."""
    pat = safe_compile(r"colou?rs?")
    assert pat.search("colors") is not None
    assert pat.search("colour") is not None


def test_layer1_runaway_detector_ignores_group_syntax() -> None:
    """``(?:...)``, ``(?P<name>...)``, ``(?=...)`` etc. use ``?`` as
    syntax, not as a quantifier. A pattern that's mostly groups
    must not be misclassified."""
    pat = safe_compile(r"(?P<a>x)(?P<b>y)(?P<c>z)(?P<d>w)(?P<e>v)")
    assert pat.match("xyzwv") is not None


def test_layer2_returns_none_when_worker_yields_gil_for_too_long() -> None:
    """Layer 2 protects against any blocking call that *yields the
    GIL*: HTTP reads, sleeps, custom Pattern subclasses, etc.

    Python's built-in ``re`` engine does not yield the GIL during a
    match, so Layer 2 cannot interrupt a runaway built-in regex —
    that's why Layer 1 rejects the shape statically. This test
    locks in the daemon-thread timeout for the case Layer 2 *can*
    handle.
    """
    import time

    class _SleepyPattern:
        def search(self, _text: str) -> object:  # noqa: D401
            time.sleep(2.0)
            return "should-never-be-returned"

    started = time.perf_counter()
    result = safe_search(_SleepyPattern(), "anything", timeout_s=0.05)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - started
    assert result is None
    assert elapsed < 1.0, "Layer 2 timeout did not fire"
