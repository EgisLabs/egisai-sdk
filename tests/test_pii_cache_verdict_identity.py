"""Caching analysis must not move a single verdict.

The cache exists to stop the same NER pass running many times per
request. That is only acceptable if the decision the engine reaches is
the one it would have reached anyway — a governance product that gets
faster by getting *different* is worse than a slow one.

These run the real engine and the real detectors (regex fallback when
Presidio isn't installed, which is representative either way: both
paths funnel through the same cached entry point). Each case is
evaluated twice, once with the cache disabled and once with it on, and
the two decisions must match field for field.
"""

from __future__ import annotations

import pytest

from egisai.policy import PolicyContext, PolicyRule, evaluate_policies
from egisai.policy import _pii_analysis_cache as cache

# Text chosen to trip several distinct detectors, including overlapping
# spans (the SSN also looks like a digit run), because span-merge order
# is the most plausible place for a caching bug to change output.
SAMPLES = [
    "Please email jane.doe@northwind-logistics.co.uk about invoice 41".ljust(80),
    "SSN 123-45-6789 and card 4111 1111 1111 1111 belong to John Smith.",
    "Call 415-555-0142 or +44 20 7946 0958; IBAN GB82WEST12345698765432.",
    "Nothing sensitive here at all, just an ordinary sentence.",
    "",
    "a@b.co " * 60,
]


def _rules(count: int) -> list[PolicyRule]:
    """``count`` distinct pii_scan policies — the shape that was slow."""
    return [
        PolicyRule(
            id=f"p{i}",
            name=f"pii-{i}",
            type="pii_scan",
            tenant="t",
            config={"action": "block"},
            phase="request",
        )
        for i in range(count)
    ]


def _decide(text: str, rules: list[PolicyRule]):
    ctx = PolicyContext(
        tenant="t",
        model="gpt-4o",
        prompt_text=text,
        prompt_chars=len(text),
        stream=False,
    )
    return evaluate_policies(rules, ctx, semantic_blocker=None, surfaces=("model",))


def _comparable(decision) -> tuple:
    """The parts of a decision an operator or an audit row can observe."""
    return (
        decision.verdict,
        decision.reason_code,
        decision.message,
        decision.matched_policy,
        tuple(sorted(decision.sanitize_types or ())),
        decision.sanitize_mask_char,
    )


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EGISAI_PII_CACHE_TTL_SECS", raising=False)
    cache.clear()
    yield
    cache.clear()


@pytest.mark.parametrize("text", SAMPLES)
@pytest.mark.parametrize("policy_count", [1, 3, 9])
def test_verdict_is_identical_with_and_without_the_cache(
    text: str,
    policy_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rules = _rules(policy_count)

    monkeypatch.setenv("EGISAI_PII_CACHE_TTL_SECS", "0")
    cache.clear()
    uncached = _decide(text, rules)

    monkeypatch.delenv("EGISAI_PII_CACHE_TTL_SECS", raising=False)
    cache.clear()
    cached = _decide(text, rules)

    assert _comparable(cached) == _comparable(uncached)


@pytest.mark.parametrize("text", SAMPLES)
def test_repeated_evaluation_is_stable(text: str) -> None:
    """A warm cache must keep returning the same verdict, not drift."""
    rules = _rules(3)
    first = _comparable(_decide(text, rules))
    for _ in range(5):
        assert _comparable(_decide(text, rules)) == first


def test_sanitized_output_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Masking must not shift by so much as a character.

    Offsets are what the cache stores, so an off-by-one here would be
    the signature of a broken key or a stale entry.
    """
    from egisai.policy.pii import sanitize

    text = "Reach Jane at jane.doe@example.org or 415-555-0142 (SSN 123-45-6789)."

    monkeypatch.setenv("EGISAI_PII_CACHE_TTL_SECS", "0")
    cache.clear()
    cold_text, cold_records = sanitize(text)

    monkeypatch.delenv("EGISAI_PII_CACHE_TTL_SECS", raising=False)
    cache.clear()
    sanitize(text)  # warm it
    warm_text, warm_records = sanitize(text)

    assert warm_text == cold_text
    assert [(r.type, r.count, r.pattern) for r in warm_records] == [
        (r.type, r.count, r.pattern) for r in cold_records
    ]


def test_label_redaction_is_byte_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    """Audit previews are built with this; drift would corrupt the trail."""
    from egisai.policy.pii import label_redact

    # Deliberately not an example.* domain: those are RFC-2606
    # reserved and the engine drops them on purpose, which would make
    # the leak assertion below vacuous.
    text = (
        "Contact bob@northwind-logistics.co.uk, "
        "card 4111 1111 1111 1111, phone 415-555-0142."
    )

    monkeypatch.setenv("EGISAI_PII_CACHE_TTL_SECS", "0")
    cache.clear()
    cold = label_redact(text)

    monkeypatch.delenv("EGISAI_PII_CACHE_TTL_SECS", raising=False)
    cache.clear()
    label_redact(text)
    warm = label_redact(text)

    assert warm == cold
    assert "bob@northwind-logistics.co.uk" not in warm
    assert "4111 1111 1111 1111" not in warm


def test_scan_findings_are_identical(monkeypatch: pytest.MonkeyPatch) -> None:
    from egisai.policy.pii import scan

    text = "jane@example.org, 415-555-0142, 123-45-6789, 4111 1111 1111 1111"

    monkeypatch.setenv("EGISAI_PII_CACHE_TTL_SECS", "0")
    cache.clear()
    cold = scan(text)

    monkeypatch.delenv("EGISAI_PII_CACHE_TTL_SECS", raising=False)
    cache.clear()
    scan(text)
    warm = scan(text)

    assert [(f.type, f.value_redacted, f.confidence, f.method) for f in warm] == [
        (f.type, f.value_redacted, f.confidence, f.method) for f in cold
    ]
