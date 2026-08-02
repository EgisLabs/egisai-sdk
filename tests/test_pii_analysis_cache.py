"""The PII analysis cache must be a pure speed-up — never a behavior change.

Everything here is about proving one claim: memoizing
``analyzer.analyze()`` changes *when* work happens and nothing about
*what* is detected. The performance win is only worth having if the
verdicts are identical, so these tests care far more about equality of
results than about timings.

The analyzer is stubbed rather than real. Presidio takes ~1 s to load
and a real NER pass would make these slow and machine-dependent; a stub
also lets us count calls exactly, which is the whole point.
"""

from __future__ import annotations

import dataclasses
import threading
from dataclasses import dataclass

import pytest

from egisai.policy import _pii_analysis_cache as cache


@dataclass
class _Result:
    """Shape-compatible stand-in for Presidio's ``RecognizerResult``."""

    entity_type: str
    start: int
    end: int
    score: float


class _CountingAnalyzer:
    """Records every ``analyze`` call so duplication is observable."""

    def __init__(self, results: list[_Result] | None = None) -> None:
        self.calls: list[tuple[str, tuple[str, ...] | None]] = []
        self._results = results if results is not None else []

    def analyze(self, *, text: str, entities: list[str] | None, language: str):
        self.calls.append((text, tuple(entities) if entities else None))
        return list(self._results)


EMAIL = _Result(entity_type="EMAIL_ADDRESS", start=11, end=27, score=0.99)


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EGISAI_PII_CACHE_TTL_SECS", raising=False)
    monkeypatch.delenv("EGISAI_PII_CACHE_MAX", raising=False)
    cache.clear()
    yield
    cache.clear()


# ── The saving ───────────────────────────────────────────────────────


def test_identical_text_is_analyzed_once() -> None:
    """The core win: N policies asking the same question cost one pass."""
    analyzer = _CountingAnalyzer([EMAIL])
    text = "contact me at a@b.com"

    spans = [
        cache.analyze_cached(analyzer, text=text, entities=None) for _ in range(9)
    ]

    assert len(analyzer.calls) == 1, "expected one analyzer pass for nine callers"
    assert all(s == spans[0] for s in spans), "callers must see identical spans"


def test_different_text_is_analyzed_separately() -> None:
    """Caching must never answer a question that wasn't asked."""
    analyzer = _CountingAnalyzer([EMAIL])

    cache.analyze_cached(analyzer, text="first", entities=None)
    cache.analyze_cached(analyzer, text="second", entities=None)

    assert len(analyzer.calls) == 2


def test_entity_filter_is_part_of_the_question() -> None:
    """Same text, different recognizers, is a different answer.

    Collapsing these would let a filtered scan return entities the
    caller explicitly excluded — a correctness bug, not a perf one.
    """
    analyzer = _CountingAnalyzer([EMAIL])
    text = "contact me at a@b.com"

    cache.analyze_cached(analyzer, text=text, entities=None)
    cache.analyze_cached(analyzer, text=text, entities=["EMAIL_ADDRESS"])
    cache.analyze_cached(analyzer, text=text, entities=["US_SSN"])

    assert len(analyzer.calls) == 3
    # Order within the filter must not create a spurious miss.
    cache.analyze_cached(analyzer, text=text, entities=["US_SSN", "EMAIL_ADDRESS"])
    cache.analyze_cached(analyzer, text=text, entities=["EMAIL_ADDRESS", "US_SSN"])
    assert len(analyzer.calls) == 4


def test_agentic_loop_only_pays_for_new_content() -> None:
    """Append-only transcripts must not re-pay for earlier turns.

    This is the shape that made latency grow every turn: messages
    1..N-1 are byte-identical to the previous turn and only the new
    ones are genuinely unseen.
    """
    analyzer = _CountingAnalyzer([EMAIL])
    transcript: list[str] = []
    for turn in range(10):
        transcript.append(f"message number {turn}")
        for message in transcript:
            cache.analyze_cached(analyzer, text=message, entities=None)

    # 10 turns re-scanning the whole history would be 55 passes.
    assert len(analyzer.calls) == 10


# ── The guarantee ────────────────────────────────────────────────────


def test_cached_spans_match_a_fresh_analysis_exactly() -> None:
    """A cache hit and a cold run must be indistinguishable."""
    results = [
        _Result("EMAIL_ADDRESS", 0, 9, 0.99),
        _Result("PHONE_NUMBER", 20, 32, 0.75),
        _Result("PERSON", 40, 50, 0.85),
    ]
    text = "a@b.com and 415-555-0142 and Jane Roe"

    cold = cache.analyze_cached(_CountingAnalyzer(results), text=text, entities=None)
    cache.clear()
    warm_analyzer = _CountingAnalyzer(results)
    cache.analyze_cached(warm_analyzer, text=text, entities=None)
    warm = cache.analyze_cached(warm_analyzer, text=text, entities=None)

    assert cold == warm
    assert [(s.entity_type, s.start, s.end, s.score) for s in warm] == [
        ("EMAIL_ADDRESS", 0, 9, 0.99),
        ("PHONE_NUMBER", 20, 32, 0.75),
        ("PERSON", 40, 50, 0.85),
    ]


def test_a_caller_cannot_corrupt_another_callers_result() -> None:
    """Cached spans are frozen, so one consumer can't poison the next."""
    analyzer = _CountingAnalyzer([EMAIL])
    first = cache.analyze_cached(analyzer, text="x", entities=None)

    assert isinstance(first, tuple), "a mutable list would be shared state"
    with pytest.raises(dataclasses.FrozenInstanceError):
        first[0].start = 999  # type: ignore[misc]

    second = cache.analyze_cached(analyzer, text="x", entities=None)
    assert second[0].start == EMAIL.start


def test_nothing_sensitive_is_retained(monkeypatch: pytest.MonkeyPatch) -> None:
    """The cache is keyed by PII-bearing text but must not store it.

    security-and-compliance.mdc rule 1: raw values never persist
    anywhere. Keys are hashes and values are offsets, so the whole
    structure can be dumped without leaking the prompt.
    """
    analyzer = _CountingAnalyzer([EMAIL])
    secret = "my ssn is 123-45-6789 and my email is victim@example.com"
    cache.analyze_cached(analyzer, text=secret, entities=None)

    dumped = repr(cache._cache)  # noqa: SLF001 — asserting on internals is the point
    assert "123-45-6789" not in dumped
    assert "victim@example.com" not in dumped
    assert secret not in dumped


# ── Bounds and invalidation ──────────────────────────────────────────


def test_cache_is_bounded_and_evicts_oldest_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long-lived process must not grow this without limit."""
    monkeypatch.setenv("EGISAI_PII_CACHE_MAX", "3")
    analyzer = _CountingAnalyzer([EMAIL])

    for i in range(5):
        cache.analyze_cached(analyzer, text=f"text-{i}", entities=None)
    assert cache.stats()["entries"] == 3

    # text-0 was evicted, so it costs a fresh pass; text-4 is still warm.
    before = len(analyzer.calls)
    cache.analyze_cached(analyzer, text="text-4", entities=None)
    assert len(analyzer.calls) == before
    cache.analyze_cached(analyzer, text="text-0", entities=None)
    assert len(analyzer.calls) == before + 1


def test_ttl_zero_disables_caching_entirely(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An operator must be able to switch the behavior off outright."""
    monkeypatch.setenv("EGISAI_PII_CACHE_TTL_SECS", "0")
    analyzer = _CountingAnalyzer([EMAIL])

    for _ in range(4):
        cache.analyze_cached(analyzer, text="same", entities=None)

    assert len(analyzer.calls) == 4


def test_expired_entries_are_re_analyzed(monkeypatch: pytest.MonkeyPatch) -> None:
    analyzer = _CountingAnalyzer([EMAIL])
    clock = [1000.0]
    monkeypatch.setattr(cache.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("EGISAI_PII_CACHE_TTL_SECS", "30")

    cache.analyze_cached(analyzer, text="same", entities=None)
    clock[0] += 29.0
    cache.analyze_cached(analyzer, text="same", entities=None)
    assert len(analyzer.calls) == 1

    clock[0] += 2.0
    cache.analyze_cached(analyzer, text="same", entities=None)
    assert len(analyzer.calls) == 2


def test_rebuilding_the_analyzer_drops_cached_spans() -> None:
    """Spans from a retired model must never be served for a new one."""
    from egisai.policy import _pii_loader

    analyzer = _CountingAnalyzer([EMAIL])
    cache.analyze_cached(analyzer, text="same", entities=None)
    assert cache.stats()["entries"] == 1

    _pii_loader.reset_for_tests()

    assert cache.stats()["entries"] == 0
    cache.analyze_cached(analyzer, text="same", entities=None)
    assert len(analyzer.calls) == 2


def test_empty_text_never_reaches_the_analyzer() -> None:
    analyzer = _CountingAnalyzer([])
    assert cache.analyze_cached(analyzer, text="", entities=None) == ()
    assert analyzer.calls == []


def test_concurrent_callers_all_get_correct_spans() -> None:
    """Governance runs on a thread pool; the cache must be safe there.

    Analysis deliberately happens outside the lock, so concurrent
    misses may duplicate work. That is a performance trade, never a
    correctness one — every caller must still get the right answer.
    """
    analyzer = _CountingAnalyzer([EMAIL])
    seen: list[tuple] = []
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker(n: int) -> None:
        try:
            barrier.wait(timeout=5)
            for _ in range(20):
                seen.append(
                    cache.analyze_cached(
                        analyzer, text=f"text-{n % 4}", entities=None
                    )
                )
        except BaseException as exc:  # noqa: BLE001 — surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors
    assert len(seen) == 160
    assert all(s == ((cache.Span("EMAIL_ADDRESS", 11, 27, 0.99)),) for s in seen)
