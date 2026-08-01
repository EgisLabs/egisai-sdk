"""The shared L2 span store: a pure speed-up that can never break a scan.

The L2 exists so a long context analyzed yesterday, or five minutes
ago on another instance, is free today. That makes it the one part of
the PII path where a foreign process supplies the answer — so these
tests care about two things above all:

* it must never change a verdict (same spans as a cold analysis, and
  a corrupt or hostile entry must be ignored rather than trusted);
* it must never fail a scan (every store failure mode degrades to
  local analysis).

Everything else — hit counting, key shape — is in service of proving
those two.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from egisai.policy import _pii_analysis_cache as cache


@dataclass
class _Result:
    entity_type: str
    start: int
    end: int
    score: float


class _CountingAnalyzer:
    def __init__(self, results: list[_Result] | None = None) -> None:
        self.calls: list[str] = []
        self._results = results if results is not None else []

    def analyze(self, *, text: str, entities: list[str] | None, language: str):
        self.calls.append(text)
        return list(self._results)


class _FakeStore:
    """An in-memory stand-in for the gateway's Redis-backed store."""

    def __init__(self) -> None:
        self.data: dict[str, tuple[cache.Span, ...]] = {}
        self.gets: list[str] = []
        self.puts: list[str] = []

    def get(self, key: str):
        self.gets.append(key)
        return self.data.get(key)

    def put(self, key: str, spans: tuple[cache.Span, ...]) -> None:
        self.puts.append(key)
        self.data[key] = spans


EMAIL = _Result("EMAIL_ADDRESS", 11, 27, 0.99)
FINGERPRINT = "v1.2.3|ettin|abc123|w512:t0.500"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EGISAI_PII_CACHE_TTL_SECS", raising=False)
    monkeypatch.delenv("EGISAI_PII_CACHE_MAX", raising=False)
    cache.clear()
    cache.set_span_store(None)
    cache.set_engine_fingerprint("")
    yield
    cache.clear()
    cache.set_span_store(None)
    cache.set_engine_fingerprint("")


@pytest.fixture
def store() -> _FakeStore:
    s = _FakeStore()
    cache.set_span_store(s)
    cache.set_engine_fingerprint(FINGERPRINT)
    return s


# ── The win ──────────────────────────────────────────────────────────


def test_a_second_process_reuses_the_first_processes_analysis(
    store: _FakeStore,
) -> None:
    """The whole point: a cold L1 does not mean a cold analyzer.

    Simulates a restart (or another instance) by wiping L1 while
    leaving the shared store intact — exactly what happens when a
    Cursor session lands on a different Cloud Run instance.
    """
    first = _CountingAnalyzer([EMAIL])
    spans = cache.analyze_cached(first, text="contact me at a@b.com", entities=None)
    assert len(first.calls) == 1

    cache.clear()  # the process died; Redis did not

    second = _CountingAnalyzer([EMAIL])
    reused = cache.analyze_cached(second, text="contact me at a@b.com", entities=None)

    assert second.calls == [], "the analyzer must not run for a shared hit"
    assert reused == spans


def test_l1_is_preferred_so_a_warm_process_never_calls_the_store(
    store: _FakeStore,
) -> None:
    """L2 is a fallback, not a tax on every scan."""
    analyzer = _CountingAnalyzer([EMAIL])
    cache.analyze_cached(analyzer, text="same", entities=None)
    gets_after_first = len(store.gets)

    for _ in range(5):
        cache.analyze_cached(analyzer, text="same", entities=None)

    assert len(store.gets) == gets_after_first


def test_long_documents_share_per_chunk_not_per_document(
    store: _FakeStore,
) -> None:
    """The append-only case: yesterday's chunks still hit today.

    A grown transcript has a novel full text, so a document-level
    entry could never match. Sharing at chunk granularity is what
    makes the unchanged prefix free.
    """
    paragraphs = [f"Paragraph {i} of routine content. " * 6 for i in range(120)]
    base = "\n\n".join(paragraphs)

    cache.analyze_cached(_CountingAnalyzer([]), text=base, entities=None)
    assert len(store.puts) > 1, "a long document must populate many chunk entries"

    cache.clear()  # restart

    grown = base + "\n\n" + "A brand new final turn arrives. " * 5
    analyzer = _CountingAnalyzer([])
    cache.analyze_cached(analyzer, text=grown, entities=None)

    analyzed = sum(len(c) for c in analyzer.calls)
    assert analyzed < len(grown) // 4, (
        "only the changed tail should reach the analyzer after a restart"
    )


def test_shared_hits_are_counted_for_diagnostics(store: _FakeStore) -> None:
    cache.analyze_cached(_CountingAnalyzer([EMAIL]), text="x", entities=None)
    cache.clear()
    cache.analyze_cached(_CountingAnalyzer([EMAIL]), text="x", entities=None)

    assert cache.stats()["l2_hits"] == 1


# ── Correctness under a hostile / broken store ───────────────────────


def test_a_store_that_raises_on_get_falls_back_to_analysis(
    store: _FakeStore,
) -> None:
    class _Exploding(_FakeStore):
        def get(self, key: str):
            raise TimeoutError("redis is having a bad day")

    cache.set_span_store(_Exploding())
    analyzer = _CountingAnalyzer([EMAIL])

    spans = cache.analyze_cached(analyzer, text="x", entities=None)

    assert len(analyzer.calls) == 1
    assert spans == (cache.Span("EMAIL_ADDRESS", 11, 27, 0.99),)


def test_a_store_that_raises_on_put_does_not_fail_the_scan(
    store: _FakeStore,
) -> None:
    class _Exploding(_FakeStore):
        def put(self, key: str, spans) -> None:
            raise ConnectionError("redis went away mid-write")

    cache.set_span_store(_Exploding())
    analyzer = _CountingAnalyzer([EMAIL])

    spans = cache.analyze_cached(analyzer, text="x", entities=None)

    assert spans == (cache.Span("EMAIL_ADDRESS", 11, 27, 0.99),)


@pytest.mark.parametrize(
    "poison",
    [
        pytest.param("not a tuple at all", id="wrong-container"),
        pytest.param(({"entity_type": "US_SSN"},), id="dicts-not-spans"),
        pytest.param((("US_SSN", 0, 9, 0.9),), id="raw-tuples-not-spans"),
        pytest.param([cache.Span("US_SSN", 0, 9, 0.9)], id="list-not-tuple"),
    ],
)
def test_an_unusable_entry_is_ignored_rather_than_trusted(poison) -> None:
    """A corrupt entry must cost a re-analysis, never a wrong verdict.

    The store is shared infrastructure, so a version skew or a bad
    actor with write access must not be able to inject spans. The
    only safe response to anything unexpected is to analyze locally.
    """

    class _Poisoned:
        def get(self, key: str):
            return poison

        def put(self, key: str, spans) -> None:
            pass

    cache.set_span_store(_Poisoned())
    cache.set_engine_fingerprint(FINGERPRINT)
    analyzer = _CountingAnalyzer([EMAIL])

    spans = cache.analyze_cached(analyzer, text="x", entities=None)

    assert len(analyzer.calls) == 1, "poisoned entry must not short-circuit"
    assert spans == (cache.Span("EMAIL_ADDRESS", 11, 27, 0.99),)


def test_shared_spans_are_identical_to_a_cold_analysis(store: _FakeStore) -> None:
    """A hit and a miss must be indistinguishable to the caller."""
    results = [
        _Result("EMAIL_ADDRESS", 0, 9, 0.99),
        _Result("PHONE_NUMBER", 20, 32, 0.75),
        _Result("PERSON", 40, 50, 0.85),
    ]
    text = "a@b.com and 415-555-0142 and Jane Roe"

    cold = cache.analyze_cached(_CountingAnalyzer(results), text=text, entities=None)
    cache.clear()
    warm = cache.analyze_cached(_CountingAnalyzer(results), text=text, entities=None)

    assert cold == warm


# ── The engine fingerprint ───────────────────────────────────────────


def test_a_model_upgrade_cannot_serve_the_old_models_spans(
    store: _FakeStore,
) -> None:
    """Redis outlives deploys; the key must not.

    Without the fingerprint this is the bug that matters most: an
    upgraded NER model would keep answering with the retired model's
    spans until the TTL expired, silently, everywhere.
    """
    cache.analyze_cached(_CountingAnalyzer([EMAIL]), text="x", entities=None)
    cache.clear()

    cache.set_engine_fingerprint("v1.2.3|ettin|DIFFERENT_WEIGHTS|w512:t0.500")
    analyzer = _CountingAnalyzer([EMAIL])
    cache.analyze_cached(analyzer, text="x", entities=None)

    assert len(analyzer.calls) == 1, "a new model must re-analyze"


def test_no_fingerprint_means_no_shared_cache(store: _FakeStore) -> None:
    """An unidentifiable engine must not read or write shared state."""
    cache.set_engine_fingerprint("")
    analyzer = _CountingAnalyzer([EMAIL])

    cache.analyze_cached(analyzer, text="x", entities=None)

    assert store.gets == []
    assert store.puts == []
    assert len(analyzer.calls) == 1


def test_the_fingerprint_prefixes_every_key(store: _FakeStore) -> None:
    cache.analyze_cached(_CountingAnalyzer([EMAIL]), text="x", entities=None)

    assert store.puts, "expected a write"
    assert all(k.startswith(f"{FINGERPRINT}|") for k in store.puts)


def test_entity_filter_still_partitions_shared_entries(store: _FakeStore) -> None:
    """Same text, different recognizers, is a different question.

    Collapsing these in the shared store would let a filtered scan
    return entities the caller explicitly excluded.
    """
    analyzer = _CountingAnalyzer([EMAIL])
    cache.analyze_cached(analyzer, text="x", entities=None)
    cache.analyze_cached(analyzer, text="x", entities=["EMAIL_ADDRESS"])
    cache.clear()

    fresh = _CountingAnalyzer([EMAIL])
    cache.analyze_cached(fresh, text="x", entities=None)
    cache.analyze_cached(fresh, text="x", entities=["EMAIL_ADDRESS"])

    assert fresh.calls == [], "both distinct questions should be shared"
    assert len(set(store.puts)) == 2, "and stored under distinct keys"


# ── Operator switches ────────────────────────────────────────────────


def test_disabling_the_cache_disables_the_shared_layer_too(
    store: _FakeStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``EGISAI_PII_CACHE_TTL_SECS=0`` must mean no caching at all.

    An operator switching caching off for a compliance review would
    reasonably expect nothing to be read from or written to a shared
    store either.
    """
    monkeypatch.setenv("EGISAI_PII_CACHE_TTL_SECS", "0")
    analyzer = _CountingAnalyzer([EMAIL])

    for _ in range(3):
        cache.analyze_cached(analyzer, text="same", entities=None)

    assert len(analyzer.calls) == 3
    assert store.gets == []
    assert store.puts == []


def test_removing_the_store_restores_process_local_behaviour(
    store: _FakeStore,
) -> None:
    cache.set_span_store(None)
    cache.analyze_cached(_CountingAnalyzer([EMAIL]), text="x", entities=None)

    assert store.gets == []
    assert store.puts == []


def test_nothing_sensitive_is_handed_to_the_store(store: _FakeStore) -> None:
    """security-and-compliance rule 1: no raw values leave the SDK.

    The store is remote infrastructure, so this is the boundary that
    matters most — keys are hashes and values are offsets.
    """
    secret = "my ssn is 123-45-6789 and my email is victim@example.com"
    cache.analyze_cached(_CountingAnalyzer([EMAIL]), text=secret, entities=None)

    handed_over = repr(store.gets) + repr(store.puts) + repr(store.data)
    assert "123-45-6789" not in handed_over
    assert "victim@example.com" not in handed_over
    assert secret not in handed_over
