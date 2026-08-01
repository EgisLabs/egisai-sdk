"""Parallel chunk analysis must be invisible in the output.

Spreading one document's chunks across threads is a pure latency
optimization, so the bar is exact equality with the sequential path —
same spans, same order, same merge of overlaps. Anything less would
mean the number of CPUs a container happens to have could change a
policy verdict.

A scripted analyzer stands in for Presidio here so the suite stays
fast and runs without the spaCy model. Thread-safety of the real
analyzer is a separate property, verified against Presidio itself
before this path was enabled: 720 concurrent analyses across 2/4/8
threads returned spans byte-identical to sequential.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass

import pytest

from egisai.policy import _pii_analysis_cache as cache


@dataclass
class _Result:
    entity_type: str
    start: int
    end: int
    score: float


class _RegexAnalyzer:
    """Finds SSN-shaped strings; records concurrency it observed."""

    def __init__(self, delay: float = 0.0) -> None:
        self._lock = threading.Lock()
        self.calls: list[str] = []
        self.concurrent = 0
        self.max_concurrent = 0
        self._delay = delay

    def analyze(self, *, text: str, entities, language: str):
        with self._lock:
            self.calls.append(text)
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self._delay:
                # Long enough that a sequential run could not overlap.
                threading.Event().wait(self._delay)
            return [
                _Result("US_SSN", m.start(), m.end(), 0.85)
                for m in re.finditer(r"\b\d{3}-\d{2}-\d{4}\b", text)
            ]
        finally:
            with self._lock:
                self.concurrent -= 1


@pytest.fixture(autouse=True)
def _fresh_state():
    cache.clear()
    cache.shutdown_pool_for_test()
    yield
    cache.clear()
    cache.shutdown_pool_for_test()


def _document(paragraphs: int = 160, ssn_every: int = 5) -> str:
    parts = []
    for i in range(paragraphs):
        body = f"Paragraph {i} carries some ordinary narrative text. " * 4
        if i % ssn_every == 0:
            body += f" SSN on file: {100 + i:03d}-45-6789."
        parts.append(body)
    return "\n\n".join(parts)


def test_parallel_matches_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Identical spans whether chunks ran on one thread or four."""
    text = _document()
    monkeypatch.setenv("EGISAI_PII_CHUNKING", "on")

    monkeypatch.setenv("EGISAI_PII_PARALLEL_WORKERS", "1")
    sequential = cache.analyze_cached(_RegexAnalyzer(), text=text, entities=None)
    cache.clear()
    cache.shutdown_pool_for_test()

    monkeypatch.setenv("EGISAI_PII_PARALLEL_WORKERS", "4")
    parallel = cache.analyze_cached(_RegexAnalyzer(), text=text, entities=None)

    assert parallel == sequential
    assert len(sequential) > 1, "fixture should produce several findings"


def test_parallel_actually_overlaps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunks really do run concurrently, not just correctly."""
    monkeypatch.setenv("EGISAI_PII_CHUNKING", "on")
    monkeypatch.setenv("EGISAI_PII_PARALLEL_WORKERS", "4")

    analyzer = _RegexAnalyzer(delay=0.05)
    cache.analyze_cached(analyzer, text=_document(), entities=None)

    assert len(analyzer.calls) > 1, "document should have chunked"
    assert analyzer.max_concurrent > 1, "chunks did not overlap"


def test_single_worker_stays_inline(monkeypatch: pytest.MonkeyPatch) -> None:
    """A one-CPU container must behave exactly as before — no threads."""
    monkeypatch.setenv("EGISAI_PII_CHUNKING", "on")
    monkeypatch.setenv("EGISAI_PII_PARALLEL_WORKERS", "1")

    analyzer = _RegexAnalyzer(delay=0.01)
    cache.analyze_cached(analyzer, text=_document(), entities=None)

    assert len(analyzer.calls) > 1, "document should have chunked"
    assert analyzer.max_concurrent == 1


def test_worker_count_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Past the measured knee more threads only add contention."""
    monkeypatch.setenv("EGISAI_PII_PARALLEL_WORKERS", "64")
    assert cache.parallel_workers() == cache._MAX_PARALLEL_WORKERS


@pytest.mark.parametrize("raw", ["0", "-3", "not-a-number", ""])
def test_bad_worker_env_falls_back_to_derived(
    monkeypatch: pytest.MonkeyPatch, raw: str
) -> None:
    """A malformed override must never yield a zero-width pool."""
    monkeypatch.setenv("EGISAI_PII_PARALLEL_WORKERS", raw)
    assert cache.parallel_workers() >= 1


def test_analyzer_exception_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worker failures surface, so pii.scan's fail-closed path still runs."""
    monkeypatch.setenv("EGISAI_PII_CHUNKING", "on")
    monkeypatch.setenv("EGISAI_PII_PARALLEL_WORKERS", "4")

    class _Boom:
        def analyze(self, *, text: str, entities, language: str):
            raise RuntimeError("presidio exploded")

    with pytest.raises(RuntimeError, match="presidio exploded"):
        cache.analyze_cached(_Boom(), text=_document(), entities=None)


def test_parallel_preserves_append_only_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chunk caching still works when the pool is in play."""
    monkeypatch.setenv("EGISAI_PII_CHUNKING", "on")
    monkeypatch.setenv("EGISAI_PII_PARALLEL_WORKERS", "4")

    base = _document(paragraphs=160)
    first = _RegexAnalyzer()
    cache.analyze_cached(first, text=base, entities=None)

    grown = base + "\n\nOne more paragraph. SSN on file: 999-45-6789."
    second = _RegexAnalyzer()
    spans = cache.analyze_cached(second, text=grown, entities=None)

    reanalyzed = sum(len(c) for c in second.calls)
    assert reanalyzed < len(base) / 2, (
        f"append re-analyzed {reanalyzed} of {len(base)} chars — the "
        "unchanged prefix should have come from cache"
    )
    # The appended SSN is found, so the reuse above didn't cost coverage.
    assert any(s.start >= len(base) for s in spans)
