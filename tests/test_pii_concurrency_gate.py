"""Analyzer concurrency is capped process-wide, not per document.

Parallelism *inside* one document was already bounded. Nothing bounded
it *across* documents, and on a server that is the number that decides
whether the box stays responsive: the gateway runs governance on a pool
of worker threads, each free to start its own chunked analysis, so N
concurrent governed requests put up to ``N x parallel_workers`` NER
passes on the machine at once. On a 2-vCPU container that was roughly
sixteen CPU-saturating threads over two cores, and the process's own
asyncio event loop is just one more thread competing for a timeslice —
which is why "a big PII scan is running" presented as "the whole
dashboard froze".

The gate lives at the one place Presidio actually runs, so it holds no
matter which path got there. These tests pin the ceiling, the ordering
guarantee it must not break, and the deadlock it must not introduce.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from egisai.policy import _pii_analysis_cache as cache


class _CountingAnalyzer:
    """Records the high-water mark of simultaneous ``analyze`` calls."""

    def __init__(self, hold_s: float = 0.02) -> None:
        self._hold_s = hold_s
        self._lock = threading.Lock()
        self.live = 0
        self.peak = 0
        self.calls = 0

    def analyze(self, *, text: str, entities: Any, language: str) -> list[Any]:
        with self._lock:
            self.live += 1
            self.calls += 1
            self.peak = max(self.peak, self.live)
        try:
            time.sleep(self._hold_s)
            return []
        finally:
            with self._lock:
                self.live -= 1


@pytest.fixture(autouse=True)
def _clean_state() -> Any:
    cache.clear()
    cache.reset_gate_for_test()
    cache.shutdown_pool_for_test()
    yield
    cache.clear()
    cache.reset_gate_for_test()
    cache.shutdown_pool_for_test()


def _hammer(analyzer: Any, *, threads: int, text_for: Any) -> None:
    """Run ``threads`` independent analyses at once and wait for all."""
    workers = [
        threading.Thread(
            target=cache.analyze_cached,
            args=(analyzer,),
            kwargs={"text": text_for(i), "entities": None},
        )
        for i in range(threads)
    ]
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=30)
    assert not any(w.is_alive() for w in workers), "analysis threads hung"


def test_concurrent_passes_are_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Eight callers, a ceiling of two, never more than two at once."""
    monkeypatch.setenv("EGISAI_PII_MAX_CONCURRENCY", "2")
    monkeypatch.setenv("EGISAI_PII_CACHE_TTL_SECS", "0")
    cache.reset_gate_for_test()
    analyzer = _CountingAnalyzer()

    _hammer(analyzer, threads=8, text_for=lambda i: f"document number {i}")

    assert analyzer.calls == 8
    assert analyzer.peak <= 2


def test_the_ceiling_does_not_serialise_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A cap of N must actually let N run — this is not a mutex.

    Sizing the gate to the CPU allowance is the whole point; collapsing
    it to one would hand back the parallelism the chunk pool exists to
    provide.
    """
    monkeypatch.setenv("EGISAI_PII_MAX_CONCURRENCY", "4")
    monkeypatch.setenv("EGISAI_PII_CACHE_TTL_SECS", "0")
    cache.reset_gate_for_test()
    analyzer = _CountingAnalyzer(hold_s=0.05)

    _hammer(analyzer, threads=4, text_for=lambda i: f"document number {i}")

    assert analyzer.peak > 1


def test_chunked_analysis_cannot_deadlock_against_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The parent must not hold the gate while awaiting its chunks.

    ``_analyze_chunked`` blocks on ``pool.map``. If the gate were taken
    around that wait instead of around the analyzer call, a single
    document would hold the only permit while its own chunk threads
    queued for it — a self-deadlock that no amount of capacity fixes.
    """
    monkeypatch.setenv("EGISAI_PII_MAX_CONCURRENCY", "1")
    monkeypatch.setenv("EGISAI_PII_PARALLEL_WORKERS", "4")
    monkeypatch.setenv("EGISAI_PII_CHUNKING", "on")
    # A positive TTL is what enables the chunk path at all — ``0`` is
    # documented as "no cache" and routes straight to a single pass.
    monkeypatch.setenv("EGISAI_PII_CACHE_TTL_SECS", "300")
    cache.reset_gate_for_test()
    cache.shutdown_pool_for_test()
    analyzer = _CountingAnalyzer(hold_s=0.0)

    from egisai.policy import _pii_chunker

    text = "\n".join(f"line {i} of the document" for i in range(20_000))
    assert len(text) > _pii_chunker.min_chunkable_chars(), "need a chunked doc"

    done = threading.Event()

    def _run() -> None:
        cache.analyze_cached(analyzer, text=text, entities=None)
        done.set()

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()

    assert done.wait(timeout=30), "chunked analysis deadlocked on the gate"
    assert analyzer.calls > 1, "expected the document to be chunked"
    assert analyzer.peak == 1


def test_default_ceiling_tracks_the_cpu_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """More passes than cores buys nothing and costs everyone else."""
    monkeypatch.delenv("EGISAI_PII_MAX_CONCURRENCY", raising=False)
    monkeypatch.setattr(cache._cpu, "available_cpus", lambda: 3)

    assert cache.max_concurrency() == 3


def test_an_explicit_override_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Operators can raise it past the core count for I/O-ish engines."""
    monkeypatch.setenv("EGISAI_PII_MAX_CONCURRENCY", "9")
    monkeypatch.setattr(cache._cpu, "available_cpus", lambda: 2)

    assert cache.max_concurrency() == 9


@pytest.mark.parametrize("bad", ["", "0", "-4", "banana"])
def test_a_nonsense_override_falls_back(
    monkeypatch: pytest.MonkeyPatch, bad: str
) -> None:
    monkeypatch.setenv("EGISAI_PII_MAX_CONCURRENCY", bad)
    monkeypatch.setattr(cache._cpu, "available_cpus", lambda: 2)

    assert cache.max_concurrency() == 2


def test_findings_are_unchanged_under_the_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A throughput control must not change a single verdict."""
    monkeypatch.setenv("EGISAI_PII_CACHE_TTL_SECS", "0")

    from egisai.policy import pii

    text = "Contact me at jane.doe@example.com or on 415-555-0199."

    monkeypatch.setenv("EGISAI_PII_MAX_CONCURRENCY", "1")
    cache.reset_gate_for_test()
    serial = pii.scan(text)

    monkeypatch.setenv("EGISAI_PII_MAX_CONCURRENCY", "8")
    cache.reset_gate_for_test()
    parallel = pii.scan(text)

    assert [(f.type, f.value_redacted) for f in serial] == [
        (f.type, f.value_redacted) for f in parallel
    ]
