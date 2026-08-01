"""Memoize Presidio analysis so identical text is never NER'd twice.

The analyzer is by far the most expensive thing on the governance
path — a spaCy NER forward pass costs roughly 45 ms per 1,000
characters — and the call path asks it the *same question* many
times per request:

* every active ``pii_scan`` policy calls :func:`egisai.policy.pii.scan`
  on the same flattened prompt, so N policies meant N passes;
* a ``sanitize`` verdict then re-analyzes to apply masks, and the
  audit preview re-analyzes again via ``label_redact``;
* in an agentic loop the transcript is append-only, so every turn
  re-analyzes all the messages that earlier turns already analyzed.

``analyzer.analyze()`` is a pure function of ``(text, entities)`` for
a fixed model, so all of that is recomputation of a known answer.
Caching it collapses each group into a single pass and leaves the
returned spans byte-identical — this is de-duplication, not
approximation, so no verdict changes.

**What is stored.** Only :class:`Span` records: an entity label plus
offsets and a confidence score. The key is a SHA-256 of the text.
Neither the text nor any detected value is retained, so the cache
holds nothing sensitive even though it is keyed by PII-bearing input
(security-and-compliance.mdc rule 1). Offsets are meaningless without
the text, which only the caller that supplied it ever has.

**Chunked reuse for long documents.** Whole-text keys can't help
when a caller re-sends an *almost* identical document (an agentic
transcript that grew by one turn, a Cursor payload whose system
prompt never changes). Long texts are therefore split at stable
newline boundaries by :mod:`egisai.policy._pii_chunker` and each
chunk is cached independently — the unchanged prefix of an
append-only document is served entirely from cache and only the new
tail is analyzed. ``EGISAI_PII_CHUNKING=off`` restores single-pass
analysis.

**Parallel chunks.** The chunks of one document are independent
analyses of disjoint strings, and Presidio spends most of its time in
spaCy's matrix ops, which release the GIL. Analyzing them on a small
thread pool therefore gives real wall-clock speedup — measured ~2.4x
at four threads, with returns going negative beyond that — which is
what makes the *first* sight of a large payload affordable. Results
are collected in chunk order, so output is identical to the
sequential path either way. Worker count is derived from the
container's true CPU allowance, so a single-vCPU deployment runs the
sequential path unchanged.

**A ceiling on concurrent analysis.** Parallelism *within* one
document is bounded, but nothing bounded parallelism *across*
documents, and on a server that is the number that matters. The
gateway runs governance on a pool of worker threads, each of which
can start its own chunked analysis, so N concurrent governed requests
put up to ``N x parallel_workers`` NER passes on the box at once. On a
2-vCPU container that was ~16 CPU-saturating threads fighting over two
cores, and the loser is everything else in the process: the asyncio
event loop stops getting scheduled, so dashboard requests, SSE
keepalives, and health probes all stall while a large scan runs. The
work was never *blocking* the loop — it was starving it.

:func:`_analyze` is the one place Presidio actually runs, on every
path (direct, chunked, and each chunk's recursion), so a semaphore
there caps concurrent passes process-wide no matter how many callers
or pools exist above it. It is held only around the analyzer call, and
``_analyze_chunked`` does not hold it while waiting on its chunk
threads, so the pool cannot deadlock against itself. Sized to the
container's CPU allowance: more threads than cores buys nothing on
CPU-bound work and costs everyone else their timeslice.

**An optional shared second level.** Everything above is
process-local, which is exactly as far as it goes: the cache dies
with the process and is not visible to the instance next door. On
the gateway that is the difference between a returning Cursor
session costing ~400 ms and costing ~9 s, because a long context
that was analyzed yesterday — or five minutes ago on another
autoscaled instance — is a total miss today.

:func:`set_span_store` lets an embedder plug in a shared store
(the gateway backs it with Redis) that survives restarts and is
seen by every instance. The SDK deliberately knows nothing about
the transport: it hands over a key and frozen spans, and every
call is wrapped so a slow or broken store degrades to L1-only
rather than failing a scan. L2 is consulted only where a real
analyzer pass would otherwise happen, so for a long document that
is once per *chunk* — which is what makes an append-only
transcript reuse yesterday's work for everything but its new tail.

The store is keyed by :func:`set_engine_fingerprint`, an identity
for the model that produced the spans. In-process this is handled
by :func:`clear` on analyzer rebuild, but a shared store outlives
deploys: without the fingerprint in the key, upgrading the NER
model would serve spans from the retired one indefinitely. Entries
under a superseded fingerprint are simply never read again and age
out on the store's own TTL.

**Tuning.** ``EGISAI_PII_CACHE_TTL_SECS`` (default 300, ``0``
disables the cache entirely — including L2),
``EGISAI_PII_CACHE_MAX`` (default 2048 entries — sized so a
200k-character document's ~50 chunks plus several whole-text
entries never thrash), ``EGISAI_PII_PARALLEL_WORKERS`` (default
``min(4, cpus)``; ``1`` forces sequential chunk analysis), and
``EGISAI_PII_MAX_CONCURRENCY`` (default ``cpus``; the process-wide
ceiling described above).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Protocol

from egisai.policy import _cpu, _pii_chunker

__all__ = [
    "Span",
    "SpanStore",
    "analyze_cached",
    "clear",
    "max_concurrency",
    "parallel_workers",
    "set_engine_fingerprint",
    "set_span_store",
    "stats",
]

LOGGER = logging.getLogger("egisai.policy.pii_cache")

#: Beyond this, GIL contention outweighs the added parallelism —
#: measured on the spaCy ``en_core_web_lg`` path, where 8 threads is
#: slower than 4. Also bounds thread creation inside SDK host
#: processes, which did not ask us for a large pool.
_MAX_PARALLEL_WORKERS = 4


@dataclass(frozen=True, slots=True)
class Span:
    """One analyzer hit, reduced to the fields the callers read.

    Presidio's own ``RecognizerResult`` is mutable and carries
    explanation objects we never touch. Storing this frozen shape
    instead keeps cached entries small and makes it impossible for
    one caller to corrupt another caller's copy.
    """

    entity_type: str
    start: int
    end: int
    score: float


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, TypeError, ValueError):
        return default


def _ttl_secs() -> float:
    return max(0.0, _env_float("EGISAI_PII_CACHE_TTL_SECS", 300.0))


def _max_entries() -> int:
    return max(1, _env_int("EGISAI_PII_CACHE_MAX", 2048))


_lock = threading.Lock()
_cache: OrderedDict[str, tuple[float, tuple[Span, ...]]] = OrderedDict()
_hits = 0
_misses = 0

_pool_lock = threading.Lock()
_pool: ThreadPoolExecutor | None = None
_pool_size = 0

_gate_lock = threading.Lock()
_gate: threading.BoundedSemaphore | None = None
_gate_size = 0


# ── L2: an optional shared, cross-process span store ─────────────────


class SpanStore(Protocol):
    """A shared cache of analyzer spans, plugged in by the embedder.

    Implementations are expected to be cheap relative to the NER pass
    they replace (single-digit milliseconds) and to bound themselves —
    the SDK calls these on the governance hot path and treats a slow
    store as a broken one only insofar as the store lets it.

    Both methods may raise; the caller swallows everything and
    continues as if the store were absent. Implementations therefore
    do **not** need internal error handling to be safe here.
    """

    def get(self, key: str) -> tuple[Span, ...] | None:
        """Return spans stored under ``key``, or ``None`` if unknown."""
        ...

    def put(self, key: str, spans: tuple[Span, ...]) -> None:
        """Store ``spans`` under ``key``. Best-effort; may be dropped."""
        ...


_l2_lock = threading.Lock()
_l2_store: SpanStore | None = None
_l2_fingerprint = ""
_l2_hits = 0


def set_span_store(store: SpanStore | None) -> None:
    """Install (or remove, with ``None``) the shared L2 span store.

    Called once by the embedder at boot — the gateway does this in
    its lifespan with a Redis-backed store. The SDK running inside a
    customer's process never calls it, so the default is unchanged
    single-process caching.
    """
    global _l2_store
    with _l2_lock:
        _l2_store = store


def set_engine_fingerprint(fingerprint: str) -> None:
    """Declare which model's spans are being cached.

    Part of every L2 key, so spans produced by a retired model can
    never be served after an upgrade. An empty fingerprint disables
    L2 entirely: we would rather pay for a re-analysis than key a
    shared store under an engine we cannot name.
    """
    global _l2_fingerprint
    with _l2_lock:
        _l2_fingerprint = fingerprint or ""


def _l2_snapshot() -> tuple[SpanStore | None, str]:
    with _l2_lock:
        return _l2_store, _l2_fingerprint


def _l2_key(key: str) -> str:
    return f"{_l2_fingerprint}|{key}"


def _l2_get(key: str) -> tuple[Span, ...] | None:
    """Ask the shared store, or ``None`` if it is absent or unhappy.

    Every failure mode — no store, no fingerprint, timeout, decode
    error, a store that returns nonsense — lands on ``None``, which
    the caller handles by analyzing the text itself. L2 can therefore
    only ever make things faster, never wrong and never broken.
    """
    global _l2_hits
    store, fingerprint = _l2_snapshot()
    if store is None or not fingerprint:
        return None
    try:
        spans = store.get(_l2_key(key))
    except Exception:  # noqa: BLE001 — fail-open by design
        LOGGER.debug("PII span store get failed; using local analysis", exc_info=True)
        return None
    if spans is None:
        return None
    if not isinstance(spans, tuple) or not all(isinstance(s, Span) for s in spans):
        # A corrupted or version-skewed entry must not become a
        # verdict. Treat it exactly like a miss.
        LOGGER.debug("PII span store returned an unusable entry; ignoring it")
        return None
    with _l2_lock:
        _l2_hits += 1
    return spans


def _l2_put(key: str, spans: tuple[Span, ...]) -> None:
    """Publish freshly computed spans, best-effort."""
    store, fingerprint = _l2_snapshot()
    if store is None or not fingerprint:
        return
    try:
        store.put(_l2_key(key), spans)
    except Exception:  # noqa: BLE001 — fail-open by design
        LOGGER.debug("PII span store put failed; entry stays local", exc_info=True)


def max_concurrency() -> int:
    """Process-wide ceiling on simultaneous analyzer passes.

    Defaults to the container's CPU allowance. Presidio/spaCy is
    CPU-bound and releases the GIL, so up to one pass per core is
    genuinely parallel and anything beyond that only takes timeslices
    away from the rest of the process — including, on the gateway, the
    asyncio event loop that serves every other request.
    """
    raw = os.environ.get("EGISAI_PII_MAX_CONCURRENCY")
    if raw:
        try:
            explicit = int(raw)
        except ValueError:
            explicit = 0
        if explicit > 0:
            return explicit
    return max(1, _cpu.available_cpus())


def _get_gate() -> threading.BoundedSemaphore:
    """The shared analyzer gate, rebuilt if the configured width moves.

    Lazy for the same reason the chunk pool is: an SDK host process
    that never scans anything should not pay for our synchronisation
    primitives at import.
    """
    global _gate, _gate_size
    width = max_concurrency()
    with _gate_lock:
        if _gate is None or _gate_size != width:
            _gate = threading.BoundedSemaphore(width)
            _gate_size = width
        return _gate


def reset_gate_for_test() -> None:
    """Drop the gate so a test can change ``EGISAI_PII_MAX_CONCURRENCY``."""
    global _gate, _gate_size
    with _gate_lock:
        _gate, _gate_size = None, 0


def parallel_workers() -> int:
    """Threads to spread one document's chunks over. ``1`` = sequential."""
    raw = os.environ.get("EGISAI_PII_PARALLEL_WORKERS")
    if raw:
        try:
            explicit = int(raw)
        except ValueError:
            explicit = 0
        if explicit > 0:
            return min(explicit, _MAX_PARALLEL_WORKERS)
    return max(1, min(_MAX_PARALLEL_WORKERS, _cpu.available_cpus()))


def _get_pool(workers: int) -> ThreadPoolExecutor | None:
    """Lazily build the shared chunk pool, or ``None`` to stay inline.

    Built on first use rather than at import so a process that never
    scans a long document never pays for the threads — the SDK runs
    inside customer applications and must stay cheap when idle. The
    pool is rebuilt if the requested width changes, which in practice
    only happens when a test overrides the env var.
    """
    global _pool, _pool_size
    if workers < 2:
        return None
    with _pool_lock:
        if _pool is None or _pool_size != workers:
            stale = _pool
            _pool = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="egis-pii-chunk"
            )
            _pool_size = workers
            if stale is not None:
                # Don't wait: in-flight chunk work belongs to another
                # caller and will finish on its own threads.
                stale.shutdown(wait=False)
        return _pool


def shutdown_pool_for_test() -> None:
    """Drop the chunk pool so a test can start from a clean state."""
    global _pool, _pool_size
    with _pool_lock:
        stale, _pool, _pool_size = _pool, None, 0
    if stale is not None:
        stale.shutdown(wait=True)


def _key(text: str, entities: list[str] | None) -> str:
    """Content hash of the exact question asked of the analyzer.

    ``entities`` selects which recognizers run, so it changes the
    answer and has to be part of the key. ``None`` ("every
    recognizer") is distinct from any explicit list, hence the
    sentinel rather than an empty join.
    """
    digest = hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()
    scope = "*" if entities is None else ",".join(sorted(entities))
    return f"{digest}|{scope}"


def analyze_cached(
    analyzer: Any,
    *,
    text: str,
    entities: list[str] | None,
) -> tuple[Span, ...]:
    """Return analyzer spans for ``text``, reusing a recent identical run.

    On a miss the analyzer runs **outside** the cache lock. Two
    threads asking the same question concurrently may therefore both
    compute it — a small, bounded duplication that is much cheaper
    than serialising every governed request behind one mutex.
    """
    if not text:
        # No entity can occur in an empty string, so skip both the
        # analyzer and a cache entry that could never be reused.
        return ()

    ttl = _ttl_secs()
    if ttl <= 0.0:
        return _analyze(analyzer, text=text, entities=entities)

    global _hits, _misses
    key = _key(text, entities)
    now = time.monotonic()

    with _lock:
        entry = _cache.get(key)
        if entry is not None:
            stored_at, spans = entry
            if now - stored_at <= ttl:
                _cache.move_to_end(key)
                _hits += 1
                return spans
            del _cache[key]
        _misses += 1

    if (
        _pii_chunker.chunking_enabled()
        and len(text) > _pii_chunker.min_chunkable_chars()
    ):
        # Each chunk recurses back into this function below
        # ``min_chunkable_chars``, so L2 is consulted per chunk rather
        # than for the whole document. That is the granularity that
        # actually reuses anything: the full text of an append-only
        # transcript is novel on every turn, its leading chunks never
        # are.
        spans = _analyze_chunked(analyzer, text=text, entities=entities)
    else:
        shared = _l2_get(key)
        if shared is not None:
            spans = shared
        else:
            spans = _analyze(analyzer, text=text, entities=entities)
            _l2_put(key, spans)

    with _lock:
        _cache[key] = (time.monotonic(), spans)
        _cache.move_to_end(key)
        limit = _max_entries()
        while len(_cache) > limit:
            _cache.popitem(last=False)

    return spans


def _analyze_chunked(
    analyzer: Any,
    *,
    text: str,
    entities: list[str] | None,
) -> tuple[Span, ...]:
    """Analyze a long document as independently cached chunks.

    Each chunk recurses through :func:`analyze_cached`, so a chunk
    that appeared in an earlier (shorter or edited) version of the
    document is a cache hit — this is what turns an append-only
    transcript from O(total length) per call into O(new content).

    Chunk-local offsets are shifted back to document coordinates.
    Where chunks overlap (only after a forced mid-line cut, see
    ``_pii_chunker``), the same value may be detected twice; exact
    duplicates and same-entity overlapping spans are unioned, which
    can only widen coverage — the fail-closed direction.

    Chunks are analyzed on a small thread pool when the container has
    the cores to use one. That is safe on three counts: the chunks are
    disjoint strings, Presidio's analyzer holds no per-call state (a
    shared analyzer returns identical spans under concurrency — see
    ``tests/test_pii_parallel_chunks.py``), and results are gathered in
    chunk order so the merge below sees exactly the sequence it would
    have seen inline. Nesting is bounded: chunks are always smaller
    than ``min_chunkable_chars``, so a chunk never re-enters this
    function and the pool cannot deadlock waiting on itself.
    """
    ranges = _pii_chunker.chunk_ranges(text)

    def analyze_range(bounds: tuple[int, int]) -> list[Span]:
        start, end = bounds
        return [
            Span(
                entity_type=span.entity_type,
                start=span.start + start,
                end=span.end + start,
                score=span.score,
            )
            for span in analyze_cached(
                analyzer, text=text[start:end], entities=entities
            )
        ]

    pool = _get_pool(parallel_workers()) if len(ranges) > 1 else None
    if pool is None:
        per_chunk = [analyze_range(bounds) for bounds in ranges]
    else:
        # ``map`` yields in submission order and re-raises worker
        # exceptions on iteration, so a Presidio failure surfaces to
        # ``pii.scan``'s fail-closed handler exactly as it does inline.
        per_chunk = list(pool.map(analyze_range, ranges))

    collected: list[Span] = [span for chunk in per_chunk for span in chunk]
    collected.sort(key=lambda s: (s.start, s.end))
    merged: list[Span] = []
    for span in collected:
        if merged:
            last = merged[-1]
            if (
                span.entity_type == last.entity_type
                and span.start < last.end
            ):
                # Same entity seen by both sides of an overlap:
                # union the ranges, keep the stronger score.
                merged[-1] = Span(
                    entity_type=last.entity_type,
                    start=last.start,
                    end=max(last.end, span.end),
                    score=max(last.score, span.score),
                )
                continue
        merged.append(span)
    return tuple(merged)


def _analyze(
    analyzer: Any,
    *,
    text: str,
    entities: list[str] | None,
) -> tuple[Span, ...]:
    """Call Presidio and reduce the result to frozen spans.

    The single chokepoint for analyzer work, and therefore where the
    process-wide concurrency ceiling is enforced — see
    :func:`max_concurrency`. Callers **queue** here rather than being
    turned away: skipping detection to relieve CPU pressure would put
    unscanned text through a ``pii_scan`` policy, which is the one
    thing this engine must never do (security-and-compliance rule 4).
    """
    with _get_gate():
        results = analyzer.analyze(text=text, entities=entities, language="en")
    return tuple(
        Span(
            entity_type=r.entity_type,
            start=int(r.start),
            end=int(r.end),
            score=float(r.score),
        )
        for r in results
    )


def clear() -> None:
    """Drop every locally cached entry.

    Called whenever the analyzer itself is rebuilt — spans from a
    previous model must never be served for a new one.

    The shared L2 store is deliberately left alone: it belongs to
    every instance, not this one, so wiping it here would throw away
    other processes' valid work to solve a local problem. Correctness
    across a model change is handled by the fingerprint in the L2 key
    instead — superseded entries become unreachable and age out on
    the store's own TTL.
    """
    global _hits, _misses, _l2_hits
    with _lock:
        _cache.clear()
        _hits = 0
        _misses = 0
    with _l2_lock:
        _l2_hits = 0


def stats() -> dict[str, int]:
    """Hit/miss counters, for tests and latency diagnostics."""
    with _l2_lock:
        l2_hits = _l2_hits
    with _lock:
        return {
            "hits": _hits,
            "misses": _misses,
            "entries": len(_cache),
            "l2_hits": l2_hits,
        }
