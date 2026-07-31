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

**Tuning.** ``EGISAI_PII_CACHE_TTL_SECS`` (default 300, ``0``
disables the cache entirely) and ``EGISAI_PII_CACHE_MAX`` (default
2048 entries — sized so a 200k-character document's ~50 chunks plus
several whole-text entries never thrash).
"""

from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from egisai.policy import _pii_chunker

__all__ = ["Span", "analyze_cached", "clear", "stats"]


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
        spans = _analyze_chunked(analyzer, text=text, entities=entities)
    else:
        spans = _analyze(analyzer, text=text, entities=entities)

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
    """
    collected: list[Span] = []
    for start, end in _pii_chunker.chunk_ranges(text):
        for span in analyze_cached(
            analyzer, text=text[start:end], entities=entities
        ):
            collected.append(
                Span(
                    entity_type=span.entity_type,
                    start=span.start + start,
                    end=span.end + start,
                    score=span.score,
                )
            )

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
    """Call Presidio and reduce the result to frozen spans."""
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
    """Drop every cached entry.

    Called whenever the analyzer itself is rebuilt — spans from a
    previous model must never be served for a new one.
    """
    global _hits, _misses
    with _lock:
        _cache.clear()
        _hits = 0
        _misses = 0


def stats() -> dict[str, int]:
    """Hit/miss counters, for tests and latency diagnostics."""
    with _lock:
        return {"hits": _hits, "misses": _misses, "entries": len(_cache)}
