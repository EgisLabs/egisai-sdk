"""Stable text chunking so PII analysis of long documents is cacheable.

The problem this solves is structural, not incidental: coding
assistants and agentic loops re-send almost the same document on
every call. Cursor's gateway requests carry a large, unchanging
system prompt plus an append-only conversation — 100k+ characters of
which perhaps 2k are new. Whole-text caching (the L1 layer in
``_pii_analysis_cache``) can't help there because the full string is
different on every call, so the analyzer re-reads everything, every
time. That is where multi-second "policy latency" on big payloads
comes from.

Chunking makes the cache see through the change: the text is split
at *stable* boundaries computed greedily left-to-right, so an edit
or append only changes the chunks it actually touches — every chunk
before it re-hashes identically and is served from cache. The first
sight of a novel document pays full price; every subsequent call
pays only for what changed.

Boundary safety (the part that has to be right):

* Cuts prefer blank lines (``\\n\\n``), then single newlines. No
  single PII value — an SSN, a card number, an API key, an email —
  contains a newline, so a newline cut can never split a value in
  two. (Multi-line structures like PEM keys are detected per line by
  the entropy recognizer, so they survive cutting too.)
* Only when a window of text contains no newline at all (minified
  JS, base64 blobs) is a hard mid-line cut taken, and then the two
  chunks **overlap** by a margin larger than any single value, so a
  value crossing the cut is complete in at least one chunk.
  Duplicated detections in the overlap are deduped by the caller.
* NER context is chunk-local. Blank-line boundaries are natural
  paragraph edges — the same units NER models are trained on — so
  this matches how the transformer engine windows its input anyway.
  The behavior is not byte-identical to one whole-document pass at
  the margins, which is why ``EGISAI_PII_CHUNKING=off`` exists; the
  direction of the difference is bounded by the overlap rule (a
  value is never made invisible, only its surrounding context is
  narrowed).
"""

from __future__ import annotations

import os

__all__ = ["chunk_ranges", "chunking_enabled", "chunk_target_chars", "min_chunkable_chars"]


# Characters a hard (mid-line) cut overlaps into the next chunk.
# Must exceed the longest single value we detect — the longest are
# JWTs and PEM lines at ~200–400 chars — so a value crossing a hard
# cut is always complete in one of the two chunks.
_HARD_CUT_OVERLAP = 512


def chunking_enabled() -> bool:
    return os.environ.get("EGISAI_PII_CHUNKING", "on").strip().lower() not in (
        "off",
        "0",
        "false",
        "no",
    )


def chunk_target_chars() -> int:
    try:
        value = int(os.environ["EGISAI_PII_CHUNK_CHARS"])
    except (KeyError, TypeError, ValueError):
        value = 4000
    # Below ~1k chars the per-chunk fixed costs dominate and NER
    # context gets too narrow; refuse pathological configs.
    return max(1000, value)


def min_chunkable_chars() -> int:
    """Texts shorter than this are analyzed whole."""
    return 2 * chunk_target_chars()


def chunk_ranges(text: str) -> list[tuple[int, int]]:
    """Split ``text`` into stable ``(start, end)`` ranges.

    Greedy, left-to-right, and deterministic: each cut depends only
    on the text before and immediately around it, so an append-only
    document produces the identical leading chunks on every call —
    that stability is what makes per-chunk caching effective.

    Ranges always cover the whole text. They are disjoint when a
    newline cut was available, and overlap by ``_HARD_CUT_OVERLAP``
    after a forced mid-line cut.
    """
    length = len(text)
    target = chunk_target_chars()
    if length <= min_chunkable_chars():
        return [(0, length)]

    ranges: list[tuple[int, int]] = []
    start = 0
    while start < length:
        if length - start <= target + target // 2:
            # Close enough to the end — take the rest rather than
            # leaving a tiny trailing chunk with no useful context.
            ranges.append((start, length))
            break

        # Look for the best cut inside [start + target/2, start + target]:
        # prefer a blank line, fall back to any newline.
        window_lo = start + target // 2
        window_hi = start + target
        cut = text.rfind("\n\n", window_lo, window_hi)
        if cut != -1:
            cut += 2  # cut *after* the blank line
            ranges.append((start, cut))
            start = cut
            continue
        cut = text.rfind("\n", window_lo, window_hi)
        if cut != -1:
            cut += 1
            ranges.append((start, cut))
            start = cut
            continue

        # No newline in the whole window (minified / binary-ish
        # content): hard cut, with overlap so no value can be lost.
        end = min(window_hi + _HARD_CUT_OVERLAP, length)
        ranges.append((start, end))
        start = window_hi

    return ranges
