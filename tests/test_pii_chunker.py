"""The chunker's two load-bearing properties: safety and stability.

Safety: a chunk boundary must never make a PII value invisible —
newline cuts can't split a single-line value, and forced mid-line
cuts overlap far enough that the value is whole in one chunk.

Stability: an append-only document must produce the identical leading
chunks on every call, because that is what makes per-chunk caching
effective for agentic transcripts.
"""

from __future__ import annotations

import pytest

from egisai.policy import _pii_chunker as chunker


def _prose(paragraphs: int, sentence: str = "The quarterly report describes routine operations. ") -> str:
    return "\n\n".join(sentence * 3 for _ in range(paragraphs))


def test_short_text_is_one_chunk() -> None:
    text = "hello world"
    assert chunker.chunk_ranges(text) == [(0, len(text))]


def test_ranges_cover_the_whole_text() -> None:
    text = _prose(80)
    ranges = chunker.chunk_ranges(text)
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(text)
    # Contiguous or overlapping — never a gap.
    for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:], strict=False):
        assert next_start <= prev_end


def test_cuts_prefer_blank_line_boundaries() -> None:
    text = _prose(80)
    ranges = chunker.chunk_ranges(text)
    assert len(ranges) > 1
    for start, _ in ranges[1:]:
        # Every non-initial chunk starts right after a blank line.
        assert text[start - 2 : start] == "\n\n"


def test_newline_cut_never_splits_a_single_line_value() -> None:
    """An SSN sits on one line; no newline-derived cut can bisect it."""
    line = "the customer's SSN is 856-45-6789 per the intake form."
    text = "\n".join(line for _ in range(400))  # no blank lines at all
    ranges = chunker.chunk_ranges(text)
    assert len(ranges) > 1
    for start, end in ranges:
        chunk = text[start:end]
        # Count SSNs whose full shape survived in this chunk.
        assert "856-45-6789" in chunk
    # And across all chunks every line is fully contained somewhere:
    covered = [text[s:e] for s, e in ranges]
    for s, _e in ranges[1:]:
        assert text[s - 1] == "\n", "cut must land right after a newline"
    assert sum(c.count("856-45-6789") for c in covered) >= text.count("856-45-6789")


def test_hard_cut_overlap_keeps_values_whole() -> None:
    """No newlines anywhere: cuts must overlap by more than a value length."""
    blob = "x" * 200
    secret = "sk-ABCDEF0123456789ABCDEF0123456789"
    # One giant line with secrets sprinkled through it.
    text = (blob + secret) * 60
    ranges = chunker.chunk_ranges(text)
    assert len(ranges) > 1
    for (_, prev_end), (next_start, _) in zip(ranges, ranges[1:], strict=False):
        assert prev_end - next_start >= len(secret), (
            "hard-cut overlap must exceed the longest single value"
        )
    # Every occurrence of the secret is whole in at least one chunk.
    expected = text.count(secret)
    found = 0
    seen_positions: set[int] = set()
    for s, e in ranges:
        idx = text.find(secret, s)
        while idx != -1 and idx + len(secret) <= e:
            if idx not in seen_positions:
                seen_positions.add(idx)
                found += 1
            idx = text.find(secret, idx + 1)
            if idx == -1 or idx >= e:
                break
    assert found == expected


def test_append_only_documents_reuse_leading_chunks() -> None:
    """The stability property that makes transcript caching work."""
    base = _prose(60)
    grown = base + "\n\n" + "New user turn with fresh content. " * 20

    base_ranges = chunker.chunk_ranges(base)
    grown_ranges = chunker.chunk_ranges(grown)

    # All but the last chunk of the shorter document reappear
    # byte-identically in the longer one.
    stable = base_ranges[:-1]
    assert stable == grown_ranges[: len(stable)]
    for s, e in stable:
        assert base[s:e] == grown[s:e]


def test_chunking_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGISAI_PII_CHUNKING", "off")
    assert not chunker.chunking_enabled()
    monkeypatch.setenv("EGISAI_PII_CHUNKING", "on")
    assert chunker.chunking_enabled()


def test_pathological_chunk_size_is_clamped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGISAI_PII_CHUNK_CHARS", "10")
    assert chunker.chunk_target_chars() == 1000
