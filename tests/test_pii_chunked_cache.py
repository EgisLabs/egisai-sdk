"""Chunked analysis through the cache: reuse without lost detections.

Uses a scripted analyzer (regex over whatever text it is handed) so
we can count exactly how much text was re-analyzed, which is the
entire point of chunking.
"""

from __future__ import annotations

import re
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
    """Finds SSN-shaped strings in whatever text it is given."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def analyze(self, *, text: str, entities, language: str):
        self.calls.append(text)
        return [
            _Result("US_SSN", m.start(), m.end(), 0.85)
            for m in re.finditer(r"\b\d{3}-\d{2}-\d{4}\b", text)
        ]

    @property
    def chars_analyzed(self) -> int:
        return sum(len(t) for t in self.calls)


@pytest.fixture(autouse=True)
def _fresh_cache():
    cache.clear()
    yield
    cache.clear()


def _paragraphs(count: int, ssn_every: int = 7) -> str:
    parts = []
    for i in range(count):
        body = f"Routine paragraph number {i} with ordinary content. " * 4
        if i % ssn_every == 0:
            body += f" SSN on file: {100 + i:03d}-45-6789."
        parts.append(body)
    return "\n\n".join(parts)


def test_chunked_results_match_unchunked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same spans whether the document was analyzed whole or chunked."""
    text = _paragraphs(120)

    monkeypatch.setenv("EGISAI_PII_CHUNKING", "off")
    whole = cache.analyze_cached(_RegexAnalyzer(), text=text, entities=None)
    cache.clear()

    monkeypatch.setenv("EGISAI_PII_CHUNKING", "on")
    chunked = cache.analyze_cached(_RegexAnalyzer(), text=text, entities=None)

    assert whole == chunked
    # Sanity: the document actually contains findings, so equality
    # is not vacuous.
    assert len(whole) > 5


def test_appended_transcript_only_analyzes_new_content() -> None:
    """The structural win: turn N+1 pays only for what changed."""
    analyzer = _RegexAnalyzer()
    base = _paragraphs(120)
    cache.analyze_cached(analyzer, text=base, entities=None)
    baseline_chars = analyzer.chars_analyzed
    assert baseline_chars >= len(base)  # the first sight pays full price

    appended = base + "\n\n" + "A new user turn arrives with SSN 987-65-4321. " * 3
    spans = cache.analyze_cached(analyzer, text=appended, entities=None)

    new_chars = analyzer.chars_analyzed - baseline_chars
    # Only the tail chunk(s) were re-analyzed — a small multiple of
    # the appended text, nowhere near the whole document.
    assert new_chars < len(appended) // 4
    # And the new SSN in the appended turn is detected at the right offset.
    tail_hits = [s for s in spans if appended[s.start : s.end] == "987-65-4321"]
    assert tail_hits, "the appended turn's finding must be present"


def test_value_at_chunk_boundary_is_never_lost() -> None:
    """Place an SSN in the last line before every blank-line cut."""
    analyzer = _RegexAnalyzer()
    text = _paragraphs(120, ssn_every=1)  # an SSN in every paragraph

    spans = cache.analyze_cached(analyzer, text=text, entities=None)

    expected = len(re.findall(r"\b\d{3}-\d{2}-\d{4}\b", text))
    assert len(spans) == expected
    for s in spans:
        assert re.fullmatch(r"\d{3}-\d{2}-\d{4}", text[s.start : s.end])


def test_offsets_are_document_absolute() -> None:
    analyzer = _RegexAnalyzer()
    text = _paragraphs(120)
    for span in cache.analyze_cached(analyzer, text=text, entities=None):
        assert re.fullmatch(r"\d{3}-\d{2}-\d{4}", text[span.start : span.end])


def test_overlap_duplicates_are_unioned() -> None:
    """Hard-cut overlaps see the same value twice; output has it once."""
    analyzer = _RegexAnalyzer()
    # One giant line (no newlines) forces hard cuts with overlap.
    blob = ("x" * 500 + " 123-45-6789 ") * 40
    spans = cache.analyze_cached(analyzer, text=blob, entities=None)

    positions = [(s.start, s.end) for s in spans]
    assert len(positions) == len(set(positions)), "no duplicate spans"
    assert len(spans) == blob.count("123-45-6789")


def test_identical_full_text_is_a_single_cache_hit() -> None:
    """The whole-document key still short-circuits before chunking."""
    analyzer = _RegexAnalyzer()
    text = _paragraphs(120)
    first = cache.analyze_cached(analyzer, text=text, entities=None)
    calls_after_first = len(analyzer.calls)

    second = cache.analyze_cached(analyzer, text=text, entities=None)

    assert second == first
    assert len(analyzer.calls) == calls_after_first, "no new analyzer work"
