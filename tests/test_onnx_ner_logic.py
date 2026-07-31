"""Windowing / decoding logic of the ONNX NER engine, without the model.

The engine's constructor takes its session, tokenizer and label map
as arguments precisely so this file can drive the span mechanics with
deterministic fakes. What must hold:

* full coverage — every token of a long input is decoded, including
  tokens that sit exactly on a window boundary (the failure mode that
  sank GLiNER and the model's own default tokenizer config);
* spans split across window edges are unioned, never trimmed away;
* the confidence threshold and the label→entity mapping are honored;
* offsets always index the original text.
"""

from __future__ import annotations

import numpy as np
import pytest

from egisai.policy._onnx_ner import (
    ENTITY_LABEL_MAP,
    NerSpan,
    OnnxNerEngine,
    _merge_spans,
    _snap_to_words,
)

# Label ids used by the fake model: 0 = O, 1 = B-first_name,
# 2 = I-first_name, 3 = B-city.
_ID2LABEL = {0: "O", 1: "B-first_name", 2: "I-first_name", 3: "B-city"}
_CLS, _SEP = 9001, 9002


class _WordTokenizer:
    """Whitespace tokenizer with real character offsets."""

    class _Encoding:
        def __init__(self, ids: list[int], offsets: list[tuple[int, int]]):
            self.ids = ids
            self.offsets = offsets

    def encode(self, text: str, add_special_tokens: bool = False):
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        pos = 0
        for word in text.split(" "):
            if word:
                ids.append(100)  # token id is irrelevant to the fake session
                offsets.append((pos, pos + len(word)))
            pos += len(word) + 1
        return self._Encoding(ids, offsets)


class _ScriptedSession:
    """Fake onnxruntime session: labels chosen by token position.

    ``script`` maps an absolute token index (in the full document) to
    a ``(label_id, prob)`` pair; everything else is O. Because the
    engine hands us windows, we recover absolute positions from the
    per-window offsets it builds — the fake is given the full offsets
    list at construction so it can invert the mapping.
    """

    def __init__(self, script: dict[int, tuple[int, float]], total_tokens: int):
        self._script = script
        self._total = total_tokens
        self.calls: list[int] = []  # batch sizes, for assertions

    def get_inputs(self):
        class _Input:
            name = "input_ids"

        class _Mask:
            name = "attention_mask"

        return [_Input(), _Mask()]

    def run(self, _outputs, feeds):
        input_ids = feeds["input_ids"]
        batch, length = input_ids.shape
        self.calls.append(batch)
        n_labels = max(_ID2LABEL) + 1
        logits = np.zeros((batch, length, n_labels), dtype=np.float32)
        logits[:, :, 0] = 5.0  # default: confidently O
        # Reconstruct which absolute tokens each row carries. The
        # engine always lays out [CLS] token... [SEP] pad...; our
        # fake token ids are all 100, so we track windows by the
        # attention mask length and a cursor the test sets up via
        # the script covering unique positions. To stay simple the
        # fake assumes windows are handed to run() in document order
        # with the engine's stride, which the engine guarantees.
        mask = feeds["attention_mask"]
        for row in range(batch):
            seq_len = int(mask[row].sum())
            window_tokens = seq_len - 2
            start = self._window_starts.pop(0)
            for t in range(window_tokens):
                absolute = start + t
                if absolute in self._script:
                    label_id, prob = self._script[absolute]
                    # logit gap that produces ~prob after softmax
                    logits[row, t + 1, :] = 0.0
                    logits[row, t + 1, label_id] = np.log(
                        prob / (1 - prob) * (n_labels - 1)
                    )
        return (logits,)

    def prime(self, window_starts: list[int]) -> None:
        self._window_starts = list(window_starts)


def _engine(script: dict[int, tuple[int, float]], total: int) -> tuple[OnnxNerEngine, _ScriptedSession]:
    session = _ScriptedSession(script, total)
    engine = OnnxNerEngine(
        session=session,
        tokenizer=_WordTokenizer(),
        id2label=_ID2LABEL,
        cls_id=_CLS,
        sep_id=_SEP,
    )
    return engine, session


def _window_starts(n_tokens: int, window: int, overlap: int = 128) -> list[int]:
    step = max(1, window - overlap)
    starts = []
    for s in range(0, n_tokens, step):
        starts.append(s)
        if s + window >= n_tokens:
            break
    return starts


def test_single_window_span_with_correct_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGISAI_ETTIN_WINDOW", "512")
    text = "please contact maria sanchez tomorrow"
    # tokens: please(0) contact(1) maria(2) sanchez(3) tomorrow(4)
    engine, session = _engine({2: (1, 0.99), 3: (2, 0.99)}, 5)
    session.prime(_window_starts(5, 512))

    spans = engine.detect(text)

    assert len(spans) == 1
    span = spans[0]
    assert span.entity == "PERSON"
    assert text[span.start : span.end] == "maria sanchez"
    assert span.score >= 0.9


def test_below_threshold_tokens_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGISAI_ETTIN_WINDOW", "512")
    text = "alpha beta gamma"
    engine, session = _engine({1: (1, 0.30)}, 3)  # below the 0.5 default
    session.prime(_window_starts(3, 512))

    assert engine.detect(text) == []


def test_unmapped_labels_are_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """A label outside the entity map must never leak through."""
    monkeypatch.setenv("EGISAI_ETTIN_WINDOW", "512")
    id2label = dict(_ID2LABEL)
    id2label[3] = "B-blood_type"  # detected by the model, not mapped by us
    session = _ScriptedSession({1: (3, 0.99)}, 3)
    engine = OnnxNerEngine(
        session=session,
        tokenizer=_WordTokenizer(),
        id2label=id2label,
        cls_id=_CLS,
        sep_id=_SEP,
    )
    session.prime(_window_starts(3, 512))

    assert engine.detect("alpha beta gamma") == []


def test_every_window_is_processed_for_long_input(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full coverage: a token near the very end must still be seen."""
    monkeypatch.setenv("EGISAI_ETTIN_WINDOW", "64")
    monkeypatch.setenv("EGISAI_ETTIN_BATCH", "4")
    n = 1000
    text = " ".join(["filler"] * (n - 1) + ["maria"])
    engine, session = _engine({n - 1: (1, 0.99)}, n)
    session.prime(_window_starts(n, 64))

    spans = engine.detect(text)

    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "maria"


def test_span_on_window_boundary_is_not_lost(monkeypatch: pytest.MonkeyPatch) -> None:
    """The union rule: a span cut by one window edge survives via its neighbor.

    With window=64 and overlap=128 clamped to step=1... instead use
    window=192: step = 192-128 = 64, so token 191/192 straddle the
    first window's edge; the second window (starting at 64) contains
    both tokens fully.
    """
    monkeypatch.setenv("EGISAI_ETTIN_WINDOW", "192")
    n = 400
    words = ["filler"] * n
    words[191] = "maria"
    words[192] = "sanchez"
    text = " ".join(words)
    engine, session = _engine({191: (1, 0.99), 192: (2, 0.99)}, n)
    session.prime(_window_starts(n, 192))

    spans = engine.detect(text)

    assert len(spans) == 1
    assert text[spans[0].start : spans[0].end] == "maria sanchez"


def test_empty_text_short_circuits() -> None:
    engine, _ = _engine({}, 0)
    assert engine.detect("") == []


def test_merge_spans_unions_overlaps_and_keeps_distinct_entities() -> None:
    spans = [
        NerSpan(start=10, end=20, entity="PERSON", score=0.8),
        NerSpan(start=15, end=25, entity="PERSON", score=0.9),  # overlap → union
        NerSpan(start=30, end=35, entity="LOCATION", score=0.7),
        NerSpan(start=30, end=35, entity="PERSON", score=0.6),  # different entity
    ]
    merged = _merge_spans(spans)
    assert NerSpan(start=10, end=25, entity="PERSON", score=0.9) in merged
    assert NerSpan(start=30, end=35, entity="LOCATION", score=0.7) in merged
    assert NerSpan(start=30, end=35, entity="PERSON", score=0.6) in merged


def test_snap_expands_fragments_to_whole_words() -> None:
    text = "wire to ROBERT tomorrow"
    # The model's confident anchor was the "ERT" fragment (11..14).
    fragment = NerSpan(start=11, end=14, entity="PERSON", score=0.8)
    (snapped,) = _snap_to_words(text, [fragment])
    assert text[snapped.start : snapped.end] == "ROBERT"


def test_snap_stops_at_punctuation_and_whitespace() -> None:
    text = "She flew to Berlin. Then home."
    span = NerSpan(start=12, end=18, entity="LOCATION", score=0.9)
    (snapped,) = _snap_to_words(text, [span])
    # Already word-aligned: the period and spaces must not be eaten.
    assert (snapped.start, snapped.end) == (12, 18)


def test_entity_label_map_only_targets_ner_families() -> None:
    """Deterministic entities (SSN, cards, keys) must stay with regex.

    If someone maps e.g. ``ssn`` into the NER engine, the checksum
    validators stop being the source of truth for it — flag that in
    review via this test.
    """
    assert set(ENTITY_LABEL_MAP.values()) == {"PERSON", "LOCATION", "NRP"}
