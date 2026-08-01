"""ONNX token-classification NER engine (Ettin — Nemotron-PII).

This is the transformer alternative to spaCy's NER for the three
entity families that need a model rather than a regex: names,
locations, and GDPR special-category text (nationality / religion /
politics). It exists because spaCy's ``en_core_web_lg`` was trained
on newswire and reads source code as prose — on a 27k-character
Python file it produces ~27 false positives ("Claude" as a person,
attribute paths as locations), each of which corrupts the sanitized
payload. The Ettin encoder fine-tuned on Nvidia's Nemotron-PII
dataset produces zero on the same file, and additionally catches
lowercase ("my name is maria sanchez") and non-English names the
spaCy model misses.

Design constraints, in order:

1. **Full coverage, enforced.** Both GLiNER and this model's own
   tokenizer default to silently truncating input (the shipped
   ``tokenizer.json`` carries ``truncation: {max_length: 1024}``).
   A PII engine that silently stops reading after the first page is
   a compliance hole, so this module force-disables truncation and
   :meth:`OnnxNerEngine.self_check` refuses to come up if a canary
   entity planted deep inside a long document is not found.
2. **Fail closed at the span level.** Documents are processed in
   overlapping windows; every window is decoded in full and the
   results are *unioned* (overlapping same-entity spans merge).
   A span that straddles a window edge is complete in the
   neighboring window, so the union can only add coverage.
3. **No torch, no transformers.** Inference is onnxruntime + the
   ``tokenizers`` Rust tokenizer — the two are the ``fast-ner``
   extra and cost ~60 MB installed, versus ~2 GB for a torch stack.
4. **Constructor injection.** The session, tokenizer, and label map
   are constructor arguments so the windowing / BIO-decoding logic
   is unit-testable without the 123 MB model file.

Everything here is engine mechanics; wiring into Presidio lives in
:mod:`egisai.policy._ettin_recognizer`, and engine *selection* (the
``EGISAI_NER_ENGINE`` switch with its spaCy fallback) lives in
:mod:`egisai.policy._pii_loader`.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("egisai.pii")

__all__ = [
    "ENTITY_LABEL_MAP",
    "NerSpan",
    "OnnxNerEngine",
    "ensure_model_files",
    "resolve_model_dir",
]


# ── Model pinning ───────────────────────────────────────────────────
#
# The default model is a community ONNX export of
# ``kalyan-ks/ettin-32m-nemotron-pii`` (Ettin-32M encoder fine-tuned
# on Nvidia's synthetic Nemotron-PII corpus; 55 PII labels, 8k-token
# context). The revision is pinned to a commit SHA, which makes the
# download URLs content-immutable — a repo owner force-pushing new
# weights cannot change what this SDK pulls. Operators who need a
# different model point ``EGISAI_ETTIN_MODEL_DIR`` at a local export.
_DEFAULT_REPO = "rulesentry-io/ettin-32m-nemotron-pii-onnx"
_DEFAULT_REVISION = "a7564cc972723bd22ccb3c7a248aadb456adb267"
_MODEL_FILES = ("model.onnx", "tokenizer.json", "config.json")

# Nemotron-PII labels → the Presidio entity names the taxonomy in
# ``_pii_taxonomy`` already understands, so nothing downstream knows
# (or cares) which engine produced a span. Only the three families
# that *need* a model are mapped — everything else (SSN, cards, keys,
# emails, …) stays with the deterministic checksum/regex recognizers,
# which are both faster and independently validated.
ENTITY_LABEL_MAP: dict[str, str] = {
    "first_name": "PERSON",
    "last_name": "PERSON",
    "street_address": "LOCATION",
    "city": "LOCATION",
    "state": "LOCATION",
    "country": "LOCATION",
    "county": "LOCATION",
    "postcode": "LOCATION",
    "race_ethnicity": "NRP",
    "religious_belief": "NRP",
    "political_view": "NRP",
}


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


def _threshold() -> float:
    # 0.5 measured as the point where the 32M model has zero false
    # positives on a 180k-character source-code corpus while every
    # entry in the recall matrix (incl. lowercase and multilingual
    # names) is still found.
    return _env_float("EGISAI_ETTIN_THRESHOLD", 0.5)


def threshold() -> float:
    """Public read of the score threshold, for cache identity."""
    return _threshold()


def _window_tokens() -> int:
    # 512 was the fastest window in benchmarking (smaller windows →
    # better batch parallelism on CPU); the model itself accepts 8k.
    return max(64, _env_int("EGISAI_ETTIN_WINDOW", 512))


def window_tokens() -> int:
    """Public read of the window size, for cache identity."""
    return _window_tokens()


def _batch_size() -> int:
    return max(1, _env_int("EGISAI_ETTIN_BATCH", 8))


# Window overlap in tokens. Must comfortably exceed the longest
# entity we expect (a multi-line postal address is < 40 tokens) so
# any span cut by one window edge is complete in the neighbor.
_WINDOW_OVERLAP = 128


@dataclass(frozen=True, slots=True)
class NerSpan:
    """One detected entity, in character offsets of the input text."""

    start: int
    end: int
    entity: str
    score: float


# ── Model file management ───────────────────────────────────────────


def resolve_model_dir() -> Path:
    """Where the model files live (or will be downloaded to)."""
    explicit = os.environ.get("EGISAI_ETTIN_MODEL_DIR", "").strip()
    if explicit:
        return Path(explicit)
    revision = os.environ.get("EGISAI_ETTIN_REVISION", _DEFAULT_REVISION).strip()
    cache_root = os.environ.get("EGISAI_MODEL_CACHE", "").strip() or str(
        Path.home() / ".cache" / "egisai" / "models"
    )
    return Path(cache_root) / "ettin" / revision


def model_fingerprint(model_dir: Path) -> str:
    """Content hash of the loaded weights, for the shared span cache.

    Deliberately derived from the bytes rather than from
    ``EGISAI_ETTIN_REVISION``: the gateway image pins
    ``EGISAI_ETTIN_MODEL_DIR`` to a baked-in directory, so the
    revision env var says nothing about what is actually in it, and
    swapping the baked model without touching the constant would keep
    the retired model's spans alive in the shared cache forever.

    Content-addressing also gets the *other* direction right — an
    unchanged model hashes identically on every deploy, so a routine
    release doesn't throw away a warm cross-instance cache. That
    property is the whole point of the L2, so a cheaper identity like
    file mtime would be actively wrong here.

    Returns ``"unknown"`` if the files can't be read, which the
    caller treats as "don't use the shared cache".
    """
    digest = hashlib.sha256()
    try:
        for name in _MODEL_FILES:
            path = model_dir / name
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    except OSError:
        return "unknown"
    return digest.hexdigest()[:32]


def ensure_model_files(model_dir: Path) -> None:
    """Download the pinned model files if ``model_dir`` lacks them.

    Raises on failure — the caller (the loader daemon thread) treats
    any exception as "use the spaCy engine instead", so a sealed
    environment without egress degrades to today's behavior rather
    than breaking.

    Downloads are atomic (tmp file + ``os.replace``) so a process
    killed mid-download can never leave a half-written model that a
    later boot would try to load.
    """
    missing = [f for f in _MODEL_FILES if not (model_dir / f).exists()]
    if not missing:
        return

    repo = os.environ.get("EGISAI_ETTIN_REPO", _DEFAULT_REPO).strip()
    revision = os.environ.get("EGISAI_ETTIN_REVISION", _DEFAULT_REVISION).strip()
    model_dir.mkdir(parents=True, exist_ok=True)

    import httpx

    for name in missing:
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        LOGGER.info("[egisai] downloading NER model file %s", url)
        fd, tmp_path = tempfile.mkstemp(dir=model_dir, suffix=".part")
        try:
            with os.fdopen(fd, "wb") as out, httpx.stream(
                "GET", url, follow_redirects=True, timeout=300.0
            ) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_bytes():
                    out.write(chunk)
            os.replace(tmp_path, model_dir / name)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


# ── The engine ──────────────────────────────────────────────────────


class OnnxNerEngine:
    """Windowed, batched token-classification over an ONNX session.

    Thread-safe: onnxruntime sessions support concurrent ``run`` and
    the ``tokenizers`` tokenizer is guarded by a lock (its ``encode``
    briefly mutates shared state when padding/truncation change, so
    we serialize it rather than trust the binding's thread story).
    """

    def __init__(
        self,
        *,
        session: Any,
        tokenizer: Any,
        id2label: dict[int, str],
        label_map: dict[str, str] | None = None,
        cls_id: int,
        sep_id: int,
    ) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._tokenizer_lock = threading.Lock()
        self._id2label = id2label
        self._label_map = dict(label_map or ENTITY_LABEL_MAP)
        self._cls_id = cls_id
        self._sep_id = sep_id
        self._input_names = {i.name for i in session.get_inputs()}
        self.supported_entities: tuple[str, ...] = tuple(
            sorted(set(self._label_map.values()))
        )

    # -- construction --------------------------------------------------

    @classmethod
    def from_dir(cls, model_dir: Path) -> OnnxNerEngine:
        """Load session + tokenizer + labels from an export directory.

        Raises on any problem — never returns a half-working engine.
        """
        import onnxruntime as ort  # type: ignore[import-untyped]
        from tokenizers import Tokenizer  # type: ignore[import-untyped]

        config = json.loads((model_dir / "config.json").read_text())
        id2label = {int(k): str(v) for k, v in config["id2label"].items()}

        tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # The shipped tokenizer.json carries a baked-in
        # ``truncation: {max_length: 1024}``. Left alone, every
        # document is silently cut at ~4k characters and everything
        # after it is invisible to detection. Force full coverage.
        tokenizer.no_truncation()
        tokenizer.no_padding()

        cls_id = tokenizer.token_to_id("[CLS]")
        sep_id = tokenizer.token_to_id("[SEP]")
        if cls_id is None or sep_id is None:
            raise RuntimeError("tokenizer lacks [CLS]/[SEP] special tokens")

        options = ort.SessionOptions()
        # Model load is one-time; spend it on graph optimization.
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        return cls(
            session=session,
            tokenizer=tokenizer,
            id2label=id2label,
            cls_id=cls_id,
            sep_id=sep_id,
        )

    # -- inference ------------------------------------------------------

    def detect(self, text: str) -> list[NerSpan]:
        """Full-coverage NER over ``text``. Returns sorted, merged spans."""
        if not text:
            return []
        with self._tokenizer_lock:
            encoding = self._tokenizer.encode(text, add_special_tokens=False)
        ids: list[int] = encoding.ids
        offsets: list[tuple[int, int]] = encoding.offsets
        if not ids:
            return []

        window = _window_tokens()
        threshold = _threshold()
        step = max(1, window - _WINDOW_OVERLAP)
        starts = list(range(0, len(ids), step))
        # Drop trailing windows fully contained in the previous one.
        windows = []
        for s in starts:
            e = min(s + window, len(ids))
            windows.append((s, e))
            if e == len(ids):
                break

        spans: list[NerSpan] = []
        batch = _batch_size()
        for b in range(0, len(windows), batch):
            spans.extend(
                self._run_batch(
                    windows[b : b + batch], ids, offsets, threshold
                )
            )
        return _merge_spans(_snap_to_words(text, spans))

    def _run_batch(
        self,
        windows: list[tuple[int, int]],
        ids: list[int],
        offsets: list[tuple[int, int]],
        threshold: float,
    ) -> list[NerSpan]:
        import numpy as np

        max_len = max(e - s for s, e in windows) + 2  # +[CLS] +[SEP]
        input_ids = np.zeros((len(windows), max_len), dtype=np.int64)
        attention = np.zeros((len(windows), max_len), dtype=np.int64)
        for row, (s, e) in enumerate(windows):
            seq = [self._cls_id, *ids[s:e], self._sep_id]
            input_ids[row, : len(seq)] = seq
            attention[row, : len(seq)] = 1

        feeds = {"input_ids": input_ids, "attention_mask": attention}
        feeds = {k: v for k, v in feeds.items() if k in self._input_names}
        (logits,) = self._session.run(None, feeds)

        shifted = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(shifted)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        best_label = probs.argmax(axis=-1)
        best_prob = probs.max(axis=-1)

        # Two-threshold hysteresis: ``threshold`` gates *starting* a
        # span, the lower bar only *extends* one already started. All
        # subword pieces of a name rarely score equally ("ROBERT" →
        # "ROB" at 0.82 then "ERT" at 0.4), and cutting the span at
        # the weak piece would mask half a name. Precision is
        # unaffected: code text never produces the confident start
        # the extension bar depends on.
        continue_threshold = min(threshold, max(0.2, threshold / 2.0))

        out: list[NerSpan] = []
        for row, (s, e) in enumerate(windows):
            # Decode the FULL window — overlap regions are decoded by
            # both neighbors and unioned later. Trimming instead of
            # unioning is how a span sitting exactly on the trim
            # boundary gets lost (observed in benchmarking).
            current: NerSpan | None = None
            for t in range(e - s):
                model_pos = t + 1  # +1 skips [CLS]
                label = self._id2label.get(int(best_label[row, model_pos]), "O")
                prob = float(best_prob[row, model_pos])
                char_start, char_end = offsets[s + t]
                entity = None
                if label != "O":
                    mapped = self._label_map.get(label.split("-", 1)[-1])
                    extending = (
                        current is not None
                        and mapped == current.entity
                        and char_start - current.end <= 1
                    )
                    bar = continue_threshold if extending else threshold
                    if mapped is not None and prob >= bar:
                        entity = mapped
                if entity is None:
                    if current is not None:
                        out.append(current)
                        current = None
                    continue
                if (
                    current is not None
                    and current.entity == entity
                    and char_start - current.end <= 1
                ):
                    # Adjacent subword piece or next word of the same
                    # entity — extend.
                    current = NerSpan(
                        start=current.start,
                        end=char_end,
                        entity=entity,
                        score=max(current.score, prob),
                    )
                else:
                    if current is not None:
                        out.append(current)
                    current = NerSpan(
                        start=char_start,
                        end=char_end,
                        entity=entity,
                        score=prob,
                    )
            if current is not None:
                out.append(current)
        return out

    # -- health ----------------------------------------------------------

    def self_check(self) -> None:
        """Refuse to come up if coverage or detection is broken.

        Two canaries:

        * a name in a short sentence — basic detection;
        * a name planted ~40k characters into a filler document —
          catches every truncation failure mode this engine family
          has produced so far (tokenizer ``max_length``, window
          arithmetic bugs, off-by-one in offset mapping).

        Raises ``RuntimeError`` on failure; the loader treats that as
        "fall back to spaCy", so a broken model can never silently
        run with partial coverage.
        """
        short = "Please contact Wilhelmina Featherstone about the invoice."
        if not any(
            s.entity == "PERSON"
            and "Featherstone" in short[s.start : s.end]
            for s in self.detect(short)
        ):
            raise RuntimeError("NER self-check failed: short-text canary")

        filler = "The quarterly report describes routine operations. " * 400
        for position in (0, len(filler) // 2, len(filler)):
            doc = (
                filler[:position]
                + " Contact Wilhelmina Featherstone urgently. "
                + filler[position:]
            )
            if not any(
                "Featherstone" in doc[s.start : s.end] for s in self.detect(doc)
            ):
                raise RuntimeError(
                    "NER self-check failed: long-document canary at "
                    f"offset {position} — coverage is not full-document"
                )


def _snap_to_words(text: str, spans: list[NerSpan]) -> list[NerSpan]:
    """Expand each span to cover the complete words it intersects.

    Subword tokenization means the model's confident anchor can be a
    word *fragment* ("ROBERT" → confident on the "ERT" piece only).
    A masked fragment is both ugly and a partial leak — the rest of
    the word is still readable — so spans grow to the nearest
    non-alphanumeric boundary on each side. Growth is union-only,
    which is the fail-closed direction, and bounded by one word, so
    it cannot swallow neighboring content.
    """
    snapped: list[NerSpan] = []
    length = len(text)
    for span in spans:
        start, end = span.start, span.end
        while start > 0 and text[start - 1].isalnum():
            start -= 1
        while end < length and text[end].isalnum():
            end += 1
        if start == span.start and end == span.end:
            snapped.append(span)
        else:
            snapped.append(
                NerSpan(start=start, end=end, entity=span.entity, score=span.score)
            )
    return snapped


def _merge_spans(spans: list[NerSpan]) -> list[NerSpan]:
    """Union overlapping / adjacent same-entity spans across windows.

    Union-only: merging can extend coverage, never shrink it, which
    is the fail-closed direction for a masking engine.
    """
    if not spans:
        return []
    spans = sorted(spans, key=lambda s: (s.start, s.end))
    merged: list[NerSpan] = [spans[0]]
    for span in spans[1:]:
        last = merged[-1]
        if span.entity == last.entity and span.start <= last.end + 1:
            merged[-1] = NerSpan(
                start=last.start,
                end=max(last.end, span.end),
                entity=last.entity,
                score=max(last.score, span.score),
            )
        else:
            merged.append(span)
    return merged
