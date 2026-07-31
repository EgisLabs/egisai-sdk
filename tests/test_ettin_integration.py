"""End-to-end checks against the real Ettin ONNX model.

These need the ~123 MB model files on disk, so they skip themselves
when the weights are absent (CI stays fast and network-free). Run
locally with the model cached — the pre-release gate for any change
touching the NER engine:

    EGISAI_ETTIN_MODEL_DIR=/path/to/export pytest tests/test_ettin_integration.py -v

What must hold, in order of importance:

1. Coverage is full-document (the canary self-check passes).
2. The recall matrix detects every planted entity — including
   lowercase, all-caps, and non-English names.
3. Source code produces zero PERSON/LOCATION/NRP false positives
   (the failure mode that corrupted sanitized payloads under spaCy).
4. The full Presidio pipeline (fast NLP engine + Ettin recognizer +
   regex recognizers + context enhancement) still detects the
   deterministic entities and honors context boosts.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")

from egisai.policy._onnx_ner import OnnxNerEngine, resolve_model_dir


def _model_dir() -> Path | None:
    candidate = resolve_model_dir()
    needed = ("model.onnx", "tokenizer.json", "config.json")
    if all((candidate / f).exists() for f in needed):
        return candidate
    return None


_MODEL_DIR = _model_dir()

pytestmark = pytest.mark.skipif(
    _MODEL_DIR is None,
    reason=(
        "Ettin model files not cached locally; set EGISAI_ETTIN_MODEL_DIR "
        "to an export directory to run the real-model gate"
    ),
)


@pytest.fixture(scope="module")
def engine() -> OnnxNerEngine:
    assert _MODEL_DIR is not None
    return OnnxNerEngine.from_dir(_MODEL_DIR)


def test_self_check_passes(engine: OnnxNerEngine) -> None:
    """Coverage canaries at start / middle / end of a long document."""
    engine.self_check()


@pytest.mark.parametrize(
    ("text", "expected_entity", "expected_fragment"),
    [
        ("Maria Sanchez called from Berlin.", "PERSON", "Sanchez"),
        ("Contact Dr. Ahmed Khan urgently.", "PERSON", "Khan"),
        (
            "She lives at 1600 Amphitheatre Parkway, Mountain View CA 94043.",
            "LOCATION",
            "Amphitheatre",
        ),
        ("Patient is a British citizen of Jewish faith.", "NRP", "British"),
        ("wire to acct holder ROBERT J MCDONALD JR", "PERSON", "ROBERT"),
        ("my name is maria sanchez and i live in oakland", "PERSON", "maria"),
        (
            "Der Kunde Hans Müller wohnt in der Hauptstraße 5, Berlin.",
            "PERSON",
            "Müller",
        ),
        (
            "O paciente Joao Silva mora na Rua Augusta 200, Sao Paulo.",
            "PERSON",
            "Silva",
        ),
    ],
)
def test_recall_matrix(
    engine: OnnxNerEngine,
    text: str,
    expected_entity: str,
    expected_fragment: str,
) -> None:
    spans = engine.detect(text)
    matches = [
        s
        for s in spans
        if s.entity == expected_entity
        and expected_fragment in text[s.start : s.end]
    ]
    assert matches, f"expected {expected_entity}({expected_fragment!r}) in {spans}"


def test_zero_false_positives_on_source_code(engine: OnnxNerEngine) -> None:
    """This very test file is the corpus: identifiers, paths, imports."""
    code = Path(__file__).read_text()
    # Strip the parametrize block — it intentionally contains names.
    marker = code.index("@pytest.mark.parametrize")
    header_only = code[:marker]
    spans = engine.detect(header_only)
    fragments = [header_only[s.start : s.end] for s in spans]
    assert spans == [], f"code produced NER false positives: {fragments}"


def test_full_pipeline_with_fast_nlp_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Build the complete ettin analyzer and prove the pipeline works.

    NER entities come from the ONNX model, deterministic entities
    from the regex/checksum recognizers, and the context enhancer
    still boosts a labelled driver's license above its floor.
    """
    monkeypatch.setenv("EGISAI_NER_ENGINE", "ettin")
    from egisai.policy import _pii_loader

    analyzer = _pii_loader._build_ettin_analyzer()

    text = (
        "My name is Maria Sanchez, SSN 856-45-6789, card "
        "4111 1111 1111 1111, driver license C4455667."
    )
    results = analyzer.analyze(text=text, entities=None, language="en")
    by_entity = {r.entity_type: r for r in results}

    assert "PERSON" in by_entity, results
    assert "US_SSN" in by_entity, results
    assert "CREDIT_CARD" in by_entity, results
    license_hits = [r for r in results if r.entity_type == "US_DRIVER_LICENSE"]
    assert any(r.score >= 0.4 for r in license_hits), (
        "context words must boost the labelled license above the floor: "
        f"{license_hits}"
    )


def test_verdict_identity_chunking_on_and_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanitized output identical whether the doc was chunked or not."""
    monkeypatch.setenv("EGISAI_NER_ENGINE", "ettin")
    from egisai.policy import _pii_analysis_cache, _pii_loader
    from egisai.policy import pii as pii_mod

    analyzer = _pii_loader._build_ettin_analyzer()
    monkeypatch.setattr(pii_mod._pii_loader, "try_get_analyzer", lambda: analyzer)

    paragraph = (
        "Sprint retro notes: deployment went fine and the dashboards "
        "look healthy. Follow-ups assigned to the platform group. "
    )
    text = "\n\n".join(paragraph * 3 for _ in range(40))
    text += (
        "\n\nEscalation: customer Maria Sanchez (SSN 856-45-6789) "
        "reported the issue from Berlin."
    )

    monkeypatch.setenv("EGISAI_PII_CHUNKING", "off")
    _pii_analysis_cache.clear()
    unchunked, _ = pii_mod.sanitize(text)

    monkeypatch.setenv("EGISAI_PII_CHUNKING", "on")
    _pii_analysis_cache.clear()
    chunked, _ = pii_mod.sanitize(text)

    assert "856-45-6789" not in chunked
    assert "Maria Sanchez" not in chunked
    assert chunked == unchunked
