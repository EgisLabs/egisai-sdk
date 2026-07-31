"""Engine selection: opt-in, and every failure lands on spaCy.

The contract from sdk-design-philosophy.mdc: swapping NER engines is
an explicit operator choice (never a side effect of installed
packages), and a broken opt-in must degrade to the working default
rather than take PII detection down.
"""

from __future__ import annotations

import pytest

from egisai.policy import _pii_loader as loader


def test_default_engine_is_spacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EGISAI_NER_ENGINE", raising=False)
    assert loader._resolve_ner_engine() == "spacy"


def test_unknown_engine_value_falls_back_to_spacy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGISAI_NER_ENGINE", "quantum")
    assert loader._resolve_ner_engine() == "spacy"


def test_ettin_is_selected_case_insensitively(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGISAI_NER_ENGINE", " Ettin ")
    assert loader._resolve_ner_engine() == "ettin"


def test_ettin_failure_falls_back_to_spacy_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any exception in the ONNX path must reach the spaCy path."""
    monkeypatch.setenv("EGISAI_NER_ENGINE", "ettin")

    def _boom() -> None:
        raise RuntimeError("model host unreachable")

    monkeypatch.setattr(loader, "_build_ettin_analyzer", _boom)

    spacy_path_used: list[bool] = []

    def _fake_ensure(*, quiet: bool) -> None:
        spacy_path_used.append(True)
        raise _StopBuild()  # avoid actually loading Presidio in a unit test

    class _StopBuild(Exception):
        pass

    monkeypatch.setattr(loader, "_ensure_spacy_model_present", _fake_ensure)

    with pytest.raises(_StopBuild):
        loader._build_analyzer(quiet=True)

    assert spacy_path_used, "the spaCy fallback path must have been entered"


def test_ettin_self_check_failure_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that fails the coverage canary must not go live."""
    monkeypatch.setenv("EGISAI_NER_ENGINE", "ettin")

    class _TruncatingEngine:
        supported_entities = ("PERSON",)

        def detect(self, text: str):
            return []  # finds nothing — exactly what truncation looks like

        def self_check(self) -> None:
            raise RuntimeError("long-document canary missed")

    import egisai.policy._onnx_ner as onnx_ner

    monkeypatch.setattr(
        onnx_ner, "ensure_model_files", lambda model_dir: None
    )
    monkeypatch.setattr(
        onnx_ner.OnnxNerEngine,
        "from_dir",
        classmethod(lambda cls, model_dir: _TruncatingEngine()),
    )

    class _StopBuild(Exception):
        pass

    def _fake_ensure(*, quiet: bool) -> None:
        raise _StopBuild()

    monkeypatch.setattr(loader, "_ensure_spacy_model_present", _fake_ensure)

    # The ettin build raises inside self_check → caught → spaCy path.
    with pytest.raises(_StopBuild):
        loader._build_analyzer(quiet=True)
