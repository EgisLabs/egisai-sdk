"""Trimming the spaCy pipeline must be invisible except in the clock.

Presidio loads ``en_core_web_lg`` with every component enabled, but it
only ever consults named entities and token lemmas. The dependency
parser builds a parse tree nothing reads, and it is one of the more
expensive components, so it is switched off after load.

The risk being tested: disabling the wrong component would silently
degrade detection — fewer entities, or the same entities with different
confidence — which is exactly the kind of accuracy regression a
latency change must never introduce.
"""

from __future__ import annotations

import pytest

from egisai.policy import _pii_loader


class _FakePipeline:
    def __init__(self, names: list[str]) -> None:
        self.pipe_names = list(names)
        self.disabled: list[str] = []

    def disable_pipe(self, name: str) -> None:
        self.pipe_names.remove(name)
        self.disabled.append(name)


class _FakeEngine:
    def __init__(self, pipelines: dict[str, _FakePipeline]) -> None:
        self.nlp = pipelines


FULL = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer", "ner"]


def test_parser_is_disabled_and_nothing_else_is() -> None:
    """Only the unused component goes.

    ``ner`` produces the entities; ``tagger`` → ``attribute_ruler`` →
    ``lemmatizer`` produce the lemmas Presidio's context enhancer
    scores on. Disabling any of those would change results.
    """
    pipeline = _FakePipeline(FULL)
    _pii_loader._disable_unused_pipes(_FakeEngine({"en": pipeline}))

    assert pipeline.disabled == ["parser"]
    assert pipeline.pipe_names == [
        "tok2vec",
        "tagger",
        "attribute_ruler",
        "lemmatizer",
        "ner",
    ]


def test_every_language_pipeline_is_trimmed() -> None:
    pipelines = {lang: _FakePipeline(FULL) for lang in ("en", "es")}
    _pii_loader._disable_unused_pipes(_FakeEngine(pipelines))

    assert all(p.disabled == ["parser"] for p in pipelines.values())


def test_absent_component_is_not_an_error() -> None:
    """A model without a parser (or a future Presidio default) is fine."""
    pipeline = _FakePipeline(["tok2vec", "ner"])
    _pii_loader._disable_unused_pipes(_FakeEngine({"en": pipeline}))

    assert pipeline.disabled == []
    assert pipeline.pipe_names == ["tok2vec", "ner"]


def test_failure_leaves_the_pipeline_usable() -> None:
    """This is an optimization; it must never break analyzer startup.

    If a future Presidio or spaCy release changes these internals, the
    correct outcome is a slower analyzer, not no analyzer.
    """

    class _Hostile:
        pipe_names = FULL

        def disable_pipe(self, name: str) -> None:
            raise RuntimeError("spaCy internals moved")

    _pii_loader._disable_unused_pipes(_FakeEngine({"en": _Hostile()}))  # no raise


@pytest.mark.parametrize("engine", [None, object(), _FakeEngine({})])
def test_unexpected_engine_shapes_are_tolerated(engine: object) -> None:
    _pii_loader._disable_unused_pipes(engine)  # no raise
