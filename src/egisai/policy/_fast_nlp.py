"""Minimal Presidio NLP engine for when a transformer replaces spaCy NER.

Presidio runs its NLP engine over the *full* text before any
recognizer executes, and with ``en_core_web_lg`` that pass (tok2vec,
tagger, lemmatizer, NER) costs ~26–31 ms per 1,000 characters — the
single most expensive step on the governance path. When the Ettin
ONNX recognizer supplies the named entities, the only things the
rest of the pipeline still needs from the NLP engine are:

* **tokens with character offsets** — consumed by the
  ``LemmaContextAwareEnhancer`` to find context words near a match
  (this is what lifts a context-confirmed driver's license from its
  pattern score of 0.3 to something believable);
* **keywords** — lemma-ish strings for that same enhancer;
* **stopword / punctuation predicates** — vocabulary lookups.

A blank ``spacy.blank("en")`` pipeline provides all three at
tokenizer speed (~1 ms per 1,000 characters): tokenization is
rule-based, and ``is_stop`` / ``is_punct`` are lexical attributes
that need no trained components. What a blank pipeline does NOT
have is a lemmatizer, so lemmas are approximated by lowercased
token text. The practical effect is that inflected context words
("licenses") no longer collapse to their lemma ("license") — the
widened context list in ``_pii_recognizers`` carries the plural
forms explicitly to compensate.

This engine is only ever used when the Ettin recognizer is active;
the default spaCy path is untouched.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from presidio_analyzer.nlp_engine import NlpArtifacts, SpacyNlpEngine

if TYPE_CHECKING:  # pragma: no cover
    from spacy.tokens import Doc

__all__ = ["FastBlankNlpEngine"]


class FastBlankNlpEngine(SpacyNlpEngine):
    """Tokenizer-only NLP engine: full artifacts, no trained pipeline.

    Subclasses ``SpacyNlpEngine`` so every Presidio integration point
    (``is_stopword``, ``is_punct``, batch processing, the engine name
    registry) keeps working — only model loading and artifact
    extraction are overridden.
    """

    engine_name = "egis_fast_blank"

    def __init__(self) -> None:
        # The parent initializer wants a model list; give it the
        # blank marker so nothing ever tries to pip-install a model.
        super().__init__(models=[{"lang_code": "en", "model_name": "blank:en"}])

    def load(self) -> None:
        """Build a blank English pipeline — no download, no weights."""
        import spacy

        # The parent class initializes ``nlp = None`` without an
        # annotation, so mypy pins the attribute type to ``None``.
        self.nlp = {"en": spacy.blank("en")}  # type: ignore[assignment]

    def _doc_to_nlp_artifact(self, doc: Doc, language: str) -> NlpArtifacts:
        """Artifacts with no entities and raw-token lemmas.

        ``entities`` is empty by construction: this engine never runs
        NER, so Presidio's ``SpacyRecognizer`` (which reads entities
        off the artifacts) becomes a no-op and the Ettin recognizer
        is the sole source of PERSON / LOCATION / NRP spans.
        """
        return NlpArtifacts(
            entities=[],
            tokens=doc,
            tokens_indices=[token.idx for token in doc],
            # Blank pipelines have no lemmatizer; lowercased surface
            # forms are the closest deterministic stand-in and are
            # exactly what the context enhancer compares against.
            lemmas=[token.text.lower() for token in doc],
            nlp_engine=self,
            language=language,
        )
