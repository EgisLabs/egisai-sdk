"""Presidio recognizer wrapping :class:`egisai.policy._onnx_ner.OnnxNerEngine`.

Thin by design: all windowing / decoding / merging mechanics live in
``_onnx_ner`` where they are unit-testable without Presidio. This
class only adapts the engine's spans to ``RecognizerResult`` so the
rest of the pipeline (taxonomy mapping, score floors, masking,
context enhancement) is engine-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from presidio_analyzer import AnalysisExplanation, EntityRecognizer, RecognizerResult

if TYPE_CHECKING:  # pragma: no cover
    from presidio_analyzer.nlp_engine import NlpArtifacts

    from egisai.policy._onnx_ner import OnnxNerEngine

__all__ = ["EttinRecognizer"]


class EttinRecognizer(EntityRecognizer):
    """Names / locations / special-category text via the ONNX engine."""

    def __init__(self, engine: OnnxNerEngine) -> None:
        self._engine = engine
        super().__init__(
            supported_entities=list(engine.supported_entities),
            name="EttinNerRecognizer",
            supported_language="en",
        )

    def load(self) -> None:  # pragma: no cover - engine is pre-loaded
        """Nothing to do — the engine is constructed fully loaded."""

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        """Run the engine and translate spans to Presidio results."""
        wanted = set(entities) if entities else None
        results: list[RecognizerResult] = []
        for span in self._engine.detect(text):
            if wanted is not None and span.entity not in wanted:
                continue
            explanation = AnalysisExplanation(
                # ``getattr`` because Presidio's base class sets but
                # never annotates ``name`` (mypy ``has-type``).
                recognizer=getattr(self, "name", "EttinNerRecognizer"),
                original_score=span.score,
                textual_explanation=(
                    f"Identified as {span.entity} by the Ettin "
                    "Nemotron-PII token-classification model"
                ),
            )
            results.append(
                RecognizerResult(
                    entity_type=span.entity,
                    start=span.start,
                    end=span.end,
                    score=span.score,
                    analysis_explanation=explanation,
                )
            )
        return results
