"""Per-entity confidence floors for Presidio recognizers.

Two of Presidio's built-in recognizers are designed to fire loosely
and lean on context to become trustworthy. Read without that context
they produce enormous numbers of false positives on any prompt that
carries source code, identifiers or file paths — which is exactly the
shape of a coding-assistant prompt.

``US_DRIVER_LICENSE``
    Its "Alphanumeric (weak)" pattern scores ``0.3`` and one of its
    alternatives is ``[A-Z][0-9]{1,12}`` — a single letter followed by
    one to twelve digits. That matches ``v1``, ``o3``, ``V4`` and
    every version string, enum member and variable name ever written.
    Presidio scores it ``0.3`` precisely because it is only meant to
    be believed once its context words (``driver``, ``license``,
    ``permit``, ``lic``, …) boost it. Presidio's own
    ``LemmaContextAwareEnhancer`` lifts a context-confirmed match to
    ``0.4`` at minimum, so a floor of ``0.4`` is not a magic number:
    it is the line between "the recognizer guessed" and "the
    surrounding words agreed".

``URL``
    Its "Non schema URL" pattern scores ``0.5`` and matches any dotted
    identifier whose last segment looks like a TLD. Puerto Rico's
    ``.pr`` and Indonesia's ``.id`` are real ccTLDs, so the Python
    module path ``app.services.pr`` and the attribute access ``m.id``
    both match. The scheme-qualified patterns score ``0.6``.

    A flat ``0.6`` floor would fix the code noise but also discard
    ``www.internal-payroll.com``, which is a URL any reader would
    recognize. So scheme-less matches are kept on one condition: they
    must start with ``www.``. That is what people actually write when
    they drop the scheme, and no identifier in any language begins
    that way, so it separates the two populations cleanly without
    guessing at TLDs.

Every other recognizer is left alone. This module deliberately does
not impose a global threshold: a blunt floor would also discard
genuine low-confidence hits from checksum-validated recognizers,
which is the one direction we must never move in
(security-and-compliance.mdc rule 4 — fail closed on PII).

The floors are applied *after* the analysis cache rather than inside
it, so the cache keeps storing raw analyzer output and a change to a
floor takes effect immediately instead of being masked by warm
entries.

Escape hatch: setting ``EGISAI_PII_SCORE_FLOORS=off`` restores the
raw Presidio behavior. That direction only ever *adds* detections, so
it is safe to reach for in production if a floor is ever found to
suppress something real.
"""

from __future__ import annotations

import os

__all__ = ["floor_for", "is_believable", "passes_floor"]


# Minimum score an entity must carry to be believed. Absent entities
# have no floor — their recognizers are checksum- or structurally
# validated and their scores are already meaningful.
_MIN_SCORE_BY_ENTITY: dict[str, float] = {
    # Believe it only once context words confirmed it (see above).
    "US_DRIVER_LICENSE": 0.4,
    # Below this score the match came from the scheme-less pattern,
    # which ``is_believable`` re-admits when it starts with ``www.``.
    "URL": 0.6,
}


def _floors_enabled() -> bool:
    return os.environ.get(
        "EGISAI_PII_SCORE_FLOORS", "on"
    ).strip().lower() not in ("off", "0", "false", "no")


def floor_for(entity_type: str) -> float:
    """Minimum believable score for ``entity_type`` (``0.0`` if unfloored)."""
    if not _floors_enabled():
        return 0.0
    return _MIN_SCORE_BY_ENTITY.get(entity_type, 0.0)


def passes_floor(entity_type: str, score: float) -> bool:
    """True when this hit clears its entity's score floor."""
    return score >= floor_for(entity_type)


def is_believable(entity_type: str, score: float, raw: str) -> bool:
    """True when this hit is confident enough to act on.

    Applied identically by ``scan``, ``sanitize`` and ``label_redact``
    so a span that is masked is always also a span that was reported,
    and the audit preview never disagrees with the verdict.

    ``raw`` is the matched text, which lets a below-floor hit be
    re-admitted when the text itself carries the evidence the score
    lacked — currently only scheme-less ``www.`` URLs.
    """
    if passes_floor(entity_type, score):
        return True
    if entity_type == "URL" and _floors_enabled():
        return raw.lower().lstrip("\"'").startswith("www.")
    return False
