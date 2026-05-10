"""Custom Presidio recognizers that fill gaps in Presidio's defaults.

Presidio ships ~60 entity recognizers out of the box, but four pieces
of behaviour from our pre-Presidio engine still belong here because no
upstream recognizer covers them:

* **``EGIS_API_KEY``** — entropy-validated API keys / bearer tokens /
  GitHub PATs / AWS access keys / JWTs. Presidio has no opinion on
  generic high-entropy secrets; we keep our prefix-bank + Shannon
  entropy detector to catch novel formats.
* **``EGIS_PASSWORD``** — plaintext passwords next to a context
  keyword (``password:``, ``passwd``, ``pwd``). Best-effort by design
  — see security-and-compliance.mdc rule 3.
* **``EGIS_DOB``** — date-of-birth narrowing on top of Presidio's
  ``DATE_TIME``. Presidio tags every date; this filter only emits
  when the date is adjacent to a DOB context keyword.
* **``EGIS_WORD_FORM_SSN`` / ``EGIS_WORD_FORM_CC``** — English
  word-form digits (``"one two three four five six seven eight nine"``)
  that decode to an SSN- or credit-card-shape sequence. Catches a
  classic obfuscation Presidio's regex-driven recognizers miss.

We also patch an **email-domain post-filter** onto Presidio's
``EmailRecognizer`` so RFC-2606 reserved domains (``example.com``,
``*.test``, ``*.invalid``, ``localhost``) don't false-fire on docs.

Every recognizer here is deterministic (regex + math), so per
security-and-compliance.mdc rule 2 they all sit in Phase 1 — they
short-circuit before anything LLM-based runs.
"""
from __future__ import annotations

import re

from presidio_analyzer import (
    AnalysisExplanation,
    EntityRecognizer,
    Pattern,
    PatternRecognizer,
    RecognizerResult,
)
from presidio_analyzer.nlp_engine import NlpArtifacts

from egisai.policy._pii_helpers import (
    is_reserved_email_domain,  # noqa: F401 — re-exported for callers
    luhn_check,
    shannon_entropy,
    word_run_to_digits,
)


# ── API-key recognizer ─────────────────────────────────────────────
#
# Two-pass: first match known prefix patterns (``sk-``, ``pk-``,
# ``ghp_``, ``AKIA``, ``xox[bpsar]``, ``eyJ`` for JWT, …) then fall
# back to high-entropy hex/base64 blobs of length 32–128. Either path
# requires a Shannon-entropy floor so deterministic-shaped strings
# (UUIDs, phone numbers padded to 32 chars) don't false-fire.

_KEY_PREFIX_RE = re.compile(
    r"(?:"
    r"(?:sk|pk|api|key|token|secret|bearer|ghp|gho|ghu|ghs|ghr|glpat|"
    r"AKIA|ASIA|xox[bpsar]|eyJ|sl\.)"
    r"[\-_]?"
    r"[A-Za-z0-9\-_\.]{20,80}"
    r")"
)
_KEY_BLOB_RE = re.compile(r"\b[A-Za-z0-9+/\-_]{32,128}={0,2}\b")
_KEY_CONTEXT_KEYWORDS = (
    "key",
    "secret",
    "token",
    "api_key",
    "apikey",
    "password",
    "credential",
    "bearer",
    "authorization",
)


class ApiKeyRecognizer(EntityRecognizer):
    """Detect API keys / secrets via known prefixes + Shannon entropy."""

    def __init__(self) -> None:
        super().__init__(
            supported_entities=["EGIS_API_KEY"],
            name="EgisApiKeyRecognizer",
            supported_language="en",
        )

    def load(self) -> None:  # pragma: no cover - required no-op
        # Required by the EntityRecognizer base class. Nothing to load
        # from disk; all state is captured in the regexes above.
        return

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        if "EGIS_API_KEY" not in entities:
            return []
        results: list[RecognizerResult] = []
        seen_spans: set[tuple[int, int]] = set()

        for match in _KEY_PREFIX_RE.finditer(text):
            raw = match.group()
            entropy = shannon_entropy(raw)
            if entropy < 3.5:
                continue
            confidence = min(1.0, 0.75 + (entropy - 3.5) * 0.1)
            confidence += _context_boost(text, match.start(), match.end())
            seen_spans.add((match.start(), match.end()))
            explanation = AnalysisExplanation(
                recognizer="EgisApiKeyRecognizer",
                pattern_name="prefix+entropy",
                pattern=_KEY_PREFIX_RE.pattern,
                original_score=min(1.0, confidence),
                score=min(1.0, confidence),
                textual_explanation=(
                    "Matches a known credential prefix and clears the "
                    f"Shannon-entropy floor ({entropy:.2f} bits/char)."
                ),
            )
            results.append(
                RecognizerResult(
                    entity_type="EGIS_API_KEY",
                    start=match.start(),
                    end=match.end(),
                    score=min(1.0, confidence),
                    analysis_explanation=explanation,
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: "EgisApiKeyRecognizer",
                    },
                )
            )

        for match in _KEY_BLOB_RE.finditer(text):
            span = (match.start(), match.end())
            # Skip if the prefix path already covered (a superset of)
            # this span — prevents double-counting the same secret.
            if any(s[0] <= span[0] and s[1] >= span[1] for s in seen_spans):
                continue
            raw = match.group()
            entropy = shannon_entropy(raw)
            if entropy < 4.5 or len(raw) < 32:
                continue
            context_bonus = _context_boost(text, match.start(), match.end())
            # Generic blobs without nearby credential context are
            # only convincing at very high entropy. Empirically, 5.0
            # bits/char roughly separates random tokens from base64
            # of structured data (UUID, repeated patterns).
            if context_bonus == 0 and entropy < 5.0:
                continue
            confidence = min(
                1.0, 0.55 + (entropy - 4.5) * 0.15 + context_bonus
            )
            results.append(
                RecognizerResult(
                    entity_type="EGIS_API_KEY",
                    start=match.start(),
                    end=match.end(),
                    score=confidence,
                    analysis_explanation=AnalysisExplanation(
                        recognizer="EgisApiKeyRecognizer",
                        pattern_name="entropy",
                        pattern=_KEY_BLOB_RE.pattern,
                        original_score=confidence,
                        score=confidence,
                        textual_explanation=(
                            "High-entropy alphanumeric blob "
                            f"({entropy:.2f} bits/char)."
                        ),
                    ),
                    recognition_metadata={
                        RecognizerResult.RECOGNIZER_NAME_KEY: "EgisApiKeyRecognizer",
                    },
                )
            )
        return results


def _context_boost(text: str, start: int, end: int, window: int = 60) -> float:
    """Up to +0.15 confidence based on credential-context keywords nearby."""
    window_start = max(0, start - window)
    window_end = min(len(text), end + window)
    snippet = text[window_start:window_end].lower()
    hits = sum(1 for kw in _KEY_CONTEXT_KEYWORDS if kw in snippet)
    return min(0.15, hits * 0.05)


# ── Password recognizer ─────────────────────────────────────────────
#
# Catches the common ``password: hunter2`` / ``passwd=…`` / ``pwd=…``
# forms. Marked best-effort because there's no validator for "is this
# really a password?" — strong passwords without a labelled context
# keyword can't be deterministically distinguished from prose.

_PASSWORD_RE = re.compile(
    r"(?i)\b(?:password|passwd|pwd|passphrase)\s*[:=]\s*([^\s\n\r,;]+)"
)


class PasswordRecognizer(PatternRecognizer):
    """Plaintext password adjacent to a context keyword."""

    def __init__(self) -> None:
        patterns = [
            Pattern(
                name="password_with_label",
                regex=_PASSWORD_RE.pattern,
                score=0.85,
            )
        ]
        super().__init__(
            supported_entity="EGIS_PASSWORD",
            patterns=patterns,
            supported_language="en",
            name="EgisPasswordRecognizer",
        )


# ── Date-of-birth filter ────────────────────────────────────────────
#
# Presidio's ``DATE_TIME`` recognizer fires on every date in a prompt
# (``"meeting on May 4th"`` is a hit). For the operator-facing
# ``date_of_birth`` type we only want dates that are clearly birth
# dates — adjacent to a context keyword. We model this as a
# ``PatternRecognizer`` over the same ASCII date shapes as Presidio,
# but the regex requires a DOB keyword in the same window.

_DOB_DATE_RE = re.compile(
    r"""(?ix)
    \b(?:
        d(?:ate)?\.?\s*o(?:f)?\.?\s*b(?:irth)?       # DOB / d.o.b / date of birth
      | birth\s*date | birthdate | born\s*(?:on)?    # birth date / born / born on
    )
    \s*[:\-]?\s*
    (
        \d{4}[-/.]\d{1,2}[-/.]\d{1,2}                # 1985-04-12
      | \d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}              # 04/12/1985
      | (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)
        [a-z]*\s+\d{1,2},?\s+\d{4}                   # April 12, 1985
    )
    \b
    """,
)


class DateOfBirthRecognizer(PatternRecognizer):
    """Date-of-birth: a date next to a birth-context keyword."""

    def __init__(self) -> None:
        patterns = [
            Pattern(
                name="date_of_birth_context",
                regex=_DOB_DATE_RE.pattern,
                score=0.85,
            )
        ]
        super().__init__(
            supported_entity="EGIS_DOB",
            patterns=patterns,
            supported_language="en",
            name="EgisDateOfBirthRecognizer",
        )


# ── Word-form digit detection ──────────────────────────────────────
#
# A run of 9–19 digit-words (``"one two three four five six seven
# eight nine"``) decodes to a plain digit string. If 9 digits, treat
# as SSN-shaped; if 13–19 digits and Luhn-valid, treat as CC. Catches
# a classic prompt-obfuscation attack where the user spells a number
# out to try to evade regex-based detectors.

_WORD_DIGIT_TOKEN = (
    r"(?:zero|oh|naught|nought|one|two|three|four|"
    r"five|six|seven|eight|nine)"
)
_WORD_DIGIT_RUN_RE = re.compile(
    rf"(?:\b{_WORD_DIGIT_TOKEN}\b[\s,.\-]*(?:and[\s,.\-]+)?){{9,19}}",
    re.IGNORECASE,
)
# ``word_run_to_digits`` and ``luhn_check`` come from
# :mod:`egisai.policy._pii_helpers` (imported at the top of this file).
# The recognizers below use them inside ``analyze`` to decode the
# matched word-spelt digit run and Luhn-validate any 13–19-digit run
# before tagging it as a credit card.


class WordFormDigitsRecognizer(EntityRecognizer):
    """Decode digit-words and tag SSN- or CC-shape runs."""

    def __init__(self) -> None:
        super().__init__(
            supported_entities=["EGIS_WORD_FORM_SSN", "EGIS_WORD_FORM_CC"],
            name="EgisWordFormDigitsRecognizer",
            supported_language="en",
        )

    def load(self) -> None:  # pragma: no cover - required no-op
        return

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: NlpArtifacts | None = None,
    ) -> list[RecognizerResult]:
        wants_ssn = "EGIS_WORD_FORM_SSN" in entities
        wants_cc = "EGIS_WORD_FORM_CC" in entities
        if not (wants_ssn or wants_cc):
            return []

        results: list[RecognizerResult] = []
        for match in _WORD_DIGIT_RUN_RE.finditer(text):
            digits = word_run_to_digits(match.group(0))
            if wants_ssn and len(digits) == 9:
                results.append(
                    RecognizerResult(
                        entity_type="EGIS_WORD_FORM_SSN",
                        start=match.start(),
                        end=match.end(),
                        score=0.95,
                        analysis_explanation=AnalysisExplanation(
                            recognizer="EgisWordFormDigitsRecognizer",
                            pattern_name="word-form-ssn",
                            pattern="<word-form digit run>",
                            original_score=0.95,
                            score=0.95,
                            textual_explanation=(
                                "9 digit-words decode to SSN-shape "
                                f"sequence ({digits})."
                            ),
                        ),
                        recognition_metadata={
                            RecognizerResult.RECOGNIZER_NAME_KEY:
                                "EgisWordFormDigitsRecognizer",
                        },
                    )
                )
            elif (
                wants_cc
                and 13 <= len(digits) <= 19
                and luhn_check(digits)
            ):
                results.append(
                    RecognizerResult(
                        entity_type="EGIS_WORD_FORM_CC",
                        start=match.start(),
                        end=match.end(),
                        score=0.95,
                        analysis_explanation=AnalysisExplanation(
                            recognizer="EgisWordFormDigitsRecognizer",
                            pattern_name="word-form-cc",
                            pattern="<word-form digit run>",
                            original_score=0.95,
                            score=0.95,
                            textual_explanation=(
                                "13–19 digit-words decode to a "
                                "Luhn-valid card number."
                            ),
                        ),
                        recognition_metadata={
                            RecognizerResult.RECOGNIZER_NAME_KEY:
                                "EgisWordFormDigitsRecognizer",
                        },
                    )
                )
        return results


# ── Registration helper ─────────────────────────────────────────────


def register_custom_recognizers(registry) -> None:  # type: ignore[no-untyped-def]
    """Add every Egis-prefixed recognizer to a Presidio ``RecognizerRegistry``.

    Called once during analyzer construction. Idempotent — adding the
    same recognizer twice raises in Presidio, so we check first.
    """
    existing_names = {r.name for r in registry.recognizers}

    custom_recognizers = (
        ApiKeyRecognizer(),
        PasswordRecognizer(),
        DateOfBirthRecognizer(),
        WordFormDigitsRecognizer(),
    )
    for rec in custom_recognizers:
        if rec.name in existing_names:
            continue
        registry.add_recognizer(rec)
