"""Multi-signal PII detection with Presidio as the primary engine.

Detection runs through two paths, in order:

1. **Presidio analyzer** (when warm) — Microsoft's open-source engine
   bundling 60+ pattern recognizers (each with its own checksum or
   structural validator) plus spaCy NER for names / locations / GDPR
   special-category text. Runs entirely on-device — see
   security-and-compliance.mdc rule 1: PII never leaves the SDK
   boundary in raw form.

2. **Regex + checksum fallback** — a hand-rolled fast path that
   covers the highest-impact entities (SSN, credit card, IBAN, email,
   phone, API key) with deterministic validation. Used when the
   spaCy NER model hasn't finished loading yet (first 1–3 s after
   ``egisai.init()``, or always in air-gapped environments where the
   model can't be downloaded). Detection of names / locations is
   deferred to the warm Presidio path; everything else degrades
   gracefully.

Public API (stable):

* ``scan(text)`` → list of ``PIIFinding``
* ``sanitize(text, types=None, mask_char='#')`` → ``(masked_text, [Sanitization])``
* ``label_redact(text, types=None)`` → masked text with ``<TYPE>`` labels
* ``compute_risk_score([PIIFinding])`` → 0.0–1.0
* ``Sanitization``, ``PIIFinding`` dataclasses

Field-rename note: as of this release the canonical attribute on
``PIIFinding`` and ``Sanitization`` is ``type`` (not ``kind``). The
old ``kind`` attribute is kept as a deprecated read-only alias for
one release; new code should always use ``type``. The keyword
arguments to ``scan/sanitize/label_redact`` accept both ``types=``
(preferred) and ``kinds=`` (deprecated alias).
"""
from __future__ import annotations

import logging
import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from egisai.policy import (
    _pii_analysis_cache,
    _pii_custom,
    _pii_loader,
    _pii_scores,
    _pii_taxonomy,
)
from egisai.policy._pii_helpers import (
    is_reserved_email_domain,
    luhn_check,
    shannon_entropy,
    word_run_to_digits,
)
from egisai.policy._pii_taxonomy import PiiTypeSpec

LOGGER = logging.getLogger("egisai.pii")


# ── Public dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True)
class PIIFinding:
    """One PII entity detected in a prompt.

    ``type`` is the operator-facing canonical type id (``"ssn"``,
    ``"passport"``, …) — see :mod:`egisai.policy._pii_taxonomy`.
    ``value_redacted`` is a shape-preserving placeholder (e.g.
    ``"4111-****-****-1111"``) safe to log; the original value is
    never carried on this object.
    """

    type: str
    value_redacted: str
    confidence: float
    method: str

    @property
    def kind(self) -> str:
        """Deprecated alias for ``type``. Removed in a future release.

        Kept for one release so existing operator code paths (and any
        third-party integrations that read the SDK's internal
        objects) keep working while we migrate. Don't write new code
        against this.
        """
        return self.type


@dataclass(frozen=True)
class Sanitization:
    """Audit record for one redaction applied to a prompt.

    Carries the count and mask shape only — never the original
    value, by design (security-and-compliance.mdc rule 1).
    """

    type: str
    count: int
    pattern: str

    @property
    def kind(self) -> str:
        """Deprecated alias for ``type``. Removed in a future release."""
        return self.type


# ── Optional remote engine (server-side CPU offload) ────────────────
#
# By default the SDK runs every detection locally, on-device — that is
# the contract for `pip install egisai` users and must never regress
# (security-and-compliance.mdc rule 1: PII never leaves the SDK
# boundary in raw form). The one supported exception is a caller that
# runs *its own* trusted, in-network engine — e.g. our control plane
# offloading the CPU-heavy spaCy NER pass to a private internal
# service it fully controls. Such a caller may install an engine here;
# the raw text then travels only to that operator-owned service, never
# to a third party.
#
# When an engine is installed, `scan`/`sanitize`/`label_redact`
# delegate to it. If the engine raises for any reason (network blip,
# service cold, malformed reply) we fall back to the local path — a
# remote outage degrades detection to the local engine, it never turns
# into an error on the user's call path (sdk-design-philosophy.mdc
# rule 5: fail open on availability, never break the call path). Note
# the fall-back is still fully local PII detection, so this fail-open
# never lets raw PII slip through unscanned.


class RemotePIIEngine(Protocol):
    """Structural type for a drop-in replacement of the local engine.

    Return values MUST match the local functions exactly so callers
    can't tell the work ran elsewhere.
    """

    def scan(self, text: str) -> list[PIIFinding]: ...

    def sanitize(
        self,
        text: str,
        types: list[str] | None = None,
        mask_char: str = "#",
    ) -> tuple[str, list[Sanitization]]: ...

    def label_redact(
        self,
        text: str,
        types: list[str] | None = None,
    ) -> str: ...


_remote_engine: RemotePIIEngine | None = None


def set_remote_engine(engine: RemotePIIEngine | None) -> None:
    """Install (or clear, with ``None``) the remote PII engine.

    Call once at process start-up, before any governance traffic. Not
    part of the public `pip install` contract — this is for a trusted
    caller (our control plane) that offloads detection to an engine it
    owns. Passing ``None`` restores the fully-local behavior.
    """
    global _remote_engine
    _remote_engine = engine
    if engine is not None:
        LOGGER.info("[egisai] remote PII engine installed: %r", type(engine).__name__)
    else:
        LOGGER.info("[egisai] remote PII engine cleared — using local detection.")


def get_remote_engine() -> RemotePIIEngine | None:
    """Return the installed remote engine, or ``None`` when local."""
    return _remote_engine


# ── Normalisation ──────────────────────────────────────────────────


def _normalize_for_pii(text: str) -> str:
    """NFKC-normalise so non-ASCII digit scripts collapse to ASCII.

    Without this, ``\\d`` regex matches but downstream ``int()``
    validation raises on fullwidth / Arabic-Indic / Devanagari
    codepoints. Per security-and-compliance.mdc rule 3: detection
    must run on NFKC-normalised text.
    """
    if not text:
        return text
    try:
        return unicodedata.normalize("NFKC", text)
    except Exception:  # pragma: no cover
        return text


def _resolve_types(
    types: list[str] | None,
    kinds: list[str] | None,
) -> list[str] | None:
    """Pick the operator-supplied filter, accepting the legacy alias.

    Treats explicit ``types=`` as canonical; falls back to ``kinds=``
    only when ``types`` was omitted entirely. Empty lists count as
    "no filter". Callers warn when the alias is used so operators
    have a chance to update their config.
    """
    if types:
        return list(types)
    if kinds:
        LOGGER.warning(
            "[egisai] policy passed deprecated kinds=%r — please rename to "
            "types=. The kinds alias will be removed next release.",
            kinds,
        )
        return list(kinds)
    return None


# ── Top-level entry points ─────────────────────────────────────────


def _scan_custom(text: str) -> list[PIIFinding]:
    """Findings for the operator's own patterns.

    Confidence is fixed at 0.9 rather than scored. There is nothing to
    score: the operator asserted this shape is sensitive in their data,
    and unlike a checksum-backed detector there is no independent
    signal to weigh that assertion against. Treating it as near-certain
    is the honest reading — and it keeps a single match above the
    default risk threshold, which is what an operator who typed a
    pattern in expects to happen.
    """
    findings: list[PIIFinding] = []
    for pattern, value in _pii_custom.find(text):
        findings.append(
            PIIFinding(
                type=pattern.type_id,
                value_redacted=_pii_custom.mask_value(value, "*"),
                confidence=0.9,
                method="custom_pattern",
            )
        )
    return findings


def scan(text: str) -> list[PIIFinding]:
    """Run every detector and return validated findings.

    Hot-path-safe: tries Presidio if warm, falls back to the regex
    chain otherwise. Always NFKC-normalises before detection.

    Operator-defined patterns run in addition to the built-ins, never
    instead of them — a custom rule adds a shape the taxonomy doesn't
    cover, it does not narrow what Egis looks for.
    """
    if not text:
        return []
    engine = _remote_engine
    if engine is not None:
        try:
            remote_built_in = engine.scan(text)
        except Exception as exc:  # noqa: BLE001 — fail open to local engine
            LOGGER.warning(
                "[egisai] remote PII scan failed (%s: %s) — running the "
                "local engine instead.",
                exc.__class__.__name__,
                exc,
            )
        else:
            # The worker only knows the built-in detectors — custom
            # operator patterns are published into THIS process's
            # policy cache (see ``_pii_custom.notify_rules``), never
            # the worker's — so overlay them here. Same union the
            # local path returns, just with the heavy pass offloaded.
            return remote_built_in + _scan_custom(_normalize_for_pii(text))
    text = _normalize_for_pii(text)
    analyzer = _pii_loader.try_get_analyzer()
    built_in: list[PIIFinding] = []
    if analyzer is not None:
        try:
            built_in = _scan_with_presidio(text, analyzer, type_filter=None)
        except Exception as exc:  # noqa: BLE001 — fail closed on PII errors
            LOGGER.warning(
                "[egisai] Presidio scan raised %s: %s — falling back to regex.",
                exc.__class__.__name__,
                exc,
            )
            built_in = _scan_with_fallback(text, type_filter=None)
    else:
        built_in = _scan_with_fallback(text, type_filter=None)
    return built_in + _scan_custom(text)


def sanitize(
    text: str,
    types: list[str] | None = None,
    mask_char: str = "#",
    *,
    kinds: list[str] | None = None,
) -> tuple[str, list[Sanitization]]:
    """Mask validated PII in ``text``, returning the masked text + audit log.

    ``types`` filters detectors to a subset of the canonical
    taxonomy (see ``_pii_taxonomy.CANONICAL_TYPES``). ``None`` runs
    every detector. ``kinds`` is a deprecated alias for ``types``.
    """
    if not text:
        return text, []
    if not mask_char:
        mask_char = "#"

    type_filter = _resolve_types(types, kinds)

    engine = _remote_engine
    if engine is not None:
        try:
            masked, records = engine.sanitize(text, type_filter, mask_char)
        except Exception as exc:  # noqa: BLE001 — fail open to local engine
            LOGGER.warning(
                "[egisai] remote PII sanitize failed (%s: %s) — running the "
                "local engine instead.",
                exc.__class__.__name__,
                exc,
            )
        else:
            # Overlay custom operator patterns locally, over the
            # worker's already-masked text — same order the local path
            # uses (built-ins first, custom last) so an audit record
            # never double-counts one value. The worker never sees the
            # custom patterns, so this is the only place they run.
            masked, custom_tally = _pii_custom.apply(
                masked, type_filter, mask_char
            )
            for type_id, (count, shape) in custom_tally.items():
                records.append(
                    Sanitization(type=type_id, count=count, pattern=shape)
                )
            return masked, records

    text = _normalize_for_pii(text)
    if type_filter is not None:
        # Custom ids are namespaced and by definition absent from the
        # canonical taxonomy, so they must not be reported as unknown.
        unknown = _pii_taxonomy.unknown_types(
            [t for t in type_filter if not _pii_custom.is_custom(t)]
        )
        if unknown:
            # Mirror the Behavior the operator wants: a clear log
            # at policy-evaluation time so misconfigurations surface
            # instead of silently no-opping.
            LOGGER.warning(
                "[egisai] sanitize() ignored unknown PII types %r — "
                "supported types: see the platform Policies page.",
                unknown,
            )

    analyzer = _pii_loader.try_get_analyzer()
    masked = text
    records = []
    if analyzer is not None:
        try:
            masked, records = _sanitize_with_presidio(
                text,
                analyzer,
                type_filter=type_filter,
                mask_char=mask_char,
            )
        except Exception as exc:  # noqa: BLE001 — fail closed on PII errors
            LOGGER.warning(
                "[egisai] Presidio sanitize raised %s: %s — falling "
                "back to regex.",
                exc.__class__.__name__,
                exc,
            )
            masked, records = _sanitize_with_fallback(
                text, type_filter=type_filter, mask_char=mask_char
            )
    else:
        masked, records = _sanitize_with_fallback(
            text, type_filter=type_filter, mask_char=mask_char
        )

    # Custom patterns run last, over the already-masked text. Running
    # them first would let a built-in mask (``###-##-####``) be re-read
    # as a custom match, double-counting one value in the audit record.
    masked, custom_tally = _pii_custom.apply(masked, type_filter, mask_char)
    for type_id, (count, shape) in custom_tally.items():
        records.append(Sanitization(type=type_id, count=count, pattern=shape))
    return masked, records


def label_redact(
    text: str,
    types: list[str] | None = None,
    *,
    kinds: list[str] | None = None,
) -> str:
    """Replace each detected PII span with a typed ``<TYPE>`` label.

    Uses the same engine as ``sanitize`` but emits human-readable
    labels (``<EMAIL>``, ``<SSN>``) rather than shape-preserving
    masks. Output is safe to persist in audit logs.
    """
    if not text:
        return text
    type_filter = _resolve_types(types, kinds)
    engine = _remote_engine
    if engine is not None:
        try:
            redacted = engine.label_redact(text, type_filter)
        except Exception as exc:  # noqa: BLE001 — fail open to local engine
            LOGGER.warning(
                "[egisai] remote PII label_redact failed (%s: %s) — running "
                "the local engine instead.",
                exc.__class__.__name__,
                exc,
            )
        else:
            # Custom labels are this process's job — overlay them on
            # the worker's built-in labelled text.
            return _pii_custom.redact_labels(redacted, type_filter)
    text = _normalize_for_pii(text)

    analyzer = _pii_loader.try_get_analyzer()
    redacted = text
    if analyzer is not None:
        try:
            redacted = _label_redact_with_presidio(
                text, analyzer, type_filter=type_filter
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "[egisai] Presidio label_redact raised %s: %s — falling "
                "back to regex.",
                exc.__class__.__name__,
                exc,
            )
            redacted = _label_redact_with_fallback(
                text, type_filter=type_filter
            )
    else:
        redacted = _label_redact_with_fallback(text, type_filter=type_filter)
    return _pii_custom.redact_labels(redacted, type_filter)


def compute_risk_score(findings: list[PIIFinding]) -> float:
    """Aggregate risk score from 0.0 (clean) to 1.0 (severe).

    Combines per-finding confidence with the number of distinct
    operator types — diversity matters because an email and a credit
    card together are riskier than three emails alone.
    """
    if not findings:
        return 0.0
    weighted_sum = sum(f.confidence for f in findings)
    type_diversity = len({f.type for f in findings})
    raw = (weighted_sum * 0.3) + (type_diversity * 0.15)
    return min(1.0, round(raw / (1.0 + raw) * 2, 4))


# ── Presidio path ─────────────────────────────────────────────────


def _entities_for_filter(
    type_filter: list[str] | None,
) -> list[str] | None:
    """Translate operator types → Presidio entity names for analyze()."""
    if type_filter is None:
        # ``None`` to Presidio means "every loaded recognizer" —
        # which is what we want since our taxonomy fans out across
        # essentially every Presidio entity we know about.
        return None
    entities = _pii_taxonomy.entities_for(type_filter)
    if entities is None:
        return None
    return list(entities)


def _scan_with_presidio(
    text: str,
    analyzer,  # type: ignore[no-untyped-def]
    type_filter: list[str] | None,
) -> list[PIIFinding]:
    """Run Presidio's analyze() and translate results into ``PIIFinding``."""
    entities = _entities_for_filter(type_filter)
    results = _pii_analysis_cache.analyze_cached(
        analyzer, text=text, entities=entities
    )
    findings: list[PIIFinding] = []
    for r in results:
        if not _pii_scores.is_believable(
            r.entity_type, float(r.score), text[r.start : r.end]
        ):
            # Recognizer fired below the confidence its own design
            # says is believable — see :mod:`egisai.policy._pii_scores`.
            continue
        op_type = _pii_taxonomy.type_for_entity(r.entity_type)
        if op_type is None:
            # Recognizer fired for an entity we don't expose to the
            # operator; ignore so we don't surface taxonomy-internal
            # noise on audit records.
            continue
        if type_filter is not None and op_type not in type_filter:
            continue
        raw = text[r.start : r.end]
        # Email post-filter: strip RFC-2606 reserved domains.
        if op_type == "email" and "@" in raw:
            domain = raw.rsplit("@", 1)[-1]
            if is_reserved_email_domain(domain):
                continue
        findings.append(
            PIIFinding(
                type=op_type,
                value_redacted=_redact_value(op_type, raw),
                confidence=float(r.score),
                method=f"presidio:{r.entity_type}",
            )
        )
    return findings


def _sanitize_with_presidio(
    text: str,
    analyzer,  # type: ignore[no-untyped-def]
    type_filter: list[str] | None,
    mask_char: str,
) -> tuple[str, list[Sanitization]]:
    """Apply shape-preserving masks driven by Presidio results."""
    entities = _entities_for_filter(type_filter)
    raw_results = _pii_analysis_cache.analyze_cached(
        analyzer, text=text, entities=entities
    )
    return _apply_results(
        text,
        raw_results,
        type_filter=type_filter,
        mask_char=mask_char,
        label_mode=False,
    )


def _label_redact_with_presidio(
    text: str,
    analyzer,  # type: ignore[no-untyped-def]
    type_filter: list[str] | None,
) -> str:
    """Replace spans with typed labels — shape doesn't matter."""
    entities = _entities_for_filter(type_filter)
    raw_results = _pii_analysis_cache.analyze_cached(
        analyzer, text=text, entities=entities
    )
    masked, _ = _apply_results(
        text,
        raw_results,
        type_filter=type_filter,
        mask_char="#",
        label_mode=True,
    )
    return masked


def _apply_results(
    text: str,
    raw_results: Sequence[Any],
    *,
    type_filter: list[str] | None,
    mask_char: str,
    label_mode: bool,
) -> tuple[str, list[Sanitization]]:
    """Common masker for Presidio results.

    Walks the analyzer output, translates each Presidio entity to an
    operator type (dropping unknowns), filters by ``type_filter``,
    drops reserved-domain emails, and rewrites the spans.

    When two recognizers overlap on the same span (e.g. an SSN that
    also looks like a sequence of digits), the higher-score entity
    wins; this matches Presidio's own "highest score wins" merge
    Behavior and prevents double-masking.
    """
    selected: list[tuple[int, int, str, float]] = []
    for r in raw_results:
        # Same test the reporting path applies, so a span that gets
        # masked here is always one ``scan`` also reported.
        if not _pii_scores.is_believable(
            r.entity_type, float(r.score), text[r.start : r.end]
        ):
            continue
        op_type = _pii_taxonomy.type_for_entity(r.entity_type)
        if op_type is None:
            continue
        if type_filter is not None and op_type not in type_filter:
            continue
        raw = text[r.start : r.end]
        if op_type == "email" and "@" in raw:
            domain = raw.rsplit("@", 1)[-1]
            if is_reserved_email_domain(domain):
                continue
        selected.append((r.start, r.end, op_type, float(r.score)))

    selected.sort(key=lambda it: (it[0], -it[3]))

    # Drop overlapping spans — keep the first (highest-score after
    # the sort), discard later ones that fall inside it.
    merged: list[tuple[int, int, str]] = []
    last_end = -1
    for start, end, op_type, _score in selected:
        if start < last_end:
            continue
        merged.append((start, end, op_type))
        last_end = end

    counts: dict[str, int] = {}
    sample_pattern: dict[str, str] = {}

    pieces: list[str] = []
    cursor = 0
    for start, end, op_type in merged:
        pieces.append(text[cursor:start])
        raw = text[start:end]
        if label_mode:
            replacement = _label_for(op_type)
        else:
            replacement = _mask_value(op_type, raw, mask_char)
            counts[op_type] = counts.get(op_type, 0) + 1
            sample_pattern.setdefault(op_type, replacement)
        pieces.append(replacement)
        cursor = end
    pieces.append(text[cursor:])

    records = [
        Sanitization(type=t, count=c, pattern=sample_pattern.get(t, "[redacted]"))
        for t, c in counts.items()
    ]
    return "".join(pieces), records


def _label_for(op_type: str) -> str:
    """Render the ``<TYPE>`` label for ``label_redact`` output."""
    return f"<{op_type.upper()}>"


def _mask_value(op_type: str, raw: str, mask_char: str) -> str:
    """Shape-preserving mask for digit-shaped PII; placeholder otherwise.

    Special-case: when the operator type is digit-shaped (SSN, credit
    card, phone) but the raw matched text contains no digits at all,
    we're masking a word-form spelling (``"one two three four five
    six seven eight nine"``). ``re.sub(r"\\d", …)`` would be a no-op
    in that case and the original text would survive. Emit the
    canonical digit-shape placeholder instead so the masked output
    actually scrubs the PII and downstream tooling can pattern-match
    ``###-##-####`` regardless of the input form.
    """
    if op_type in _DIGIT_SHAPE_TYPES:
        if any(ch.isdigit() for ch in raw):
            return re.sub(r"\d", mask_char, raw)
        # Word-form (or other zero-digit) match: render the canonical
        # shape for this type so the masked output is unambiguously
        # PII-free and uniform across input variants.
        canonical = _CANONICAL_DIGIT_SHAPE.get(op_type)
        if canonical is not None:
            return canonical.replace("#", mask_char)
    return _WORD_PLACEHOLDERS.get(op_type, f"[{op_type}-redacted]")


# Canonical "what this PII looks like masked" shape per digit-shaped
# operator type. Used by ``_mask_value`` when the matched span has
# zero digits (word-form spelling) — substituting against ``\d`` would
# leave the original text intact.
_CANONICAL_DIGIT_SHAPE: dict[str, str] = {
    "ssn": "###-##-####",
    "credit_card": "####-####-####-####",
    "phone": "###-###-####",
}


def _redact_value(op_type: str, raw: str) -> str:
    """Build the safe-to-log preview shown in the ``PIIFinding`` audit blob.

    Keeps the first/last few characters where doing so doesn't leak
    enough to reconstruct the original value, and masks the middle.
    For everything that isn't shape-preserving we ship a typed
    placeholder.
    """
    if op_type == "ssn":
        # NOTE: the inner ``re.sub`` is extracted into a local so the
        # f-string body stays free of backslash escapes — Python 3.11
        # (one of our supported runtimes) disallows ``\`` inside
        # ``{ ... }`` expression slots. Python 3.12+ added support;
        # we keep the workaround to stay 3.11-compatible.
        last4 = re.sub(r"\D", "", raw)[-4:].zfill(4)
        return f"***-**-{last4}"
    if op_type == "credit_card":
        digits = re.sub(r"\D", "", raw)
        if len(digits) >= 8:
            return digits[:4] + "-****-****-" + digits[-4:]
        return "****-****"
    if op_type == "iban":
        clean = re.sub(r"\s", "", raw)
        if len(clean) >= 8:
            return clean[:4] + "****" + clean[-4:]
        return "****"
    if op_type == "email":
        local, _, domain = raw.partition("@")
        if local and domain:
            return local[0] + "***@" + domain
        return "<EMAIL>"
    if op_type == "phone":
        digits = re.sub(r"\D", "", raw)
        if len(digits) >= 4:
            return "(***) ***-" + digits[-4:]
        return "(***) ***-****"
    if op_type == "api_key" and len(raw) >= 10:
        return raw[:6] + "..." + raw[-4:]
    return f"<{op_type.upper()}>"


# Operator types whose literal value is digit-shaped enough that
# preserving the shape (SSN: ``###-##-####``, CC groups, phone
# digits) is more useful to downstream tooling than a generic
# placeholder. Anything not in this set gets a word placeholder
# from ``_WORD_PLACEHOLDERS``.
_DIGIT_SHAPE_TYPES: frozenset[str] = frozenset(
    {"ssn", "credit_card", "phone"}
)
_WORD_PLACEHOLDERS: dict[str, str] = {
    "email": "[email-redacted]",
    "iban": "[iban-redacted]",
    "api_key": "[secret-redacted]",
    "password": "[password-redacted]",
    "passport": "[passport-redacted]",
    "drivers_license": "[license-redacted]",
    "national_id": "[id-redacted]",
    "person_name": "[name-redacted]",
    "address": "[address-redacted]",
    "url": "[url-redacted]",
    "ip_address": "[ip-redacted]",
    "mac_address": "[mac-redacted]",
    "crypto_wallet": "[wallet-redacted]",
    "medical_license": "[license-redacted]",
    "bank_account": "[account-redacted]",
    "vehicle_registration": "[plate-redacted]",
    "date_of_birth": "[dob-redacted]",
    "nationality_or_religion": "[demographic-redacted]",
}


# ── Regex / checksum fallback path ─────────────────────────────────
#
# Mirrors a subset of the Presidio detectors so the SDK keeps working
# during the spaCy warm-up window or in air-gapped installs. Coverage
# is intentionally narrower than the warm path — names / locations /
# multi-country IDs / GDPR special-category text only land when the
# Presidio engine is up. Per security-and-compliance.mdc rule 4 we
# fail-closed on detector errors (a missed scan would beat a missed
# detection).


_CC_RE = re.compile(
    r"\b"
    r"(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"
    r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,4}"
    r"\b"
)
_SSN_RE = re.compile(r"\b(\d{3})[\s\-](\d{2})[\s\-](\d{4})\b")
_INVALID_SSN_AREAS = {0, 666} | set(range(900, 1000))
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+1[\s\-]?)?"
    r"\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"
    r"(?!\d)"
)
_IBAN_RE = re.compile(
    r"\b([A-Z]{2}\d{2}[\s]?[\dA-Z]{4}[\s]?(?:[\dA-Z]{4}[\s]?){1,7}[\dA-Z]{1,4})\b"
)
_KEY_RE = re.compile(
    r"(?:"
    r"(?:sk|pk|api|key|token|secret|bearer|ghp|gho|ghu|ghs|ghr|glpat|"
    r"AKIA|ASIA|xox[bpsar]|eyJ|sl\.)"
    r"[\-_]?"
    r"[A-Za-z0-9\-_\.]{20,80}"
    r")"
)
# Word-form digit obfuscation: ``"one two three four five six seven
# eight nine"`` is a SSN-shape sequence; the same mechanism Luhn-
# validates a 13–19-token run as a credit card. Mirrors the
# Presidio-side ``WordFormDigitsRecognizer`` so detection coverage
# is constant between the warm Presidio path and the regex fallback
# (no feature regression during the spaCy warm-up window).
_WORD_DIGIT_TOKEN = (
    r"(?:zero|oh|naught|nought|one|two|three|four|"
    r"five|six|seven|eight|nine)"
)
_WORD_DIGIT_RUN_RE = re.compile(
    rf"(?:\b{_WORD_DIGIT_TOKEN}\b[\s,.\-]*(?:and[\s,.\-]+)?){{9,19}}",
    re.IGNORECASE,
)
_BLOB_RE = re.compile(r"\b[A-Za-z0-9+/\-_]{32,128}={0,2}\b")
_CONTEXT_KEYWORDS: dict[str, list[str]] = {
    "credit_card": ["card", "visa", "mastercard", "amex", "payment", "credit", "debit"],
    "ssn": ["ssn", "social security", "social-security", "tax id", "taxpayer"],
    "api_key": [
        "key", "secret", "token", "api_key", "apikey",
        "password", "credential", "bearer", "authorization",
    ],
    "email": ["email", "e-mail", "mailto", "contact"],
    "phone": ["phone", "mobile", "cell", "tel", "call", "sms", "whatsapp"],
    "iban": ["iban", "bank", "account", "transfer", "swift", "bic"],
}
_CONTEXT_WINDOW = 60


def _iban_mod97(iban: str) -> bool:
    """ISO 7064 mod-97 IBAN validation."""
    clean = re.sub(r"\s", "", iban)
    if len(clean) < 15 or len(clean) > 34:
        return False
    rearranged = clean[4:] + clean[:4]
    numeric = ""
    for ch in rearranged:
        if ch.isdigit():
            numeric += ch
        elif ch.isalpha():
            numeric += str(ord(ch.upper()) - 55)
        else:
            return False
    return int(numeric) % 97 == 1


def _context_boost(text: str, start: int, end: int, op_type: str) -> float:
    window_start = max(0, start - _CONTEXT_WINDOW)
    window_end = min(len(text), end + _CONTEXT_WINDOW)
    window = text[window_start:window_end].lower()
    keywords = _CONTEXT_KEYWORDS.get(op_type, [])
    hits = sum(1 for kw in keywords if kw in window)
    return min(0.15, hits * 0.05)


def _scan_with_fallback(
    text: str,
    type_filter: list[str] | None,
) -> list[PIIFinding]:
    """Regex+checksum scan covering the highest-impact entities only.

    Coverage parity with the pre-Presidio engine: SSN, credit card,
    IBAN, email, phone, API key — all checksum or context validated.
    Anything Presidio adds (passport, names, multi-country IDs, …)
    requires the warm path; callers degrade gracefully.
    """
    findings: list[PIIFinding] = []
    wants = lambda op_type: type_filter is None or op_type in type_filter  # noqa: E731

    if wants("credit_card"):
        for m in _CC_RE.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if 13 <= len(digits) <= 19 and luhn_check(digits):
                conf = 0.85 + _context_boost(text, m.start(), m.end(), "credit_card")
                findings.append(
                    PIIFinding(
                        type="credit_card",
                        value_redacted=digits[:4] + "-****-****-" + digits[-4:],
                        confidence=min(1.0, conf),
                        method="luhn",
                    )
                )
    if wants("ssn"):
        for m in _SSN_RE.finditer(text):
            area, group, serial = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if area in _INVALID_SSN_AREAS or group == 0 or serial == 0:
                continue
            conf = 0.70 + _context_boost(text, m.start(), m.end(), "ssn")
            findings.append(
                PIIFinding(
                    type="ssn",
                    value_redacted=f"***-**-{serial:04d}",
                    confidence=min(1.0, conf),
                    method="area_validation+context",
                )
            )
    if wants("iban"):
        for m in _IBAN_RE.finditer(text):
            raw = m.group()
            if not _iban_mod97(raw):
                continue
            clean = re.sub(r"\s", "", raw)
            conf = 0.95 + _context_boost(text, m.start(), m.end(), "iban")
            findings.append(
                PIIFinding(
                    type="iban",
                    value_redacted=clean[:4] + "****" + clean[-4:],
                    confidence=min(1.0, conf),
                    method="mod97",
                )
            )
    if wants("email"):
        for m in _EMAIL_RE.finditer(text):
            raw = m.group()
            local, _, domain = raw.partition("@")
            if not domain or is_reserved_email_domain(domain):
                continue
            conf = 0.80 + _context_boost(text, m.start(), m.end(), "email")
            findings.append(
                PIIFinding(
                    type="email",
                    value_redacted=local[0] + "***@" + domain,
                    confidence=min(1.0, conf),
                    method="structural",
                )
            )
    if wants("phone"):
        for m in _PHONE_RE.finditer(text):
            digits = re.sub(r"\D", "", m.group())
            if len(digits) < 10 or len(digits) > 11:
                continue
            conf = 0.55 + _context_boost(text, m.start(), m.end(), "phone")
            if conf < 0.65:
                continue
            findings.append(
                PIIFinding(
                    type="phone",
                    value_redacted="(***) ***-" + digits[-4:],
                    confidence=min(1.0, conf),
                    method="pattern+context",
                )
            )
    if wants("ssn") or wants("credit_card"):
        for m in _WORD_DIGIT_RUN_RE.finditer(text):
            digits = word_run_to_digits(m.group(0))
            if wants("ssn") and len(digits) == 9:
                findings.append(
                    PIIFinding(
                        type="ssn",
                        value_redacted=f"***-**-{digits[-4:]}",
                        confidence=0.95,
                        method="word-form",
                    )
                )
            elif (
                wants("credit_card")
                and 13 <= len(digits) <= 19
                and luhn_check(digits)
            ):
                findings.append(
                    PIIFinding(
                        type="credit_card",
                        value_redacted=digits[:4] + "-****-****-" + digits[-4:],
                        confidence=0.95,
                        method="word-form+luhn",
                    )
                )
    if wants("api_key"):
        seen: set[tuple[int, int]] = set()
        for m in _KEY_RE.finditer(text):
            raw = m.group()
            entropy = shannon_entropy(raw)
            if entropy < 3.5:
                continue
            seen.add((m.start(), m.end()))
            conf = 0.75 + (entropy - 3.5) * 0.1 + _context_boost(
                text, m.start(), m.end(), "api_key"
            )
            findings.append(
                PIIFinding(
                    type="api_key",
                    value_redacted=raw[:6] + "..." + raw[-4:],
                    confidence=min(1.0, conf),
                    method="prefix+entropy",
                )
            )
        for m in _BLOB_RE.finditer(text):
            span = (m.start(), m.end())
            if any(s[0] <= span[0] and s[1] >= span[1] for s in seen):
                continue
            raw = m.group()
            entropy = shannon_entropy(raw)
            if entropy < 4.5 or len(raw) < 32:
                continue
            ctx = _context_boost(text, m.start(), m.end(), "api_key")
            if ctx == 0 and entropy < 5.0:
                continue
            conf = 0.55 + (entropy - 4.5) * 0.15 + ctx
            findings.append(
                PIIFinding(
                    type="api_key",
                    value_redacted=raw[:6] + "..." + raw[-4:],
                    confidence=min(1.0, conf),
                    method="entropy",
                )
            )
    return findings


def _sanitize_with_fallback(
    text: str,
    type_filter: list[str] | None,
    mask_char: str,
) -> tuple[str, list[Sanitization]]:
    """Mask validated PII using the regex+checksum engine.

    Mirrors the operator-facing types our regex chain knows about:
    ssn, credit_card, iban, email, phone, api_key. Other types are
    silently no-ops in this code path — they require the warm
    Presidio engine.
    """
    counts: dict[str, int] = {}
    sample_pattern: dict[str, str] = {}

    def _bump(op_type: str, rendered: str) -> None:
        counts[op_type] = counts.get(op_type, 0) + 1
        sample_pattern.setdefault(op_type, rendered)

    wants = lambda op_type: type_filter is None or op_type in type_filter  # noqa: E731

    if wants("ssn"):
        def _ssn_repl(m: re.Match[str]) -> str:
            area, group, serial = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if area in _INVALID_SSN_AREAS or group == 0 or serial == 0:
                return m.group(0)
            rendered = re.sub(r"\d", mask_char, m.group(0))
            _bump("ssn", rendered)
            return rendered
        text = _SSN_RE.sub(_ssn_repl, text)

    if wants("credit_card"):
        def _cc_repl(m: re.Match[str]) -> str:
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) < 13 or len(digits) > 19 or not luhn_check(digits):
                return m.group(0)
            rendered = re.sub(r"\d", mask_char, m.group(0))
            _bump("credit_card", rendered)
            return rendered
        text = _CC_RE.sub(_cc_repl, text)

    if wants("ssn") or wants("credit_card"):
        # Word-form digit run: decode to digits, route to ``ssn`` if
        # length 9 or to ``credit_card`` if Luhn-valid 13–19. Same
        # mask shape as the digit-form spans above so downstream
        # tooling can't tell them apart.
        def _word_repl(m: re.Match[str]) -> str:
            digits = word_run_to_digits(m.group(0))
            if wants("ssn") and len(digits) == 9:
                rendered = mask_char * 3 + "-" + mask_char * 2 + "-" + mask_char * 4
                _bump("ssn", rendered)
                return rendered
            if (
                wants("credit_card")
                and 13 <= len(digits) <= 19
                and luhn_check(digits)
            ):
                rendered = "-".join([mask_char * 4] * 4)
                _bump("credit_card", rendered)
                return rendered
            return m.group(0)
        text = _WORD_DIGIT_RUN_RE.sub(_word_repl, text)

    if wants("iban"):
        def _iban_repl(m: re.Match[str]) -> str:
            if not _iban_mod97(m.group(0)):
                return m.group(0)
            rendered = _WORD_PLACEHOLDERS["iban"]
            _bump("iban", rendered)
            return rendered
        text = _IBAN_RE.sub(_iban_repl, text)

    if wants("email"):
        def _email_repl(m: re.Match[str]) -> str:
            domain = m.group(0).rsplit("@", 1)[-1]
            if is_reserved_email_domain(domain):
                return m.group(0)
            rendered = _WORD_PLACEHOLDERS["email"]
            _bump("email", rendered)
            return rendered
        text = _EMAIL_RE.sub(_email_repl, text)

    if wants("phone"):
        def _phone_repl(m: re.Match[str]) -> str:
            raw = m.group(0)
            digits = re.sub(r"\D", "", raw)
            if len(digits) < 10 or len(digits) > 11:
                return raw
            conf = 0.55 + _context_boost(text, m.start(), m.end(), "phone")
            if conf < 0.65:
                return raw
            rendered = re.sub(r"\d", mask_char, raw)
            _bump("phone", rendered)
            return rendered
        text = _PHONE_RE.sub(_phone_repl, text)

    if wants("api_key"):
        def _key_repl(m: re.Match[str]) -> str:
            raw = m.group(0)
            if shannon_entropy(raw) < 3.5:
                return raw
            rendered = _WORD_PLACEHOLDERS["api_key"]
            _bump("api_key", rendered)
            return rendered
        text = _KEY_RE.sub(_key_repl, text)

    records = [
        Sanitization(type=t, count=c, pattern=sample_pattern.get(t, "[redacted]"))
        for t, c in counts.items()
    ]
    return text, records


def _label_redact_with_fallback(
    text: str,
    type_filter: list[str] | None,
) -> str:
    """Replace each detected PII span with a typed ``<TYPE>`` label.

    Same coverage as ``_sanitize_with_fallback`` — anything richer
    (names, multi-country IDs, …) requires the warm Presidio engine.
    """
    wants = lambda op_type: type_filter is None or op_type in type_filter  # noqa: E731

    if wants("ssn"):
        def _ssn_repl(m: re.Match[str]) -> str:
            area, group, serial = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if area in _INVALID_SSN_AREAS or group == 0 or serial == 0:
                return m.group(0)
            return _label_for("ssn")
        text = _SSN_RE.sub(_ssn_repl, text)

    if wants("credit_card"):
        def _cc_repl(m: re.Match[str]) -> str:
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) < 13 or len(digits) > 19 or not luhn_check(digits):
                return m.group(0)
            return _label_for("credit_card")
        text = _CC_RE.sub(_cc_repl, text)

    if wants("ssn") or wants("credit_card"):
        def _word_repl(m: re.Match[str]) -> str:
            digits = word_run_to_digits(m.group(0))
            if wants("ssn") and len(digits) == 9:
                return _label_for("ssn")
            if (
                wants("credit_card")
                and 13 <= len(digits) <= 19
                and luhn_check(digits)
            ):
                return _label_for("credit_card")
            return m.group(0)
        text = _WORD_DIGIT_RUN_RE.sub(_word_repl, text)

    if wants("iban"):
        def _iban_repl(m: re.Match[str]) -> str:
            if not _iban_mod97(m.group(0)):
                return m.group(0)
            return _label_for("iban")
        text = _IBAN_RE.sub(_iban_repl, text)

    if wants("email"):
        def _email_repl(m: re.Match[str]) -> str:
            domain = m.group(0).rsplit("@", 1)[-1]
            if is_reserved_email_domain(domain):
                return m.group(0)
            return _label_for("email")
        text = _EMAIL_RE.sub(_email_repl, text)

    if wants("phone"):
        def _phone_repl(m: re.Match[str]) -> str:
            raw = m.group(0)
            digits = re.sub(r"\D", "", raw)
            if len(digits) < 10 or len(digits) > 11:
                return raw
            return _label_for("phone")
        text = _PHONE_RE.sub(_phone_repl, text)

    if wants("api_key"):
        def _key_repl(m: re.Match[str]) -> str:
            raw = m.group(0)
            if shannon_entropy(raw) < 3.5:
                return raw
            return _label_for("api_key")
        text = _KEY_RE.sub(_key_repl, text)

    return text


# ── Public catalog passthrough ─────────────────────────────────────


def available_types() -> tuple[PiiTypeSpec, ...]:
    """Operator-facing PII types this SDK can detect.

    Equivalent to ``_pii_taxonomy.all_types()`` — exposed here so the
    backend's ``GET /v1/sdk/pii-types`` endpoint can ship the same
    list to the dashboard without importing internals.
    """
    return _pii_taxonomy.all_types()


# ── Re-export utilities still imported by the engine ──────────────


__all__ = [
    "PIIFinding",
    "Sanitization",
    "available_types",
    "compute_risk_score",
    "label_redact",
    "sanitize",
    "scan",
]


# Keep ``math`` referenced so unused-import linters don't strip it —
# the entropy fallback above uses it indirectly via shannon_entropy
# (imported from _pii_recognizers).
_ = math
