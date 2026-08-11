"""Operator-defined PII patterns, alongside the canonical taxonomy.

The 21 types in :mod:`egisai.policy._pii_taxonomy` are the shapes that
are the same at every company — an IBAN is an IBAN whether you are a
bank or a bakery. The shapes that leak in practice often are not:
``EMP-004182``, ``PT-99213``, an internal case number that means
nothing outside one org's ticketing system but is still the thing an
auditor asks about.

Those can't ship in the taxonomy. There is no detector to write,
because the pattern is a fact about one customer's data, not about the
world. So the operator supplies it and Egis runs it beside the
built-ins.

What a custom pattern is not
----------------------------
It is a regex, and that buys exactly what a regex buys. The built-in
types are backed by checksums (Luhn, mod-97, Verhoeff) or by a named-
entity model, which is why they can claim a false-positive rate. A
custom pattern claims nothing — it matches what it matches. That is
the right trade for ``EMP-\\d{6}``, which is unambiguous by
construction, and the wrong one for "names of our customers", which is
what ``person_name`` and the semantic judge are for.

Safety
------
Every pattern is compiled through :func:`_regex_safe.safe_compile`, so
the nested-quantifier and runaway-optional shapes behind essentially
all real ReDoS reports are rejected outright. That happens twice: once
in the platform when the policy is saved, so the operator gets an error
in the form instead of a slow agent, and once here, because a rule can
reach an SDK from a backend that predates the validation.

A pattern that fails to compile is dropped with a warning rather than
raising. One bad regex in one policy must not take out PII detection
for every other type the operator configured.

Identity on the wire
--------------------
A custom type is addressed as ``custom:<id>``, which is what appears in
``sanitize_types``, in the audit record's ``type`` field, and on the
dashboard. The prefix is what keeps the two namespaces from ever
colliding: no canonical id contains a colon, so a custom pattern can
never shadow ``ssn`` no matter what the operator names it.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from egisai.policy._regex_safe import UnsafePatternError, safe_compile

LOGGER = logging.getLogger("egisai.pii.custom")

#: Namespace marker. Every operator-defined type id carries it, and no
#: canonical type ever will — the taxonomy ids are bare identifiers.
CUSTOM_TYPE_PREFIX = "custom:"

#: Ceiling on how many patterns one org can run. Each one is another
#: full pass over every prompt, so this is a latency budget as much as
#: a sanity limit. Well past what any real deployment has needed.
MAX_CUSTOM_PATTERNS = 25

#: Non-alphanumerics are preserved when masking so the shape of the
#: value survives — ``EMP-004182`` masks to ``###-######``, which tells
#: an operator reading the audit row what was caught without telling
#: them what it said.
_MASKABLE = re.compile(r"[^\W_]", re.UNICODE)


@dataclass(frozen=True)
class CustomPattern:
    """One operator-defined shape, ready to run."""

    #: Wire id, always ``custom:``-prefixed.
    type_id: str
    #: What the operator called it. Rendered on the dashboard and in
    #: policy messages; never used for matching.
    label: str
    regex: re.Pattern[str]


def is_custom(type_id: str) -> bool:
    return type_id.startswith(CUSTOM_TYPE_PREFIX)


def _clean_id(raw: object, *, fallback: str) -> str:
    """Normalise an id to ``[a-z0-9_]`` with a ``custom:`` prefix."""
    text = str(raw or "").strip().lower()
    if text.startswith(CUSTOM_TYPE_PREFIX):
        text = text[len(CUSTOM_TYPE_PREFIX) :]
    slug = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    if not slug:
        slug = re.sub(r"[^a-z0-9]+", "_", fallback.strip().lower()).strip("_")
    return f"{CUSTOM_TYPE_PREFIX}{slug or 'pattern'}"


def parse(raw: object) -> list[CustomPattern]:
    """Read a policy's ``custom_types`` config into runnable patterns.

    Tolerant by design: anything malformed is skipped with a warning
    rather than raising. This runs against config that may have been
    written by a newer dashboard than this SDK knows about, and the
    correct response to one unreadable entry is to keep detecting
    everything else.
    """
    if not isinstance(raw, list):
        return []
    out: list[CustomPattern] = []
    seen: set[str] = set()
    for entry in raw[:MAX_CUSTOM_PATTERNS]:
        if not isinstance(entry, dict):
            continue
        pattern = entry.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            continue
        label = str(entry.get("label") or "").strip()
        type_id = _clean_id(entry.get("id"), fallback=label or pattern)
        if type_id in seen:
            continue
        flags = 0 if entry.get("case_sensitive") else re.IGNORECASE
        try:
            compiled = safe_compile(pattern, flags)
        except (UnsafePatternError, re.error) as exc:
            LOGGER.warning(
                "[egisai] custom PII pattern %r was not loaded: %s",
                label or type_id,
                exc,
            )
            continue
        seen.add(type_id)
        out.append(
            CustomPattern(
                type_id=type_id,
                label=label or type_id[len(CUSTOM_TYPE_PREFIX) :],
                regex=compiled,
            )
        )
    return out


# ── Active set ──────────────────────────────────────────────────────
#
# Published once whenever the policy cache changes, read on every call.
# A plain module global rebound as a whole is enough for that shape:
# readers take one reference and the tuple they get can never be
# mutated underneath them, so no lock is needed on the hot path.

_ACTIVE: tuple[CustomPattern, ...] = ()
_BY_ID: Mapping[str, CustomPattern] = {}


def notify_rules(rules: Iterable[Any]) -> None:
    """Publish the union of custom patterns across all ``pii_scan`` rules.

    Called from the policy cache on every refresh, mirroring
    ``limits.notify_rules``. Doing it here rather than at match time
    means the compile cost is paid once per policy change instead of
    once per model call, and — the part that actually matters — the
    patterns are in place before anything asks to sanitize with them.

    That ordering is the whole design. ``sanitize_types`` travels from
    the decision to a dozen call sites that mask the payload, and none
    of them know what a policy is. They pass a list of type ids and
    trust that the ids resolve. If a custom id could arrive at one of
    those call sites unregistered, Egis would report a sanitize it did
    not perform.
    """
    collected: list[CustomPattern] = []
    seen: set[str] = set()
    for rule in rules or ():
        if getattr(rule, "type", None) != "pii_scan":
            continue
        config = getattr(rule, "config", None)
        if not isinstance(config, dict):
            continue
        for item in parse(config.get("custom_types")):
            if item.type_id in seen:
                continue
            seen.add(item.type_id)
            collected.append(item)
            if len(collected) >= MAX_CUSTOM_PATTERNS:
                break
    _publish(tuple(collected))


def _publish(patterns: tuple[CustomPattern, ...]) -> None:
    global _ACTIVE, _BY_ID
    _ACTIVE = patterns
    _BY_ID = {p.type_id: p for p in patterns}


def active() -> tuple[CustomPattern, ...]:
    return _ACTIVE


def label_for(type_id: str) -> str:
    """Operator's name for a custom type, or the bare id if unknown."""
    found = _BY_ID.get(type_id)
    if found is not None:
        return found.label
    return type_id[len(CUSTOM_TYPE_PREFIX) :] if is_custom(type_id) else type_id


def reset_for_tests() -> None:
    _publish(())


# ── Matching ────────────────────────────────────────────────────────


def _selected(
    type_filter: Sequence[str] | None,
) -> tuple[CustomPattern, ...]:
    patterns = _ACTIVE
    if type_filter is None:
        return patterns
    wanted = set(type_filter)
    return tuple(p for p in patterns if p.type_id in wanted)


def find(
    text: str, type_filter: Sequence[str] | None = None
) -> list[tuple[CustomPattern, str]]:
    """Every custom match in ``text`` as ``(pattern, matched_text)``.

    The matched text is returned so the caller can render a masked
    shape from it. It is never retained, logged, or attached to a
    finding.
    """
    if not text:
        return []
    out: list[tuple[CustomPattern, str]] = []
    for pattern in _selected(type_filter):
        try:
            for match in pattern.regex.finditer(text):
                value = match.group(0)
                if value:
                    out.append((pattern, value))
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "[egisai] custom PII pattern %r failed while scanning: %s",
                pattern.label,
                exc,
            )
    return out


def mask_value(value: str, mask_char: str) -> str:
    """Shape-preserving mask for one matched value."""
    return _MASKABLE.sub(mask_char or "#", value)


def apply(
    text: str,
    type_filter: Sequence[str] | None,
    mask_char: str,
) -> tuple[str, dict[str, tuple[int, str]]]:
    """Mask every custom match, returning the text and per-type counts.

    The second element maps ``type_id`` to ``(count, mask_shape)`` —
    the two things the audit record carries, and the only two it is
    ever allowed to.
    """
    if not text:
        return text, {}
    tally: dict[str, tuple[int, str]] = {}
    for pattern in _selected(type_filter):

        def _replace(match: re.Match[str], _p: CustomPattern = pattern) -> str:
            rendered = mask_value(match.group(0), mask_char)
            count, shape = tally.get(_p.type_id, (0, rendered))
            tally[_p.type_id] = (count + 1, shape)
            return rendered

        try:
            text = pattern.regex.sub(_replace, text)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "[egisai] custom PII pattern %r failed while masking: %s",
                pattern.label,
                exc,
            )
    return text, tally


def redact_labels(text: str, type_filter: Sequence[str] | None) -> str:
    """Replace custom matches with ``<LABEL>`` for audit-safe display."""
    if not text:
        return text
    for pattern in _selected(type_filter):
        label = re.sub(r"[^A-Za-z0-9]+", "_", pattern.label).strip("_").upper()
        try:
            text = pattern.regex.sub(f"<{label or 'CUSTOM'}>", text)
        except Exception:  # noqa: BLE001
            continue
    return text


__all__ = [
    "CUSTOM_TYPE_PREFIX",
    "MAX_CUSTOM_PATTERNS",
    "CustomPattern",
    "active",
    "apply",
    "find",
    "is_custom",
    "label_for",
    "notify_rules",
    "parse",
    "redact_labels",
    "reset_for_tests",
]
