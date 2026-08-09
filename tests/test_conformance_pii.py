"""The Python detector's conformance run against the shared PII corpus.

Companion to ``test_conformance_corpus.py``, which covers the policy
engine. This module covers the layer underneath it: the deterministic
PII detector that every language has to reimplement.

Everything here runs against the **regex + checksum tier only**. The
Presidio/NER tier is deliberately switched off for the duration of each
test, for two reasons:

1. It is not reproducible across machines. Whether the analyzer is warm
   depends on a background thread, an optional dependency, and how long
   the process has been alive. A corpus that passed or failed on that
   basis would be worse than no corpus.
2. It is not portable. A TypeScript engine ships regexes, not spaCy.
   Measuring it against detections it was never meant to make would
   push whoever writes it to fake them, which is the opposite of what
   this file is for.

So the contract pinned here is the floor: what every engine must find,
in every language, with no model loaded. The ML tier is strictly
additive on top of it and is covered by the tests in
``test_pii_scores.py`` and friends.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from conftest import CONFORMANCE_ROOT, skip_without_corpus

from egisai.policy import pii as pii_module

CORPUS_ROOT = CONFORMANCE_ROOT
PII_CORPUS = CORPUS_ROOT / "pii-corpus"

# See the sibling policy-corpus module and the ``IN_MONOREPO`` note in
# ``conftest.py``: absent on the mirror is expected, absent here is a
# broken gate.
pytestmark = skip_without_corpus


def _load_cases() -> list[tuple[str, dict[str, Any]]]:
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(PII_CORPUS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            found.append((f"{path.stem}::{case['id']}", case))
    return found


CASES = _load_cases()


@pytest.fixture(autouse=True)
def deterministic_tier_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the regex tier for every case in this module.

    ``scan``/``sanitize``/``label_redact`` all consult
    ``_pii_loader.try_get_analyzer()`` and take the Presidio path when
    it returns an engine. Returning ``None`` is the same signal the
    real code sees during warm-up, so this exercises a genuine
    production path rather than a test-only branch.
    """
    monkeypatch.setattr(
        pii_module._pii_loader, "try_get_analyzer", lambda: None
    )


def test_the_pii_corpus_is_actually_present() -> None:
    """Guard against a silently empty parametrization.

    Zero cases and all cases passing look identical in pytest output.
    """
    assert len(CASES) >= 20, (
        f"Only {len(CASES)} PII conformance cases loaded from "
        f"{PII_CORPUS}. Either the corpus moved or the loader is not "
        "finding it."
    )


@pytest.mark.parametrize(
    "case", [c for _, c in CASES], ids=[i for i, _ in CASES]
)
def test_the_detector_agrees_with_the_corpus(case: dict[str, Any]) -> None:
    text = case["text"]
    why = case["description"]
    type_filter = case.get("sanitize_types")
    mask_char = case.get("mask_char", "#")

    if "types" in case:
        found = sorted({f.type for f in pii_module.scan(text)})
        assert found == sorted(case["types"]), (
            f"{case['id']}: expected to detect {sorted(case['types'])}, "
            f"found {found}.\n{why}"
        )

    if "sanitized" in case:
        masked, records = pii_module.sanitize(
            text, types=type_filter, mask_char=mask_char
        )
        assert masked == case["sanitized"], (
            f"{case['id']}: expected {case['sanitized']!r}, got "
            f"{masked!r}.\n{why}"
        )

        if "records" in case:
            actual = [
                {"type": r.type, "count": r.count, "pattern": r.pattern}
                for r in records
            ]
            assert actual == case["records"], (
                f"{case['id']}: expected sanitization records "
                f"{case['records']}, got {actual}.\n{why}"
            )

        if "record_must_not_contain" in case:
            raw = case["record_must_not_contain"]
            blob = repr([(r.type, r.count, r.pattern) for r in records])
            assert raw not in blob, (
                f"{case['id']}: the raw value {raw!r} survived into the "
                f"sanitization record. Audit rows are persisted, so this "
                f"is the leak the mask exists to prevent.\n{why}"
            )
            assert raw not in masked, (
                f"{case['id']}: the raw value {raw!r} survived into the "
                f"masked text.\n{why}"
            )

    if "label_redacted" in case:
        labelled = pii_module.label_redact(text, types=type_filter)
        assert labelled == case["label_redacted"], (
            f"{case['id']}: expected {case['label_redacted']!r}, got "
            f"{labelled!r}.\n{why}"
        )
