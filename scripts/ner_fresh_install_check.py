#!/usr/bin/env python3
"""Fresh-install NER gate.

Proves that a from-scratch install of the ``egisai`` wheel — exactly what a
customer gets from ``pip install egisai`` — loads the spaCy / Presidio NER
model and actually masks a person name and a street address, rather than
silently falling back to the regex-only detectors.

This is the gate that catches a broken upstream native dependency (the
``murmurhash 1.0.16`` tokenizer regression that raised
``OverflowError: Python int too large to convert to C ssize_t`` inside
``spacy.load`` and tripped ``pii_ner_loader_failed``). A regex-only fallback
cannot detect free-form PERSON names, so a successful name mask is definitive
proof that the heavy NER model loaded.

Run it against a wheel installed in a CLEAN virtualenv (never the source
tree), on each supported interpreter::

    python -m venv /tmp/v
    /tmp/v/bin/pip install dist/egisai-*.whl
    /tmp/v/bin/python egisai/scripts/ner_fresh_install_check.py

Exits non-zero with an actionable message if the model never warms (download
or ``spacy.load`` failed) or if a name / address survives sanitization.
"""

from __future__ import annotations

import os
import sys
from typing import NoReturn

# Be explicit about the engine so the check is deterministic regardless of any
# ambient EGISAI_NER_ENGINE. spaCy ``en_core_web_lg`` is the SDK default and is
# the engine a stock ``pip install egisai`` uses.
os.environ["EGISAI_NER_ENGINE"] = "spacy"

# Generous because a cold run auto-downloads the ~400 MB model wheel exactly
# like a customer's first ``egisai.init()`` does.
_WARM_TIMEOUT_SECS = 300.0

_TEXT = "John Smith lives at 1600 Pennsylvania Avenue, Washington DC"


def _fail(msg: str) -> NoReturn:
    print(f"FRESH-INSTALL NER CHECK FAILED: {msg}", flush=True)
    raise SystemExit(1)


def main() -> None:
    import egisai

    print(f"python     : {sys.version.split()[0]}", flush=True)
    print(f"egisai     : {egisai.__version__}", flush=True)

    import spacy

    print(f"spacy      : {spacy.__version__}", flush=True)
    try:
        import murmurhash

        print(f"murmurhash : {murmurhash.__version__}", flush=True)
    except Exception:  # noqa: BLE001 - informational only
        pass

    from egisai.policy import _pii_loader, pii

    # ``prime_analyzer_async`` is exactly what ``egisai.init()`` calls to start
    # the background NER warm-up (and, off on-prem, to auto-download the model).
    # Calling it directly keeps this check independent of a reachable backend.
    _pii_loader.prime_analyzer_async()

    # Block until the spaCy / Presidio analyzer is actually warm. ``False`` means
    # the model download or ``spacy.load`` crashed — the exact murmurhash-1.0.16
    # failure mode we are gating against.
    if not _pii_loader.wait_for_warm(timeout_secs=_WARM_TIMEOUT_SECS):
        _fail(
            "spaCy / Presidio NER never warmed within "
            f"{_WARM_TIMEOUT_SECS:.0f}s — en_core_web_lg download or spacy.load "
            "failed (e.g. a regressed native dependency like murmurhash 1.0.16)."
        )

    masked, records = pii.sanitize(_TEXT)
    types = sorted({r.type for r in records})
    print(f"input      : {_TEXT}", flush=True)
    print(f"sanitized  : {masked}", flush=True)
    print(f"record types: {types}", flush=True)

    # Definitive NER proof: a free-form PERSON name can only be detected by the
    # NER model, never by the regex/checksum fallback.
    if "John Smith" in masked:
        _fail(f"PERSON name not masked — NER degraded to regex fallback: {masked!r}")
    if "person_name" not in types:
        _fail(f"no person_name detection recorded — NER not active: {types}")
    # Address recall: the city/state component is masked by the NER engine.
    if "Washington" in masked:
        _fail(f"street ADDRESS not masked — NER degraded to regex fallback: {masked!r}")
    if "address" not in types:
        _fail(f"no address detection recorded — NER not active: {types}")

    print("OK — fresh install loaded the NER model and masked name + address.", flush=True)


if __name__ == "__main__":
    main()
