"""Confidence floors: kill code-shaped noise, keep every real detection.

The floors exist because two Presidio recognizers are *designed* to
over-fire and rely on context to become believable. These tests pin
the exact line between "the recognizer guessed" (dropped) and "the
evidence agreed" (kept), plus the escape hatch.
"""

from __future__ import annotations

import pytest

from egisai.policy import _pii_scores as scores


def test_unfloored_entities_pass_at_any_score() -> None:
    # Checksum-validated recognizers own their scores; a low-score
    # SSN must never be suppressed (fail closed on PII).
    assert scores.is_believable("US_SSN", 0.05, "856-45-6789")
    assert scores.is_believable("CREDIT_CARD", 0.1, "4111 1111 1111 1111")


def test_uncontexted_license_guess_is_dropped() -> None:
    # 0.3 is the raw pattern score for "[A-Z][0-9]{1,12}" — this is
    # what fires on "v1", "o3" and every version string.
    assert not scores.is_believable("US_DRIVER_LICENSE", 0.3, "v1")


def test_context_boosted_license_is_kept() -> None:
    # Presidio's context enhancer lifts confirmed matches to >= 0.4.
    assert scores.is_believable("US_DRIVER_LICENSE", 0.4, "C4455667")


def test_schemeless_url_noise_is_dropped() -> None:
    # "app.services.pr" matches the non-schema URL pattern at 0.5.
    assert not scores.is_believable("URL", 0.5, "app.services.pr")


def test_scheme_qualified_url_is_kept() -> None:
    assert scores.is_believable("URL", 0.6, "https://internal-payroll.com")


def test_www_prefixed_url_is_readmitted_despite_low_score() -> None:
    # People drop the scheme but keep "www." — no identifier starts
    # that way, so the raw text is the evidence the score lacked.
    assert scores.is_believable("URL", 0.5, "www.internal-payroll.com")
    assert scores.is_believable("URL", 0.5, '"www.internal-payroll.com')


def test_floors_can_be_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGISAI_PII_SCORE_FLOORS", "off")
    # Disabling only ever ADDS detections — the safe direction.
    assert scores.is_believable("US_DRIVER_LICENSE", 0.3, "v1")
    assert scores.is_believable("URL", 0.5, "app.services.pr")


def test_floor_for_reports_configured_minimums() -> None:
    assert scores.floor_for("US_DRIVER_LICENSE") == 0.4
    assert scores.floor_for("URL") == 0.6
    assert scores.floor_for("US_SSN") == 0.0
