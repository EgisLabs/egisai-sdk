"""Audit drops are accounted, ``diagnostics()`` exposes runtime health.

Before 0.11.0, the audit queue dropped the oldest event silently
when full — operators had no way to detect a multi-hour outage was
costing them audit completeness. The drop counter + ``diagnostics()``
gives them that signal without affecting the call hot path.
"""

from __future__ import annotations

from typing import Any

import egisai
from egisai import _logger


def _fill_queue(payload: dict[str, Any] | None = None) -> None:
    payload = payload or {"verdict": "allow"}
    for _ in range(_logger._QUEUE_MAX + 5):
        _logger.enqueue(dict(payload))


def test_dropped_total_starts_at_zero() -> None:
    _logger.reset_dropped_total()
    assert _logger.get_dropped_total() == 0


def test_overflow_increments_drop_counter() -> None:
    _logger.reset_dropped_total()
    _fill_queue()
    # We pushed _QUEUE_MAX + 5 events; the first _QUEUE_MAX fit, the
    # next 5 forced drops of the oldest event. Drop counter must
    # reflect those overflows.
    assert _logger.get_dropped_total() >= 5


def test_drop_warning_does_not_raise(caplog) -> None:
    """The drop warning is emitted via standard logging, not print —
    so consumers can suppress it via log levels."""
    import logging

    _logger.reset_dropped_total()
    with caplog.at_level(logging.WARNING, logger="egisai.logger"):
        _fill_queue()
    # First overflow should emit at least one warning record.
    relevant = [r for r in caplog.records if "audit queue full" in r.message]
    assert len(relevant) >= 1


def test_diagnostics_before_init_reports_uninitialised() -> None:
    """``egisai.diagnostics()`` is safe to call before ``init()``
    so a /healthz endpoint can use it without ordering constraints."""
    snap = egisai.diagnostics()
    assert snap["initialized"] is False
    assert snap["sdk_version"] == egisai.__version__


def test_diagnostics_after_init_exposes_full_health(fake_backend) -> None:
    fake_backend.set_rules(
        [
            {
                "id": "1",
                "name": "block-pii",
                "type": "pii_scan",
                "tenant": None,
                "config": {"action": "block"},
            }
        ],
        etag='"diag"',
    )
    egisai.init(
        api_key="egis_live_test",
        app="diag-app",
        env="dev",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
        semantic_on_outage="block",
    )

    snap = egisai.diagnostics()
    assert snap["initialized"] is True
    assert snap["app"] == "diag-app"
    assert snap["env"] == "dev"
    assert snap["on_block"] == "raise"
    assert snap["semantic_on_outage"] == "block"
    assert snap["policy_rule_count"] == 1
    # Both fields exist; values aren't asserted strictly because
    # background workers may have flushed by the time we check.
    assert "audit_queue_size" in snap
    assert "audit_dropped_total" in snap
