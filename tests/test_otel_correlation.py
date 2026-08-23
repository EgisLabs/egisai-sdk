"""Ambient OpenTelemetry ids ride along with every governed event.

Why this exists
---------------
A customer who runs OpenTelemetry debugs in their traces. Without the
span ids on the audit row, lining an Egis record up against their own
span means comparing timestamps — which stops working the moment two
agents run concurrently, i.e. exactly when someone is trying to work
out what happened.

What's pinned
-------------
* The ids are read from the *active* span and shipped as
  ``otel_trace_id`` / ``otel_span_id``, formatted as W3C lowercase hex
  (32 and 16 chars).
* They never replace the SDK's own ``trace_id``. That one groups the
  steps of a single run; an OTel trace spans several agents and
  services, so merging them would break run grouping in the dashboard.
* Absence is the normal case, not an error: no OTel installed, no
  active span, or an invalid span context each yield ``None`` and a
  perfectly good event.
* Nothing here may raise. This runs on the hot path of every governed
  call, so a broken OTel setup must cost the customer nothing worse
  than an empty column.

``opentelemetry`` is deliberately not a test dependency — it isn't a
runtime one either, and pinning behavior against a stub is what keeps
it that way. The stubs below implement only the two methods the SDK
touches.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import egisai
import egisai._context as ctx_mod
from egisai._events import build_event


@pytest.fixture
def initialised(fake_backend: Any) -> None:
    """``build_event`` reads the live config, so the SDK must be up."""
    egisai.init(
        api_key="egis_live_x",
        app="otel-test",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )


class _SpanContext:
    def __init__(self, trace_id: int, span_id: int, valid: bool = True) -> None:
        self.trace_id = trace_id
        self.span_id = span_id
        self.is_valid = valid


class _Span:
    def __init__(self, sc: Any) -> None:
        self._sc = sc

    def get_span_context(self) -> Any:
        return self._sc


def _install_otel(monkeypatch: pytest.MonkeyPatch, span: Any) -> None:
    """Put a minimal ``opentelemetry.trace`` in front of the SDK."""
    trace_mod = types.ModuleType("opentelemetry.trace")
    trace_mod.get_current_span = lambda: span  # type: ignore[attr-defined]
    pkg = types.ModuleType("opentelemetry")
    pkg.trace = trace_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "opentelemetry", pkg)
    monkeypatch.setitem(sys.modules, "opentelemetry.trace", trace_mod)
    _reset_probe(monkeypatch)


def _reset_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the module-level import cache between cases.

    The cache exists so the *miss* path — an ImportError raised and
    caught on every model call — is paid once. That makes it state,
    and state has to be reset per test.
    """
    monkeypatch.setattr(ctx_mod, "_otel_checked", False, raising=False)
    monkeypatch.setattr(ctx_mod, "_otel_trace_mod", None, raising=False)


def test_the_active_span_is_what_lands_on_the_event(
    monkeypatch: pytest.MonkeyPatch, initialised: None
) -> None:
    _install_otel(
        monkeypatch, _Span(_SpanContext(trace_id=0x4BF92F, span_id=0x51F))
    )

    ev = build_event(source="openai", target="chat", payload={})

    # W3C wants fixed-width lowercase hex, zero-padded.
    assert ev["otel_trace_id"] == "000000000000000000000000004bf92f"
    assert ev["otel_span_id"] == "000000000000051f"


def test_our_own_trace_id_is_untouched(
    monkeypatch: pytest.MonkeyPatch, initialised: None
) -> None:
    """The two groupings must stay separate.

    An OTel trace covers several agents and services. If it overwrote
    ``trace_id``, unrelated agent runs would collapse into one row
    group on the Requests page.
    """
    _install_otel(
        monkeypatch, _Span(_SpanContext(trace_id=0xABC, span_id=0xDEF))
    )

    ev = build_event(source="openai", target="chat", payload={})

    assert ev["trace_id"] != ev["otel_trace_id"]
    assert len(ev["trace_id"]) == 32  # our uuid4 hex, not a W3C id


def test_no_opentelemetry_installed_is_the_normal_case(
    monkeypatch: pytest.MonkeyPatch, initialised: None
) -> None:
    monkeypatch.setitem(sys.modules, "opentelemetry", None)
    _reset_probe(monkeypatch)

    ev = build_event(source="openai", target="chat", payload={})

    assert ev["otel_trace_id"] is None
    assert ev["otel_span_id"] is None


def test_an_invalid_span_context_is_not_a_correlation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``INVALID_SPAN`` carries all-zero ids.

    Recording those would claim a correlation that does not exist —
    someone would paste it into Jaeger and find nothing.
    """
    _install_otel(
        monkeypatch, _Span(_SpanContext(trace_id=0, span_id=0, valid=False))
    )

    assert ctx_mod.ambient_otel_ids() == (None, None)


def test_no_active_span_is_not_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_otel(monkeypatch, None)

    assert ctx_mod.ambient_otel_ids() == (None, None)


def test_a_broken_otel_never_breaks_the_call(
    monkeypatch: pytest.MonkeyPatch, initialised: None
) -> None:
    """Fail open. An OTel install that throws is their bug, not an
    outage in the customer's product."""

    class _Exploding:
        def get_span_context(self) -> Any:
            raise RuntimeError("provider misconfigured")

    _install_otel(monkeypatch, _Exploding())

    assert ctx_mod.ambient_otel_ids() == (None, None)
    ev = build_event(source="openai", target="chat", payload={})
    assert ev["otel_trace_id"] is None
