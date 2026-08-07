"""Routing-quality probe — SDK-side tests.

Layers under test:

1. ``_routing.set_quality_sample`` clamps / tolerates junk, and
   ``should_sample_quality`` is a hard off switch at 0.0, always-on at
   1.0.
2. ``_run_route_quality_probe`` (the daemon-thread body) redacts the
   answer, calls the backend, and enqueues a ``routing.quality`` event
   carrying the verdict + the original ``trace_id`` — and enqueues
   NOTHING when the backend declines or the answer is empty.
3. ``_maybe_probe_route_quality`` only fires on a sampled downgrade and
   passes the right correlation data through to the probe.
"""

from __future__ import annotations

from typing import Any

from egisai import _routing
from egisai._patches import _common


def _reset() -> None:
    _routing.reset()


# ── sample-rate hint ─────────────────────────────────────────────────


class TestSampleHint:
    def test_none_and_junk_disable(self) -> None:
        _reset()
        _routing.set_quality_sample(None)
        assert _routing.should_sample_quality() is False
        _routing.set_quality_sample("nonsense")  # type: ignore[arg-type]
        assert _routing.should_sample_quality() is False

    def test_zero_is_off(self) -> None:
        _reset()
        _routing.set_quality_sample(0.0)
        assert _routing.should_sample_quality() is False

    def test_one_is_always_on(self) -> None:
        _reset()
        _routing.set_quality_sample(1.0)
        assert _routing.should_sample_quality() is True

    def test_clamps_above_one(self) -> None:
        _reset()
        _routing.set_quality_sample(5.0)
        assert _routing.should_sample_quality() is True

    def test_reset_clears(self) -> None:
        _routing.set_quality_sample(1.0)
        _routing.reset()
        assert _routing.should_sample_quality() is False


# ── probe body (daemon-thread target) ────────────────────────────────


class TestRunProbe:
    def test_enqueues_verdict_with_trace_id(
        self, monkeypatch: Any
    ) -> None:
        captured: list[dict[str, Any]] = []
        monkeypatch.setattr(_common, "enqueue", captured.append)

        import egisai._backend as backend

        def _fake_route_quality(**kwargs: Any) -> dict[str, Any]:
            # The answer must have been redacted before it reached here.
            assert kwargs["answer_preview"]
            return {"quality": "degraded", "score": 0.62}

        monkeypatch.setattr(backend, "route_quality", _fake_route_quality)

        _common._run_route_quality_probe(
            trace_id="trace-123",
            requested_model="gpt-4o",
            served_model="gpt-4o-mini",
            requested_tier=4,
            served_tier=2,
            prompt_preview="summarize the doc",
            answer_text="a rather thin summary",
        )

        assert len(captured) == 1
        ev = captured[0]
        assert ev["kind"] == "routing.quality"
        assert ev["trace_id"] == "trace-123"
        assert ev["routing_quality"] == "degraded"
        assert ev["routing_quality_score"] == 0.62

    def test_backend_declines_enqueues_nothing(
        self, monkeypatch: Any
    ) -> None:
        captured: list[dict[str, Any]] = []
        monkeypatch.setattr(_common, "enqueue", captured.append)
        import egisai._backend as backend

        monkeypatch.setattr(backend, "route_quality", lambda **k: None)
        _common._run_route_quality_probe(
            trace_id="t",
            requested_model="a",
            served_model="b",
            requested_tier=None,
            served_tier=None,
            prompt_preview="p",
            answer_text="an answer",
        )
        assert captured == []

    def test_empty_answer_enqueues_nothing(self, monkeypatch: Any) -> None:
        captured: list[dict[str, Any]] = []
        monkeypatch.setattr(_common, "enqueue", captured.append)
        import egisai._backend as backend

        called = {"n": 0}

        def _rq(**k: Any) -> dict[str, Any] | None:
            called["n"] += 1
            return None

        monkeypatch.setattr(backend, "route_quality", _rq)
        _common._run_route_quality_probe(
            trace_id="t",
            requested_model="a",
            served_model="b",
            requested_tier=None,
            served_tier=None,
            prompt_preview="p",
            answer_text="    ",
        )
        assert captured == []
        assert called["n"] == 0  # never even asked the backend

    def test_probe_never_raises(self, monkeypatch: Any) -> None:
        import egisai._backend as backend

        monkeypatch.setattr(
            backend,
            "route_quality",
            lambda **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        # Must swallow — the customer's call already returned.
        _common._run_route_quality_probe(
            trace_id="t",
            requested_model="a",
            served_model="b",
            requested_tier=None,
            served_tier=None,
            prompt_preview="p",
            answer_text="an answer",
        )


# ── scheduler gate ───────────────────────────────────────────────────


class _FakeThread:
    """Runs the target synchronously so tests stay deterministic."""

    def __init__(self, *, target: Any, kwargs: Any, **_: Any) -> None:
        self._target = target
        self._kwargs = kwargs

    def start(self) -> None:
        self._target(**self._kwargs)


def _extract(_response: Any, _payload: Any) -> tuple[str, list, list, list]:
    return ("the served answer", [], [], [])


class TestScheduler:
    def _run(self, monkeypatch: Any, *, direction: str, sample: float) -> list:
        captured: list[dict[str, Any]] = []
        monkeypatch.setattr(_common, "enqueue", captured.append)
        monkeypatch.setattr(_common.threading, "Thread", _FakeThread)
        import egisai._backend as backend

        monkeypatch.setattr(
            backend,
            "route_quality",
            lambda **k: {"quality": "adequate", "score": 0.9},
        )
        _reset()
        _routing.set_quality_sample(sample)

        ev = {"trace_id": "abc", "prompt_preview": "p"}
        state = {
            "decision": {
                "model": "gpt-4o-mini",
                "direction": direction,
                "factors": {"requested_tier": 4, "served_tier": 2},
            },
            "requested_model": "gpt-4o",
            "applied": True,
        }
        _common._maybe_probe_route_quality(
            ev=ev,
            route_state=state,
            response=object(),
            payload={},
            extract_output_signals=_extract,
        )
        return captured

    def test_sampled_downgrade_probes(self, monkeypatch: Any) -> None:
        captured = self._run(monkeypatch, direction="downgrade", sample=1.0)
        assert len(captured) == 1
        assert captured[0]["kind"] == "routing.quality"
        assert captured[0]["trace_id"] == "abc"

    def test_upgrade_never_probes(self, monkeypatch: Any) -> None:
        captured = self._run(monkeypatch, direction="upgrade", sample=1.0)
        assert captured == []

    def test_unsampled_never_probes(self, monkeypatch: Any) -> None:
        captured = self._run(monkeypatch, direction="downgrade", sample=0.0)
        assert captured == []
