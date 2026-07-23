"""Run / step lifecycle tests for 0.18.0.

These exercise the core invariants of the new audit model:

  * One framework entry point => ONE Run (not N audit rows).
  * Identity is LOCKED at run open — inner LLM calls cannot drift
    to a different agent even if their per-call payload would have
    fingerprinted differently (the "4 agents from 1 task" bug).
  * Tokens / latency / cost / verdict aggregate across all steps.
  * Wire format ships ``run.start`` + ``run.step`` (one per step) +
    ``run.end`` so the dashboard can render live timelines.
  * Sub-agent (nested framework entry) opens a child Run with
    ``parent_run_id`` linkage.
  * Streaming runs close cleanly when the iterator exhausts AND
    when the caller breaks out early.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest

from egisai import _logger, _run
from egisai._auto_agent import IdentityRecord

_FB: Any = None  # Module-level pointer used by ``_drain_queue``.


@pytest.fixture(autouse=True)
def _init_sdk(fake_backend: Any) -> Iterator[Any]:
    """Initialise the SDK so the event builders can read config.

    The ``fake_backend`` fixture wires our in-memory transport into
    ``egisai._backend.get_client`` so handshake / ensure / events all
    land in a list we can inspect — same pattern as
    ``test_claude_agent_sdk_governance``.
    """
    global _FB
    import egisai
    egisai.init(
        api_key="egis_live_test",
        app="run-lifecycle-test",
        env="t",
        on_block="raise",
        enable_sse=False,
    )
    _FB = fake_backend
    yield fake_backend
    _FB = None


def _make_record(name: str = "Test Agent") -> IdentityRecord:
    """Build an IdentityRecord for tests without going through the resolver."""
    return IdentityRecord(
        agent_id=f"id-{name.lower().replace(' ', '-')}",
        display_name=name,
        identity_key=f"test:{name}",
        identity_hash="a" * 64,
        source="framework:test",
        push_to_stack=True,
    )


def _drain_queue(fb: Any = None) -> list[dict[str, Any]]:
    """Return every event the SDK has emitted in this test.

    First flushes the SDK (stops the worker, drains everything to the
    fake backend over HTTP), then reads the events list.
    """
    from egisai import shutdown
    shutdown()
    fb = fb or _FB
    if fb is None:
        # Fallback for direct invocations w/o the fixture.
        out: list[dict[str, Any]] = []
        while not _logger._q.empty():
            try:
                out.append(_logger._q.get_nowait())
            except Exception:  # noqa: BLE001
                break
        return out
    events = list(fb.events_received)
    fb.events_received.clear()
    return events


# ── Basic lifecycle ─────────────────────────────────────────────────


def test_open_close_emits_start_and_end() -> None:
    """A bare open/close cycle ships exactly run.start + run.end."""
    record = _make_record()
    _run.open_run(framework="test", identity=record, prompt_text="hi")
    _run.close_run()

    events = _drain_queue()
    kinds = [e.get("kind") for e in events]
    assert kinds == ["run.start", "run.end"], kinds
    start, end = events
    assert start["run_id"] == end["run_id"]
    assert start["framework"] == "test"
    assert start["agent_id"] == record.agent_id
    assert end["agent_id"] == record.agent_id
    assert end["step_count"] == 0
    assert end["verdict"] == "allow"


def test_append_step_emits_run_step_envelope() -> None:
    """Steps ride a ``kind=run.step`` envelope with the legacy event nested."""
    record = _make_record()
    _run.open_run(framework="test", identity=record)
    step = _run.append_step(
        event={
            "source": "openai",
            "target": "openai.chat.completions.create",
            "model": "gpt-4o",
            "verdict": "allow",
            "tokens_in": 100,
            "tokens_out": 50,
            "prompt_preview": "first",
            "response_preview": "answer 1",
        },
        kind="model_call",
    )
    assert step is not None
    _run.close_run()

    events = _drain_queue()
    kinds = [e.get("kind") for e in events]
    assert kinds == ["run.start", "run.step", "run.end"], kinds
    step_ev = events[1]
    assert step_ev["step_kind"] == "model_call"
    assert step_ev["seq"] == 0
    assert step_ev["model"] == "gpt-4o"
    assert step_ev["tokens_in"] == 100
    end_ev = events[2]
    assert end_ev["step_count"] == 1
    assert end_ev["tokens_in"] == 100
    assert end_ev["tokens_out"] == 50
    assert end_ev["model"] == "gpt-4o"


def test_append_step_returns_none_when_no_run() -> None:
    """Falling back to legacy ``enqueue`` is the correct contract."""
    out = _run.append_step(event={"verdict": "allow"}, kind="model_call")
    assert out is None
    events = _drain_queue()
    assert events == [], "append_step should not enqueue when no run is open"


# ── Aggregates (tokens / latency / cost / verdict) ─────────────────


def test_aggregates_sum_across_steps() -> None:
    """Run.end carries the SUM of all step tokens / cost, MAX latency."""
    _run.open_run(framework="test", identity=_make_record())
    for i in range(4):
        _run.append_step(
            event={
                "source": "openai",
                "model": "gpt-4o",
                "verdict": "allow",
                "tokens_in": 100 * (i + 1),
                "tokens_out": 50,
                "cost_usd": 0.01,
                "latency_ms": 100,
            },
            kind="model_call",
        )
    _run.close_run()

    events = _drain_queue()
    end_ev = events[-1]
    assert end_ev["tokens_in"] == 100 + 200 + 300 + 400
    assert end_ev["tokens_out"] == 4 * 50
    assert end_ev["cost_usd"] == pytest.approx(0.04)
    assert end_ev["step_count"] == 4


def test_run_end_latency_is_sum_of_step_latencies() -> None:
    """``run.end.latency_ms`` = SUM of per-step model latencies.

    Regression — pre-0.41.1 the run.end event shipped the run's whole
    wall clock (open→close), which the backend trusted over its own
    step sums. Since the wall clock includes policy-evaluation time,
    a run blocked at the input policy showed nonzero "Model" latency
    on the dashboard for a model that was never called.
    """
    _run.open_run(framework="test", identity=_make_record())
    for _ in range(4):
        _run.append_step(
            event={
                "model": "gpt-4o",
                "verdict": "allow",
                "latency_ms": 100,
            },
            kind="model_call",
        )
    _run.close_run()

    events = _drain_queue()
    step_events = [e for e in events if e.get("kind") == "run.step"]
    assert [e["latency_ms"] for e in step_events] == [100, 100, 100, 100]
    assert events[-1]["latency_ms"] == 400


def test_step_latency_prestamped_zero_survives_emission() -> None:
    """A patch-stamped ``latency_ms`` (here the explicit 0 of an
    input-side block) MUST survive the run.step emission untouched.

    Regression — pre-0.41.1 ``_safe_emit_run_step`` recomputed
    ``latency_ms`` from the step's timestamps, which span the whole
    gate (input policy + model + output policy). For blocked calls
    that stamped 0 (the model was never contacted) the recomputation
    replaced the 0 with the policy-evaluation wall clock, so the
    dashboard showed phantom "model latency" on blocked requests.
    """
    import time

    _run.open_run(framework="test", identity=_make_record())
    # ``started_at`` far in the past simulates a slow policy phase
    # (e.g. a semantic_guard judge round-trip) before the block.
    _run.append_step(
        event={
            "model": "gpt-4o",
            "verdict": "block",
            "latency_ms": 0,
            "policy_latency_ms": 5000,
        },
        kind="model_call",
        started_at=time.monotonic() - 5.0,
    )
    _run.close_run(error="input policy block")

    events = _drain_queue()
    step_ev = next(e for e in events if e.get("kind") == "run.step")
    assert step_ev["latency_ms"] == 0, (
        "blocked step stamped latency_ms=0; emission must not clobber "
        f"it with gate wall clock (got {step_ev['latency_ms']})"
    )
    end_ev = events[-1]
    assert end_ev["kind"] == "run.end"
    assert end_ev["latency_ms"] == 0, (
        "run.end must sum step latencies (0), not ship the run's "
        f"wall clock (got {end_ev['latency_ms']})"
    )


def test_step_latency_falls_back_to_timestamps_when_unstamped() -> None:
    """Events that never stamped ``latency_ms`` keep the legacy
    behavior: latency derives from the step's own timestamps."""
    import time

    _run.open_run(framework="test", identity=_make_record())
    _run.append_step(
        event={"model": "gpt-4o", "verdict": "allow"},
        kind="model_call",
        started_at=time.monotonic() - 0.25,
    )
    _run.close_run()

    events = _drain_queue()
    step_ev = next(e for e in events if e.get("kind") == "run.step")
    # ~250 ms span; generous bounds for CI jitter.
    assert 200 <= step_ev["latency_ms"] < 5000
    assert events[-1]["latency_ms"] == step_ev["latency_ms"]


def test_finalize_in_place_respects_stamped_latency() -> None:
    """The claude_agent_sdk path: placeholder seq 0 ``model_call`` is
    finalized in place with the terminal event — the stamped
    ``latency_ms`` (model turn only) wins over the placeholder's
    timestamp span."""
    import time

    _run.open_run(framework="test", identity=_make_record())
    _run.append_initial_model_call_step(
        event={"model": "claude-3-5-sonnet", "verdict": "allow"},
        started_at=time.monotonic() - 5.0,
    )
    _run.finalize_or_append_model_call_step(
        event={
            "model": "claude-3-5-sonnet",
            "verdict": "allow",
            "latency_ms": 123,
            "tokens_in": 10,
            "tokens_out": 20,
        },
    )
    _run.close_run()

    events = _drain_queue()
    step_events = [e for e in events if e.get("kind") == "run.step"]
    # Placeholder emission + in-place finalization, same seq.
    assert {e["seq"] for e in step_events} == {0}
    assert step_events[-1]["latency_ms"] == 123
    assert events[-1]["latency_ms"] == 123


def test_worst_verdict_propagates() -> None:
    """A single ``block`` step makes the run's verdict ``block``."""
    _run.open_run(framework="test", identity=_make_record())
    _run.append_step(event={"verdict": "allow", "model": "x"}, kind="model_call")
    _run.append_step(event={"verdict": "sanitize", "model": "x"}, kind="model_call")
    _run.append_step(event={"verdict": "block", "model": "x"}, kind="model_call")
    _run.append_step(event={"verdict": "allow", "model": "x"}, kind="model_call")
    _run.close_run()

    end_ev = _drain_queue()[-1]
    assert end_ev["verdict"] == "block"


# ── Identity lock — the "4 agents from 1 task" bug ─────────────────


def test_identity_locked_at_run_open() -> None:
    """A step's agent_id always matches the run's agent_id, even when
    the inner event arrived without one (the legacy code path used to
    fall through to Tier 5 and register a fresh agent per turn)."""
    record = _make_record("Locked Agent")
    _run.open_run(framework="test", identity=record)
    _run.append_step(
        event={"source": "openai", "model": "gpt-4o", "verdict": "allow"},
        kind="model_call",
    )
    _run.close_run()

    events = _drain_queue()
    step_ev = next(e for e in events if e.get("kind") == "run.step")
    assert step_ev["agent_id"] == "id-locked-agent"
    assert step_ev["app"] == "Locked Agent"


def test_multiple_steps_share_one_agent_id() -> None:
    """The four-agents-from-one-task fix: every step inherits the
    run's locked agent_id."""
    record = _make_record("Customer Support Agent")
    _run.open_run(framework="openai_agents", identity=record)
    for i in range(5):
        _run.append_step(
            event={
                # Simulate inner LLM calls with NO agent_id of their
                # own — the old gate's _attribute_event would have
                # let Tier 5 fingerprint a different name per turn.
                "model": "gpt-4o",
                "verdict": "allow",
                "tokens_in": 100,
                "tokens_out": 50,
                "prompt_preview": f"turn-{i} different system prompt",
            },
            kind="model_call",
        )
    _run.close_run()

    step_events = [e for e in _drain_queue() if e.get("kind") == "run.step"]
    assert len(step_events) == 5
    agent_ids = {e["agent_id"] for e in step_events}
    assert agent_ids == {"id-customer-support-agent"}, (
        f"all 5 steps must share one agent_id, got: {agent_ids}"
    )


# ── Trace_id is one-per-Run (was: one-per-LLM-call) ────────────────


def test_trace_id_constant_across_steps() -> None:
    """Steps in one Run share the Run's trace_id, not per-step trace_ids."""
    _run.open_run(framework="test", identity=_make_record())
    for _ in range(3):
        _run.append_step(
            event={"model": "gpt-4o", "verdict": "allow"},
            kind="model_call",
        )
    _run.close_run()

    events = _drain_queue()
    trace_ids = {e["trace_id"] for e in events}
    assert len(trace_ids) == 1, (
        "all events for a Run must share one trace_id"
    )


# ── Nested run = sub-agent / handoff ───────────────────────────────


def test_nested_open_links_via_parent_run_id() -> None:
    """A second open_run inside an open run opens a CHILD."""
    parent = _run.open_run(framework="parent", identity=_make_record("Parent"))
    child = _run.open_run(framework="child", identity=_make_record("Child"))

    assert parent.parent_run_id is None
    assert child.parent_run_id == parent.run_id
    assert _run.current_run() is child

    _run.close_run()  # closes child
    assert _run.current_run() is None or _run.current_run().run_id != child.run_id
    # NOTE: in the current minimal implementation, close_run sets the
    # ContextVar to None rather than restoring the parent. That's a
    # known limitation we'll address with a stack-based current_run
    # if a framework patch actually does parent <-> child <-> parent
    # transitions. For v1 the only nested case is "framework re-entry"
    # which closes both with the outer wrap's finally.


# ── Close is idempotent / never raises ─────────────────────────────


def test_close_run_idempotent() -> None:
    _run.open_run(framework="test", identity=_make_record())
    _run.close_run()
    # Second close is a no-op, must not raise.
    _run.close_run()
    events = _drain_queue()
    # Exactly one start + one end.
    kinds = [e.get("kind") for e in events]
    assert kinds.count("run.start") == 1
    assert kinds.count("run.end") == 1


def test_close_run_with_no_open_run_is_safe() -> None:
    # Should never raise; should not emit anything.
    _run.close_run()
    assert _drain_queue() == []


# ── Framework wrap integration ─────────────────────────────────────


def test_sync_entrypoint_wrap_opens_and_closes_a_run() -> None:
    """``wrap_sync_entrypoint`` opens a Run before the inner call and
    closes it after, even when the inner call raises."""
    from egisai._patches._framework import wrap_sync_entrypoint

    record = _make_record("Sync Agent")

    def derive(_self: Any, *a: Any, **kw: Any) -> IdentityRecord:
        return record

    seen_runs: list[Any] = []

    def inner(_self: Any) -> str:
        seen_runs.append(_run.current_run())
        return "done"

    wrapped = wrap_sync_entrypoint(inner, derive)
    out = wrapped(object())
    assert out == "done"
    assert seen_runs[0] is not None
    assert seen_runs[0].agent_id == record.agent_id

    events = _drain_queue()
    kinds = [e.get("kind") for e in events]
    assert kinds == ["run.start", "run.end"]


async def _await_async_wrap(inner_recorder: list[Any]) -> str:
    from egisai._patches._framework import wrap_async_entrypoint

    record = _make_record("Async Agent")

    def derive(_self: Any, *a: Any, **kw: Any) -> IdentityRecord:
        return record

    async def inner(_self: Any) -> str:
        inner_recorder.append(_run.current_run())
        return "ok"

    wrapped = wrap_async_entrypoint(inner, derive)
    return await wrapped(object())


def test_async_entrypoint_wrap_opens_and_closes_a_run() -> None:
    seen: list[Any] = []
    out = asyncio.run(_await_async_wrap(seen))
    assert out == "ok"
    assert seen[0] is not None

    kinds = [e.get("kind") for e in _drain_queue()]
    assert kinds == ["run.start", "run.end"]


# ── Streaming (async-iter) closes on iterator exhaustion ───────────


async def _consume_async_iter_wrap() -> int:
    from egisai._patches._framework import wrap_async_iter_entrypoint

    record = _make_record("Stream Agent")

    def derive(_self: Any, *a: Any, **kw: Any) -> IdentityRecord:
        return record

    async def inner(_self: Any):  # type: ignore[no-untyped-def]
        for i in range(3):
            yield i

    wrapped = wrap_async_iter_entrypoint(inner, derive)
    count = 0
    async for _ in wrapped(object()):
        count += 1
    return count


def test_async_iter_wrap_closes_on_exhaustion() -> None:
    n = asyncio.run(_consume_async_iter_wrap())
    assert n == 3

    kinds = [e.get("kind") for e in _drain_queue()]
    assert kinds == ["run.start", "run.end"]


async def _break_out_of_async_iter_wrap() -> None:
    """Caller breaks out of the loop early — Run still closes on
    generator cleanup (we explicitly aclose in the wrap)."""
    from egisai._patches._framework import wrap_async_iter_entrypoint

    record = _make_record("Break Agent")

    def derive(_self: Any, *a: Any, **kw: Any) -> IdentityRecord:
        return record

    async def inner(_self: Any):  # type: ignore[no-untyped-def]
        for i in range(100):
            yield i

    wrapped = wrap_async_iter_entrypoint(inner, derive)
    async for i in wrapped(object()):
        if i >= 2:
            break


def test_async_iter_wrap_closes_on_early_break() -> None:
    asyncio.run(_break_out_of_async_iter_wrap())

    kinds = [e.get("kind") for e in _drain_queue()]
    assert "run.start" in kinds
    assert "run.end" in kinds


# ── Polymorphic wrap handles all return shapes ──────────────────────


async def _exercise_polymorphic_coro() -> str:
    from egisai._patches._framework import wrap_polymorphic_entrypoint

    record = _make_record("Poly Coro")

    def derive(_self: Any, *a: Any, **kw: Any) -> IdentityRecord:
        return record

    def inner(_self: Any) -> Any:
        async def _co() -> str:
            assert _run.current_run() is not None
            return "coro-done"
        return _co()

    wrapped = wrap_polymorphic_entrypoint(inner, derive)
    return await wrapped(object())


def test_polymorphic_wrap_coroutine() -> None:
    out = asyncio.run(_exercise_polymorphic_coro())
    assert out == "coro-done"
    kinds = [e.get("kind") for e in _drain_queue()]
    assert kinds == ["run.start", "run.end"]


def test_polymorphic_wrap_plain_value() -> None:
    from egisai._patches._framework import wrap_polymorphic_entrypoint

    record = _make_record("Poly Sync")

    def derive(_self: Any, *a: Any, **kw: Any) -> IdentityRecord:
        return record

    def inner(_self: Any) -> str:
        assert _run.current_run() is not None
        return "value"

    wrapped = wrap_polymorphic_entrypoint(inner, derive)
    assert wrapped(object()) == "value"

    kinds = [e.get("kind") for e in _drain_queue()]
    assert kinds == ["run.start", "run.end"]


# ── Failure paths still close the run cleanly ──────────────────────


def test_sync_wrap_closes_run_on_exception() -> None:
    from egisai._patches._framework import wrap_sync_entrypoint

    def derive(_self: Any, *a: Any, **kw: Any) -> IdentityRecord:
        return _make_record("Crash")

    def inner(_self: Any) -> None:
        raise RuntimeError("boom")

    wrapped = wrap_sync_entrypoint(inner, derive)
    with pytest.raises(RuntimeError, match="boom"):
        wrapped(object())

    kinds = [e.get("kind") for e in _drain_queue()]
    assert kinds == ["run.start", "run.end"]
    # ``run.end`` was the second event; structurally sanity-check
    # that the wrap captured the exception path as a closed run
    # rather than leaving the run open.
    assert len(kinds) == 2
