"""Claude Agent SDK governance must never run on the caller's loop.

Every other async patch routes policy evaluation through
``asyncio.to_thread``; the Claude Agent SDK paths did not. That matters
more here than anywhere else, because this patch governs a long-lived
streaming agent loop: the input phase runs Presidio/spaCy NER and can
make a blocking ``semantic_guard`` round-trip, so a single slow judge
stalled the customer's entire async application — every other coroutine
they had in flight, not just the turn being governed.

These tests assert the property directly: while governance is running,
the loop still gets scheduled. They record the thread the work lands on
rather than timing anything, so they don't flake on a loaded CI box.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

import egisai._patches.claude_agent_sdk as patch


class _ThreadRecorder:
    """Stands in for a governance function; remembers where it ran."""

    def __init__(self, result: Any = None) -> None:
        self.result = result
        self.thread: str | None = None
        self.calls = 0

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        self.thread = threading.current_thread().name
        self.calls += 1
        return self.result


async def _loop_thread_name() -> str:
    return threading.current_thread().name


@pytest.mark.asyncio
async def test_input_phase_runs_off_the_loop_for_client_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _ThreadRecorder(result=_allow())
    monkeypatch.setattr(patch, "_run_input_phase", rec)

    await _drive_client_query(monkeypatch)

    assert rec.calls == 1, "input phase did not run"
    assert rec.thread != await _loop_thread_name()


@pytest.mark.asyncio
async def test_input_phase_runs_off_the_loop_for_module_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _ThreadRecorder(result=_allow())
    monkeypatch.setattr(patch, "_run_input_phase", rec)

    await _drive_module_query(monkeypatch)

    assert rec.calls == 1, "input phase did not run"
    assert rec.thread != await _loop_thread_name()


@pytest.mark.asyncio
async def test_a_slow_input_phase_does_not_stall_the_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The behaviour the operator actually feels.

    A blocking judge holds a worker thread for as long as it likes; the
    customer's other coroutines must keep making progress meanwhile.
    """
    release = threading.Event()
    entered = threading.Event()

    def _slow(*_a: Any, **_k: Any) -> Any:
        entered.set()
        release.wait(timeout=10)
        return _allow()

    monkeypatch.setattr(patch, "_run_input_phase", _slow)

    ticks = 0

    async def heartbeat() -> None:
        nonlocal ticks
        while not release.is_set():
            await asyncio.sleep(0.005)
            ticks += 1

    beat = asyncio.create_task(heartbeat())
    governed = asyncio.create_task(_drive_client_query(monkeypatch))

    await asyncio.get_running_loop().run_in_executor(None, entered.wait, 10)
    await asyncio.sleep(0.1)  # loop must keep turning while it blocks
    mid_flight = ticks

    release.set()
    await governed
    beat.cancel()

    assert mid_flight > 0, "event loop was frozen while governance ran"


# ── Harness ─────────────────────────────────────────────────────────
#
# The patch wraps real Claude Agent SDK entry points. Rather than
# install that SDK, drive the wrapper directly with a stand-in original
# and stub out everything the gate touches besides the phase under test.


def _allow() -> Any:
    from egisai.policy.engine import PolicyDecision

    return PolicyDecision(
        verdict="allow",
        reason_code=None,
        message=None,
        matched_policy=None,
    )


def _neutralise(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "_report_declared_access",
        "_safe_enqueue",
        "reset_policy_usage",
        "set_policy_checked",
        "reset_init_latency",
    ):
        if hasattr(patch, name):
            monkeypatch.setattr(patch, name, lambda *a, **k: None)
    monkeypatch.setattr(patch, "_derive", lambda *a, **k: None)
    monkeypatch.setattr(patch, "_build_input_event", lambda **k: {})
    monkeypatch.setattr(patch, "get_policy_usage", lambda: (0, 0))


async def _drive_client_query(monkeypatch: pytest.MonkeyPatch) -> None:
    _neutralise(monkeypatch)

    async def orig(_self: Any, _prompt: Any, *_a: Any, **_k: Any) -> None:
        return None

    wrapped = patch._wrap_client_query(orig)

    class _Client:
        pass

    await wrapped(_Client(), "hello")


async def _drive_module_query(monkeypatch: pytest.MonkeyPatch) -> None:
    _neutralise(monkeypatch)

    async def orig(_prompt: Any, *_a: Any, **_k: Any) -> Any:
        return
        yield  # pragma: no cover — makes ``orig`` an async generator

    wrapped = patch._wrap_module_query(orig)

    async for _ in wrapped("hello"):
        pass


def test_hook_gated_steps_skip_the_thread_hop() -> None:
    """The common path must stay free.

    When the PreToolUse hook already emitted a step, dispatch is a
    no-op, so the async callers check this first instead of paying a
    thread hop per tool on every message.
    """
    assert patch._tool_step_already_shipped({"abc": "allow"}, "abc") is True
    assert patch._tool_step_already_shipped({"abc": "allow"}, "xyz") is False
    assert patch._tool_step_already_shipped(None, "abc") is False
    assert patch._tool_step_already_shipped({"abc": "allow"}, None) is False
