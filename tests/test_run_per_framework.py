"""Per-framework: one entry-point call => one Run with N steps.

The user-reported "4 agents, 5 requests" bug was specifically about
OpenAI Agents (``Runner.run`` with 4 tool calls firing 5 inner LLM
calls). This file exercises that exact pattern across a representative
sample of the 14 frameworks we patch, using realistic stubs that mirror
each upstream's calling convention.

For each framework:
  * Patch a stub entry point so ``apply()`` succeeds.
  * Drive the entry point in a way that internally fires multiple
    nested gate_call() invocations.
  * Assert that the audit stream contains EXACTLY ONE
    ``run.start`` + N ``run.step`` + ONE ``run.end`` envelope,
    every step inheriting the run's locked ``agent_id``.

This is the smoke test that demonstrates the architectural fix end-
to-end inside the SDK; the e2e harnesses (``agents-test/*.py``) do the
same against the real backend later.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from typing import Any

import pytest

from egisai import _logger
from egisai._auto_agent import IdentityRecord

_FB: Any = None


@pytest.fixture(autouse=True)
def _init_sdk(fake_backend: Any) -> Iterator[Any]:
    global _FB
    import egisai
    egisai.init(
        api_key="egis_live_test",
        app="framework-run-test",
        env="t",
        on_block="raise",
        enable_sse=False,
    )
    _FB = fake_backend
    yield fake_backend
    _FB = None


def _events() -> list[dict[str, Any]]:
    from egisai import shutdown
    shutdown()
    fb = _FB
    if fb is None:
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


def _make_record(name: str) -> IdentityRecord:
    return IdentityRecord(
        agent_id=f"id-{name.lower().replace(' ', '-')}",
        display_name=name,
        identity_key=f"test:{name}",
        identity_hash="a" * 64,
        source="framework:test",
        push_to_stack=True,
    )


# ── Shared assertion helpers ────────────────────────────────────────


def assert_one_run_n_steps(events: list[dict[str, Any]], expected_steps: int) -> None:
    """Assert the wire stream describes exactly ONE run with N steps."""
    starts = [e for e in events if e.get("kind") == "run.start"]
    steps = [e for e in events if e.get("kind") == "run.step"]
    ends = [e for e in events if e.get("kind") == "run.end"]
    assert len(starts) == 1, (
        f"expected exactly 1 run.start, got {len(starts)}; events={events!r}"
    )
    assert len(ends) == 1, (
        f"expected exactly 1 run.end, got {len(ends)}; events={events!r}"
    )
    assert len(steps) == expected_steps, (
        f"expected {expected_steps} run.steps, got {len(steps)};\n"
        f"steps={steps!r}"
    )
    run_id = starts[0]["run_id"]
    for s in steps:
        assert s["run_id"] == run_id, (
            f"step {s.get('seq')} belongs to a different run: {s.get('run_id')} != {run_id}"
        )
    assert ends[0]["run_id"] == run_id


def assert_all_steps_one_agent(
    events: list[dict[str, Any]], expected_agent_id: str
) -> None:
    """Every step's ``agent_id`` matches the locked run owner."""
    steps = [e for e in events if e.get("kind") == "run.step"]
    assert steps, "expected at least one run.step"
    agent_ids = {s.get("agent_id") for s in steps}
    assert agent_ids == {expected_agent_id}, (
        f"identity leak across steps: {agent_ids} (expected just {expected_agent_id!r})"
    )


# ── Framework: OpenAI Agents (sync wrap) ───────────────────────────


def test_openai_agents_like_sync_run_one_agent_many_steps() -> None:
    """Simulate ``Runner.run`` (sync wrap) with N inner LLM calls."""
    from egisai._patches._framework import wrap_async_entrypoint
    from egisai._run import append_step

    record = _make_record("Customer Support Agent")

    def derive(*a: Any, **kw: Any) -> IdentityRecord:
        return record

    async def runner_run_body(_self: Any, _agent: Any, _task: str) -> str:
        # Pretend the framework's agent loop fires 5 inner LLM calls.
        for i in range(5):
            append_step(
                event={
                    "source": "openai",
                    "model": "gpt-4o",
                    "verdict": "allow",
                    "tokens_in": 100 + i,
                    "tokens_out": 50 + i,
                    "prompt_preview": f"turn-{i}",
                    "response_preview": f"answer-{i}",
                },
                kind="model_call",
            )
        return "final answer"

    wrapped = wrap_async_entrypoint(runner_run_body, derive)
    out = asyncio.run(wrapped(object(), object(), "Find Maria's account"))
    assert out == "final answer"

    events = _events()
    assert_one_run_n_steps(events, expected_steps=5)
    assert_all_steps_one_agent(events, expected_agent_id="id-customer-support-agent")
    end = next(e for e in events if e.get("kind") == "run.end")
    # Tokens aggregate.
    assert end["tokens_in"] == sum(100 + i for i in range(5))
    assert end["tokens_out"] == sum(50 + i for i in range(5))
    assert end["step_count"] == 5


# ── Framework: LangGraph / CrewAI / AutoGen / Strands ──────────────


def test_async_iter_streaming_run_aggregates_steps() -> None:
    """Simulate ``Pregel.astream`` — async generator yielding events,
    with inner gate_calls firing per yield. One Run, N steps."""
    from egisai._patches._framework import wrap_async_iter_entrypoint
    from egisai._run import append_step

    record = _make_record("Stream Agent")

    def derive(*a: Any, **kw: Any) -> IdentityRecord:
        return record

    async def pregel_astream(_self: Any, _input: dict[str, Any]):  # type: ignore[no-untyped-def]
        for i in range(3):
            append_step(
                event={
                    "source": "openai",
                    "model": "gpt-4o-mini",
                    "verdict": "allow",
                    "tokens_in": 40,
                    "tokens_out": 20,
                },
                kind="model_call",
            )
            yield {"step": i}

    wrapped = wrap_async_iter_entrypoint(pregel_astream, derive)

    async def consume() -> int:
        count = 0
        async for _ in wrapped(object(), {"prompt": "hi"}):
            count += 1
        return count

    n = asyncio.run(consume())
    assert n == 3

    events = _events()
    assert_one_run_n_steps(events, expected_steps=3)
    assert_all_steps_one_agent(events, "id-stream-agent")


# ── Framework: polymorphic (Agno, smolagents, claude_agent_sdk module) ──


def test_polymorphic_streaming_iter_one_run() -> None:
    """``agno.Agent.arun(stream=True)`` returns async-iter; we must
    still produce ONE Run for the full streaming session."""
    from egisai._patches._framework import wrap_polymorphic_entrypoint
    from egisai._run import append_step

    record = _make_record("Poly Agent")

    def derive(*a: Any, **kw: Any) -> IdentityRecord:
        return record

    def arun_polymorphic(_self: Any) -> Any:
        async def _gen():  # type: ignore[no-untyped-def]
            for i in range(2):
                append_step(
                    event={
                        "source": "anthropic",
                        "model": "claude-3-5-sonnet",
                        "verdict": "allow",
                        "tokens_in": 60,
                        "tokens_out": 30,
                    },
                    kind="model_call",
                )
                yield {"chunk": i}
        return _gen()

    wrapped = wrap_polymorphic_entrypoint(arun_polymorphic, derive)

    async def consume() -> None:
        async for _ in wrapped(object()):
            pass

    asyncio.run(consume())
    events = _events()
    assert_one_run_n_steps(events, expected_steps=2)
    assert_all_steps_one_agent(events, "id-poly-agent")


def test_polymorphic_plain_value_one_step_run() -> None:
    """Plain-value return (``llamaindex.FunctionAgent.run`` handler):
    Run scope spans the call only — one step recorded inside, run
    closes when the call returns."""
    from egisai._patches._framework import wrap_polymorphic_entrypoint
    from egisai._run import append_step

    record = _make_record("Plain Agent")

    def derive(*a: Any, **kw: Any) -> IdentityRecord:
        return record

    def llamaindex_run(_self: Any) -> str:
        append_step(
            event={
                "source": "openai",
                "model": "gpt-4o",
                "verdict": "allow",
                "tokens_in": 100,
                "tokens_out": 50,
            },
            kind="model_call",
        )
        return "result"

    wrapped = wrap_polymorphic_entrypoint(llamaindex_run, derive)
    assert wrapped(object()) == "result"

    events = _events()
    assert_one_run_n_steps(events, expected_steps=1)
    assert_all_steps_one_agent(events, "id-plain-agent")


# ── Sub-agent / handoff scenario ───────────────────────────────────


def test_nested_wraps_open_child_run_with_parent_link() -> None:
    """The user's exception: 'if the agent initiated a sub-agent,
    that is fine to add the second agent'. Confirm the child run
    has parent_run_id."""
    from egisai._patches._framework import (
        wrap_async_entrypoint,
        wrap_sync_entrypoint,
    )
    from egisai._run import append_step

    parent_rec = _make_record("Parent Agent")
    child_rec = _make_record("Child Agent")

    def parent_derive(*a: Any, **kw: Any) -> IdentityRecord:
        return parent_rec

    def child_derive(*a: Any, **kw: Any) -> IdentityRecord:
        return child_rec

    def child_body(_self: Any) -> str:
        append_step(
            event={
                "source": "openai",
                "model": "gpt-4o",
                "verdict": "allow",
                "tokens_in": 50, "tokens_out": 25,
            },
            kind="model_call",
        )
        return "child-done"

    child_wrap = wrap_sync_entrypoint(child_body, child_derive)

    async def parent_body(_self: Any) -> str:
        append_step(
            event={
                "source": "openai",
                "model": "gpt-4o",
                "verdict": "allow",
                "tokens_in": 100, "tokens_out": 50,
            },
            kind="model_call",
        )
        # Spawn the sub-agent inside the parent run.
        out = child_wrap(object())
        append_step(
            event={
                "source": "openai",
                "model": "gpt-4o",
                "verdict": "allow",
                "tokens_in": 30, "tokens_out": 15,
            },
            kind="model_call",
        )
        return out

    parent_wrap = wrap_async_entrypoint(parent_body, parent_derive)
    asyncio.run(parent_wrap(object()))

    events = _events()
    starts = [e for e in events if e.get("kind") == "run.start"]
    ends = [e for e in events if e.get("kind") == "run.end"]
    # One parent run + one child run.
    assert len(starts) == 2, [s["framework"] for s in starts]
    assert len(ends) == 2

    # Order: parent.start, child.start, child.end, parent.end.
    parent_start = starts[0]
    child_start = starts[1]
    assert parent_start["parent_run_id"] is None
    assert child_start["parent_run_id"] == parent_start["run_id"]


# ── Failure: inner call raises -> run closes, error captured ───────


def test_inner_failure_closes_run_with_error() -> None:
    """A framework entry that raises mid-task still closes the run."""
    from egisai._patches._framework import wrap_async_entrypoint
    from egisai._run import append_step

    record = _make_record("Boom Agent")

    def derive(*a: Any, **kw: Any) -> IdentityRecord:
        return record

    async def body(_self: Any) -> None:
        append_step(
            event={"source": "openai", "model": "gpt-4o", "verdict": "allow"},
            kind="model_call",
        )
        raise RuntimeError("framework boom")

    wrapped = wrap_async_entrypoint(body, derive)
    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(wrapped(object()))

    events = _events()
    assert_one_run_n_steps(events, expected_steps=1)
    end = next(e for e in events if e.get("kind") == "run.end")
    assert end["error"] is not None
    assert "boom" in str(end["error"])
