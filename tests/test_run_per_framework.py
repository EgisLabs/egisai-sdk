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
import hashlib
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
    # ``identity_hash`` is keyed on ``name`` so two records built from
    # different agent names produce different hashes. This mirrors the
    # real world: ``make_identity`` derives the digest from the
    # (framework, prompt, tools, model) bundle, so a Parent Agent and
    # a Child Agent always disagree on at least one bundle slot. The
    # ``_RunScope.__enter__`` re-entry guard short-circuits only on
    # identity_hash equality, so the suite must reflect that two
    # logically distinct agents carry distinct hashes — otherwise the
    # child run merges into the parent and the "sub-agent / handoff"
    # parent_run_id assertion can never fire.
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    return IdentityRecord(
        agent_id=f"id-{name.lower().replace(' ', '-')}",
        display_name=name,
        identity_key=f"test:{name}",
        identity_hash=digest,
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


# ── Re-entry guard: same identity nested wraps merge into one Run ──


def test_same_identity_nested_wraps_emit_one_run() -> None:
    """LangGraph regression: ``Pregel.invoke`` calls ``self.stream``
    internally; both methods sit behind separate ``_RunScope`` wraps
    keyed on the same compiled-graph identity. Without the re-entry
    guard the SDK emits two ``run.start`` events (one outer, one
    inner) per agent invocation — the outer Run ends up empty,
    the inner one carries the real step, and the dashboard's
    "average step count" tile / billing roll-up double-count traces.

    The guard keys on ``identity_hash`` so a *true* sub-agent
    (different bundle → different hash) still spawns a child Run
    with ``parent_run_id`` wired up — that contract is exercised
    by ``test_nested_wraps_open_child_run_with_parent_link`` above.
    """
    from egisai._patches._framework import (
        wrap_sync_entrypoint,
        wrap_sync_iter_entrypoint,
    )
    from egisai._run import append_step

    record = _make_record("Re-entry Agent")

    def derive(*a: Any, **kw: Any) -> IdentityRecord:
        return record

    # ``stream`` is the inner wrap — yields chunks. We expect its
    # ``_RunScope.__enter__`` to detect the outer Run for the same
    # identity and skip opening a duplicate Run.
    def stream_body(_self: Any) -> Iterator[str]:
        append_step(
            event={
                "source": "openai",
                "model": "gpt-4o",
                "verdict": "allow",
                "tokens_in": 100, "tokens_out": 50,
            },
            kind="model_call",
        )
        yield "chunk-1"
        yield "chunk-2"

    stream_wrap = wrap_sync_iter_entrypoint(stream_body, derive)

    def invoke_body(_self: Any) -> str:
        # Mirror LangGraph: ``invoke`` consumes ``stream`` internally.
        final = ""
        for chunk in stream_wrap(_self):
            final = chunk
        return final

    invoke_wrap = wrap_sync_entrypoint(invoke_body, derive)
    result = invoke_wrap(object())
    assert result == "chunk-2"

    events = _events()
    starts = [e for e in events if e.get("kind") == "run.start"]
    ends = [e for e in events if e.get("kind") == "run.end"]
    steps = [e for e in events if e.get("kind") == "run.step"]
    assert len(starts) == 1, "re-entry guard should suppress the inner run.start"
    assert len(ends) == 1
    assert len(steps) == 1, "the single step should live under one run"
    assert steps[0]["run_id"] == starts[0]["run_id"]


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


# ── Block-raised PermissionError must NOT taint run.error ───────────


def test_sdk_block_permission_error_stamps_short_reason_not_full_repr() -> None:
    """``_RunScope`` must recognise the SDK's own block-raise
    PermissionError (message starts with ``[egisai]``) and translate
    it into the short canonical reason string ``"policy block"``
    instead of stamping the full ``repr(PermissionError(...))`` on
    ``run.error``.

    Rationale: ``_block_response`` in ``_patches._common`` already
    dispatched a step with ``verdict='block'`` and a fully populated
    ``prompt_decision`` block before raising. Putting the
    PermissionError's repr on ``run.error`` would falsely classify a
    blocked Run as an uncaught-exception crash — the agents-test
    validator's allowed-list reserves ``run.error`` for either NULL
    or one of the canonical short strings (``"input policy block"``,
    ``"output policy block"``, ``"policy block"``) used by
    ``claude_agent_sdk``'s own ``close_run`` sites. Mirror that
    shape across every framework wrap so a refused Bedrock /
    Bedrock-Agent / langchain / etc. turn ends up with the same
    audit signature on the wire.
    """
    from egisai._patches._framework import wrap_async_entrypoint
    from egisai._run import append_step

    record = _make_record("Blocked Agent")

    def derive(*a: Any, **kw: Any) -> IdentityRecord:
        return record

    async def body(_self: Any) -> None:
        append_step(
            event={"source": "openai", "model": "gpt-4o", "verdict": "block"},
            kind="model_call",
        )
        raise PermissionError(
            "[egisai] Refused: agent attempted to refund "
            "(matched=Block refund issuing)"
        )

    wrapped = wrap_async_entrypoint(body, derive)
    with pytest.raises(PermissionError, match=r"\[egisai\] Refused"):
        asyncio.run(wrapped(object()))

    events = _events()
    end = next(e for e in events if e.get("kind") == "run.end")
    assert end["error"] == "policy block", (
        f"SDK-raised block should stamp the short canonical reason; got {end['error']!r}"
    )

    # A non-egisai PermissionError MUST still get its full repr — the
    # ``[egisai]`` prefix is the only signal we trust to mean "this
    # exception is the SDK's own refuse-by-raise, not a real crash".
    async def bare_perm_body(_self: Any) -> None:
        append_step(
            event={"source": "openai", "model": "gpt-4o", "verdict": "allow"},
            kind="model_call",
        )
        raise PermissionError("os: permission denied")

    bare_wrapped = wrap_async_entrypoint(bare_perm_body, derive)
    with pytest.raises(PermissionError, match="permission denied"):
        asyncio.run(bare_wrapped(object()))
    events = _events()
    end = next(e for e in events if e.get("kind") == "run.end")
    assert end["error"] is not None
    assert "os: permission denied" in str(end["error"])
