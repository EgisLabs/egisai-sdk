"""Sub-agent topology for the Claude Agent SDK patch.

An orchestrator agent that delegates work spawns a *second*
``ClaudeSDKClient`` from inside its own turn — usually from an MCP
tool handler, so the child client is constructed and driven while the
parent's Run is still open on the ContextVar.

Before 0.46.1 the patch treated "a Run is open" as "the Run is mine".
Three sites called ``close_run()`` after only checking
``current_run() is not None``, so the child client ended the PARENT's
Run: ``_flush_stale_inflight`` fired the moment the child's
``query()`` started, the parent's timeline was truncated to whatever
had happened before the delegation, and every later parent event
(tool results, the final model call) arrived with no Run open and got
synthesized into orphan single-step rows on the dashboard. The child
then saw no open Run and started a top-level Run of its own, so the
parent → child link was lost too.

The contract these tests lock in is the one every other patch already
follows via ``_framework._RunScope``:

* a turn closes only the Run it opened,
* a nested turn for a *different* agent opens a CHILD Run carrying
  ``parent_run_id``,
* a nested turn for the *same* agent rides along on the open Run
  rather than duplicating it.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

# ── Fake upstream (class names must match real ones — the patch
# duck-types on ``type(message).__name__``) ─────────────────────────


class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class AssistantMessage:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class ResultMessage:
    def __init__(
        self, *, input_tokens: int = 10, output_tokens: int = 20
    ) -> None:
        self.usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        self.total_cost_usd = 0.01


class _Options:
    def __init__(
        self,
        *,
        system_prompt: str = "You are the orchestrator.",
        allowed_tools: list[str] | None = None,
        model: str = "claude-3-5-sonnet",
    ) -> None:
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools or []
        self.permission_mode = "auto"
        self.model = model
        self.mcp_servers: dict[str, Any] = {}


class _Client:
    """Per-instance script, so parent and child can differ."""

    def __init__(self, options: Any = None) -> None:
        self.options = options
        self.script: list[Any] = []
        self._sent: list[Any] = []

    async def __aenter__(self) -> _Client:
        return self

    async def __aexit__(self, *exc: Any) -> bool:
        return False

    async def query(self, prompt: Any, session_id: str = "default") -> None:
        self._sent.append(prompt)

    async def receive_messages(self) -> AsyncIterator[Any]:
        # An exception in the script is raised mid-stream — the shape a
        # dead CLI subprocess produces. Subclassing to override this
        # method would shadow the patch (which wraps it on the class
        # registered in the fake module), so the script carries it.
        for msg in self.script:
            if isinstance(msg, BaseException):
                raise msg
            yield msg

    async def receive_response(self) -> AsyncIterator[Any]:
        async for msg in self.receive_messages():
            yield msg
            if isinstance(msg, ResultMessage):
                return


@pytest.fixture
def fake_claude(fake_backend: Any) -> Iterator[Any]:
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="claude-subagent-test",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )
    mod = types.ModuleType("claude_agent_sdk")
    mod.ClaudeSDKClient = _Client
    mod.AssistantMessage = AssistantMessage
    mod.TextBlock = TextBlock
    mod.ResultMessage = ResultMessage
    sys.modules["claude_agent_sdk"] = mod

    from egisai._patches import claude_agent_sdk

    assert claude_agent_sdk.apply() is True

    # The logger queue is process-global and outlives a single test's
    # backend. Drain whatever the previous test left behind so this
    # test's ``run.start`` / ``run.end`` counts describe only its own
    # traffic.
    from egisai import _logger

    while not _logger._q.empty():
        try:
            _logger._q.get_nowait()
        except Exception:  # noqa: BLE001
            break
    fake_backend.events_received.clear()

    yield fake_backend
    sys.modules.pop("claude_agent_sdk", None)


def _flush() -> None:
    from egisai import shutdown

    shutdown()


def _by_kind(events: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [e for e in events if e.get("kind") == kind]


async def _drive(client: _Client, prompt: str, script: list[Any]) -> None:
    """One complete turn: query + full response iteration."""
    client.script = script
    await client.query(prompt)
    async for _ in client.receive_response():
        pass


def _turn_script(text: str) -> list[Any]:
    return [AssistantMessage([TextBlock(text)]), ResultMessage()]


async def _delegate_during(parent: _Client, spawn: Any) -> None:
    """Drive the parent's stream, running ``spawn`` once mid-turn.

    The delegation fires on the first AssistantMessage — i.e. while
    the parent's Run is open and its stream is still being consumed,
    which is where a real MCP tool handler would spawn a sub-agent.
    """
    delegated = False
    async for msg in parent.receive_response():
        if not delegated and isinstance(msg, AssistantMessage):
            delegated = True
            await spawn()
    assert delegated, "parent stream never produced an AssistantMessage"


PARENT_OPTS = {"system_prompt": "You are the payments orchestrator."}
CHILD_OPTS = {"system_prompt": "You are the fraud specialist sub-agent."}


# ── 1. Parent → child linkage ───────────────────────────────────────


def test_sub_agent_turn_opens_child_run_linked_to_parent(
    fake_claude: Any,
) -> None:
    """A second client driven inside the parent's turn produces a
    CHILD Run carrying the parent's ``run_id``."""
    fake_backend = fake_claude

    async def spawn() -> None:
        async with _Client(options=_Options(**CHILD_OPTS)) as child:
            await _drive(child, "score this payment", _turn_script("score 0.9"))

    async def run() -> None:
        async with _Client(options=_Options(**PARENT_OPTS)) as parent:
            parent.script = _turn_script("delegating to the specialist")
            await parent.query("investigate payment PMT-1")
            await _delegate_during(parent, spawn)

    asyncio.run(run())
    _flush()

    starts = _by_kind(fake_backend.events_received, "run.start")
    assert len(starts) == 2, [s.get("app") for s in starts]
    parent_start, child_start = starts[0], starts[1]
    assert parent_start["parent_run_id"] is None
    assert child_start["parent_run_id"] == parent_start["run_id"], (
        "sub-agent Run must link to the orchestrator's Run"
    )
    assert child_start["run_id"] != parent_start["run_id"]


def test_sub_agent_does_not_close_the_parent_run(fake_claude: Any) -> None:
    """The regression itself: the parent's Run must stay open across
    the whole delegation and end exactly once, after the child's."""
    fake_backend = fake_claude

    async def spawn() -> None:
        async with _Client(options=_Options(**CHILD_OPTS)) as child:
            await _drive(child, "score this payment", _turn_script("score 0.9"))

    async def run() -> None:
        async with _Client(options=_Options(**PARENT_OPTS)) as parent:
            parent.script = _turn_script("done")
            await parent.query("investigate payment PMT-1")
            await _delegate_during(parent, spawn)

    asyncio.run(run())
    _flush()

    events = fake_backend.events_received
    starts = _by_kind(events, "run.start")
    ends = _by_kind(events, "run.end")
    parent_run_id = starts[0]["run_id"]
    child_run_id = starts[1]["run_id"]

    assert [e["run_id"] for e in ends] == [child_run_id, parent_run_id], (
        "child must end first; the parent's Run must outlive the delegation"
    )
    assert len(_by_kind(events, "run.end")) == 2, "no Run may end twice"
    parent_end = ends[-1]
    assert parent_end["error"] is None, (
        f"parent Run must not be marked failed; got {parent_end['error']!r}"
    )


def test_parent_steps_after_delegation_stay_on_the_parent_run(
    fake_claude: Any,
) -> None:
    """Everything the orchestrator does *after* the sub-agent returns
    must still be recorded under the orchestrator's Run.

    With the parent's Run closed out from under it, these landed as
    run-less legacy rows that the backend synthesized into separate
    one-step Runs — the "duplicate orchestrator" the dashboard showed.
    """
    fake_backend = fake_claude

    async def spawn() -> None:
        async with _Client(options=_Options(**CHILD_OPTS)) as child:
            await _drive(child, "score this payment", _turn_script("score 0.9"))

    async def run() -> None:
        async with _Client(options=_Options(**PARENT_OPTS)) as parent:
            parent.script = _turn_script("summary of the investigation")
            await parent.query("investigate payment PMT-1")
            await _delegate_during(parent, spawn)

    asyncio.run(run())
    _flush()

    events = fake_backend.events_received
    parent_run_id = _by_kind(events, "run.start")[0]["run_id"]
    steps = _by_kind(events, "run.step")
    parent_steps = [s for s in steps if s["run_id"] == parent_run_id]
    # Provisional seq-0 at query time + the terminal model_call.
    assert len(parent_steps) >= 2, (
        f"parent Run lost its post-delegation steps; got {parent_steps!r}"
    )
    assert any(
        s.get("tokens_in") for s in parent_steps
    ), "the parent's finalized model_call step must land on the parent Run"
    # No event may escape the Run framework into a legacy single row.
    legacy = [
        e for e in events
        if e.get("kind") not in {"run.start", "run.step", "run.end"}
    ]
    assert legacy == [], f"orphan run-less audit rows: {legacy!r}"


# ── 2. Concurrent sub-agents ───────────────────────────────────────


def test_parallel_sub_agents_each_open_their_own_child_run(
    fake_claude: Any,
) -> None:
    """Fan-out: two specialists dispatched with ``asyncio.gather``.

    ``gather`` wraps each coroutine in a Task, and a Task copies the
    context at creation — so both children see the parent's Run as
    their parent and neither can pop the other's Run off the shared
    ContextVar.
    """
    fake_backend = fake_claude

    async def spawn(name: str) -> None:
        opts = _Options(system_prompt=f"You are the {name} sub-agent.")
        async with _Client(options=opts) as child:
            await _drive(child, f"{name} task", _turn_script(f"{name} done"))

    async def fan_out() -> None:
        await asyncio.gather(spawn("fraud"), spawn("compliance"))

    async def run() -> None:
        async with _Client(options=_Options(**PARENT_OPTS)) as parent:
            parent.script = _turn_script("fan-out complete")
            await parent.query("investigate payment PMT-1")
            await _delegate_during(parent, fan_out)

    asyncio.run(run())
    _flush()

    events = fake_backend.events_received
    starts = _by_kind(events, "run.start")
    ends = _by_kind(events, "run.end")
    assert len(starts) == 3, [s.get("app") for s in starts]
    assert len(ends) == 3

    parent_run_id = starts[0]["run_id"]
    children = starts[1:]
    for child in children:
        assert child["parent_run_id"] == parent_run_id
    assert len({c["run_id"] for c in children}) == 2, "children must not share a Run"
    # The parent is always the last Run to end.
    assert ends[-1]["run_id"] == parent_run_id


# ── 3. Same-identity re-entry still merges ─────────────────────────


def test_same_identity_nested_turn_rides_along_on_one_run(
    fake_claude: Any,
) -> None:
    """A nested client with the SAME identity bundle is re-entry, not
    delegation — it must not duplicate the Run.

    Mirrors ``_framework._RunScope``'s guard (and the LangGraph
    ``invoke`` → ``stream`` case it was written for): an empty
    duplicate Run row skews the dashboard's step-count tile and the
    billing roll-up.
    """
    fake_backend = fake_claude

    async def spawn() -> None:
        async with _Client(options=_Options(**PARENT_OPTS)) as inner:
            await _drive(inner, "inner turn", _turn_script("inner done"))

    async def run() -> None:
        async with _Client(options=_Options(**PARENT_OPTS)) as outer:
            outer.script = _turn_script("outer done")
            await outer.query("same-identity turn")
            await _delegate_during(outer, spawn)

    asyncio.run(run())
    _flush()

    events = fake_backend.events_received
    assert len(_by_kind(events, "run.start")) == 1
    assert len(_by_kind(events, "run.end")) == 1


# ── 4. Sub-agent failures leave the parent alone ───────────────────


def test_failing_sub_agent_does_not_fail_the_parent_run(
    fake_claude: Any,
) -> None:
    """A crash inside the sub-agent's stream ends the CHILD Run with
    the error and leaves the parent's Run clean."""
    fake_backend = fake_claude
    boom_script = [
        AssistantMessage([TextBlock("partial")]),
        RuntimeError("subprocess died"),
    ]

    async def spawn() -> None:
        async with _Client(options=_Options(**CHILD_OPTS)) as child:
            with pytest.raises(RuntimeError, match="subprocess died"):
                await _drive(child, "score", boom_script)

    async def run() -> None:
        async with _Client(options=_Options(**PARENT_OPTS)) as parent:
            parent.script = _turn_script("recovered without the specialist")
            await parent.query("investigate payment PMT-1")
            await _delegate_during(parent, spawn)

    asyncio.run(run())
    _flush()

    events = fake_backend.events_received
    starts = _by_kind(events, "run.start")
    ends = _by_kind(events, "run.end")
    assert len(starts) == 2
    by_run = {e["run_id"]: e for e in ends}
    parent_end = by_run[starts[0]["run_id"]]
    child_end = by_run[starts[1]["run_id"]]
    assert child_end["error"] == "stream failed"
    assert parent_end["error"] is None, (
        "a sub-agent crash must not mark the orchestrator's Run failed"
    )
