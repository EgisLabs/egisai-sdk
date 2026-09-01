"""Outbound MCP client gate — what an agent says to its tools.

The patch in ``egisai._patches.mcp_client`` wraps the official MCP
SDK's ``ClientSession.call_tool``. It is the only place we can see a
tool call *before* it leaves the process, so these tests care about
three things in order of severity:

1. **It cannot leak.** A ``sanitize`` verdict must mask the arguments
   that actually go over the wire, and the audit row must be sampled
   from the masked copy — never the original
   (``security-and-compliance.mdc`` §1 and §5).
2. **It cannot break the agent.** Every failure mode inside the gate
   falls through to the real ``call_tool``, and a block returns a
   readable error result instead of raising into the agent's loop
   (``sdk-design-philosophy.mdc`` §5).
3. **It records the truth.** The step row says which tool, which
   server, and what the verdict was.

There is no real ``mcp`` package in CI, so the tests stand up an
in-process double with the same attribute shapes the patch reaches
for. That is deliberate: the patch's contract with the SDK is exactly
"an async ``call_tool(name, arguments)`` on a class we can find", and
the double pins that contract without a heavyweight dependency.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

from egisai import _config
from egisai._patches import mcp_client
from egisai.policy import PolicyDecision

SSN = "123-45-6789"


# ── Test doubles ────────────────────────────────────────────────────


class FakeSession:
    """Stands in for ``mcp.client.session.ClientSession``."""

    def __init__(self, label: str | None = "slack.com") -> None:
        if label is not None:
            self._egisai_server_label = label
        self.calls: list[tuple[str, Any]] = []
        self.raises: BaseException | None = None

    async def call_tool(self, name: str, arguments: Any = None) -> Any:
        self.calls.append((name, arguments))
        if self.raises is not None:
            raise self.raises
        return FakeResult()


class FakeResult:
    """An MCP ``CallToolResult`` as far as the patch is concerned."""

    def __init__(self, is_error: bool = False) -> None:
        self.isError = is_error  # noqa: N815 — mirrors the MCP wire name


@pytest.fixture
def mcp_types(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a stand-in for ``mcp.types``.

    In production the patch only runs when ``mcp`` is installed, so
    ``_blocked_result`` can always build a real ``CallToolResult``.
    CI has no ``mcp``, so without this stub the tests would only ever
    exercise the ``PermissionError`` fallback and the path customers
    actually hit would go untested.
    """

    class CallToolResult:
        # Mirrors current ``mcp``: the pydantic field is ``is_error``
        # (the ``isError`` wire alias is exposed via serialization, not
        # the constructor). The ``.isError`` attribute is kept so the
        # tests can assert on the wire-facing name.
        def __init__(self, content: Any, is_error: bool) -> None:
            self.content = content
            self.isError = is_error  # noqa: N815

    class TextContent:
        def __init__(self, type: str, text: str) -> None:  # noqa: A002
            self.type = type
            self.text = text

    module = types.ModuleType("mcp.types")
    module.CallToolResult = CallToolResult  # type: ignore[attr-defined]
    module.TextContent = TextContent  # type: ignore[attr-defined]
    package = types.ModuleType("mcp")
    package.types = module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mcp", package)
    monkeypatch.setitem(sys.modules, "mcp.types", module)


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch, mcp_types: None) -> dict[str, Any]:
    """Wrap ``FakeSession.call_tool`` and capture what the gate emits.

    Returns a dict the test can steer: set ``decision`` to choose the
    verdict, read ``steps`` to see the audit rows, read ``access`` to
    see what landed on the agent's inventory.
    """
    state: dict[str, Any] = {
        "decision": PolicyDecision.allow(),
        "raise_in_policy": None,
        "steps": [],
        "access": [],
    }

    def fake_evaluate(call: Any) -> PolicyDecision:
        state["call"] = call
        if state["raise_in_policy"] is not None:
            raise state["raise_in_policy"]
        return state["decision"]

    def fake_dispatch(ev: dict[str, Any], **kwargs: Any) -> None:
        state["steps"].append(ev)

    def fake_attribute(ev: dict[str, Any], payload: Any) -> None:
        ev["agent_id"] = "agent-1"

    def fake_access(agent_id: str | None, items: list[Any], **kwargs: Any) -> None:
        state["access"].append((agent_id, items))

    _config.set_config(
        _config.EgisaiConfig(api_key="egis_test", app="tests", env="test")
    )

    monkeypatch.setattr(mcp_client, "evaluate_output", fake_evaluate)
    monkeypatch.setattr(mcp_client, "_dispatch_step", fake_dispatch)
    monkeypatch.setattr(mcp_client, "_attribute_event", fake_attribute)
    monkeypatch.setattr(mcp_client, "maybe_report_access_items", fake_access)
    monkeypatch.setattr(mcp_client, "_reported", set())

    original = FakeSession.call_tool
    monkeypatch.setattr(
        FakeSession, "call_tool", mcp_client._make_wrapper(original)
    )
    return state


def call(session: FakeSession, name: str = "post_message", **kwargs: Any) -> Any:
    return asyncio.run(session.call_tool(name, **kwargs))


# ── 1. It cannot leak ───────────────────────────────────────────────


def test_a_masked_argument_is_what_the_tool_actually_receives(
    gate: dict[str, Any],
) -> None:
    """The point of sitting inline: the third party sees the mask."""
    gate["decision"] = PolicyDecision.sanitize(
        types=["ssn"],
        reason_code="pii",
        message="ssn",
        matched_policy="pii_scan",
    )
    session = FakeSession()

    call(session, arguments={"text": f"employee ssn is {SSN}"})

    _, forwarded = session.calls[0]
    assert SSN not in forwarded["text"]
    assert "#" in forwarded["text"]


def test_the_audit_row_never_carries_the_raw_value(
    gate: dict[str, Any],
) -> None:
    """Previews are sampled after masking, not before."""
    gate["decision"] = PolicyDecision.sanitize(
        types=["ssn"],
        reason_code="pii",
        message="ssn",
        matched_policy="pii_scan",
    )
    session = FakeSession()

    call(session, arguments={"text": f"ssn {SSN}"})

    step = gate["steps"][0]
    assert SSN not in repr(step)


def test_a_blocked_call_never_reaches_the_server(
    gate: dict[str, Any],
) -> None:
    gate["decision"] = PolicyDecision.deny(
        reason_code="denied_mcp",
        message="slack is off limits",
        matched_policy="deny_mcp_call",
    )
    session = FakeSession()

    result = call(session, arguments={"text": "hello"})

    assert session.calls == []
    assert getattr(result, "isError", None) is True


def test_masking_records_the_shape_not_the_secret(
    gate: dict[str, Any],
) -> None:
    """Sanitization records are count + pattern only."""
    gate["decision"] = PolicyDecision.sanitize(
        types=["ssn"],
        reason_code="pii",
        message="ssn",
        matched_policy="pii_scan",
    )
    session = FakeSession()

    call(session, arguments={"a": SSN, "nested": {"b": SSN}})

    records = gate["steps"][0]["sanitizations"]
    assert records
    for record in records:
        assert set(record) == {"type", "count", "pattern"}
        assert SSN not in str(record)


def test_masking_reaches_every_string_in_the_tree(
    gate: dict[str, Any],
) -> None:
    """Tool arguments nest; a shallow walk would miss the payload."""
    gate["decision"] = PolicyDecision.sanitize(
        types=["ssn"],
        reason_code="pii",
        message="ssn",
        matched_policy="pii_scan",
    )
    session = FakeSession()

    call(
        session,
        arguments={"rows": [{"note": f"x {SSN}"}], "top": SSN, "n": 7},
    )

    _, forwarded = session.calls[0]
    assert SSN not in str(forwarded)
    assert forwarded["n"] == 7  # non-strings survive untouched


# ── 2. It cannot break the agent ────────────────────────────────────


def test_a_policy_engine_that_explodes_still_calls_the_tool(
    gate: dict[str, Any],
) -> None:
    """Fail open. Governance breaking is not the customer's problem."""
    gate["raise_in_policy"] = RuntimeError("engine down")
    session = FakeSession()

    call(session, arguments={"text": "hi"})

    assert session.calls == [("post_message", {"text": "hi"})]


def test_an_uninitialised_sdk_is_completely_transparent(
    gate: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No ``init()`` means we do not even evaluate."""
    monkeypatch.setattr(mcp_client, "get_config_optional", lambda: None)
    session = FakeSession()

    call(session, arguments={"text": "hi"})

    assert session.calls == [("post_message", {"text": "hi"})]
    assert gate["steps"] == []


def test_a_block_returns_a_result_rather_than_raising(
    gate: dict[str, Any],
) -> None:
    """An exception here would kill the agent's whole run.

    The MCP spec already has a shape for "this tool failed", so the
    model reads the refusal and can route around it.
    """
    gate["decision"] = PolicyDecision.deny(
        reason_code="denied_mcp",
        message="nope",
        matched_policy="deny_mcp_call",
    )
    session = FakeSession()

    result = call(session, arguments={})

    assert getattr(result, "isError", None) is True


def test_a_block_falls_back_to_raising_if_mcp_types_are_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hard stop beats quietly letting a blocked call through.

    This path should be unreachable in production — the patch only
    installs when ``mcp`` is importable — but "unreachable" is not a
    reason to let a refusal turn into an allow.

    Force the "mcp not importable" branch deterministically by blanking
    it out of ``sys.modules`` (a ``None`` entry makes ``import`` raise).
    Otherwise this test's outcome depends on whether ``mcp`` happens to
    be installed in the environment — absent in CI, present in a local
    combined venv — which made it flake between the two.
    """
    monkeypatch.setitem(sys.modules, "mcp", None)
    monkeypatch.setitem(sys.modules, "mcp.types", None)
    with pytest.raises(PermissionError, match="nope"):
        mcp_client._blocked_result("nope")


def test_a_failing_tool_still_propagates_its_error(
    gate: dict[str, Any],
) -> None:
    """We audit the failure but never swallow it."""
    session = FakeSession()
    session.raises = ValueError("upstream 500")

    with pytest.raises(ValueError, match="upstream 500"):
        call(session, arguments={})

    assert gate["steps"][0]["error"] == "tool call failed"


def test_an_audit_failure_does_not_fail_the_call(
    gate: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("logger down")

    monkeypatch.setattr(mcp_client, "_dispatch_step", boom)
    session = FakeSession()

    result = call(session, arguments={"text": "hi"})

    assert session.calls
    assert isinstance(result, FakeResult)


def test_a_keyword_call_stays_a_keyword_call(
    gate: dict[str, Any],
) -> None:
    """Rewriting arguments must not change the calling convention."""
    gate["decision"] = PolicyDecision.sanitize(
        types=["ssn"],
        reason_code="pii",
        message="ssn",
        matched_policy="pii_scan",
    )
    seen: dict[str, Any] = {}

    async def original(self: Any, name: str, arguments: Any = None) -> Any:
        seen["name"] = name
        seen["arguments"] = arguments
        return FakeResult()

    wrapped = mcp_client._make_wrapper(original)
    asyncio.run(wrapped(FakeSession(), "t", arguments={"text": SSN}))

    assert SSN not in str(seen["arguments"])


# ── 3. It records the truth ─────────────────────────────────────────


def test_the_step_says_which_tool_and_which_server(
    gate: dict[str, Any],
) -> None:
    session = FakeSession(label="slack.com")

    call(session, "post_message", arguments={"text": "hi"})

    step = gate["steps"][0]
    assert step["step_kind"] == "tool_call"
    assert step["tool_name"] == "post_message"
    assert step["target"] == "mcp.slack.com.tools/call"
    assert step["verdict"] == "allow"
    assert step["enforcement_status"] == "enforced"


def test_the_policy_call_is_scoped_to_the_mcp_surface(
    gate: dict[str, Any],
) -> None:
    """A ``deny_tool_call`` rule scoped to "tool" must not fire here."""
    session = FakeSession()

    call(session, "post_message", arguments={"text": "hi"})

    assert gate["call"].surfaces == ("mcp",)
    assert gate["call"].mcp_targets == ["slack.com"]
    assert gate["call"].tool_names == ["post_message"]


def test_sanitize_is_allowed_because_we_hold_the_arguments(
    gate: dict[str, Any],
) -> None:
    """Paths with no mutation point must block instead of mask.

    We have one, so ``allow_sanitize`` is honest here.
    """
    session = FakeSession()

    call(session, arguments={"text": "hi"})

    assert gate["call"].allow_sanitize is True


def test_an_observed_server_lands_on_the_access_inventory(
    gate: dict[str, Any],
) -> None:
    """Declared inventory is a claim; this is evidence."""
    session = FakeSession(label="slack.com")

    call(session, "post_message", arguments={})

    agent_id, items = gate["access"][0]
    assert agent_id == "agent-1"
    kinds = {(i["kind"], i["name"]) for i in items}
    assert ("mcp_server", "slack.com") in kinds
    assert ("tool", "post_message") in kinds


def test_the_same_server_is_only_reported_once(
    gate: dict[str, Any],
) -> None:
    """Steady state is a set lookup, not a round-trip per call."""
    session = FakeSession(label="slack.com")

    call(session, arguments={})
    call(session, arguments={})
    call(session, arguments={})

    assert len(gate["access"]) == 1


def test_an_unnameable_server_is_still_governed(
    gate: dict[str, Any],
) -> None:
    """We would rather audit an anonymous call than skip the gate."""
    session = FakeSession(label=None)

    call(session, arguments={"text": "hi"})

    assert gate["steps"][0]["target"] == "mcp.mcp.tools/call"
    assert gate["access"] == []  # nothing useful to inventory


def test_a_tool_that_reports_its_own_failure_is_recorded_as_one(
    gate: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """MCP tool errors are results, not exceptions."""

    async def original(self: Any, name: str, arguments: Any = None) -> Any:
        return FakeResult(is_error=True)

    wrapped = mcp_client._make_wrapper(original)
    asyncio.run(wrapped(FakeSession(), "t", arguments={}))

    assert gate["steps"][0]["error"] == "tool call failed"


# ── Server labelling ────────────────────────────────────────────────


def test_a_url_transport_is_named_by_its_host() -> None:
    """A full URL in an audit row is noise; the host is the identity."""

    class Transport:
        url = "https://mcp.example.com/v1/sse?token=abc"

    class Session:
        _transport = Transport()

    assert mcp_client._server_label(Session()) == "mcp.example.com"


def test_the_handshake_name_wins_over_a_transport_guess() -> None:
    class Session:
        class _server_info:  # noqa: N801
            name = "GitHub MCP"

    assert mcp_client._server_label(Session()) == "GitHub MCP"


# ── apply() ─────────────────────────────────────────────────────────


def test_apply_is_a_noop_without_the_mcp_package() -> None:
    """``mcp`` is an optional dependency; absence must be silent."""
    assert mcp_client.apply() is False


def test_patching_twice_does_not_double_wrap() -> None:
    class Once:
        async def call_tool(self, name: str, arguments: Any = None) -> Any:
            return None

    assert mcp_client._patch_class_method(Once, "call_tool") is True
    assert mcp_client._patch_class_method(Once, "call_tool") is False


def test_a_sync_call_tool_is_left_alone() -> None:
    """We only know how to gate the async form."""

    class Sync:
        def call_tool(self, name: str, arguments: Any = None) -> Any:
            return None

    assert mcp_client._patch_class_method(Sync, "call_tool") is False
