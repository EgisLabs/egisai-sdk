"""Inbound MCP server gate — what reaches a tool a customer HOSTS.

The patch in ``egisai._patches.mcp_server`` wraps FastMCP's
``call_tool`` on the *serving* side, so it is the inbound twin of the
outbound client gate. These tests pin the three operator controls that
must be enforced here before a tool ever runs:

1. **A stopped server refuses every call.** When an operator hits Stop
   on the dashboard, the server id rides the policy snapshot into the
   SDK and the gate blocks — the inbound twin of the agent pause gate.
2. **A disapproved tool refuses its own calls.** A ``"<server_id>:<tool>"``
   block key refuses just that tool, leaving the rest of the server
   live.
3. **It records real policy latency.** The time spent evaluating policy
   is stamped on the gate so the request detail's "Policy" latency is a
   true number instead of always 0.
"""

from __future__ import annotations

from typing import Any

import pytest

from egisai import _policy_cache
from egisai._patches import mcp_server
from egisai.policy import PolicyDecision

SERVER_ID = "ab12cd34-0000-0000-0000-000000000000"


class _FakeInstance:
    """Stands in for a FastMCP server instance."""


@pytest.fixture
def prepared(monkeypatch: pytest.MonkeyPatch):
    """Drive ``_prepare_gate`` with a known identity and policy result.

    Returns a callable ``run(tool_name, arguments, *, decision)`` that
    returns the resulting ``_Gate``.
    """
    identity = mcp_server._ServerIdentity(
        name="GithubTools",
        identity_hash="h",
        transport="stdio",
    )
    identity.server_id = SERVER_ID
    identity.registered = True

    monkeypatch.setattr(mcp_server, "_resolve_identity", lambda _inst: identity)
    monkeypatch.setattr(mcp_server, "get_rules", lambda: [object()])
    # A rule that applies org-wide so policy actually runs.
    monkeypatch.setattr(mcp_server, "_scope_rules", lambda rules, sid: rules)

    state: dict[str, Any] = {"decision": PolicyDecision.allow()}

    def fake_eval(rules: Any, ctx: Any, *, surfaces: Any = None) -> PolicyDecision:
        return state["decision"]

    monkeypatch.setattr(mcp_server, "evaluate_output_policies", fake_eval)

    # Default: nothing stopped, nothing blocked.
    monkeypatch.setattr(
        _policy_cache, "get_paused_mcp_server_ids", lambda: frozenset()
    )
    monkeypatch.setattr(
        _policy_cache, "get_blocked_mcp_tools", lambda: frozenset()
    )

    def run(
        tool_name: str = "list_repos",
        arguments: Any = None,
        *,
        decision: PolicyDecision | None = None,
    ):
        if decision is not None:
            state["decision"] = decision
        return mcp_server._prepare_gate(_FakeInstance(), tool_name, arguments)

    return run


def test_a_stopped_server_refuses_every_call(
    prepared: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        _policy_cache,
        "get_paused_mcp_server_ids",
        lambda: frozenset({SERVER_ID}),
    )
    gate = prepared("list_repos", {"q": "x"})
    assert gate.blocked is True
    assert "stopped" in gate.message.lower()


def test_a_disapproved_tool_refuses_only_itself(
    prepared: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        _policy_cache,
        "get_blocked_mcp_tools",
        lambda: frozenset({f"{SERVER_ID}:delete_repo"}),
    )
    # The blocked tool is refused…
    blocked = prepared("delete_repo", {})
    assert blocked.blocked is True
    assert "disapproved" in blocked.message.lower()
    # …but a sibling tool on the same server is not.
    ok = prepared("list_repos", {})
    assert ok.blocked is False


def test_policy_latency_is_recorded(prepared: Any) -> None:
    gate = prepared("list_repos", {"q": "x"})
    assert gate.blocked is False
    # The field is populated (non-negative), not left at an unstamped
    # sentinel — so the request detail shows a real number.
    assert isinstance(gate.policy_latency_ms, int)
    assert gate.policy_latency_ms >= 0


def test_an_unregistered_server_fails_open_on_stop(
    prepared: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a resolved server id we can't match the stop list — the
    gate must fall open rather than block an unattributed server."""
    identity = mcp_server._ServerIdentity(name="x", identity_hash="h", transport=None)
    identity.server_id = None
    monkeypatch.setattr(mcp_server, "_resolve_identity", lambda _inst: identity)
    monkeypatch.setattr(
        _policy_cache,
        "get_paused_mcp_server_ids",
        lambda: frozenset({SERVER_ID}),
    )
    gate = prepared("list_repos", {})
    assert gate.blocked is False
