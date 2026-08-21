"""``deny_resource_access`` — the identity-aware access "scalpel".

``deny_tool_call`` blocks a tool for everyone. ``deny_resource_access``
blocks one *resource* (a file id, record id, path, MCP URI found inside
the call's arguments) for the wrong *people*, leaving the same tool
working for everyone else and every other resource.

These tests hit the engine helper directly via
``evaluate_output_policies`` with an identity-carrying
``OutputPolicyContext`` so the matching + fail-closed semantics are
pinned without the full SDK init flow.
"""

from __future__ import annotations

import json
from typing import Any

from egisai.policy.engine import (
    OutputPolicyContext,
    PolicyRule,
    evaluate_output_policies,
)


def _ctx(
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    mcp_targets: list[str] | None = None,
    tool_names: list[str] | None = None,
    user_role: str = "",
    end_user_id: str = "",
    user_id: str = "",
) -> OutputPolicyContext:
    return OutputPolicyContext(
        tenant="tenant-x",
        model="gpt-4o",
        text="",
        tool_names=list(tool_names or []),
        tool_calls=list(tool_calls or []),
        mcp_targets=list(mcp_targets or []),
        stream=False,
        user_role=user_role,
        end_user_id=end_user_id,
        user_id=user_id,
    )


def _rule(config: dict[str, Any], *, name: str = "resource-rule") -> PolicyRule:
    return PolicyRule(
        id=None,
        name=name,
        type="deny_resource_access",
        tenant="tenant-x",
        config=config,
        phase="response",
    )


def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """A model-response tool call — arguments as a JSON string, the shape
    the provider patches normalize to."""
    return {"name": name, "arguments": json.dumps(args)}


# ── allowlist: only permitted roles get the resource ──────────────────


def test_allow_roles_blocks_a_user_not_on_the_list() -> None:
    rule = _rule(
        {
            "resource_patterns": [r"q4_board_deck"],
            "allow_roles": ["finance"],
        }
    )
    decision = evaluate_output_policies(
        [rule],
        _ctx(
            tool_calls=[_call("drive_get_file", {"file_id": "q4_board_deck"})],
            user_role="support",
        ),
    )
    assert decision.verdict == "block"
    assert decision.reason_code == "resource_access_blocked"


def test_allow_roles_permits_a_user_on_the_list() -> None:
    rule = _rule(
        {
            "resource_patterns": [r"q4_board_deck"],
            "allow_roles": ["finance"],
        }
    )
    decision = evaluate_output_policies(
        [rule],
        _ctx(
            tool_calls=[_call("drive_get_file", {"file_id": "q4_board_deck"})],
            user_role="finance",
        ),
    )
    assert decision.verdict == "allow"


def test_allow_roles_is_case_insensitive() -> None:
    rule = _rule(
        {"resource_patterns": [r"secret"], "allow_roles": ["Finance"]}
    )
    decision = evaluate_output_policies(
        [rule],
        _ctx(
            tool_calls=[_call("read", {"path": "/secret/x"})],
            user_role="FINANCE",
        ),
    )
    assert decision.verdict == "allow"


# ── fail-closed: an unknown identity is never "permitted" ─────────────


def test_unknown_identity_is_blocked_under_an_allowlist() -> None:
    """The critical security property: if the caller never set an
    identity, an allowlist rule refuses rather than leaks."""
    rule = _rule(
        {"resource_patterns": [r"payroll"], "allow_roles": ["hr"]}
    )
    decision = evaluate_output_policies(
        [rule],
        _ctx(tool_calls=[_call("read", {"path": "/payroll/2026.csv"})]),
    )
    assert decision.verdict == "block"


# ── blocklist: named roles / users are refused ────────────────────────


def test_deny_roles_blocks_that_role() -> None:
    rule = _rule({"resource_patterns": [r"prod_db"], "deny_roles": ["intern"]})
    decision = evaluate_output_policies(
        [rule],
        _ctx(
            tool_calls=[_call("query", {"target": "prod_db"})],
            user_role="intern",
        ),
    )
    assert decision.verdict == "block"


def test_deny_roles_leaves_other_roles_alone() -> None:
    rule = _rule({"resource_patterns": [r"prod_db"], "deny_roles": ["intern"]})
    decision = evaluate_output_policies(
        [rule],
        _ctx(
            tool_calls=[_call("query", {"target": "prod_db"})],
            user_role="dba",
        ),
    )
    assert decision.verdict == "allow"


def test_deny_end_users_blocks_that_subject() -> None:
    rule = _rule(
        {"resource_patterns": [r"case_"], "deny_end_users": ["cust_42"]}
    )
    decision = evaluate_output_policies(
        [rule],
        _ctx(
            tool_calls=[_call("get_case", {"id": "case_991"})],
            end_user_id="cust_42",
        ),
    )
    assert decision.verdict == "block"


def test_deny_wins_over_allow() -> None:
    """A denied user can't slip through by also being on an allowlist."""
    rule = _rule(
        {
            "resource_patterns": [r"vault"],
            "allow_roles": ["admin"],
            "deny_end_users": ["contractor_7"],
        }
    )
    decision = evaluate_output_policies(
        [rule],
        _ctx(
            tool_calls=[_call("read", {"path": "/vault/keys"})],
            user_role="admin",
            end_user_id="contractor_7",
        ),
    )
    assert decision.verdict == "block"


# ── scoping: only the matched resource / tool is gated ────────────────


def test_a_different_resource_is_untouched() -> None:
    """The scalpel: the same tool, a different file, no block — the
    whole point versus deny_tool_call."""
    rule = _rule(
        {"resource_patterns": [r"q4_board_deck"], "allow_roles": ["finance"]}
    )
    decision = evaluate_output_policies(
        [rule],
        _ctx(
            tool_calls=[_call("drive_get_file", {"file_id": "team_lunch"})],
            user_role="support",
        ),
    )
    assert decision.verdict == "allow"


def test_tool_patterns_narrow_the_rule_to_named_tools() -> None:
    rule = _rule(
        {
            "tool_patterns": [r"^drive_"],
            "resource_patterns": [r"secret"],
            "allow_roles": ["finance"],
        }
    )
    # A non-drive tool touching the same string is out of scope.
    decision = evaluate_output_policies(
        [rule],
        _ctx(
            tool_calls=[_call("log_event", {"note": "secret"})],
            user_role="support",
        ),
    )
    assert decision.verdict == "allow"


def test_empty_resource_patterns_gate_every_resource_of_the_tool() -> None:
    rule = _rule(
        {"tool_patterns": [r"^payroll_export$"], "allow_roles": ["hr"]}
    )
    decision = evaluate_output_policies(
        [rule],
        _ctx(
            tool_calls=[_call("payroll_export", {"quarter": "q1"})],
            user_role="support",
        ),
    )
    assert decision.verdict == "block"


# ── MCP resources ─────────────────────────────────────────────────────


def test_matches_resource_in_an_mcp_target() -> None:
    rule = _rule(
        {
            "resource_patterns": [r"finance/reports"],
            "allow_roles": ["finance"],
        }
    )
    decision = evaluate_output_policies(
        [rule],
        _ctx(
            mcp_targets=["docs-server/finance/reports/q4.pdf"],
            user_role="support",
        ),
    )
    assert decision.verdict == "block"


def test_matches_resource_in_a_dict_input_payload() -> None:
    """PreToolUse / MCP-client gates pass ``input`` as a dict rather
    than an ``arguments`` JSON string — the rule must still see it."""
    rule = _rule(
        {"resource_patterns": [r"ssn_export"], "deny_roles": ["support"]}
    )
    decision = evaluate_output_policies(
        [rule],
        _ctx(
            tool_calls=[{"name": "read", "input": {"file": "ssn_export.csv"}}],
            user_role="support",
        ),
    )
    assert decision.verdict == "block"


# ── misconfiguration: no identity predicate = no-op ───────────────────


def test_no_identity_predicate_is_a_noop() -> None:
    """A rule with no allow/deny lists gates nobody. It no-ops rather
    than blocking everyone (that's a mislabelled deny_tool_call) — the
    backend rejects it at creation, this is the runtime safety net."""
    rule = _rule({"resource_patterns": [r"anything"]})
    decision = evaluate_output_policies(
        [rule],
        _ctx(
            tool_calls=[_call("read", {"path": "/anything/here"})],
            user_role="support",
        ),
    )
    assert decision.verdict == "allow"


def test_no_tool_calls_at_all_is_a_noop() -> None:
    rule = _rule(
        {"resource_patterns": [r"secret"], "allow_roles": ["finance"]}
    )
    decision = evaluate_output_policies([rule], _ctx(user_role="support"))
    assert decision.verdict == "allow"
