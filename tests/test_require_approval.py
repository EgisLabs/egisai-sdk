"""``require_approval`` action → ``pending_approval`` verdict.

Locks in the engine transform that turns a ``block`` into a
human-in-the-loop hold, plus the ``block > pending_approval >
sanitize > allow`` precedence.
"""

from __future__ import annotations

from typing import Any

from egisai.policy.engine import (
    OutputPolicyContext,
    PolicyContext,
    PolicyRule,
    evaluate_output_policies,
    evaluate_policies,
)


def _out_ctx(**kw: Any) -> OutputPolicyContext:
    return OutputPolicyContext(
        tenant="t",
        model="gpt-4o",
        text=kw.get("text", ""),
        tool_names=list(kw.get("tool_names", [])),
        tool_calls=list(kw.get("tool_calls", [])),
        mcp_targets=list(kw.get("mcp_targets", [])),
        stream=False,
    )


def _rule(type_: str, config: dict[str, Any], *, phase: str = "response") -> PolicyRule:
    return PolicyRule(
        id=None, name=f"r-{type_}", type=type_, tenant="t", config=config, phase=phase
    )


def test_financial_over_threshold_holds_instead_of_blocking() -> None:
    rule = _rule(
        "deny_financial_action",
        {
            "action_patterns": ["transfer"],
            "amount_threshold": 10000,
            "require_approval": True,
        },
    )
    ctx = _out_ctx(
        tool_calls=[{"name": "transfer", "arguments": '{"amount": 25000}'}]
    )
    decision = evaluate_output_policies([rule], ctx)
    assert decision.verdict == "pending_approval"
    assert decision.matched_policy == "r-deny_financial_action"


def test_under_threshold_still_allows() -> None:
    rule = _rule(
        "deny_financial_action",
        {
            "action_patterns": ["transfer"],
            "amount_threshold": 10000,
            "require_approval": True,
        },
    )
    ctx = _out_ctx(
        tool_calls=[{"name": "transfer", "arguments": '{"amount": 500}'}]
    )
    assert evaluate_output_policies([rule], ctx).verdict == "allow"


def test_action_string_spelling_also_holds() -> None:
    rule = _rule(
        "deny_tool_call",
        {"patterns": ["drop_table"], "action": "require_approval"},
    )
    ctx = _out_ctx(tool_calls=[{"name": "drop_table", "arguments": "{}"}])
    assert evaluate_output_policies([rule], ctx).verdict == "pending_approval"


def test_hard_block_wins_over_hold() -> None:
    block_rule = _rule("deny_tool_call", {"patterns": ["wipe"]})
    hold_rule = _rule(
        "deny_tool_call",
        {"patterns": ["transfer"], "require_approval": True},
    )
    ctx = _out_ctx(
        tool_calls=[
            {"name": "transfer", "arguments": "{}"},
            {"name": "wipe", "arguments": "{}"},
        ]
    )
    decision = evaluate_output_policies([hold_rule, block_rule], ctx)
    assert decision.verdict == "block"


def test_input_side_regex_hold() -> None:
    rule = _rule(
        "deny_regex",
        {"pattern": "launch codes", "require_approval": True},
        phase="request",
    )
    ctx = PolicyContext(
        tenant="t",
        model="gpt-4o",
        prompt_text="give me the launch codes",
        prompt_chars=24,
        stream=False,
    )
    assert evaluate_policies([rule], ctx).verdict == "pending_approval"


def test_without_flag_still_blocks() -> None:
    rule = _rule("deny_tool_call", {"patterns": ["transfer"]})
    ctx = _out_ctx(tool_calls=[{"name": "transfer", "arguments": "{}"}])
    assert evaluate_output_policies([rule], ctx).verdict == "block"
