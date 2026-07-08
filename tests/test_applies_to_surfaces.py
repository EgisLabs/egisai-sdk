"""``PolicyRule.applies_to`` scopes a rule to specific call surfaces.

Empty ``applies_to`` (the default, and the shape every pre-0.32.0
rule has) means "all surfaces" — nothing changes for existing
policies. A non-empty tuple narrows the rule: it only fires when the
evaluation's ``surfaces`` intersects it.

Surface semantics per evaluation site:

- model-call prompt              → ``("model",)``
- model-call output              → ``("model", "tool", "mcp")``
  (the completion text is model surface; the model's tool-call
  requests are tool/mcp surface — one pass covers all three)
- claude_agent_sdk per-tool hook → ``("tool",)`` or ``("mcp",)``
- inbound MCP ``tools/call``     → ``("mcp",)``
"""

from __future__ import annotations

from egisai.policy import (
    OutputPolicyContext,
    PolicyContext,
    PolicyRule,
    evaluate_output_policies,
    evaluate_policies,
)


def _pii_block(applies_to: tuple[str, ...] = ()) -> PolicyRule:
    return PolicyRule(
        id="r1",
        name="pii-block",
        type="pii_scan",
        tenant=None,
        config={"action": "block"},
        phase="both",
        applies_to=applies_to,
    )


def _deny_tool(applies_to: tuple[str, ...] = ()) -> PolicyRule:
    return PolicyRule(
        id="r2",
        name="no-bash",
        type="deny_tool_call",
        tenant=None,
        config={"patterns": ["bash"]},
        phase="response",
        applies_to=applies_to,
    )


_INPUT_CTX = PolicyContext(
    tenant="t",
    model="gpt-4o",
    prompt_text="My SSN is 123-45-6789.",
    prompt_chars=22,
    stream=False,
)

_TOOL_OUTPUT_CTX = OutputPolicyContext(
    tenant="t",
    model="gpt-4o",
    text="",
    tool_names=["bash"],
    tool_calls=[{"name": "bash", "arguments": "{}"}],
    mcp_targets=[],
    stream=False,
)


# ── Empty applies_to = legacy "all surfaces" behavior ─────────────────


def test_unscoped_rule_fires_on_model_surface() -> None:
    decision = evaluate_policies([_pii_block()], _INPUT_CTX)
    assert decision.verdict == "block"


def test_unscoped_rule_fires_on_tool_surface() -> None:
    decision = evaluate_output_policies(
        [_deny_tool()], _TOOL_OUTPUT_CTX, surfaces=("tool",)
    )
    assert decision.verdict == "block"


# ── Narrowing ─────────────────────────────────────────────────────────


def test_model_scoped_rule_skipped_on_tool_surface() -> None:
    """A PII rule scoped to the model surface must NOT fire when
    the evaluation covers only a tool invocation."""
    rule = _pii_block(applies_to=("model",))
    ctx = OutputPolicyContext(
        tenant="t",
        model="gpt-4o",
        text="SSN 123-45-6789 in a tool result",
        tool_names=[],
        tool_calls=[],
        mcp_targets=[],
        stream=False,
    )
    decision = evaluate_output_policies([rule], ctx, surfaces=("tool",))
    assert decision.verdict == "allow"


def test_tool_scoped_rule_skipped_on_model_prompt() -> None:
    rule = _pii_block(applies_to=("tool",))
    decision = evaluate_policies([rule], _INPUT_CTX)  # surfaces=("model",)
    assert decision.verdict == "allow"


def test_tool_scoped_rule_fires_on_tool_surface() -> None:
    rule = _deny_tool(applies_to=("tool",))
    decision = evaluate_output_policies(
        [rule], _TOOL_OUTPUT_CTX, surfaces=("tool",)
    )
    assert decision.verdict == "block"


def test_tool_scoped_rule_fires_on_model_output_default_surfaces() -> None:
    """The output phase of a model call covers all three surfaces —
    the model's tool-call requests ARE tool governance, so a
    tool-scoped deny rule fires there."""
    rule = _deny_tool(applies_to=("tool",))
    decision = evaluate_output_policies([rule], _TOOL_OUTPUT_CTX)
    assert decision.verdict == "block"


def test_mcp_scoped_rule_skipped_on_plain_tool_surface() -> None:
    rule = _deny_tool(applies_to=("mcp",))
    decision = evaluate_output_policies(
        [rule], _TOOL_OUTPUT_CTX, surfaces=("tool",)
    )
    assert decision.verdict == "allow"


def test_multi_surface_scope_intersects() -> None:
    rule = _deny_tool(applies_to=("tool", "mcp"))
    decision = evaluate_output_policies(
        [rule], _TOOL_OUTPUT_CTX, surfaces=("mcp",)
    )
    assert decision.verdict == "block"


# ── Wire-shape parser ────────────────────────────────────────────────


def test_to_rule_defaults_applies_to_empty() -> None:
    """Backends that pre-date ``applies_to`` don't ship the field —
    the rule must apply on every surface (legacy behavior)."""
    from egisai._policy_cache import _to_rule

    rule = _to_rule({"id": "x", "name": "n", "type": "pii_scan"})
    assert rule.applies_to == ()


def test_to_rule_parses_applies_to() -> None:
    from egisai._policy_cache import _to_rule

    rule = _to_rule(
        {
            "id": "x",
            "name": "n",
            "type": "pii_scan",
            "applies_to": ["model", "tool"],
        }
    )
    assert rule.applies_to == ("model", "tool")


def test_to_rule_drops_unknown_surfaces() -> None:
    """A future backend may ship surfaces this SDK doesn't know.
    Unknown entries are dropped (rather than crashing or disabling
    the rule) so the rule stays active on the surfaces we DO
    understand — over-application, the safe direction."""
    from egisai._policy_cache import _to_rule

    rule = _to_rule(
        {
            "id": "x",
            "name": "n",
            "type": "pii_scan",
            "applies_to": ["model", "browser"],
        }
    )
    assert rule.applies_to == ("model",)
