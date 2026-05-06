"""Phase-symmetric rule evaluation (0.12.5).

The phase picker is fully open: every rule type accepts every
phase. The engine evaluates each rule on whichever side it has
meaningful signals for. These tests pin the symmetry contract:

- Text-detector types (``pii_scan``, ``deny_regex``,
  ``deny_output_regex``, ``max_prompt_chars``, ``allow_model``,
  ``semantic_guard``) fire on either side, with side-specific
  reason codes so audit narratives can phrase the outcome
  correctly.
- Structural response-side types (``deny_tool_call``,
  ``deny_bash_command``, ``deny_mcp_call``) silently no-op on the
  prompt side (no tool / MCP signals there yet) and fire normally
  on the response side.
- ``pii_scan`` with ``action="sanitize"`` is honored on the
  prompt side and coerced to block on the response side
  (sanitization isn't wired through provider responses).
"""

from __future__ import annotations

from egisai.policy import (
    OutputPolicyContext,
    PolicyContext,
    PolicyRule,
    evaluate_output_policies,
    evaluate_policies,
)


def _rule(
    type_: str,
    *,
    phase: str = "both",
    name: str | None = None,
    config: dict | None = None,
) -> PolicyRule:
    return PolicyRule(
        id="r",
        name=name or f"{type_}-{phase}",
        type=type_,
        tenant=None,
        config=config or {},
        phase=phase,
    )


def _input_ctx(text: str = "Hello world!") -> PolicyContext:
    return PolicyContext(
        tenant="t",
        model="gpt-4o",
        prompt_text=text,
        prompt_chars=len(text),
        stream=False,
    )


def _output_ctx(
    text: str = "Hi.",
    *,
    tool_names: list[str] | None = None,
    tool_calls: list[dict[str, str]] | None = None,
    mcp_targets: list[str] | None = None,
    model: str = "gpt-4o",
) -> OutputPolicyContext:
    return OutputPolicyContext(
        tenant="t",
        model=model,
        text=text,
        tool_names=tool_names or [],
        tool_calls=tool_calls or [],
        mcp_targets=mcp_targets or [],
        stream=False,
    )


# ── pii_scan: prompt side keeps existing semantics ────────────────────


def test_pii_scan_prompt_side_blocks() -> None:
    rule = _rule("pii_scan", phase="pre_model", config={"action": "block"})
    decision = evaluate_policies(
        [rule], _input_ctx("My SSN is 123-45-6789.")
    )
    assert decision.verdict == "block"
    assert decision.reason_code == "pii_detected"


def test_pii_scan_prompt_side_sanitize_honored() -> None:
    rule = _rule(
        "pii_scan", phase="pre_model", config={"action": "sanitize"}
    )
    decision = evaluate_policies(
        [rule], _input_ctx("My SSN is 123-45-6789.")
    )
    assert decision.verdict == "sanitize"
    assert decision.reason_code == "pii_sanitized"


# ── pii_scan: response side, new symmetric path ──────────────────────


def test_pii_scan_response_side_blocks_with_dedicated_reason_code() -> None:
    """Output-side ``pii_scan`` blocks with ``pii_in_output`` so the
    dashboard can render a response-aware narrative."""
    rule = _rule("pii_scan", phase="post_model", config={"action": "block"})
    decision = evaluate_output_policies(
        [rule], _output_ctx(text="The user's SSN is 123-45-6789 by the way.")
    )
    assert decision.verdict == "block"
    assert decision.reason_code == "pii_in_output"


def test_pii_scan_response_side_sanitize_coerced_to_block() -> None:
    """``action="sanitize"`` on the response side coerces to block —
    SDK can't safely mutate provider responses, so the operator's
    intent (catch leaked PII) is preserved by refusing the response.
    """
    rule = _rule(
        "pii_scan", phase="post_model", config={"action": "sanitize"}
    )
    decision = evaluate_output_policies(
        [rule], _output_ctx(text="Sure, the SSN is 123-45-6789.")
    )
    assert decision.verdict == "block"
    assert decision.reason_code == "pii_in_output"


def test_pii_scan_response_clean_text_passes() -> None:
    rule = _rule("pii_scan", phase="post_model", config={"action": "block"})
    decision = evaluate_output_policies(
        [rule], _output_ctx(text="No regulated content here.")
    )
    assert decision.verdict == "allow"


# ── deny_regex: works on either side ─────────────────────────────────


def test_deny_regex_prompt_side_blocks_with_prompt_reason() -> None:
    rule = _rule(
        "deny_regex", phase="pre_model", config={"pattern": r"forbidden"}
    )
    decision = evaluate_policies(
        [rule], _input_ctx("This contains FORBIDDEN content.")
    )
    assert decision.verdict == "block"
    assert decision.reason_code == "prompt_blocked"


def test_deny_regex_response_side_blocks_with_output_reason() -> None:
    """Operators can target ``deny_regex`` on the response (effectively
    duplicating ``deny_output_regex``); the engine uses the
    ``output_blocked`` reason code so the audit narrative reads
    correctly."""
    rule = _rule(
        "deny_regex", phase="post_model", config={"pattern": r"secret"}
    )
    decision = evaluate_output_policies(
        [rule], _output_ctx(text="here is your secret token")
    )
    assert decision.verdict == "block"
    assert decision.reason_code == "output_blocked"


def test_deny_output_regex_on_prompt_side_blocks() -> None:
    """The mirror image: ``deny_output_regex`` set to ``pre_model``
    runs on prompt text just like ``deny_regex`` would."""
    rule = _rule(
        "deny_output_regex", phase="pre_model", config={"pattern": r"badword"}
    )
    decision = evaluate_policies([rule], _input_ctx("contains badword now"))
    assert decision.verdict == "block"
    assert decision.reason_code == "prompt_blocked"


# ── max_prompt_chars: phase-aware reason code ────────────────────────


def test_max_chars_prompt_side_uses_prompt_reason() -> None:
    rule = _rule(
        "max_prompt_chars", phase="pre_model", config={"max_chars": 5}
    )
    decision = evaluate_policies([rule], _input_ctx("abcdefgh"))
    assert decision.verdict == "block"
    assert decision.reason_code == "prompt_too_large"


def test_max_chars_response_side_uses_output_reason() -> None:
    rule = _rule(
        "max_prompt_chars", phase="post_model", config={"max_chars": 5}
    )
    decision = evaluate_output_policies(
        [rule], _output_ctx(text="this is a long response")
    )
    assert decision.verdict == "block"
    assert decision.reason_code == "output_too_large"


# ── allow_model: identical check on both sides ───────────────────────


def test_allow_model_blocks_on_prompt_side() -> None:
    rule = _rule(
        "allow_model", phase="pre_model", config={"models": ["gpt-3.5"]}
    )
    decision = evaluate_policies([rule], _input_ctx())
    assert decision.verdict == "block"
    assert decision.reason_code == "model_not_allowed"


def test_allow_model_blocks_on_response_side() -> None:
    """Same rule on the response side fires identically — the model
    name doesn't change between phases."""
    rule = _rule(
        "allow_model", phase="post_model", config={"models": ["gpt-3.5"]}
    )
    decision = evaluate_output_policies([rule], _output_ctx(model="gpt-4o"))
    assert decision.verdict == "block"
    assert decision.reason_code == "model_not_allowed"


# ── Tool / bash / MCP types: no-op on prompt side ────────────────────


def test_deny_tool_call_on_prompt_side_silently_noops() -> None:
    """The pre-model context doesn't carry tool definitions yet, so
    ``deny_tool_call`` set to pre-only quietly returns allow rather
    than crash the call. The user's freedom to pick any phase
    cannot break the gate."""
    rule = _rule(
        "deny_tool_call",
        phase="pre_model",
        config={"patterns": ["bash"]},
    )
    decision = evaluate_policies([rule], _input_ctx())
    assert decision.verdict == "allow"


def test_deny_bash_command_on_prompt_side_silently_noops() -> None:
    rule = _rule(
        "deny_bash_command",
        phase="pre_model",
        config={"command_patterns": [r"rm\s+-rf"]},
    )
    decision = evaluate_policies([rule], _input_ctx())
    assert decision.verdict == "allow"


def test_deny_mcp_call_on_prompt_side_silently_noops() -> None:
    rule = _rule(
        "deny_mcp_call",
        phase="pre_model",
        config={"patterns": [r"prod"]},
    )
    decision = evaluate_policies([rule], _input_ctx())
    assert decision.verdict == "allow"


# ── Tool / bash / MCP types: still fire on response side ─────────────


def test_deny_tool_call_response_side_still_fires() -> None:
    rule = _rule(
        "deny_tool_call",
        phase="post_model",
        config={"patterns": ["bash"]},
    )
    decision = evaluate_output_policies(
        [rule], _output_ctx(tool_names=["bash"])
    )
    assert decision.verdict == "block"
    assert decision.reason_code == "tool_call_blocked"


def test_deny_bash_command_response_side_still_fires() -> None:
    rule = _rule(
        "deny_bash_command",
        phase="post_model",
        config={"command_patterns": [r"rm\s+-rf"]},
    )
    decision = evaluate_output_policies(
        [rule],
        _output_ctx(
            tool_calls=[{"name": "bash", "arguments": "rm -rf /"}],
        ),
    )
    assert decision.verdict == "block"
    assert decision.reason_code == "bash_command_blocked"


def test_deny_mcp_call_response_side_still_fires() -> None:
    rule = _rule(
        "deny_mcp_call",
        phase="post_model",
        config={"patterns": ["prod"]},
    )
    decision = evaluate_output_policies(
        [rule], _output_ctx(mcp_targets=["prod-finance"])
    )
    assert decision.verdict == "block"
    assert decision.reason_code == "mcp_call_blocked"
