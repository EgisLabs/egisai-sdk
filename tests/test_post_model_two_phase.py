"""Post-model evaluation runs deterministic-first, LLM-second.

Same security contract the prompt side honors
(security-and-compliance.mdc §2): Phase 1 (local, deterministic
checks like ``pii_scan``, ``deny_output_regex``,
``deny_tool_call``) must run before Phase 2 (LLM-backed checks
like ``semantic_guard``). When Phase 1 blocks, Phase 2 MUST NOT
execute — no judge call, no token spend, no risk of a
sensitive payload reaching the LLM.

These tests pin that contract on the response side. The pre-model
counterpart lives in ``test_phase_ordering_and_pii_robustness.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from egisai.policy import (
    OutputPolicyContext,
    PolicyRule,
    evaluate_output_policies,
)
from egisai.policy.semantic import SemanticMatch

# ── Test stub: counts judge calls ──────────────────────────────


@dataclass
class _RecordingBlocker:
    """Stub ``SemanticBlocker`` that counts ``.check()`` calls.

    The engine duck-types on the ``check`` method, so we don't need
    to instantiate the real HTTP-backed blocker. ``return_match``
    controls whether the judge says match or no-match — letting one
    stub express both "judge says block" and "judge says allow"
    paths in different tests.
    """

    return_match: bool = False
    calls: int = 0

    def check(
        self, text: str, config: dict[str, object]
    ) -> SemanticMatch | None:
        self.calls += 1
        if not self.return_match:
            return None
        return SemanticMatch(intent="dangerous", similarity=0.95)


# ── Rule + context fixtures ────────────────────────────────────


def _rule(
    type_: str,
    *,
    name: str,
    config: dict | None = None,
    phase: str = "response",
) -> PolicyRule:
    return PolicyRule(
        id="r",
        name=name,
        type=type_,
        tenant=None,
        config=config or {},
        phase=phase,
    )


def _ctx(
    text: str = "Some response.",
    *,
    tool_names: list[str] | None = None,
    tool_calls: list[dict[str, str]] | None = None,
    mcp_targets: list[str] | None = None,
) -> OutputPolicyContext:
    return OutputPolicyContext(
        tenant="t",
        model="gpt-4o",
        text=text,
        tool_names=tool_names or [],
        tool_calls=tool_calls or [],
        mcp_targets=mcp_targets or [],
        stream=False,
    )


# ── Phase 1 block short-circuits Phase 2 ───────────────────────


def test_pii_block_in_phase1_skips_semantic_guard() -> None:
    """A ``pii_scan`` block on the response side MUST short-circuit
    before ``semantic_guard`` is consulted. The judge's ``check``
    counter pinning at zero proves no judge call happened."""
    blocker = _RecordingBlocker(return_match=False)
    pii = _rule(
        "pii_scan", name="block-ssn-out", config={"action": "block"}
    )
    sg = _rule(
        "semantic_guard",
        name="guard-out",
        config={"intents": ["dangerous response"]},
    )

    decision = evaluate_output_policies(
        [pii, sg],
        _ctx(text="Sure, the SSN is 123-45-6789."),
        semantic_blocker=blocker,
    )

    assert decision.verdict == "block"
    assert decision.reason_code == "pii_in_output"
    assert blocker.calls == 0, (
        "semantic_guard MUST NOT run after a Phase 1 block — "
        "any judge invocation here is a token-spend / leak bug."
    )


def test_deny_output_regex_block_in_phase1_skips_semantic_guard() -> None:
    """Same contract for ``deny_output_regex``: a Phase 1 block
    via regex must skip the LLM judge entirely."""
    blocker = _RecordingBlocker(return_match=False)
    regex = _rule(
        "deny_output_regex",
        name="block-secret",
        config={"pattern": r"secret token"},
    )
    sg = _rule(
        "semantic_guard",
        name="guard-out",
        config={"intents": ["leak credentials"]},
    )

    decision = evaluate_output_policies(
        [regex, sg],
        _ctx(text="here is your secret token: abc"),
        semantic_blocker=blocker,
    )

    assert decision.verdict == "block"
    assert decision.reason_code == "output_blocked"
    assert blocker.calls == 0


def test_deny_tool_call_block_in_phase1_skips_semantic_guard() -> None:
    """Tool-call deny is deterministic too — and a block from it
    must short-circuit just like the text detectors."""
    blocker = _RecordingBlocker(return_match=False)
    tool_deny = _rule(
        "deny_tool_call",
        name="no-bash",
        config={"patterns": ["bash"]},
    )
    sg = _rule(
        "semantic_guard",
        name="guard-out",
        config={"intents": ["destructive shell"]},
    )

    decision = evaluate_output_policies(
        [tool_deny, sg],
        _ctx(tool_names=["bash"]),
        semantic_blocker=blocker,
    )

    assert decision.verdict == "block"
    assert decision.reason_code == "tool_call_blocked"
    assert blocker.calls == 0


# ── Phase 1 allow → Phase 2 still runs ────────────────────────


def test_phase1_allow_lets_semantic_guard_run() -> None:
    """When Phase 1 doesn't block, the engine proceeds to Phase 2
    and consults the judge. Whether the judge says block or allow
    is the next test's problem — here we just prove ``check`` was
    called."""
    blocker = _RecordingBlocker(return_match=False)
    pii = _rule(
        "pii_scan", name="block-ssn-out", config={"action": "block"}
    )
    sg = _rule(
        "semantic_guard",
        name="guard-out",
        config={"intents": ["dangerous response"]},
    )

    decision = evaluate_output_policies(
        [pii, sg],
        _ctx(text="No regulated content here."),
        semantic_blocker=blocker,
    )

    assert decision.verdict == "allow"
    assert blocker.calls == 1


def test_phase2_block_still_works_when_phase1_empty() -> None:
    """An org with only ``semantic_guard`` rules on the response
    side still gets Phase 2 evaluation. Empty-Phase-1 must not
    accidentally short-circuit the call."""
    blocker = _RecordingBlocker(return_match=True)
    sg = _rule(
        "semantic_guard",
        name="guard-out",
        config={"intents": ["dangerous response"]},
    )

    decision = evaluate_output_policies(
        [sg], _ctx(text="anything"), semantic_blocker=blocker
    )

    assert decision.verdict == "block"
    assert decision.reason_code == "semantic_blocked"
    assert blocker.calls == 1


def test_phase2_block_combines_with_phase1_records() -> None:
    """A Phase 1 ``pii_scan`` that found nothing (Phase 1 allowed)
    plus a Phase 2 ``semantic_guard`` block: the final decision is
    ``block`` with ``semantic_blocked`` reason. The judge must have
    been consulted exactly once."""
    blocker = _RecordingBlocker(return_match=True)
    pii = _rule(
        "pii_scan",
        name="block-ssn-out",
        config={"action": "block"},
    )
    sg = _rule(
        "semantic_guard",
        name="guard-out",
        config={"intents": ["dangerous response"]},
    )

    decision = evaluate_output_policies(
        [pii, sg],
        _ctx(text="No regulated content here."),
        semantic_blocker=blocker,
    )

    assert decision.verdict == "block"
    assert decision.reason_code == "semantic_blocked"
    assert blocker.calls == 1


# ── Order independence: phase split ignores rule list order ────


def test_phase1_order_does_not_matter_to_short_circuit() -> None:
    """The phase split is type-driven, not list-order-driven. A
    ``semantic_guard`` rule listed FIRST must not pre-empt a
    ``pii_scan`` block that comes after it in the list."""
    blocker = _RecordingBlocker(return_match=False)
    sg = _rule(
        "semantic_guard",
        name="guard-out",
        config={"intents": ["dangerous response"]},
    )
    pii = _rule(
        "pii_scan", name="block-ssn-out", config={"action": "block"}
    )

    decision = evaluate_output_policies(
        [sg, pii],  # <- semantic_guard listed first on purpose
        _ctx(text="The SSN is 123-45-6789."),
        semantic_blocker=blocker,
    )

    assert decision.verdict == "block"
    assert decision.reason_code == "pii_in_output"
    assert blocker.calls == 0, (
        "Even when listed first in the rule list, semantic_guard "
        "must yield to the deterministic Phase 1 short-circuit."
    )
