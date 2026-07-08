"""``PolicyRule.phase`` controls which side of a call a rule fires on.

Three phase values (call-relative vocabulary, 0.32.0):

- ``"request"``  — only ``evaluate_policies`` (inbound side)
- ``"response"`` — only ``evaluate_output_policies`` (outbound side)
- ``"both"``     — runs on whichever side the rule's type supports
                   (the default; preserves pre-0.12.4 Behavior for
                   older platform responses that don't carry the
                   field at all)

The legacy model-centric spellings (``pre_model`` / ``post_model``)
are still accepted on the wire and normalized by
``_policy_cache._to_rule`` so an un-upgraded backend keeps working.
"""

from __future__ import annotations

from egisai.policy import (
    OutputPolicyContext,
    PolicyContext,
    PolicyRule,
    evaluate_output_policies,
    evaluate_policies,
)


def _semantic_guard(phase: str) -> PolicyRule:
    return PolicyRule(
        id="r1",
        name=f"semantic-{phase}",
        type="semantic_guard",
        tenant=None,
        config={"intents": ["jailbreak"]},
        phase=phase,
    )


def _pii_scan(phase: str) -> PolicyRule:
    return PolicyRule(
        id="r2",
        name=f"pii-{phase}",
        type="pii_scan",
        tenant=None,
        config={"action": "block"},
        phase=phase,
    )


def _deny_output_regex(phase: str) -> PolicyRule:
    return PolicyRule(
        id="r3",
        name=f"deny-out-{phase}",
        type="deny_output_regex",
        tenant=None,
        config={"pattern": r"sk-[A-Za-z0-9]{16,}"},
        phase=phase,
    )


_INPUT_CTX = PolicyContext(
    tenant="t",
    model="gpt-4o",
    prompt_text="My SSN is 123-45-6789.",
    prompt_chars=22,
    stream=False,
)

_OUTPUT_CTX = OutputPolicyContext(
    tenant="t",
    model="gpt-4o",
    text="Here's the key sk-abcdefghijklmnopqr",
    tool_names=[],
    tool_calls=[],
    mcp_targets=[],
    stream=False,
)


# ── Request side ──────────────────────────────────────────────────────


def test_request_rule_runs_on_request_side() -> None:
    decision = evaluate_policies([_pii_scan("request")], _INPUT_CTX)
    assert decision.verdict == "block"
    assert decision.matched_policy == "pii-request"


def test_response_rule_skipped_on_request_side() -> None:
    """A rule scoped to ``response`` MUST NOT fire on the input phase.

    This is the central guarantee of phase scoping: the operator's
    "only enforce on the response" intent is honored even when the
    rule's *type* is technically valid on the input side
    (``semantic_guard``).
    """
    decision = evaluate_policies([_semantic_guard("response")], _INPUT_CTX)
    assert decision.verdict == "allow"


def test_both_phase_rule_runs_on_request_side() -> None:
    decision = evaluate_policies([_pii_scan("both")], _INPUT_CTX)
    assert decision.verdict == "block"


# ── Response side ─────────────────────────────────────────────────────


def test_response_rule_runs_on_response_side() -> None:
    decision = evaluate_output_policies(
        [_deny_output_regex("response")], _OUTPUT_CTX
    )
    assert decision.verdict == "block"


def test_request_rule_skipped_on_response_side() -> None:
    """A rule scoped to ``request`` MUST NOT fire on the output
    phase, even if it could (``semantic_guard`` is supported on
    both sides at the type level).
    """
    decision = evaluate_output_policies(
        [_semantic_guard("request")],
        OutputPolicyContext(
            tenant="t",
            model="gpt-4o",
            text="malicious payload",
            tool_names=[],
            tool_calls=[],
            mcp_targets=[],
            stream=False,
        ),
    )
    assert decision.verdict == "allow"


def test_both_phase_rule_runs_on_response_side() -> None:
    decision = evaluate_output_policies(
        [_deny_output_regex("both")], _OUTPUT_CTX
    )
    assert decision.verdict == "block"


# ── Wire-shape parser: defaults + legacy-spelling normalization ──────


def test_to_rule_defaults_to_both_when_phase_field_absent() -> None:
    """Older platform responses don't carry ``phase`` — the SDK
    must default to ``"both"`` so each rule fires on whichever side
    its type supports (the pre-0.12.4 Behavior).
    """
    from egisai._policy_cache import _to_rule

    rule = _to_rule({"id": "x", "name": "n", "type": "pii_scan"})
    assert rule.phase == "both"


def test_to_rule_rejects_garbage_phase_values() -> None:
    """A malformed wire payload (e.g. ``"phase": "before"``) must
    fall back to ``"both"`` rather than crash the rule loader."""
    from egisai._policy_cache import _to_rule

    rule = _to_rule({"id": "x", "name": "n", "type": "pii_scan", "phase": "before"})
    assert rule.phase == "both"


def test_to_rule_preserves_explicit_phase() -> None:
    from egisai._policy_cache import _to_rule

    for phase in ("request", "response", "both"):
        rule = _to_rule(
            {"id": "x", "name": "n", "type": "pii_scan", "phase": phase}
        )
        assert rule.phase == phase


def test_to_rule_normalizes_legacy_phase_spellings() -> None:
    """Backends that pre-date the vocabulary rename ship the
    model-centric spellings; the parser maps them to the canonical
    call-relative ones so the engine only ever sees one vocabulary.
    """
    from egisai._policy_cache import _to_rule

    for legacy, canonical in (
        ("pre_model", "request"),
        ("post_model", "response"),
    ):
        rule = _to_rule(
            {"id": "x", "name": "n", "type": "pii_scan", "phase": legacy}
        )
        assert rule.phase == canonical


def test_legacy_spelling_round_trips_through_engine() -> None:
    """End-to-end: a legacy-spelled wire rule still enforces on the
    correct side after normalization."""
    from egisai._policy_cache import _to_rule

    rule = _to_rule(
        {
            "id": "x",
            "name": "legacy-pii",
            "type": "pii_scan",
            "config": {"action": "block"},
            "phase": "pre_model",
        }
    )
    decision = evaluate_policies([rule], _INPUT_CTX)
    assert decision.verdict == "block"
    # And it must NOT fire on the response side.
    out = evaluate_output_policies([rule], _OUTPUT_CTX)
    assert out.verdict == "allow"
