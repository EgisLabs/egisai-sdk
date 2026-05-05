"""Tests for the compositional Phase 1 / Phase 2 policy engine.

The engine now walks every policy (instead of short-circuiting on
the first match) and synthesizes a single decision via
``_synthesize_decision``. ``decision.matched_policies`` carries
the full audit trail; the legacy ``decision.matched_policy`` is
the *primary* — the rule whose verdict became the final verdict.

Coverage: every interaction the SOC 2 / ISO 27001 evidence trail
asks about — multiple sanitize policies merging kinds, a block
overriding a sanitize, the sanitize match still recorded, Phase 2
block overriding Phase 1 sanitize, the legacy ``matched_policy``
field correctly identifying the primary.
"""

from __future__ import annotations

from typing import Any

from egisai.policy import (
    PolicyContext,
    PolicyRule,
    evaluate_policies,
)

# ── Fixture builders ───────────────────────────────────────────────


def _ctx(prompt: str = "hello", model: str = "gpt-4o-mini") -> PolicyContext:
    return PolicyContext(
        tenant="acme",
        model=model,
        prompt_text=prompt,
        prompt_chars=len(prompt),
        stream=False,
    )


def _pii(
    name: str,
    *,
    action: str,
    kinds: list[str] | None = None,
    mask_char: str = "#",
    threshold: float = 0.3,
) -> PolicyRule:
    cfg: dict[str, Any] = {"action": action, "threshold": threshold}
    if kinds is not None:
        cfg["kinds"] = kinds
    if mask_char != "#":
        cfg["mask_char"] = mask_char
    return PolicyRule(id=None, name=name, type="pii_scan", tenant=None, config=cfg)


def _max_chars(name: str, n: int) -> PolicyRule:
    return PolicyRule(
        id=None,
        name=name,
        type="max_prompt_chars",
        tenant=None,
        config={"max_chars": n},
    )


def _allow_model(name: str, models: list[str]) -> PolicyRule:
    return PolicyRule(
        id=None,
        name=name,
        type="allow_model",
        tenant=None,
        config={"models": models},
    )


# ── Compositional sanitize ─────────────────────────────────────────


def test_two_sanitize_policies_merge_kinds() -> None:
    # Policy A sanitizes SSN; policy B sanitizes credit_card.
    # Prompt has both. Final decision: sanitize, kinds union of
    # {ssn, credit_card}, both matches recorded.
    prompt = "my SSN is 123-45-6789 and card 4111-1111-1111-1111"
    rules = [
        _pii("sanitize-ssn", action="sanitize", kinds=["ssn"]),
        _pii("sanitize-cc", action="sanitize", kinds=["credit_card"]),
    ]
    d = evaluate_policies(rules, _ctx(prompt))
    assert d.verdict == "sanitize"
    # Union: both kinds present (order is encounter order).
    assert "ssn" in d.sanitize_kinds
    assert "credit_card" in d.sanitize_kinds
    # Audit: both records.
    names = [r.name for r in d.matched_policies]
    assert names == ["sanitize-ssn", "sanitize-cc"]
    # Primary = first match in priority order.
    assert d.matched_policy == "sanitize-ssn"


def test_first_sanitize_policy_wins_mask_char() -> None:
    # When two sanitize policies disagree on mask_char, the FIRST
    # in priority order wins. Deterministic for operators.
    prompt = "ssn 123-45-6789 cc 4111-1111-1111-1111"
    rules = [
        _pii("first-X", action="sanitize", kinds=["ssn"], mask_char="X"),
        _pii("second-0", action="sanitize", kinds=["credit_card"], mask_char="0"),
    ]
    d = evaluate_policies(rules, _ctx(prompt))
    assert d.verdict == "sanitize"
    assert d.sanitize_mask_char == "X"


# ── Block beats sanitize, but both are recorded ────────────────────


def test_block_overrides_sanitize_but_both_recorded() -> None:
    # Block credit_card (priority 1, picked because it appears first
    # in the list passed in). Sanitize SSN runs after. Final
    # verdict = block; matched_policy = the block; matched_policies
    # = both, in encounter order. Operator sees in the modal:
    # "credit_card was going to be blocked AND ssn would have been
    # masked — both rules saw the data."
    prompt = "card 4111-1111-1111-1111 and ssn 123-45-6789"
    rules = [
        _pii("block-cc", action="block", kinds=["credit_card"]),
        _pii("sanitize-ssn", action="sanitize", kinds=["ssn"]),
    ]
    d = evaluate_policies(rules, _ctx(prompt))
    assert d.verdict == "block"
    assert d.matched_policy == "block-cc"
    names = [r.name for r in d.matched_policies]
    verdicts = [r.verdict for r in d.matched_policies]
    assert names == ["block-cc", "sanitize-ssn"]
    assert verdicts == ["block", "sanitize"]


def test_first_block_wins_when_multiple_blocks() -> None:
    # Two block policies both match. Final primary = first in list.
    # Both recorded.
    prompt = "x" * 5000
    rules = [
        _max_chars("max-1k", 1000),     # blocks: too large
        _max_chars("max-2k", 2000),     # would also block
    ]
    d = evaluate_policies(rules, _ctx(prompt))
    assert d.verdict == "block"
    assert d.matched_policy == "max-1k"
    assert [r.name for r in d.matched_policies] == ["max-1k", "max-2k"]


# ── Allow path ─────────────────────────────────────────────────────


def test_allow_when_no_policies_match() -> None:
    rules = [
        _pii("block-cc", action="block", kinds=["credit_card"]),
        _allow_model("approved", models=["gpt-4o-mini"]),
    ]
    d = evaluate_policies(rules, _ctx("hello world"))
    assert d.verdict == "allow"
    assert d.matched_policy is None
    assert d.matched_policies == ()


# ── Backward-compat: matched_policy still set ──────────────────────


def test_matched_policy_singular_is_set_to_primary_block() -> None:
    rules = [
        _pii("block-cc", action="block", kinds=["credit_card"]),
        _pii("sanitize-ssn", action="sanitize", kinds=["ssn"]),
    ]
    d = evaluate_policies(rules, _ctx("card 4111-1111-1111-1111"))
    assert d.verdict == "block"
    assert d.matched_policy == "block-cc"


def test_matched_policy_singular_is_set_to_primary_sanitize() -> None:
    rules = [
        _pii("sanitize-A", action="sanitize", kinds=["ssn"]),
        _pii("sanitize-B", action="sanitize", kinds=["email"]),
    ]
    d = evaluate_policies(
        rules,
        _ctx("123-45-6789 a@b.com"),
    )
    assert d.verdict == "sanitize"
    assert d.matched_policy == "sanitize-A"


# ── Phase 2 block overrides Phase 1 sanitize ───────────────────────
#
# Phase 2 (semantic_guard) needs a SemanticBlocker stub. We use a
# small fake that says "yes, blocked" for any text containing a
# trigger substring — sufficient to exercise the integration
# without spinning up the real LLM judge.


class _FakeSemanticMatch:
    def __init__(self, intent: str) -> None:
        self.intent = intent


class _FakeSemanticBlocker:
    """Returns a match whenever ``trigger`` substring appears."""

    def __init__(self, trigger: str) -> None:
        self.trigger = trigger

    def check(self, text: str, _config: dict) -> _FakeSemanticMatch | None:
        return _FakeSemanticMatch(intent="dangerous") if self.trigger in text else None


def test_phase2_block_overrides_phase1_sanitize_records_both() -> None:
    # Phase 1 sanitizes SSN. Phase 2's LLM judge sees the masked
    # text and decides it's still injection — block. Final verdict
    # = block; matched_policies includes the sanitize from P1 + the
    # block from P2.
    prompt = "ssn 123-45-6789 please-block-this"
    rules = [
        _pii("sanitize-ssn", action="sanitize", kinds=["ssn"]),
        PolicyRule(
            id=None,
            name="guard-injection",
            type="semantic_guard",
            tenant=None,
            config={"intents": ["dangerous"], "message": "blocked by judge"},
        ),
    ]
    d = evaluate_policies(
        rules,
        _ctx(prompt),
        semantic_blocker=_FakeSemanticBlocker(trigger="please-block"),
    )
    assert d.verdict == "block"
    assert d.matched_policy == "guard-injection"
    names = [r.name for r in d.matched_policies]
    assert names == ["sanitize-ssn", "guard-injection"]


def test_phase1_block_short_circuits_phase2() -> None:
    # Phase 1 blocks → Phase 2 LLM judge never runs. The fake
    # blocker would have matched if called, but the guard policy
    # is never reached. Audit reflects only the Phase 1 match.
    prompt = "ssn 123-45-6789 please-block-this"
    rules = [
        _pii("block-ssn", action="block", kinds=["ssn"]),
        PolicyRule(
            id=None,
            name="guard-injection",
            type="semantic_guard",
            tenant=None,
            config={"intents": ["dangerous"], "message": "judge"},
        ),
    ]
    d = evaluate_policies(
        rules,
        _ctx(prompt),
        semantic_blocker=_FakeSemanticBlocker(trigger="please-block"),
    )
    assert d.verdict == "block"
    assert d.matched_policy == "block-ssn"
    # CRITICAL — the LLM judge was NOT consulted (PII never left
    # the SDK boundary). Only the Phase 1 match in the audit row.
    assert [r.name for r in d.matched_policies] == ["block-ssn"]
