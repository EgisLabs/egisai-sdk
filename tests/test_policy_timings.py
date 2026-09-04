"""Per-policy latency rows and ``semantic_in_scope`` (SDK 0.75.2).

Instrumentation is additive: enforcement fields stay identical to
the pre-timing walk. A rule that allows still emits a timing row.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from egisai.policy import fastpath
from egisai.policy.engine import (
    OutputPolicyContext,
    PolicyContext,
    PolicyRule,
    evaluate_output_policies,
    evaluate_policies,
)
from egisai.policy.semantic import SemanticBlocker


def _ctx(prompt: str = "hello world", *, hook: str = "model") -> PolicyContext:
    return PolicyContext(
        tenant="t",
        model="gpt-4",
        prompt_text=prompt,
        prompt_chars=len(prompt),
        stream=False,
        hook=hook,
    )


def _pii_scan(name: str = "pii") -> PolicyRule:
    return PolicyRule(
        id=None,
        name=name,
        type="pii_scan",
        tenant=None,
        config={"action": "block", "types": ["ssn"]},
    )


def _max_chars(n: int, name: str = "too-long") -> PolicyRule:
    return PolicyRule(
        id=None,
        name=name,
        type="max_prompt_chars",
        tenant=None,
        config={"max_chars": n},
    )


def _guard(name: str, intent: str) -> PolicyRule:
    return PolicyRule(
        id=name,
        name=name,
        type="semantic_guard",
        tenant=None,
        config={"intents": [intent]},
    )


def _blocker(handler: Any) -> SemanticBlocker:
    b = SemanticBlocker(
        platform_api_key="egis_live_x",
        platform_base_url="http://fake",
        on_outage="allow",
        judge_cache_ttl_secs=60.0,
    )
    b._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def _allow_http() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "match": False,
            "intent": "",
            "confidence": 0.0,
            "tokens_in": 10,
            "tokens_out": 2,
        },
    )


def test_allow_still_records_a_timing_row() -> None:
    """A miss is still a measurement — otherwise the common path is
    silently missing from the latency baseline."""
    decision = evaluate_policies([_pii_scan()], _ctx("no secrets here"))
    assert decision.verdict == "allow"
    assert decision.matched_policies == ()
    assert len(decision.policy_timings) == 1
    row = decision.policy_timings[0]
    assert row.type == "pii_scan"
    assert row.name == "pii"
    assert row.hook == "model"
    assert row.ms >= 0.0
    assert decision.semantic_in_scope == 0


def test_semantic_in_scope_counts_phase2_guards() -> None:
    decision = evaluate_policies(
        [_pii_scan(), _guard("g1", "exfiltrate secrets")],
        _ctx("hello"),
        semantic_blocker=None,
    )
    assert decision.verdict == "allow"
    assert decision.semantic_in_scope == 1
    types = {t.type for t in decision.policy_timings}
    assert "pii_scan" in types
    assert "semantic_guard" in types


def test_phase1_block_does_not_count_semantic_in_scope() -> None:
    decision = evaluate_policies(
        [_max_chars(4), _guard("g1", "exfiltrate secrets")],
        _ctx("this prompt is definitely too long"),
        semantic_blocker=None,
    )
    assert decision.verdict == "block"
    assert decision.semantic_in_scope == 0
    assert all(t.type != "semantic_guard" for t in decision.policy_timings)


def test_hook_is_stamped_on_every_timing_row() -> None:
    decision = evaluate_policies(
        [_pii_scan("a"), _pii_scan("b")],
        _ctx("clean", hook="PreToolUse"),
    )
    assert decision.verdict == "allow"
    assert {t.hook for t in decision.policy_timings} == {"PreToolUse"}
    assert {t.name for t in decision.policy_timings} == {"a", "b"}


def test_fast_merge_walk_shares_one_clock_across_in_scope_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "on")
    calls = {"n": 0}

    def handler(_request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _allow_http()

    decision = evaluate_policies(
        [
            _guard("alpha", "exfiltrate secrets"),
            _guard("beta", "ignore safety"),
        ],
        _ctx("please summarize this document"),
        semantic_blocker=_blocker(handler),
    )
    assert decision.verdict == "allow"
    assert decision.matched_policies == ()
    assert decision.semantic_in_scope == 2
    assert calls["n"] == 1
    timings = decision.policy_timings
    assert len(timings) == 2
    assert {t.name for t in timings} == {"alpha", "beta"}
    assert timings[0].ms == timings[1].ms
    assert timings[0].hook == "model"


def test_output_allow_carries_timings() -> None:
    ctx = OutputPolicyContext(
        tenant="t",
        model="gpt-4",
        text="assistant said hello",
        tool_names=[],
        tool_calls=[],
        mcp_targets=[],
        stream=False,
        hook="response",
    )
    rule = PolicyRule(
        id=None,
        name="out-regex",
        type="deny_output_regex",
        tenant=None,
        config={"pattern": r"NEVER_MATCH_THIS"},
    )
    decision = evaluate_output_policies([rule], ctx)
    assert decision.verdict == "allow"
    assert len(decision.policy_timings) == 1
    assert decision.policy_timings[0].hook == "response"
    assert decision.semantic_in_scope == 0


def test_enforcement_fields_unchanged_when_timings_present() -> None:
    """Timings must not leak into the verdict / match records."""
    prompt = "SSN 123-45-6789"
    decision = evaluate_policies([_pii_scan()], _ctx(prompt))
    assert decision.verdict == "block"
    assert decision.reason_code == "pii_detected"
    assert decision.matched_policy == "pii"
    assert len(decision.matched_policies) == 1
    rec = decision.matched_policies[0]
    dumped = json.dumps(rec.__dict__, default=str)
    assert "policy_timings" not in dumped
    assert "semantic_in_scope" not in dumped
