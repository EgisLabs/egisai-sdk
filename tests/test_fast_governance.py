"""Fast-governance mode (EGISAI_FAST_GOVERNANCE): merged judge calls,
windowed judge text, normalized cache keys, and the shadow harness.

The mode's whole promise is "fewer judge questions, provably-equal
answers", so the suite is organised around the equivalence claims:

* OFF (default) is byte-identical to the previous release — the
  legacy walk runs, one call per policy.
* ON asks one merged question per threshold group and attributes the
  verdict back to the policy that owns the cited intent, with the
  same record shape (name/type/reason_code/message) the legacy walk
  produces.
* SHADOW never changes the enforced decision, never books tokens to
  the governed call, and says AGREE/DISAGREE out loud.
* Windowing bounds what the judge reads without ever touching what
  Phase 1 (regex/PII) reads.
* Cache-key normalization collapses opaque ids but never amounts.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from egisai._context import get_policy_usage, reset_policy_usage
from egisai.policy import fastpath
from egisai.policy.engine import (
    OutputPolicyContext,
    PolicyContext,
    PolicyRule,
    evaluate_output_policies,
    evaluate_policies,
)
from egisai.policy.semantic import SemanticBlocker

# ── Harness ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_fastpath_state(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        fastpath.MODE_ENV,
        fastpath.WINDOW_ENV,
        fastpath.NORMALIZE_ENV,
        fastpath.SHADOW_SAMPLE_ENV,
    ):
        monkeypatch.delenv(var, raising=False)
    fastpath.reset_shadow_stats_for_tests()
    fastpath._last_shadow_thread = None


def _blocker(
    handler: Any, *, cache_ttl: float = 60.0, on_outage: str = "allow"
) -> SemanticBlocker:
    b = SemanticBlocker(
        platform_api_key="egis_live_x",
        platform_base_url="http://fake",
        on_outage=on_outage,
        judge_cache_ttl_secs=cache_ttl,
    )
    b._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def _guard(
    name: str,
    intent: str,
    *,
    threshold: float | None = None,
    targets: list[str] | None = None,
    message: str | None = None,
) -> PolicyRule:
    config: dict[str, Any] = {"intents": [intent]}
    if threshold is not None:
        config["threshold"] = threshold
    if targets is not None:
        config["targets"] = targets
    if message is not None:
        config["message"] = message
    return PolicyRule(
        id=name, name=name, type="semantic_guard", tenant=None, config=config
    )


def _ctx(prompt_text: str = "hello") -> PolicyContext:
    return PolicyContext(
        tenant="t",
        model="gpt-4",
        prompt_text=prompt_text,
        prompt_chars=len(prompt_text),
        stream=False,
    )


def _octx(
    text: str = "assistant reply",
    tool_calls: list[dict[str, Any]] | None = None,
) -> OutputPolicyContext:
    return OutputPolicyContext(
        tenant="t",
        model="gpt-4",
        text=text,
        tool_names=[],
        tool_calls=tool_calls or [],
        mcp_targets=[],
        stream=False,
    )


def _allow() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "match": False, "intent": "", "confidence": 0.0,
            "tokens_in": 10, "tokens_out": 2,
        },
    )


def _block(intent: str, confidence: float = 0.99) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "match": True, "intent": intent, "confidence": confidence,
            "tokens_in": 10, "tokens_out": 2,
        },
    )


def _body(request: httpx.Request) -> dict[str, Any]:
    return json.loads(request.read().decode("utf-8"))


# ── Knob parsing ─────────────────────────────────────────────────────


def test_mode_defaults_to_on(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fast governance is the production default since 0.50.0."""
    monkeypatch.delenv(fastpath.MODE_ENV, raising=False)
    assert fastpath.mode() == "on"


def test_invalid_mode_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "turbo")
    assert fastpath.mode() == "on"


def test_mode_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("off", "shadow", "on", " ON ", "Shadow"):
        monkeypatch.setenv(fastpath.MODE_ENV, value)
        assert fastpath.mode() == value.strip().lower()


def test_window_default_and_override(monkeypatch: pytest.MonkeyPatch) -> None:
    assert fastpath.window_chars() == 16_000
    monkeypatch.setenv(fastpath.WINDOW_ENV, "20000")
    assert fastpath.window_chars() == 20_000
    monkeypatch.setenv(fastpath.WINDOW_ENV, "0")
    assert fastpath.window_chars() == 0  # disabled
    monkeypatch.setenv(fastpath.WINDOW_ENV, "50")
    assert fastpath.window_chars() == 1_000  # clamped to the floor
    monkeypatch.setenv(fastpath.WINDOW_ENV, "banana")
    assert fastpath.window_chars() == 16_000


def test_window_text_keeps_short_text_and_tails_long_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert fastpath.window_text("short") == "short"
    monkeypatch.setenv(fastpath.WINDOW_ENV, "1000")
    long = "x" * 5_000 + "THE NEW INSTRUCTION"
    windowed = fastpath.window_text(long)
    assert len(windowed) == 1_000
    assert windowed.endswith("THE NEW INSTRUCTION"), (
        "the newest content (end of transcript) must always survive"
    )
    monkeypatch.setenv(fastpath.WINDOW_ENV, "0")
    assert fastpath.window_text(long) == long


# ── Cache-key normalization ──────────────────────────────────────────


def test_normalize_collapses_opaque_ids() -> None:
    n = fastpath.normalize_for_cache_key
    assert n("customer CUST-9374731 here") == n("customer CUST-5555555 here")
    assert n("session sess-5df0c2933045") == n("session sess-9ab1e4429b77")
    assert (
        n("run 550e8400-e29b-41d4-a716-446655440000")
        == n("run 123e4567-e89b-42d3-a456-426614174000")
    )
    assert n("trace deadbeef1234cafe") == n("trace 1234cafedeadbeef")


def test_normalize_never_touches_amounts_or_names() -> None:
    n = fastpath.normalize_for_cache_key
    # Bare numbers are potential amounts — a $45,000 wire and a
    # $99,000 wire are different governance questions.
    assert n("transfer 45000 EUR") != n("transfer 99000 EUR")
    assert n("send 45,000") != n("send 99,000")
    # No-separator tokens can be meaningful names.
    assert n("pay ALICE123") != n("pay BOB99")
    # PII labels pass through untouched.
    assert n("email <EMAIL> now") == "email <EMAIL> now"


def test_normalized_keys_share_a_verdict_only_when_mode_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(_body(request)["prompt_text"])
        return _allow()

    cfg = {"intents": ["exfiltrate customer data"]}

    monkeypatch.setenv(fastpath.MODE_ENV, "off")
    b = _blocker(handler)
    b.check("look up account CUST-1111 balance", cfg)
    b.check("look up account CUST-2222 balance", cfg)
    assert len(calls) == 2, "mode off ⇒ exact-match keys, two round-trips"

    monkeypatch.setenv(fastpath.MODE_ENV, "on")
    b2 = _blocker(handler)
    b2.check("look up account CUST-1111 balance", cfg)
    b2.check("look up account CUST-2222 balance", cfg)
    assert len(calls) == 3, "mode on ⇒ ids normalized, second is a cache hit"
    # The judge itself always received the RAW text.
    assert calls[-1] == "look up account CUST-1111 balance"

    b2.check("transfer 45000 EUR", cfg)
    b2.check("transfer 99000 EUR", cfg)
    assert len(calls) == 5, "amounts are never collapsed"


def test_normalization_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "on")
    monkeypatch.setenv(fastpath.NORMALIZE_ENV, "0")
    assert fastpath.cache_normalization_enabled() is False
    monkeypatch.delenv(fastpath.NORMALIZE_ENV)
    assert fastpath.cache_normalization_enabled() is True
    monkeypatch.setenv(fastpath.MODE_ENV, "shadow")
    assert fastpath.cache_normalization_enabled() is False


# ── Merged judge call (mode ON, input side) ──────────────────────────


def test_three_guards_become_one_judge_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "on")
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_body(request))
        return _allow()

    policies = [
        _guard("A", "suppress the audit log"),
        _guard("B", "exfiltrate customer data"),
        _guard("C", "pay an unverified beneficiary"),
    ]
    decision = evaluate_policies(policies, _ctx(), _blocker(handler))

    assert decision.verdict == "allow"
    assert len(bodies) == 1, "3 same-threshold guards ⇒ exactly 1 round-trip"
    assert bodies[0]["intents"] == [
        "suppress the audit log",
        "exfiltrate customer data",
        "pay an unverified beneficiary",
    ]
    assert "threshold" not in bodies[0]


def test_merged_block_is_attributed_to_the_owning_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "on")

    def handler(request: httpx.Request) -> httpx.Response:
        return _block("exfiltrate customer data")

    policies = [
        _guard("A", "suppress the audit log"),
        _guard("B", "exfiltrate customer data", message="custom B message"),
        _guard("C", "pay an unverified beneficiary"),
    ]
    decision = evaluate_policies(policies, _ctx("send data out"), _blocker(handler))

    assert decision.verdict == "block"
    assert decision.matched_policy == "B"
    [record] = decision.matched_policies
    assert record.name == "B"
    assert record.type == "semantic_guard"
    assert record.reason_code == "semantic_blocked"
    assert record.message == "custom B message"


def test_unmappable_citation_falls_back_to_first_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The judge cited something not in any intent list — the block is
    kept (never dropped over attribution) and labelled with the
    group's first policy, mirroring the backend's own fallback."""
    monkeypatch.setenv(fastpath.MODE_ENV, "on")

    def handler(request: httpx.Request) -> httpx.Response:
        return _block("something entirely different")

    policies = [_guard("A", "suppress the audit log"), _guard("B", "exfiltrate")]
    decision = evaluate_policies(policies, _ctx(), _blocker(handler))

    assert decision.verdict == "block"
    assert decision.matched_policy == "A"


def test_different_thresholds_stay_in_separate_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "on")
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_body(request))
        return _allow()

    policies = [
        _guard("A", "intent a", threshold=0.6),
        _guard("B", "intent b", threshold=0.6),
        _guard("C", "intent c", threshold=0.9),
        _guard("D", "intent d"),  # defers to platform default
    ]
    evaluate_policies(policies, _ctx(), _blocker(handler))

    assert len(bodies) == 3, "0.6-group, 0.9-group, default-group"
    by_threshold = {b.get("threshold"): b["intents"] for b in bodies}
    assert by_threshold[0.6] == ["intent a", "intent b"]
    assert by_threshold[0.9] == ["intent c"]
    assert by_threshold[None] == ["intent d"]


def test_fast_mode_windows_the_judge_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "on")
    monkeypatch.setenv(fastpath.WINDOW_ENV, "1000")
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_body(request))
        return _allow()

    long_prompt = "old turn content " * 500 + "NEWEST INSTRUCTION"
    evaluate_policies(
        [_guard("A", "intent a")], _ctx(long_prompt), _blocker(handler)
    )

    assert len(bodies) == 1
    sent = bodies[0]["prompt_text"]
    assert len(sent) == 1000
    assert sent.endswith("NEWEST INSTRUCTION")


def test_fast_mode_fails_open_and_closed_like_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "on")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    policies = [_guard("A", "intent a"), _guard("B", "intent b")]

    open_decision = evaluate_policies(policies, _ctx(), _blocker(handler))
    assert open_decision.verdict == "allow"

    closed_decision = evaluate_policies(
        policies, _ctx(), _blocker(handler, on_outage="block")
    )
    assert closed_decision.verdict == "block"


def test_mode_off_still_runs_one_call_per_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The kill switch restores the legacy walk, byte-identical to the
    pre-0.49 release: one judge question per guard."""
    monkeypatch.setenv(fastpath.MODE_ENV, "off")
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_body(request))
        return _allow()

    policies = [_guard("A", "intent a"), _guard("B", "intent b")]
    evaluate_policies(policies, _ctx(), _blocker(handler))

    assert len(bodies) == 2
    assert [b["intents"] for b in bodies] == [["intent a"], ["intent b"]]


# ── Merged judge call (mode ON, output side + tools) ─────────────────


def test_output_side_merges_and_dedupes_tool_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """3 guards × (1 text + 4 tool calls, one duplicated) was 15 legacy
    round-trips; fast mode asks 4 questions (1 text + 3 unique tools)."""
    monkeypatch.setenv(fastpath.MODE_ENV, "on")
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_body(request))
        return _allow()

    policies = [
        _guard("A", "intent a", targets=["text", "tool_calls"]),
        _guard("B", "intent b", targets=["text", "tool_calls"]),
        _guard("C", "intent c", targets=["text", "tool_calls"]),
    ]
    tool_calls = [
        {"name": "ToolSearch", "arguments": '{"q": "rates"}'},
        {"name": "ToolSearch", "arguments": '{"q": "rates"}'},  # duplicate
        {"name": "lookup_customer_account", "arguments": '{"id": 1}'},
        {"name": "answer_product_question", "arguments": '{"q": "tiers"}'},
    ]
    decision = evaluate_output_policies(
        policies,
        _octx("the assistant replied", tool_calls),
        _blocker(handler, cache_ttl=0),
    )

    assert decision.verdict == "allow"
    assert len(bodies) == 4, "1 text question + 3 unique tool questions"
    for b in bodies:
        assert b["intents"] == ["intent a", "intent b", "intent c"]


def test_legacy_output_side_cost_for_comparison(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the pre-fast cost so the 15→4 claim in the docs stays honest."""
    monkeypatch.setenv(fastpath.MODE_ENV, "off")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return _allow()

    policies = [
        _guard("A", "intent a", targets=["text", "tool_calls"]),
        _guard("B", "intent b", targets=["text", "tool_calls"]),
        _guard("C", "intent c", targets=["text", "tool_calls"]),
    ]
    tool_calls = [
        {"name": "ToolSearch", "arguments": '{"q": "rates"}'},
        {"name": "ToolSearch", "arguments": '{"q": "rates"}'},
        {"name": "lookup_customer_account", "arguments": '{"id": 1}'},
        {"name": "answer_product_question", "arguments": '{"q": "tiers"}'},
    ]
    evaluate_output_policies(
        policies,
        _octx("the assistant replied", tool_calls),
        _blocker(handler, cache_ttl=0),
    )
    assert len(calls) == 15


def test_tool_block_names_the_tool_and_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "on")

    def handler(request: httpx.Request) -> httpx.Response:
        body = _body(request)
        if "execute_payment" in body["prompt_text"]:
            return _block("pay an unverified beneficiary")
        return _allow()

    policies = [
        _guard("A", "suppress the audit log", targets=["tool_calls"]),
        _guard("B", "pay an unverified beneficiary", targets=["tool_calls"]),
    ]
    decision = evaluate_output_policies(
        policies,
        _octx("ok", [{"name": "execute_payment", "arguments": '{"iban": "X"}'}]),
        _blocker(handler),
    )

    assert decision.verdict == "block"
    [record] = decision.matched_policies
    assert record.name == "B"
    assert record.reason_code == "semantic_blocked_tool"
    assert "execute_payment" in record.message


def test_at_most_one_record_per_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Text match and tool match for the same policy collapse to one
    record — same as the legacy walk, where ``_semantic_guard_match``
    returns that policy's first match only."""
    monkeypatch.setenv(fastpath.MODE_ENV, "on")

    def handler(request: httpx.Request) -> httpx.Response:
        return _block("intent a")

    policies = [_guard("A", "intent a", targets=["text", "tool_calls"])]
    decision = evaluate_output_policies(
        policies,
        _octx("bad text", [{"name": "bad_tool", "arguments": "{}"}]),
        _blocker(handler),
    )

    assert decision.verdict == "block"
    assert len(decision.matched_policies) == 1
    assert decision.matched_policies[0].reason_code == "semantic_blocked"


# ── Shadow mode ──────────────────────────────────────────────────────


def test_shadow_agrees_and_never_changes_the_decision(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "shadow")
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(_body(request))
        return _allow()

    policies = [_guard("A", "intent a"), _guard("B", "intent b")]
    decision = evaluate_policies(policies, _ctx(), _blocker(handler))
    fastpath.wait_for_shadow()

    assert decision.verdict == "allow"
    # 2 legacy questions + 1 merged shadow question.
    assert len(bodies) == 3
    merged = [b for b in bodies if len(b["intents"]) == 2]
    assert len(merged) == 1
    err = capsys.readouterr().err
    assert "fast-governance shadow (prompt): AGREE" in err
    assert fastpath.shadow_stats()[:2] == (1, 0)


def test_shadow_disagreement_is_loud_but_harmless(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "shadow")

    def handler(request: httpx.Request) -> httpx.Response:
        body = _body(request)
        # Contrived: block ONLY the merged (multi-intent) question, so
        # legacy=allow while fast=block.
        if len(body["intents"]) > 1:
            return _block("intent a")
        return _allow()

    policies = [_guard("A", "intent a"), _guard("B", "intent b")]
    decision = evaluate_policies(policies, _ctx(), _blocker(handler))
    fastpath.wait_for_shadow()

    assert decision.verdict == "allow", "legacy path stays the decider"
    err = capsys.readouterr().err
    assert "DISAGREE" in err
    assert fastpath.shadow_stats()[:2] == (0, 1)


def test_shadow_tokens_never_pollute_the_governed_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "shadow")

    def handler(request: httpx.Request) -> httpx.Response:
        return _allow()

    policies = [_guard("A", "intent a"), _guard("B", "intent b")]
    reset_policy_usage()
    evaluate_policies(policies, _ctx(), _blocker(handler))
    fastpath.wait_for_shadow()

    # 2 legacy calls × (10 in, 2 out); the shadow's merged call spent
    # tokens too but on its own thread-local accumulator.
    assert get_policy_usage() == (20, 4)


def test_shadow_sampling_zero_disables_comparisons(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "shadow")
    monkeypatch.setenv(fastpath.SHADOW_SAMPLE_ENV, "0")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return _allow()

    evaluate_policies([_guard("A", "intent a")], _ctx(), _blocker(handler))
    fastpath.wait_for_shadow()

    assert len(calls) == 1, "legacy only — no shadow round-trip"
    assert "fast-governance shadow" not in capsys.readouterr().err


def test_shadow_output_side_with_tools(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "shadow")

    def handler(request: httpx.Request) -> httpx.Response:
        return _allow()

    policies = [_guard("A", "intent a", targets=["text", "tool_calls"])]
    decision = evaluate_output_policies(
        policies,
        _octx("reply", [{"name": "t1", "arguments": "{}"}]),
        _blocker(handler, cache_ttl=0),
    )
    fastpath.wait_for_shadow()

    assert decision.verdict == "allow"
    assert "fast-governance shadow (output): AGREE" in capsys.readouterr().err


# ── Equivalence sweep: same inputs, same verdicts in both modes ──────


@pytest.mark.parametrize("blocked_intent", [None, "intent a", "intent c"])
def test_on_and_off_agree_on_verdicts(
    monkeypatch: pytest.MonkeyPatch, blocked_intent: str | None
) -> None:
    """For any single-intent verdict the judge can return, both modes
    produce the same enforcement outcome and the same policy name.

    The mock honours the backend contract: a BLOCK always cites an
    intent from the request's own intent list (``_build_verdict``
    canonicalises the citation server-side), so a question that
    doesn't carry the blocked intent is an ALLOW."""
    def handler(request: httpx.Request) -> httpx.Response:
        if blocked_intent is None or blocked_intent not in _body(request)["intents"]:
            return _allow()
        return _block(blocked_intent)

    policies = [
        _guard("A", "intent a"),
        _guard("B", "intent b"),
        _guard("C", "intent c"),
    ]

    legacy = evaluate_policies(policies, _ctx("q"), _blocker(handler))
    monkeypatch.setenv(fastpath.MODE_ENV, "on")
    fast = evaluate_policies(policies, _ctx("q"), _blocker(handler))

    assert fast.verdict == legacy.verdict
    if blocked_intent is not None:
        assert fast.matched_policy == legacy.matched_policy


# ── Shadow DISAGREE diagnostics ──────────────────────────────────────


def test_diagnose_returns_raw_verdict_and_skips_cache() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return _block("intent a", confidence=0.55)

    b = _blocker(handler)
    cfg = {"intents": ["intent a"]}

    first = b.diagnose("question", cfg)
    assert b.diagnose("question", cfg) is not None

    assert first is not None
    assert first["match"] is True
    assert first["confidence"] == 0.55
    assert len(calls) == 2, "diagnosis always asks the judge NOW — no cache"


def test_diagnose_fails_quiet() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    b = _blocker(handler)
    assert b.diagnose("question", {"intents": ["intent a"]}) is None
    assert b.diagnose("", {"intents": ["intent a"]}) is None
    assert b.diagnose("question", {"intents": []}) is None


def test_disagreement_prints_per_question_diagnosis(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """After a DISAGREE, one shadow-diagnosis line per merged question,
    carrying the judge's confidence — numbers and counts only."""
    monkeypatch.setenv(fastpath.MODE_ENV, "shadow")

    def handler(request: httpx.Request) -> httpx.Response:
        body = _body(request)
        if len(body["intents"]) > 1:
            # Merged question: sub-threshold near-miss shape.
            return httpx.Response(
                200,
                json={
                    "match": False, "intent": "", "confidence": 0.62,
                    "tokens_in": 10, "tokens_out": 2,
                },
            )
        return _block("intent a")

    policies = [_guard("A", "intent a"), _guard("B", "intent b")]
    decision = evaluate_policies(policies, _ctx(), _blocker(handler))
    fastpath.wait_for_shadow()

    assert decision.verdict == "block", "legacy path stays the decider"
    err = capsys.readouterr().err
    assert "DISAGREE" in err
    assert "shadow-diagnosis (prompt/text q0)" in err
    assert "confidence=0.62" in err
    assert "policies=2" in err
    assert "intents=2" in err
    # Compliance: neither the prompt nor any intent string is printed.
    assert "intent a" not in err
    assert "hello" not in err


def test_agreement_never_triggers_diagnosis_calls(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "shadow")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return _allow()

    policies = [_guard("A", "intent a"), _guard("B", "intent b")]
    evaluate_policies(policies, _ctx(), _blocker(handler))
    fastpath.wait_for_shadow()

    # 2 legacy + 1 merged shadow — and nothing more.
    assert len(calls) == 3
    assert "shadow-diagnosis" not in capsys.readouterr().err


# ── Bin-packing at the judge endpoint's 16-intent cap ────────────────


def _multi_guard(name: str, n_intents: int) -> PolicyRule:
    return PolicyRule(
        id=name,
        name=name,
        type="semantic_guard",
        tenant=None,
        config={
            "intents": [f"{name} intent {i}" for i in range(n_intents)],
            "threshold": 0.75,
        },
    )


def test_merged_questions_never_exceed_the_backend_intent_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production bug: 5+8+10 = 23 intents merged into one question
    is rejected by the platform's 16-intent schema cap. Bin-packing
    must split it into questions of ≤16 intents covering all 23."""
    monkeypatch.setenv(fastpath.MODE_ENV, "on")
    seen: list[list[str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        intents = _body(request)["intents"]
        seen.append(intents)
        if len(intents) > 16:
            return httpx.Response(422, json={"detail": "too many intents"})
        return _allow()

    policies = [
        _multi_guard("Suppress-audit", 5),
        _multi_guard("Unverified beneficiary block", 8),
        _multi_guard("Data exfiltration guard", 10),
    ]
    decision = evaluate_policies(policies, _ctx(), _blocker(handler))

    assert decision.verdict == "allow"
    assert all(len(intents) <= 16 for intents in seen)
    covered = {i for intents in seen for i in intents}
    assert len(covered) == 23, "every intent must still be asked"
    assert len(seen) == 2, "5+8 fits one bin, 10 goes to the second"


def test_block_attribution_survives_binning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A match in the second bin attributes to its owning policy."""
    monkeypatch.setenv(fastpath.MODE_ENV, "on")

    def handler(request: httpx.Request) -> httpx.Response:
        intents = _body(request)["intents"]
        target = "Data exfiltration guard intent 7"
        if target in intents:
            return _block(target)
        return _allow()

    policies = [
        _multi_guard("Suppress-audit", 5),
        _multi_guard("Unverified beneficiary block", 8),
        _multi_guard("Data exfiltration guard", 10),
    ]
    decision = evaluate_policies(policies, _ctx(), _blocker(handler))

    assert decision.verdict == "block"
    assert decision.matched_policy == "Data exfiltration guard"


def test_small_unions_stay_one_merged_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "on")
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(len(_body(request)["intents"]))
        return _allow()

    policies = [
        _multi_guard("A", 4),
        _multi_guard("B", 6),
        _multi_guard("C", 6),
    ]
    evaluate_policies(policies, _ctx(), _blocker(handler))
    assert calls == [16], "4+6+6 = 16 fits exactly in one question"


def test_single_policy_over_the_cap_sits_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lone 20-intent policy gets its own question — identical to
    what the legacy walk sends for it (no new failure mode)."""
    monkeypatch.setenv(fastpath.MODE_ENV, "on")
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(len(_body(request)["intents"]))
        return _allow()

    policies = [_multi_guard("huge", 20), _multi_guard("small", 3)]
    evaluate_policies(policies, _ctx(), _blocker(handler))
    assert sorted(seen) == [3, 20]


# ── 4xx judge rejections are loud, not "outages" ─────────────────────


def test_judge_4xx_logs_the_status_code(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "rejected"})

    b = _blocker(handler)
    with caplog.at_level("ERROR", logger="egisai"):
        result = b.check("text", {"intents": ["intent a"]})

    assert result is None, "fail-open posture unchanged"
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "HTTP 422" in joined
    assert "REJECTED" in joined


def test_judge_5xx_stays_a_plain_outage(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    b = _blocker(handler)
    with caplog.at_level("WARNING", logger="egisai"):
        b.check("text", {"intents": ["intent a"]})

    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "HTTP 503" in joined
    assert "REJECTED" not in joined
