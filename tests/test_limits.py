"""``rate_limit`` / ``budget_limit`` policy kinds + the limits counters.

Covers, in order:

1. The ``egisai.policy.limits`` counter module in isolation — local
   sliding-window recording, snapshot merge semantics (drop local
   entries the backend already counted), budget fail-open without a
   snapshot, and garbage-tolerant snapshot parsing.
2. The engine's Phase-1 evaluators (``_rate_limit_match`` /
   ``_budget_limit_match``) through the public
   ``evaluate_policies`` entrypoint.
3. End-to-end enforcement through ``_evaluator.evaluate`` — the same
   path every framework patch takes — including the "blocked calls
   don't extend the lockout" contract.
4. The usage-sync worker lifecycle: starts when a limit rule lands
   in the policy cache, polls ``/v1/sdk/usage``, applies the
   snapshot, and never starts for orgs without limit rules.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from egisai.policy import PolicyContext, PolicyRule, evaluate_policies, limits

AGENT_A = "00000000-0000-0000-0000-00000000000a"
AGENT_B = "00000000-0000-0000-0000-00000000000b"
# The agent id the FakeBackend handshake binds the API key to.
HANDSHAKE_AGENT = "00000000-0000-0000-0000-000000000002"


def _rate_rule(
    max_requests: int,
    window_seconds: int = 60,
    scope: str = "per_agent",
    **extra: Any,
) -> PolicyRule:
    return PolicyRule(
        id="rl1",
        name="rate-cap",
        type="rate_limit",
        tenant=None,
        config={
            "max_requests": max_requests,
            "window_seconds": window_seconds,
            "scope": scope,
            **extra,
        },
    )


def _budget_rule(
    max_usd: float,
    window: str = "daily",
    scope: str = "per_agent",
) -> PolicyRule:
    return PolicyRule(
        id="bl1",
        name="budget-cap",
        type="budget_limit",
        tenant=None,
        config={"max_usd": max_usd, "window": window, "scope": scope},
    )


def _ctx(agent_id: str = AGENT_A) -> PolicyContext:
    return PolicyContext(
        tenant="",
        model="gpt-test",
        prompt_text="hello",
        prompt_chars=5,
        stream=False,
        agent_id=agent_id,
    )


def _snapshot(
    *,
    agents: dict[str, dict[str, Any]] | None = None,
    org: dict[str, Any] | None = None,
    computed_at: str | None = None,
) -> dict[str, Any]:
    return {
        "computed_at": computed_at
        or datetime.now(UTC).isoformat(),
        "limits_active": True,
        "agents": agents or {},
        "org": org,
    }


# ── 1. limits module unit semantics ─────────────────────────────────


class TestLimitsModule:
    def test_local_only_counting_per_agent_and_org(self) -> None:
        for _ in range(3):
            limits.record_model_call(AGENT_A)
        limits.record_model_call(AGENT_B)

        assert limits.rate_limit_usage(AGENT_A, 60, "per_agent") == 3
        assert limits.rate_limit_usage(AGENT_B, 60, "per_agent") == 1
        # Org bucket aggregates every recorded call.
        assert limits.rate_limit_usage("", 60, "per_org") == 4
        # Unknown agent has no history.
        assert limits.rate_limit_usage("ghost", 60, "per_agent") == 0

    def test_empty_agent_per_agent_scope_counts_zero(self) -> None:
        limits.record_model_call("")
        # Per-agent with no identity can't attribute — reads as 0
        # (the engine skips the rule before ever asking, this is
        # defense in depth).
        assert limits.rate_limit_usage("", 60, "per_agent") == 0
        # …but the org bucket still saw the call.
        assert limits.rate_limit_usage("", 60, "per_org") == 1

    def test_snapshot_merge_drops_already_counted_local_entries(self) -> None:
        # Two calls happen, then a snapshot computed *after* them
        # arrives claiming the backend saw 10 in the last minute.
        limits.record_model_call(AGENT_A)
        limits.record_model_call(AGENT_A)
        limits.replace_snapshot(
            _snapshot(
                agents={AGENT_A: {"requests": {"60": 10}}},
                org={"requests": {"60": 10}},
            )
        )
        # Local entries at/before computed_at were dropped — the
        # effective count is the snapshot's, not snapshot + locals.
        assert limits.rate_limit_usage(AGENT_A, 60, "per_agent") == 10

        # A call made after the snapshot bridges on top of it.
        limits.record_model_call(AGENT_A)
        assert limits.rate_limit_usage(AGENT_A, 60, "per_agent") == 11

    def test_budget_fails_open_without_snapshot(self) -> None:
        assert limits.budget_usage_usd(AGENT_A, "daily", "per_agent") is None

    def test_budget_reads_snapshot_spend(self) -> None:
        limits.replace_snapshot(
            _snapshot(
                agents={
                    AGENT_A: {
                        "requests": {},
                        "spend_usd": {"daily": "12.50", "monthly": "99.00"},
                    }
                },
                org={"spend_usd": {"daily": "40.00"}},
            )
        )
        assert limits.budget_usage_usd(AGENT_A, "daily", "per_agent") == 12.50
        assert limits.budget_usage_usd(AGENT_A, "monthly", "per_agent") == 99.00
        assert limits.budget_usage_usd("", "daily", "per_org") == 40.00
        # Agent known, window missing ⇒ 0 spend (not fail-open —
        # the snapshot IS present, it just has nothing booked).
        assert limits.budget_usage_usd(AGENT_A, "weekly", "per_agent") == 0.0
        # Unknown agent with a snapshot present ⇒ 0 spend.
        assert limits.budget_usage_usd(AGENT_B, "daily", "per_agent") == 0.0

    def test_garbage_snapshot_is_ignored(self) -> None:
        limits.record_model_call(AGENT_A)
        limits.replace_snapshot(None)  # type: ignore[arg-type]
        limits.replace_snapshot({"computed_at": "not-a-date"})
        limits.replace_snapshot({"agents": "nope"})  # no computed_at
        # Local state survives, no snapshot installed.
        assert limits.rate_limit_usage(AGENT_A, 60, "per_agent") == 1
        assert limits.budget_usage_usd(AGENT_A, "daily", "per_agent") is None

    def test_clear_resets_everything(self) -> None:
        limits.record_model_call(AGENT_A)
        limits.replace_snapshot(_snapshot(org={"requests": {"60": 5}}))
        limits.clear()
        assert limits.rate_limit_usage(AGENT_A, 60, "per_agent") == 0
        assert limits.budget_usage_usd(AGENT_A, "daily", "per_agent") is None


# ── 2. Engine Phase-1 evaluators ────────────────────────────────────


class TestRateLimitEngine:
    def test_blocks_at_threshold(self) -> None:
        for _ in range(3):
            limits.record_model_call(AGENT_A)
        decision = evaluate_policies([_rate_rule(3)], _ctx())
        assert decision.verdict == "block"
        assert decision.reason_code == "rate_limit_exceeded"
        assert decision.matched_policy == "rate-cap"
        assert "3 of 3" in (decision.message or "")

    def test_allows_below_threshold(self) -> None:
        limits.record_model_call(AGENT_A)
        decision = evaluate_policies([_rate_rule(3)], _ctx())
        assert decision.verdict == "allow"

    def test_per_agent_counters_are_isolated(self) -> None:
        for _ in range(5):
            limits.record_model_call(AGENT_A)
        decision = evaluate_policies([_rate_rule(3)], _ctx(AGENT_B))
        assert decision.verdict == "allow"

    def test_per_agent_rule_skips_unknown_identity(self) -> None:
        for _ in range(5):
            limits.record_model_call(AGENT_A)
        # No resolvable identity ⇒ fail open (same posture as the
        # pause gate: per-agent verbs need an agent).
        decision = evaluate_policies([_rate_rule(1)], _ctx(agent_id=""))
        assert decision.verdict == "allow"

    def test_per_org_scope_counts_every_agent(self) -> None:
        limits.record_model_call(AGENT_A)
        limits.record_model_call(AGENT_B)
        decision = evaluate_policies(
            [_rate_rule(2, scope="per_org")], _ctx(AGENT_A)
        )
        assert decision.verdict == "block"

    def test_inert_without_positive_max(self) -> None:
        limits.record_model_call(AGENT_A)
        for bad in (0, -5, None, "nope"):
            decision = evaluate_policies(
                [_rate_rule(bad)],  # type: ignore[arg-type]
                _ctx(),
            )
            assert decision.verdict == "allow", f"max_requests={bad!r}"

    def test_unsupported_window_snaps_up(self) -> None:
        # 120 s isn't in the closed set — snaps up to 3600 (strict
        # direction). The count is the same either way here; the
        # assertion is simply "doesn't crash + still enforces".
        for _ in range(2):
            limits.record_model_call(AGENT_A)
        decision = evaluate_policies(
            [_rate_rule(2, window_seconds=120)], _ctx()
        )
        assert decision.verdict == "block"

    def test_respects_response_phase_scoping(self) -> None:
        # phase="response" ⇒ the rule never runs on the input side,
        # so it cannot block a prompt.
        for _ in range(5):
            limits.record_model_call(AGENT_A)
        rule = PolicyRule(
            id="rl2",
            name="rate-cap-response",
            type="rate_limit",
            tenant=None,
            config={"max_requests": 1, "window_seconds": 60},
            phase="response",
        )
        decision = evaluate_policies([rule], _ctx())
        assert decision.verdict == "allow"


class TestBudgetLimitEngine:
    def test_blocks_when_spend_reaches_cap(self) -> None:
        limits.replace_snapshot(
            _snapshot(
                agents={AGENT_A: {"spend_usd": {"daily": "50.00"}}},
            )
        )
        decision = evaluate_policies([_budget_rule(50.0)], _ctx())
        assert decision.verdict == "block"
        assert decision.reason_code == "budget_exceeded"
        assert "$50.0000" in (decision.message or "")

    def test_allows_below_cap(self) -> None:
        limits.replace_snapshot(
            _snapshot(
                agents={AGENT_A: {"spend_usd": {"daily": "49.99"}}},
            )
        )
        decision = evaluate_policies([_budget_rule(50.0)], _ctx())
        assert decision.verdict == "allow"

    def test_fails_open_without_snapshot(self) -> None:
        decision = evaluate_policies([_budget_rule(0.01)], _ctx())
        assert decision.verdict == "allow"

    def test_per_org_scope(self) -> None:
        limits.replace_snapshot(
            _snapshot(org={"spend_usd": {"monthly": "1000.00"}})
        )
        decision = evaluate_policies(
            [_budget_rule(500.0, window="monthly", scope="per_org")],
            _ctx(),
        )
        assert decision.verdict == "block"

    def test_unknown_window_defaults_to_monthly(self) -> None:
        limits.replace_snapshot(
            _snapshot(
                agents={AGENT_A: {"spend_usd": {"monthly": "10.00"}}},
            )
        )
        decision = evaluate_policies(
            [_budget_rule(5.0, window="hourly")], _ctx()
        )
        assert decision.verdict == "block"

    def test_inert_without_positive_cap(self) -> None:
        limits.replace_snapshot(
            _snapshot(
                agents={AGENT_A: {"spend_usd": {"daily": "1000.00"}}},
            )
        )
        for bad in (0, -1, None, "nope"):
            decision = evaluate_policies(
                [_budget_rule(bad)],  # type: ignore[arg-type]
                _ctx(),
            )
            assert decision.verdict == "allow", f"max_usd={bad!r}"


# ── 3. End-to-end through the evaluator gate ────────────────────────


def _init_with_rules(fake_backend: Any, *rules: dict[str, Any]) -> None:
    fake_backend.set_rules(list(rules), etag='"limits-v1"')

    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="limits-test",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        quiet=True,
    )


def _rate_rule_wire(max_requests: int, **cfg: Any) -> dict[str, Any]:
    return {
        "id": "11111111-1111-1111-1111-111111111111",
        "name": "rate-cap",
        "type": "rate_limit",
        "tenant": None,
        "config": {
            "max_requests": max_requests,
            "window_seconds": 60,
            "scope": "per_agent",
            **cfg,
        },
    }


def _evaluate_once() -> Any:
    from egisai._evaluator import InputCall, evaluate

    return evaluate(
        InputCall(
            source="openai",
            target="openai.chat.completions.create",
            model="gpt-test",
            prompt_text="hello world",
        )
    )


def test_evaluator_enforces_rate_limit_end_to_end(fake_backend: Any) -> None:
    """Two allowed calls, then the third is refused — through the
    exact gate every framework patch calls."""
    _init_with_rules(fake_backend, _rate_rule_wire(2))

    assert _evaluate_once().verdict == "allow"
    assert _evaluate_once().verdict == "allow"
    third = _evaluate_once()
    assert third.verdict == "block"
    assert third.reason_code == "rate_limit_exceeded"


def test_blocked_calls_do_not_extend_the_lockout(fake_backend: Any) -> None:
    """Refused calls are not recorded — a retry loop can't push its
    own usage further above the limit."""
    _init_with_rules(fake_backend, _rate_rule_wire(1))

    assert _evaluate_once().verdict == "allow"
    for _ in range(5):
        assert _evaluate_once().verdict == "block"
    # Only the single allowed call is on the books.
    assert limits.rate_limit_usage(HANDSHAKE_AGENT, 60, "per_agent") == 1


def test_tool_surface_evaluations_do_not_count(fake_backend: Any) -> None:
    """Per-tool input gates re-enter ``evaluate`` with narrowed
    surfaces; they must not double-count the parent model call."""
    _init_with_rules(fake_backend, _rate_rule_wire(10))

    from egisai._evaluator import InputCall, evaluate

    evaluate(
        InputCall(
            source="claude_agent_sdk",
            target="tool",
            model="claude-test",
            prompt_text="{}",
            surfaces=("tool",),
        )
    )
    assert limits.rate_limit_usage(HANDSHAKE_AGENT, 60, "per_agent") == 0


def test_rate_limit_scoped_to_other_agent_does_not_fire(
    fake_backend: Any,
) -> None:
    rule = _rate_rule_wire(1)
    rule["agent_ids"] = [AGENT_B]  # not the handshake agent
    _init_with_rules(fake_backend, rule)

    assert _evaluate_once().verdict == "allow"
    assert _evaluate_once().verdict == "allow"


# ── 4. Usage-sync worker lifecycle ──────────────────────────────────


def _wait_until(predicate: Any, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_sync_worker_polls_and_applies_snapshot(
    fake_backend: Any, monkeypatch: Any
) -> None:
    monkeypatch.setenv("EGISAI_USAGE_SYNC_SECONDS", "0.05")
    fake_backend.usage_response = _snapshot(
        agents={HANDSHAKE_AGENT: {"spend_usd": {"daily": "7.77"}}},
    )
    _init_with_rules(fake_backend, _rate_rule_wire(100))

    assert _wait_until(lambda: fake_backend.usage_calls >= 1), (
        "usage-sync worker never polled /v1/sdk/usage"
    )
    assert _wait_until(
        lambda: limits.budget_usage_usd(
            HANDSHAKE_AGENT, "daily", "per_agent"
        )
        == 7.77
    ), "snapshot was fetched but never applied"


def test_sync_worker_not_started_without_limit_rules(
    fake_backend: Any, monkeypatch: Any
) -> None:
    monkeypatch.setenv("EGISAI_USAGE_SYNC_SECONDS", "0.05")
    fake_backend.usage_response = _snapshot()
    _init_with_rules(
        fake_backend,
        {
            "id": "22222222-2222-2222-2222-222222222222",
            "name": "regex",
            "type": "deny_regex",
            "tenant": None,
            "config": {"pattern": "x"},
        },
    )
    time.sleep(0.2)
    assert fake_backend.usage_calls == 0


def test_sync_worker_survives_404_from_older_backend(
    fake_backend: Any, monkeypatch: Any
) -> None:
    """A backend without ``/v1/sdk/usage`` (404) leaves the SDK in
    local-only counting mode — enforcement still works in-process."""
    monkeypatch.setenv("EGISAI_USAGE_SYNC_SECONDS", "0.05")
    fake_backend.usage_response = None  # endpoint 404s
    _init_with_rules(fake_backend, _rate_rule_wire(2))

    assert _wait_until(lambda: fake_backend.usage_calls >= 1)
    assert _evaluate_once().verdict == "allow"
    assert _evaluate_once().verdict == "allow"
    assert _evaluate_once().verdict == "block"
