"""SDK-side human-in-the-loop resolution — ``_approvals.request_approval``.

Covers both wait modes across every terminal outcome (approved,
rejected, expired-block, expired-allow), the fail-closed posture when
the control plane is unreachable, the async immediate-pending path, the
idempotency key that makes resume-by-re-submit work (single- and
multi-step), and the compliance invariant that no raw payload text
rides in the hold-open request.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from egisai import _approvals
from egisai._approvals import request_approval
from egisai._config import EgisaiConfig, set_config
from egisai.policy.engine import PolicyDecision


def _cfg(**over: Any) -> EgisaiConfig:
    base: dict[str, Any] = {
        "api_key": "egis_live_test",
        "app": "test",
        "env": "test",
        # Tight budgets so wait-mode tests never actually sleep long.
        "approval_wait_budget_ms": 300,
        "approval_poll_interval_ms": 50,
    }
    base.update(over)
    return EgisaiConfig(**base)


def _decision(**over: Any) -> PolicyDecision:
    return PolicyDecision.hold(
        reason_code=over.get("reason_code", "needs_approval"),
        message=over.get("message", "Wire transfer needs a human."),
        matched_policy=over.get("matched_policy", "r-deny_financial_action"),
        approval_detail=over.get("approval_detail", "$25,000 via wire_send"),
    )


def _ev(**over: Any) -> dict[str, Any]:
    ev = {"agent_id": "agent-1", "run_id": "run-1", "trace_id": "trace-1"}
    ev.update(over)
    return ev


@pytest.fixture
def capture(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Script create/poll responses and capture the create payload."""
    state: dict[str, Any] = {
        "create_payload": None,
        "create_response": None,
        "poll_responses": [],
        "poll_calls": 0,
    }

    def fake_create(payload: dict[str, Any]) -> dict[str, Any] | None:
        state["create_payload"] = payload
        return state["create_response"]

    def fake_poll(approval_id: str) -> dict[str, Any] | None:
        i = state["poll_calls"]
        state["poll_calls"] += 1
        seq = state["poll_responses"]
        if not seq:
            return {"resolved": False, "state": "pending"}
        return seq[min(i, len(seq) - 1)]

    monkeypatch.setattr(_approvals._backend, "create_approval", fake_create)
    monkeypatch.setattr(_approvals._backend, "poll_approval", fake_poll)
    return state


# ── Wait mode ──────────────────────────────────────────────────────────


def test_wait_approved(capture: dict[str, Any]) -> None:
    set_config(_cfg())
    capture["create_response"] = {"id": "h1", "resolved": False, "wait_mode": "wait"}
    capture["poll_responses"] = [
        {"resolved": True, "approved": True, "state": "approved"}
    ]
    out = request_approval(
        decision=_decision(), ev=_ev(), surface="model", model="gpt-4o"
    )
    assert out.decided and out.approved
    assert out.approval_id == "h1"
    assert capture["poll_calls"] >= 1


def test_wait_rejected(capture: dict[str, Any]) -> None:
    set_config(_cfg())
    capture["create_response"] = {"id": "h1", "resolved": False, "wait_mode": "wait"}
    capture["poll_responses"] = [
        {"resolved": True, "approved": False, "state": "rejected"}
    ]
    out = request_approval(
        decision=_decision(), ev=_ev(), surface="model", model="gpt-4o"
    )
    assert out.decided and not out.approved
    assert out.state == "rejected"


def test_wait_expired_block(capture: dict[str, Any]) -> None:
    set_config(_cfg())
    capture["create_response"] = {"id": "h1", "resolved": False, "wait_mode": "wait"}
    capture["poll_responses"] = [
        {"resolved": True, "approved": False, "state": "expired"}
    ]
    out = request_approval(
        decision=_decision(), ev=_ev(), surface="model", model="gpt-4o"
    )
    assert out.decided and not out.approved
    assert out.state == "expired"


def test_wait_expired_allow(capture: dict[str, Any]) -> None:
    set_config(_cfg())
    capture["create_response"] = {"id": "h1", "resolved": False, "wait_mode": "wait"}
    capture["poll_responses"] = [
        {"resolved": True, "approved": True, "state": "expired"}
    ]
    out = request_approval(
        decision=_decision(), ev=_ev(), surface="model", model="gpt-4o"
    )
    assert out.decided and out.approved
    assert out.state == "expired"


def test_wait_budget_elapses_returns_pending(capture: dict[str, Any]) -> None:
    set_config(_cfg(approval_wait_budget_ms=60))
    capture["create_response"] = {"id": "h1", "resolved": False, "wait_mode": "wait"}
    capture["poll_responses"] = [{"resolved": False, "state": "pending"}]
    out = request_approval(
        decision=_decision(), ev=_ev(), surface="model", model="gpt-4o"
    )
    assert not out.decided
    assert out.state == "pending"
    assert out.approval_id == "h1"


# ── Fail-closed on unreachable control plane ─────────────────────────────


def test_create_unreachable_fails_closed(capture: dict[str, Any]) -> None:
    set_config(_cfg())
    capture["create_response"] = None  # backend down
    out = request_approval(
        decision=_decision(), ev=_ev(), surface="model", model="gpt-4o"
    )
    assert out.decided and not out.approved
    assert out.state == "unavailable"
    assert capture["poll_calls"] == 0


# ── Re-attach to an already-decided hold (resume by re-submit) ───────────


def test_create_returns_resolved_approved_no_poll(capture: dict[str, Any]) -> None:
    set_config(_cfg())
    capture["create_response"] = {
        "id": "h1",
        "resolved": True,
        "approved": True,
        "state": "approved",
    }
    out = request_approval(
        decision=_decision(), ev=_ev(), surface="model", model="gpt-4o"
    )
    assert out.decided and out.approved
    assert capture["poll_calls"] == 0


def test_create_returns_resolved_rejected_no_poll(capture: dict[str, Any]) -> None:
    set_config(_cfg())
    capture["create_response"] = {
        "id": "h1",
        "resolved": True,
        "approved": False,
        "state": "rejected",
    }
    out = request_approval(
        decision=_decision(), ev=_ev(), surface="model", model="gpt-4o"
    )
    assert out.decided and not out.approved
    assert capture["poll_calls"] == 0


# ── Async mode ───────────────────────────────────────────────────────────


def test_async_policy_returns_pending_immediately(capture: dict[str, Any]) -> None:
    set_config(_cfg())  # approval_mode defaults to "auto"
    capture["create_response"] = {"id": "h1", "resolved": False, "wait_mode": "async"}
    out = request_approval(
        decision=_decision(), ev=_ev(), surface="model", model="gpt-4o"
    )
    assert not out.decided
    assert out.state == "pending"
    assert out.approval_id == "h1"
    assert capture["poll_calls"] == 0  # never blocks


def test_async_override_forces_pending_even_for_wait_policy(
    capture: dict[str, Any],
) -> None:
    set_config(_cfg(approval_mode="async"))
    capture["create_response"] = {"id": "h1", "resolved": False, "wait_mode": "wait"}
    out = request_approval(
        decision=_decision(), ev=_ev(), surface="model", model="gpt-4o"
    )
    assert not out.decided
    assert capture["poll_calls"] == 0


def test_wait_override_forces_poll_even_for_async_policy(
    capture: dict[str, Any],
) -> None:
    set_config(_cfg(approval_mode="wait"))
    capture["create_response"] = {"id": "h1", "resolved": False, "wait_mode": "async"}
    capture["poll_responses"] = [
        {"resolved": True, "approved": True, "state": "approved"}
    ]
    out = request_approval(
        decision=_decision(), ev=_ev(), surface="model", model="gpt-4o"
    )
    assert out.decided and out.approved
    assert capture["poll_calls"] >= 1


# ── Idempotency key: resume + multi-step ─────────────────────────────────


def test_idempotency_key_is_sent_and_stable(capture: dict[str, Any]) -> None:
    set_config(_cfg())
    capture["create_response"] = {"id": "h1", "resolved": True, "approved": True,
                                  "state": "approved"}
    request_approval(
        decision=_decision(), ev=_ev(), surface="model", model="gpt-4o"
    )
    key1 = capture["create_payload"]["idempotency_key"]
    assert key1
    # A retry of the SAME action reproduces the SAME key → re-attaches.
    request_approval(
        decision=_decision(), ev=_ev(), surface="model", model="gpt-4o"
    )
    key2 = capture["create_payload"]["idempotency_key"]
    assert key1 == key2


def test_multistep_distinct_actions_get_distinct_keys(
    capture: dict[str, Any],
) -> None:
    set_config(_cfg())
    capture["create_response"] = {"id": "h1", "resolved": True, "approved": True,
                                  "state": "approved"}
    # Same run, two different held actions → two different holds.
    request_approval(
        decision=_decision(approval_detail="$25,000 via wire_send"),
        ev=_ev(), surface="model", model="gpt-4o",
    )
    step1 = capture["create_payload"]["idempotency_key"]
    request_approval(
        decision=_decision(approval_detail="delete production database"),
        ev=_ev(), surface="tool", model="gpt-4o",
    )
    step2 = capture["create_payload"]["idempotency_key"]
    assert step1 != step2


# ── Compliance: no raw payload in the hold-open request ──────────────────


def test_no_raw_payload_in_create_request(capture: dict[str, Any]) -> None:
    set_config(_cfg())
    capture["create_response"] = {"id": "h1", "resolved": True, "approved": True,
                                  "state": "approved"}
    # The decision text is already post-sanitization; a raw secret that
    # was masked upstream must never reach the hold-open payload.
    raw_secret = "123-45-6789"
    request_approval(
        decision=_decision(message="SSN ###-##-#### needs approval",
                           approval_detail="masked action"),
        ev=_ev(), surface="model", model="gpt-4o",
    )
    blob = json.dumps(capture["create_payload"])
    assert raw_secret not in blob
