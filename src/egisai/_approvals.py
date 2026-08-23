"""Human-in-the-loop hold coordinator (SDK side).

When the policy engine returns a ``pending_approval`` verdict, the gate
calls :func:`request_approval`. This opens a hold on the control plane
and polls it, blocking the call in-line up to
``config.approval_wait_budget_ms``:

* Approved in time  → the gate resumes and runs the real forward.
* Rejected / expired-with-block → the gate blocks the call.
* Budget elapsed with no decision → the gate applies ``on_pending``
  (raise :class:`ApprovalPendingError` or return a "pending" stub).

The approval wait is measured here and returned as ``wait_ms`` so the
gate can book it on a dedicated ``approval_wait_ms`` field — never on
``latency_ms`` / ``policy_latency_ms`` — keeping the dashboard's
latency and the trust / behavior math undistorted by human think-time.

Fail-closed contract: if the control plane can't be reached to open or
poll the hold, the call is treated as NOT approved (blocked). A rule
that asked for human approval is important enough that "couldn't ask a
human" must not silently proceed. This is the deliberate approvals
exception to the SDK's usual availability fail-open posture.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from egisai import _backend
from egisai._config import get_config
from egisai.policy.engine import PolicyDecision


class ApprovalPendingError(RuntimeError):
    """Raised (when ``on_pending='raise'``) once the in-line wait budget
    elapses without a human decision. The action is parked on the
    dashboard; a retry re-checks the same hold."""

    def __init__(self, message: str, *, approval_id: str | None = None) -> None:
        super().__init__(message)
        self.approval_id = approval_id


@dataclass(frozen=True)
class ApprovalOutcome:
    """Result of the in-line approval wait."""

    decided: bool  # a terminal decision arrived within the budget
    approved: bool  # only meaningful when decided
    approval_id: str | None
    wait_ms: int
    state: str
    message: str | None = None


# Poll cadence — start tight so a fast Slack/email click resumes almost
# immediately, then back off so a long wait isn't a busy-loop.
_POLL_MIN_S = 0.5
_POLL_MAX_S = 3.0


def request_approval(
    *,
    decision: PolicyDecision,
    ev: dict[str, Any],
    surface: str,
    model: str | None,
) -> ApprovalOutcome:
    """Open a hold and block in-line until decided or the budget elapses."""
    cfg = get_config()
    started = time.monotonic()

    agent_id = ev.get("agent_id")
    payload: dict[str, Any] = {
        "matched_policy": decision.matched_policy,
        "reason_code": decision.reason_code,
        "message": decision.message,
        "approval_detail": decision.approval_detail or decision.message,
        "surface": surface,
        "model": model,
        "agent_id": agent_id,
        "run_id": ev.get("run_id"),
        "trace_id": ev.get("trace_id"),
    }

    created = _backend.create_approval(payload)
    if not created or not created.get("id"):
        # Fail closed — couldn't open the hold, so we couldn't ask a
        # human. Treat as not approved.
        return ApprovalOutcome(
            decided=True,
            approved=False,
            approval_id=None,
            wait_ms=int((time.monotonic() - started) * 1000),
            state="unavailable",
            message="Approval could not be requested (control plane unreachable).",
        )

    approval_id = str(created["id"])
    if created.get("resolved"):
        return ApprovalOutcome(
            decided=True,
            approved=bool(created.get("approved")),
            approval_id=approval_id,
            wait_ms=int((time.monotonic() - started) * 1000),
            state=str(created.get("state") or "resolved"),
            message=created.get("message"),
        )

    budget_s = max(0.0, cfg.approval_wait_budget_ms / 1000.0)
    delay = _POLL_MIN_S
    while (time.monotonic() - started) < budget_s:
        time.sleep(min(delay, max(0.0, budget_s - (time.monotonic() - started))))
        status = _backend.poll_approval(approval_id)
        if status and status.get("resolved"):
            return ApprovalOutcome(
                decided=True,
                approved=bool(status.get("approved")),
                approval_id=approval_id,
                wait_ms=int((time.monotonic() - started) * 1000),
                state=str(status.get("state") or "resolved"),
                message=status.get("message"),
            )
        delay = min(_POLL_MAX_S, delay * 1.5)

    # Budget elapsed with no decision.
    return ApprovalOutcome(
        decided=False,
        approved=False,
        approval_id=approval_id,
        wait_ms=int((time.monotonic() - started) * 1000),
        state="pending",
        message="Awaiting human approval.",
    )
