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

import hashlib
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


# Backoff ceiling for the in-line poll. The starting interval is
# operator-configurable (``approval_poll_interval_ms``); the cap keeps a
# long wait from busy-looping while staying responsive to a fast click.
_POLL_MAX_S = 3.0


def _idempotency_key(
    *,
    decision: PolicyDecision,
    ev: dict[str, Any],
    surface: str,
    model: str | None,
) -> str | None:
    """A stable key for this logical action so a retried call re-attaches
    to the same hold (and inherits its decision) instead of opening a
    duplicate.

    Derived from the run + the tripped policy + the surface/model + a
    hash of what needs approving (``approval_detail`` / ``message``).
    Distinct steps of a multi-step run carry distinct ``approval_detail``
    so each step gets its own hold, while a retry of the *same* step
    reproduces the same key and resumes cleanly.
    """
    parts = [
        str(ev.get("run_id") or ""),
        str(ev.get("agent_id") or ""),
        str(decision.matched_policy or ""),
        str(surface or ""),
        str(model or ""),
        str(decision.approval_detail or decision.message or ""),
    ]
    raw = "|".join(parts)
    if not raw.strip("|"):
        return None
    return hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()[:128]


def request_approval(
    *,
    decision: PolicyDecision,
    ev: dict[str, Any],
    surface: str,
    model: str | None,
) -> ApprovalOutcome:
    """Open (or re-attach to) a hold and resolve it per the wait mode.

    In 'wait' mode the call blocks in-line polling up to the budget; in
    'async' mode it returns a pending outcome immediately. Either way a
    terminal decision that is already known at create time (a re-attach
    to a decided hold) resumes instantly.
    """
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
        "idempotency_key": _idempotency_key(
            decision=decision, ev=ev, surface=surface, model=model
        ),
    }
    # Structured approver-card hints, when the action carried them
    # (already post-sanitization on the SDK side).
    for field in ("amount", "currency", "amount_threshold"):
        val = ev.get(field)
        if val is not None:
            payload[field] = val

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
        # Already decided — a re-attach to a resolved hold (retry after
        # the decision) or an instant auto-resolve. Resume immediately.
        return ApprovalOutcome(
            decided=True,
            approved=bool(created.get("approved")),
            approval_id=approval_id,
            wait_ms=int((time.monotonic() - started) * 1000),
            state=str(created.get("state") or "resolved"),
            message=created.get("message"),
        )

    # Resolve the effective mode: an explicit SDK ``approval_mode`` wins;
    # "auto" follows the policy's ``wait_mode`` reported on the create
    # response.
    mode: str = cfg.approval_mode
    if mode == "auto":
        mode = str(created.get("wait_mode") or "wait").strip().lower()
    if mode not in ("wait", "async"):
        mode = "wait"

    if mode == "async":
        # Don't block: return pending now. A later re-submit reproduces
        # the idempotency key, re-attaches to this hold, and returns the
        # decision. The gate applies ``on_pending`` (raise/stub) so the
        # caller learns the action is parked.
        return ApprovalOutcome(
            decided=False,
            approved=False,
            approval_id=approval_id,
            wait_ms=int((time.monotonic() - started) * 1000),
            state="pending",
            message="Held for human approval (async).",
        )

    budget_s = max(0.0, cfg.approval_wait_budget_ms / 1000.0)
    poll_min_s = max(0.05, cfg.approval_poll_interval_ms / 1000.0)
    poll_max_s = max(poll_min_s, _POLL_MAX_S)
    delay = poll_min_s
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
        delay = min(poll_max_s, delay * 1.5)

    # Budget elapsed with no decision.
    return ApprovalOutcome(
        decided=False,
        approved=False,
        approval_id=approval_id,
        wait_ms=int((time.monotonic() - started) * 1000),
        state="pending",
        message="Awaiting human approval.",
    )
