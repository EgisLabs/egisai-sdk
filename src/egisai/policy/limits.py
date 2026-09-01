"""Per-agent rate-limit + budget counters for the ``rate_limit`` /
``budget_limit`` policy kinds.

Design (mirrors ``pre-push-ci-parity`` + ``sdk-design-philosophy``
contracts):

* **No network on the hot path.** ``rate_limit_usage`` /
  ``budget_usage_usd`` are pure in-memory reads: a backend-synced
  snapshot plus a local sliding-window counter. The engine's
  Phase-1 evaluators call them like any other deterministic check.
* **Authoritative usage comes from the backend.** A daemon worker
  polls ``GET /v1/sdk/usage`` (see ``_backend.fetch_usage``) every
  ``EGISAI_USAGE_SYNC_SECONDS`` (default 15 s) while at least one
  limit rule is active. The snapshot carries per-agent + org-wide
  request counts for the three supported rate windows and spend
  for the three supported budget windows, so multiple SDK
  processes attributing calls to the same agent converge on the
  same numbers within one sync interval.
* **Local counting bridges the sync gap.** Every allowed
  model-surface evaluation records one local timestamp per agent.
  Effective usage = snapshot count + local calls made *after* the
  snapshot was computed. The overlap semantics deliberately lean
  strict (slight over-count is possible right at the window edge)
  because for a limiter the safe failure direction is "block one
  call early", never "let a runaway loop through".
* **Fail-open on availability.** No snapshot (older backend, first
  seconds after a rule is created, network outage) degrades to
  local-only counting for ``rate_limit`` and to a silent allow for
  ``budget_limit`` (spend can't be observed locally — the SDK does
  not price calls; the backend does at ingest). Never refuse a
  call because usage data is missing. This posture matches
  ``security-and-compliance.mdc`` §4: budgets/rate limits are an
  availability control, not a PII control, so they fail open.

Wire shape of the snapshot (``GET /v1/sdk/usage``)::

    {
      "computed_at": "2026-07-26T07:00:00Z",
      "limits_active": true,
      "agents": {
        "<agent-uuid>": {
          "requests": {"60": 4, "3600": 118, "86400": 953},
          "spend_usd": {"daily": "0.4413", "weekly": "2.10", "monthly": "9.87"}
        }
      },
      "org": {"requests": {...}, "spend_usd": {...}}
    }

Unknown keys are ignored; missing keys degrade to zero — an older
backend that doesn't serve the endpoint at all simply leaves the
snapshot ``None``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from datetime import UTC, datetime
from typing import Any

LOGGER = logging.getLogger("egisai.limits")

# Supported sliding windows (seconds) for ``rate_limit`` rules. The
# backend validates rule config against the same trio; keeping the
# set closed makes the backend's usage aggregation a single grouped
# query instead of per-rule ad-hoc scans.
RATE_WINDOWS: tuple[int, ...] = (60, 3600, 86400)

# Supported calendar windows for ``budget_limit`` rules. All are
# UTC-anchored on the backend (midnight / ISO-Monday / 1st).
BUDGET_WINDOWS: tuple[str, ...] = ("daily", "weekly", "monthly")

# Policy kinds this module serves. Referenced by ``_policy_cache``
# (worker lifecycle) and the engine's deterministic-kind set.
LIMIT_KINDS: frozenset[str] = frozenset({"rate_limit", "budget_limit"})

# Hard cap on local per-key deque length. A process that made 100k
# calls inside the largest window has certainly tripped any sane
# limit long ago; the cap only exists to bound memory on runaway
# loops that keep calling after the block (blocked calls are not
# recorded, so in practice the deque stays tiny).
_LOCAL_MAXLEN = 100_000

# Org-wide local counts are kept under this reserved key so one
# deque map serves both scopes.
_ORG_KEY = "__org__"

_DEFAULT_SYNC_SECONDS = 15.0


def _sync_interval_seconds() -> float:
    raw = os.getenv("EGISAI_USAGE_SYNC_SECONDS")
    if not raw:
        return _DEFAULT_SYNC_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_SYNC_SECONDS
    return value if value > 0 else _DEFAULT_SYNC_SECONDS


# ── Module state ────────────────────────────────────────────────────

_lock = threading.RLock()

# Parsed backend snapshot. ``None`` until the first successful sync.
_snap_agents: dict[str, dict[str, Any]] = {}
_snap_org: dict[str, Any] | None = None
# Wall-clock epoch seconds of the snapshot's ``computed_at``. Local
# entries at or before this instant are assumed ingested (drop on
# merge) so they aren't counted twice.
_snap_epoch: float | None = None

# Per-user (member) limit block for the calling key's owner, delivered
# additively by ``GET /v1/sdk/usage`` (backend ≥ the member-limits
# release). ``None`` when the owner has no active per-user limits, or
# when talking to an older backend that doesn't serve the field. Shape::
#
#     {
#       "budget_usd": "50.00", "budget_period": "monthly",
#       "max_requests": 1000, "requests_period": "daily",
#       "max_tokens": 200000, "tokens_period": "monthly",
#       "spend_usd": "12.30", "requests_used": 40,
#       "tokens_used": 1234, "access_expired": false
#     }
_snap_member: dict[str, Any] | None = None

# Local sliding-window call log. Keyed by lower-case agent UUID plus
# the reserved ``_ORG_KEY`` aggregate. Values are wall-clock epoch
# timestamps of allowed model-surface evaluations.
_local: dict[str, deque[float]] = {}

# Sync worker state.
_worker: threading.Thread | None = None
_worker_stop = threading.Event()

# One-shot flag so the "budget rule active but no usage snapshot"
# warning doesn't spam stderr on every call.
_warned_no_snapshot = False


# ── Recording (hot path — O(1), never raises) ───────────────────────


def record_model_call(agent_id: str) -> None:
    """Record one allowed model-surface call for limit accounting.

    Called by ``_evaluator.evaluate`` after a non-block decision on
    the model surface. Cheap (two deque appends under the lock) and
    exception-proof — a failure here must never break the user's
    call path.
    """
    try:
        now = time.time()
        key = (agent_id or "").strip().lower()
        with _lock:
            if key:
                _bucket(key).append(now)
            _bucket(_ORG_KEY).append(now)
    except Exception:  # noqa: BLE001
        LOGGER.debug("record_model_call failed", exc_info=True)


def _bucket(key: str) -> deque[float]:
    dq = _local.get(key)
    if dq is None:
        dq = deque(maxlen=_LOCAL_MAXLEN)
        _local[key] = dq
    return dq


def _count_local(key: str, since_epoch: float) -> int:
    """Count local entries strictly newer than ``since_epoch``.

    Also prunes entries older than the largest supported window so
    long-running processes don't accumulate stale timestamps.
    """
    dq = _local.get(key)
    if not dq:
        return 0
    horizon = time.time() - max(RATE_WINDOWS)
    while dq and dq[0] < horizon:
        dq.popleft()
    # deque is append-ordered; walk from the right for recency.
    count = 0
    for ts in reversed(dq):
        if ts > since_epoch:
            count += 1
        else:
            break
    return count


# ── Reads (hot path) ────────────────────────────────────────────────


def rate_limit_usage(agent_id: str, window_seconds: int, scope: str) -> int:
    """Effective request count for one rate window.

    Snapshot count (backend truth for everything ingested by
    ``computed_at``) plus local calls recorded after the snapshot.
    Without a snapshot this degrades to local-only counting — the
    documented single-process fallback for older backends.
    """
    key = _ORG_KEY if scope == "per_org" else (agent_id or "").strip().lower()
    if not key:
        return 0
    with _lock:
        snap_count = 0
        local_since = 0.0  # no snapshot ⇒ count the whole local window
        if _snap_epoch is not None:
            block = _snap_org if scope == "per_org" else _snap_agents.get(key)
            requests = (block or {}).get("requests") or {}
            try:
                snap_count = int(requests.get(str(window_seconds)) or 0)
            except (TypeError, ValueError):
                snap_count = 0
            local_since = _snap_epoch
        window_start = time.time() - window_seconds
        return snap_count + _count_local(key, max(local_since, window_start))


def budget_usage_usd(agent_id: str, window: str, scope: str) -> float | None:
    """Spend in USD for one calendar window, or ``None`` when unknown.

    Spend is backend-priced at ingest; the SDK cannot observe cost
    locally, so no snapshot ⇒ ``None`` and the budget rule fails
    open (with a one-time warning so operators can see the gap).
    """
    global _warned_no_snapshot
    with _lock:
        if _snap_epoch is None:
            if not _warned_no_snapshot:
                _warned_no_snapshot = True
                LOGGER.warning(
                    "[egisai] budget_limit rule is active but no usage "
                    "snapshot has been received from the platform yet — "
                    "budget enforcement is deferred until the first "
                    "sync succeeds (older backends without /v1/sdk/usage "
                    "never enforce budgets locally)."
                )
            return None
        if scope == "per_org":
            block = _snap_org
        else:
            key = (agent_id or "").strip().lower()
            if not key:
                return None
            block = _snap_agents.get(key)
        spend = (block or {}).get("spend_usd") or {}
        raw = spend.get(window)
        if raw is None:
            return 0.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0


# ── Per-user (member) limits ────────────────────────────────────────


def member_limit_block() -> tuple[str, str] | None:
    """Block reason for the calling key owner's per-user limit, or None.

    Returns ``(reason_code, message)`` when the owning member is over a
    hard per-user cap (budget / requests / tokens) or their access has
    expired, else ``None``.

    Fail-open posture (``security-and-compliance.mdc`` §4): budgets are
    an availability control, so a missing snapshot (older backend,
    first seconds after a rule appears, network blip) yields ``None``.
    Spend and tokens are backend-priced at ingest and cannot be
    observed locally, so those caps enforce off the snapshot alone;
    the request count bridges the sync gap with the local org counter
    (every call in this process belongs to the key's owner).

    The inline Gateway is the authoritative real-time hard block for
    per-user limits; this local check governs auto-patch traffic that
    never touches the Gateway, on a best-effort basis.
    """
    with _lock:
        member = _snap_member
        snap_epoch = _snap_epoch
    if not member:
        return None

    if member.get("access_expired"):
        return (
            "member_access_expired",
            "This member's access has expired. Ask an admin to extend it.",
        )

    # Budget — snapshot spend only (SDK cannot price locally).
    budget = member.get("budget_usd")
    if budget is not None:
        try:
            cap = float(budget)
            spent = float(member.get("spend_usd") or 0)
        except (TypeError, ValueError):
            cap = 0.0
            spent = 0.0
        if cap > 0 and spent >= cap:
            return (
                "member_budget_exceeded",
                f"Per-user budget exceeded: ${spent:.4f} of the "
                f"${cap:.2f} {member.get('budget_period') or 'monthly'} "
                "budget for this member has been spent.",
            )

    # Request count — snapshot count plus local calls since the snapshot.
    max_requests = member.get("max_requests")
    if max_requests is not None:
        try:
            capr = int(max_requests)
        except (TypeError, ValueError):
            capr = 0
        if capr > 0:
            used = int(member.get("requests_used") or 0)
            if snap_epoch is not None:
                with _lock:
                    used += _count_local(_ORG_KEY, snap_epoch)
            if used >= capr:
                return (
                    "member_requests_exceeded",
                    f"Per-user request budget exceeded: {used} of "
                    f"{capr} {member.get('requests_period') or 'monthly'} "
                    "model calls for this member.",
                )

    # Token count — snapshot only (tokens are backend-priced).
    max_tokens = member.get("max_tokens")
    if max_tokens is not None:
        try:
            capt = int(max_tokens)
            used_tok = int(member.get("tokens_used") or 0)
        except (TypeError, ValueError):
            capt = 0
            used_tok = 0
        if capt > 0 and used_tok >= capt:
            return (
                "member_tokens_exceeded",
                f"Per-user token budget exceeded: {used_tok} of "
                f"{capt} {member.get('tokens_period') or 'monthly'} "
                "tokens for this member.",
            )

    return None


# ── Snapshot ingestion (sync worker / tests) ────────────────────────


def replace_snapshot(payload: dict[str, Any] | None) -> None:
    """Install a fresh backend usage snapshot atomically.

    Local entries at or before the snapshot's ``computed_at`` are
    dropped — the backend has (or will have, within ingest lag)
    counted them; keeping them would double-count. Entries recorded
    after ``computed_at`` stay and bridge until the next sync.
    """
    if not isinstance(payload, dict):
        return
    computed_at = _parse_epoch(payload.get("computed_at"))
    if computed_at is None:
        return
    agents_raw = payload.get("agents")
    agents: dict[str, dict[str, Any]] = {}
    if isinstance(agents_raw, dict):
        for k, v in agents_raw.items():
            if isinstance(v, dict):
                agents[str(k).strip().lower()] = v
    org_raw = payload.get("org")
    org = org_raw if isinstance(org_raw, dict) else None
    member_raw = payload.get("member")
    member = member_raw if isinstance(member_raw, dict) else None

    global _snap_agents, _snap_org, _snap_epoch, _snap_member
    with _lock:
        _snap_agents = agents
        _snap_org = org
        _snap_member = member
        _snap_epoch = computed_at
        for dq in _local.values():
            while dq and dq[0] <= computed_at:
                dq.popleft()


def _parse_epoch(raw: Any) -> float | None:
    if not raw:
        return None
    try:
        text = str(raw).strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.timestamp()
    except Exception:  # noqa: BLE001
        return None


# ── Sync worker lifecycle ───────────────────────────────────────────


def notify_rules(rules: list[Any]) -> None:
    """Called by ``_policy_cache.replace_rules`` on every cache write.

    Starts the usage-sync worker when at least one limit rule is
    active and stops it when the last one disappears. The worker is
    what keeps the snapshot fresh even in SSE mode, where the
    policy refresher only fires on policy-change events (usage
    moves continuously between those events).
    """
    try:
        has_limits = any(
            getattr(r, "type", None) in LIMIT_KINDS
            or (isinstance(r, dict) and (r.get("type") or r.get("kind")) in LIMIT_KINDS)
            for r in rules
        )
    except Exception:  # noqa: BLE001
        has_limits = False
    if has_limits:
        _start_worker()
    else:
        _stop_worker()


def _start_worker() -> None:
    global _worker
    with _lock:
        if _worker is not None and _worker.is_alive():
            return
        _worker_stop.clear()
        _worker = threading.Thread(
            target=_sync_loop, name="egisai-usage-sync", daemon=True
        )
        _worker.start()


def _stop_worker() -> None:
    global _worker
    with _lock:
        thread = _worker
        _worker = None
    if thread is None:
        return
    _worker_stop.set()
    # Daemon thread; don't join on the caller's (possibly hot) path.


def _sync_loop() -> None:
    # First sync immediately so a freshly-created limit rule starts
    # enforcing against real numbers within one round-trip, not one
    # full interval later.
    while not _worker_stop.is_set():
        try:
            from egisai._backend import fetch_usage

            payload = fetch_usage()
            if payload is not None:
                replace_snapshot(payload)
        except Exception:  # noqa: BLE001
            # Fail open — keep the previous snapshot (or none) and
            # retry next cycle.
            LOGGER.debug("usage sync failed", exc_info=True)
        if _worker_stop.wait(timeout=_sync_interval_seconds()):
            return


def stop() -> None:
    """Shutdown hook — stop the worker. Idempotent."""
    _stop_worker()


def clear() -> None:
    """Reset all state (tests + ``shutdown``)."""
    global _snap_agents, _snap_org, _snap_epoch, _snap_member
    global _warned_no_snapshot
    _stop_worker()
    with _lock:
        _snap_agents = {}
        _snap_org = None
        _snap_member = None
        _snap_epoch = None
        _local.clear()
        _warned_no_snapshot = False
