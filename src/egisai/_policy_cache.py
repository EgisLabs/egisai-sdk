"""Thread-safe in-process cache of policy rules + paused-agent set.

Every governed call reads from this cache. The refresher worker
updates it when the platform signals a change.

The cache holds two pieces of state in lockstep:

* ``_rules``             — the org's active policy rules.
* ``_paused_agent_ids`` — UUIDs (lower-case) of agents an
                            operator has flipped into the paused
                            state from the dashboard.

Both are ETag-versioned by the same backend response so a single
``replace_*`` write is atomic from the gate's point of view: a
newly-paused agent never lands inconsistently against an
out-of-date rule list, and vice versa.
"""

from __future__ import annotations

import threading

from egisai._backend import fetch_policies
from egisai.policy import PolicyRule

_lock = threading.RLock()
_etag: str | None = None
_rules: list[PolicyRule] = []
# Lower-case canonical UUID strings of paused agents. Stored as a
# frozenset so the gate's hot-path containment check is O(1) and
# the read snapshot is safely shareable across threads without
# defensive copying.
_paused_agent_ids: frozenset[str] = frozenset()


_VALID_PHASES = ("pre_model", "post_model", "both")


def _to_rule(d: dict) -> PolicyRule:
    """Wire-shape → ``PolicyRule`` dataclass.

    ``agent_ids`` defaults to an empty tuple ("applies to all").
    ``phase`` defaults to ``"both"`` so older platform responses
    (which don't carry the field) keep their previous Behavior:
    each rule fires on whichever side its type supports.
    Both ``"type"`` and the legacy ``"kind"`` field are accepted.
    """
    raw_ids = d.get("agent_ids")
    if raw_ids is None:
        agent_ids: tuple[str, ...] = ()
    else:
        agent_ids = tuple(
            str(x).strip().lower() for x in raw_ids if x
        )
    # MCP Servers add-on scope. ``None`` / missing ⇒ empty tuple
    # ("not targeted at a specific MCP server"). Older backends that
    # don't ship the field leave this empty, which is the safe
    # default for every existing org.
    raw_mcp_ids = d.get("mcp_server_ids")
    if raw_mcp_ids is None:
        mcp_server_ids: tuple[str, ...] = ()
    else:
        mcp_server_ids = tuple(
            str(x).strip().lower() for x in raw_mcp_ids if x
        )
    raw_id = d.get("id")
    rule_id: str | None = (
        None if raw_id is None or raw_id == "" else str(raw_id)
    )
    raw_phase = d.get("phase")
    phase = raw_phase if raw_phase in _VALID_PHASES else "both"
    return PolicyRule(
        id=rule_id,
        name=d.get("name", ""),
        type=d.get("type") or d.get("kind") or "",
        tenant=d.get("tenant"),
        config=dict(d.get("config") or {}),
        agent_ids=agent_ids,
        phase=phase,
        mcp_server_ids=mcp_server_ids,
    )


def get_rules() -> list[PolicyRule]:
    """Read-only snapshot — returns a copy of the cached list."""
    with _lock:
        return list(_rules)


def get_etag() -> str | None:
    with _lock:
        return _etag


def get_paused_agent_ids() -> frozenset[str]:
    """Snapshot of the operator-paused agent ID set.

    Returns a ``frozenset`` so the gate (``_evaluator``) can
    perform O(1) containment without copying. The cache stores
    the same frozenset under the lock; readers see whatever was
    most recently published by ``replace_*``. No defensive copy
    is needed because frozensets are immutable.
    """
    with _lock:
        return _paused_agent_ids


def replace_rules(
    new_etag: str | None,
    raw_rules: list[dict],
    paused_agent_ids: list[str] | None = None,
) -> int:
    """Replace the cache atomically. Returns the new rule count.

    ``paused_agent_ids`` lands in lockstep with the new rule
    list. ``None`` (the default) preserves the existing
    paused-agent set — the wire-side contract is "if the
    backend didn't ship the field, leave the cache alone" so
    older backends that pre-date the rollout never accidentally
    clear an active pause.

    An empty list explicitly clears the set (the backend said
    "no agents are paused" — believe it).
    """
    with _lock:
        global _etag, _rules, _paused_agent_ids
        _etag = new_etag
        _rules = [_to_rule(r) for r in raw_rules]
        if paused_agent_ids is not None:
            _paused_agent_ids = frozenset(
                str(a).strip().lower() for a in paused_agent_ids if a
            )
        return len(_rules)


def refresh_now() -> bool:
    """Hit the platform once. Returns ``True`` iff the cache was updated."""
    current_etag = get_etag()
    new_etag, rules, paused = fetch_policies(etag=current_etag)
    if rules is None:
        # 304 — nothing changed since ``current_etag``. Both the
        # rule list AND the paused-agent set stay as-is, in
        # lockstep with the same ETag.
        return False
    replace_rules(new_etag, rules, paused_agent_ids=paused)
    return True


def clear() -> None:
    with _lock:
        global _etag, _rules, _paused_agent_ids
        _etag = None
        _rules = []
        _paused_agent_ids = frozenset()
