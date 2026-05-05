"""Thread-safe in-process cache of policy rules with ETag-driven freshness.

Every governed call reads from the cache. The refresher worker
updates it when the platform signals a change.
"""

from __future__ import annotations

import threading

from egisai._backend import fetch_policies
from egisai.policy import PolicyRule

_lock = threading.RLock()
_etag: str | None = None
_rules: list[PolicyRule] = []


def _to_rule(d: dict) -> PolicyRule:
    """Wire-shape → ``PolicyRule`` dataclass.

    ``agent_ids`` defaults to an empty tuple ("applies to all").
    Both ``"type"`` and the legacy ``"kind"`` field are accepted.
    """
    raw_ids = d.get("agent_ids")
    if raw_ids is None:
        agent_ids: tuple[str, ...] = ()
    else:
        agent_ids = tuple(
            str(x).strip().lower() for x in raw_ids if x
        )
    raw_id = d.get("id")
    rule_id: str | None = (
        None if raw_id is None or raw_id == "" else str(raw_id)
    )
    return PolicyRule(
        id=rule_id,
        name=d.get("name", ""),
        type=d.get("type") or d.get("kind") or "",
        tenant=d.get("tenant"),
        config=dict(d.get("config") or {}),
        agent_ids=agent_ids,
    )


def get_rules() -> list[PolicyRule]:
    """Read-only snapshot — returns a copy of the cached list."""
    with _lock:
        return list(_rules)


def get_etag() -> str | None:
    with _lock:
        return _etag


def replace_rules(new_etag: str | None, raw_rules: list[dict]) -> int:
    """Replace the cache atomically. Returns the new rule count."""
    with _lock:
        global _etag, _rules
        _etag = new_etag
        _rules = [_to_rule(r) for r in raw_rules]
        return len(_rules)


def refresh_now() -> bool:
    """Hit the platform once. Returns ``True`` iff the cache was updated."""
    current_etag = get_etag()
    new_etag, rules = fetch_policies(etag=current_etag)
    if rules is None:
        return False
    replace_rules(new_etag, rules)
    return True


def clear() -> None:
    with _lock:
        global _etag, _rules
        _etag = None
        _rules = []
