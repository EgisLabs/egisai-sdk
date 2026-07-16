"""Governance opt-out gate: ungoverned agents skip every policy phase.

What's pinned
-------------
* The ``/v1/sdk/policies`` snapshot's ``ungoverned_agent_ids``
  field lands in the cache atomically with the rule list, and the
  wire contract mirrors the paused set: missing field ⇒ preserve,
  empty list ⇒ clear.
* The evaluator's gate: an agent in the ungoverned set gets a
  plain ``allow`` on BOTH sides (input + output) even when an
  unscoped rule would otherwise block. Enforcement is skipped;
  the framework patches still build + ship their audit events, so
  monitoring is unaffected (that path is covered by the existing
  framework tests — nothing in the gate touches it).
* Precedence: the operator pause kill switch WINS over the
  governance opt-out. A paused agent is refused even if it is
  also ungoverned — the emergency control must never be maskable
  by the softer opt-out.
* Fail-safe: an empty agent id (unattributable traffic) stays
  fully governed.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from egisai import _config, _context, _evaluator, _policy_cache
from egisai._evaluator import InputCall, OutputCall

AGENT = "11111111-1111-1111-1111-111111111111"
OTHER = "22222222-2222-2222-2222-222222222222"

#: An unscoped deny rule that blocks any prompt containing "kaboom".
_DENY_RULE = {
    "id": 1,
    "name": "no-kaboom",
    "type": "deny_regex",
    "tenant": None,
    "config": {"pattern": "kaboom"},
}


@pytest.fixture(autouse=True)
def _clean_state() -> Iterator[None]:
    _policy_cache.clear()
    _config._CONFIG = None
    _context._ctx.set(_context.EgisaiContext())
    yield
    _policy_cache.clear()
    _config._CONFIG = None
    _context._ctx.set(_context.EgisaiContext())


def _set_active_agent(agent_id: str) -> None:
    cfg = _config.EgisaiConfig(
        api_key="x", app="test", env="dev", agent_id=agent_id
    )
    _config.set_config(cfg)


def _input_call(text: str = "kaboom please") -> InputCall:
    return InputCall(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4o",
        prompt_text=text,
        stream=False,
    )


def _output_call(text: str = "kaboom indeed") -> OutputCall:
    return OutputCall(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4o",
        text=text,
        tool_names=[],
        tool_calls=[],
        mcp_targets=[],
    )


# ── Cache wire contract ──────────────────────────────────────────────


def test_replace_rules_lands_ungoverned_set_in_lockstep() -> None:
    _policy_cache.replace_rules(
        '"v1"', [_DENY_RULE], ungoverned_agent_ids=[AGENT.upper()]
    )
    # Normalised to lower-case canonical form on the way in.
    assert _policy_cache.get_ungoverned_agent_ids() == frozenset({AGENT})


def test_missing_field_preserves_existing_set() -> None:
    _policy_cache.replace_rules(
        '"v1"', [], ungoverned_agent_ids=[AGENT]
    )
    # Older backend response without the field (None default) must
    # NOT clear an active opt-out mid-session.
    _policy_cache.replace_rules('"v2"', [_DENY_RULE])
    assert _policy_cache.get_ungoverned_agent_ids() == frozenset({AGENT})


def test_empty_list_explicitly_clears_the_set() -> None:
    _policy_cache.replace_rules(
        '"v1"', [], ungoverned_agent_ids=[AGENT]
    )
    _policy_cache.replace_rules('"v2"', [], ungoverned_agent_ids=[])
    assert _policy_cache.get_ungoverned_agent_ids() == frozenset()


def test_snapshot_round_trip_from_backend(fake_backend) -> None:
    """End-to-end: the backend ships ``ungoverned_agent_ids`` on the
    policies response and the cache picks it up via ``refresh_now``."""
    fake_backend.set_rules(
        [_DENY_RULE], etag='"v1"', ungoverned_agent_ids=[AGENT]
    )

    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    assert _policy_cache.get_ungoverned_agent_ids() == frozenset({AGENT})

    # Flip the agent back to governed on the backend; the next
    # refresh (new ETag) must clear the cached set.
    fake_backend.set_rules(
        [_DENY_RULE], etag='"v2"', ungoverned_agent_ids=[]
    )
    assert _policy_cache.refresh_now() is True
    assert _policy_cache.get_ungoverned_agent_ids() == frozenset()


# ── Evaluator gate ───────────────────────────────────────────────────


def test_ungoverned_agent_skips_input_policies() -> None:
    _set_active_agent(AGENT)
    _policy_cache.replace_rules(
        '"v1"', [_DENY_RULE], ungoverned_agent_ids=[AGENT]
    )
    decision = _evaluator.evaluate(_input_call())
    assert decision.verdict == "allow"
    # No synthetic policy record — the audit event reads exactly
    # like a call no policy matched (full visibility, zero noise).
    assert decision.matched_policy is None


def test_ungoverned_agent_skips_output_policies() -> None:
    _set_active_agent(AGENT)
    _policy_cache.replace_rules(
        '"v1"', [_DENY_RULE], ungoverned_agent_ids=[AGENT]
    )
    decision = _evaluator.evaluate_output(_output_call())
    assert decision.verdict == "allow"


def test_governed_agent_still_blocks() -> None:
    """Control case: the same rule set blocks when the active agent
    is NOT in the ungoverned set — the gate must be per-agent."""
    _set_active_agent(OTHER)
    _policy_cache.replace_rules(
        '"v1"', [_DENY_RULE], ungoverned_agent_ids=[AGENT]
    )
    decision = _evaluator.evaluate(_input_call())
    assert decision.verdict == "block"


def test_pause_wins_over_ungovern() -> None:
    """An agent that is BOTH paused and ungoverned is refused: the
    kill switch is the emergency control and must never be masked
    by the softer governance opt-out."""
    _set_active_agent(AGENT)
    _policy_cache.replace_rules(
        '"v1"',
        [],
        paused_agent_ids=[AGENT],
        ungoverned_agent_ids=[AGENT],
    )
    for decision in (
        _evaluator.evaluate(_input_call("hello")),
        _evaluator.evaluate_output(_output_call("hello")),
    ):
        assert decision.verdict == "block"
        assert decision.reason_code == "agent_paused"


def test_unattributable_traffic_stays_governed() -> None:
    """No active agent id ⇒ the opt-out can't match ⇒ the call runs
    the normal policy phases (the enforcing direction)."""
    _set_active_agent("")
    _policy_cache.replace_rules(
        '"v1"', [_DENY_RULE], ungoverned_agent_ids=[AGENT]
    )
    decision = _evaluator.evaluate(_input_call())
    assert decision.verdict == "block"
