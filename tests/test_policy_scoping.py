"""SDK-side scope filtering: rules with non-empty ``agent_ids`` only
fire for the active agent.

Every test patches the cache directly (no backend) and inspects the
filtered list the evaluator computes via ``_scope_filter``.
"""

from __future__ import annotations

import contextvars
from collections.abc import Iterator

import pytest

from egisai import _config, _context, _evaluator, _policy_cache
from egisai.policy import PolicyRule


@pytest.fixture(autouse=True)
def _clean_cache() -> Iterator[None]:
    _policy_cache.clear()
    yield
    _policy_cache.clear()


@pytest.fixture
def reset_ctx() -> Iterator[None]:
    """Each test gets a fresh contextvar copy so ``set_context`` calls
    don't leak across tests (frozen ContextVars persist process-wide
    otherwise)."""
    ctx = contextvars.copy_context()

    def _run(test):
        ctx.run(test)

    saved_ctx = _context._ctx.get()
    try:
        yield
    finally:
        _context._ctx.set(saved_ctx)


def _seed_rules(*rules: PolicyRule) -> None:
    """Helper: load arbitrary ``PolicyRule`` objects into the cache."""
    _policy_cache._lock.acquire()
    try:
        _policy_cache._etag = "test"
        _policy_cache._rules = list(rules)
    finally:
        _policy_cache._lock.release()


def test_to_rule_parses_agent_ids_and_normalises_case() -> None:
    rule = _policy_cache._to_rule(
        {
            "id": 1,
            "name": "scoped",
            "type": "deny_regex",
            "tenant": None,
            "config": {},
            "agent_ids": [
                "AABBCC11-1111-2222-3333-444455556666",
                "ZZZZ",  # bogus but normalised — never matched
            ],
        }
    )
    assert rule.agent_ids == (
        "aabbcc11-1111-2222-3333-444455556666",
        "zzzz",
    )


def test_to_rule_default_agent_ids_is_empty_tuple() -> None:
    rule = _policy_cache._to_rule(
        {
            "id": 1,
            "name": "unscoped",
            "type": "deny_regex",
            "tenant": None,
            "config": {},
        }
    )
    assert rule.agent_ids == ()


def test_to_rule_id_accepts_uuid_string_and_legacy_int() -> None:
    """The platform may ship ``policies.id`` either as a small
    integer (legacy wire shape) or as a UUID string. The SDK
    parser normalises both wire shapes to ``str`` so the SDK
    works against either flavour of platform without code edits."""

    new_shape = _policy_cache._to_rule(
        {
            "id": "5eeeadd5-1735-4b86-9234-3d2590923314",
            "name": "uuid-policy",
            "type": "deny_regex",
            "tenant": None,
            "config": {},
        }
    )
    assert new_shape.id == "5eeeadd5-1735-4b86-9234-3d2590923314"

    legacy = _policy_cache._to_rule(
        {
            "id": 12,
            "name": "int-policy",
            "type": "deny_regex",
            "tenant": None,
            "config": {},
        }
    )
    assert legacy.id == "12"

    missing = _policy_cache._to_rule(
        {
            "name": "no-id-policy",
            "type": "deny_regex",
            "tenant": None,
            "config": {},
        }
    )
    assert missing.id is None


def test_scope_filter_keeps_unscoped_rules_for_any_agent() -> None:
    rule = PolicyRule(
        id=1, name="open", type="deny_regex", tenant=None, config={}
    )
    assert _evaluator._scope_filter([rule], agent_id="anything") == [rule]
    assert _evaluator._scope_filter([rule], agent_id="") == [rule]


def test_scope_filter_drops_targeted_rule_for_other_agent() -> None:
    rule = PolicyRule(
        id=1,
        name="scoped",
        type="deny_regex",
        tenant=None,
        config={},
        agent_ids=("11111111-1111-1111-1111-111111111111",),
    )
    assert (
        _evaluator._scope_filter(
            [rule], agent_id="22222222-2222-2222-2222-222222222222"
        )
        == []
    )


def test_scope_filter_keeps_targeted_rule_for_listed_agent() -> None:
    rule = PolicyRule(
        id=1,
        name="scoped",
        type="deny_regex",
        tenant=None,
        config={},
        agent_ids=(
            "11111111-1111-1111-1111-111111111111",
            "22222222-2222-2222-2222-222222222222",
        ),
    )
    assert _evaluator._scope_filter(
        [rule], agent_id="22222222-2222-2222-2222-222222222222"
    ) == [rule]


def test_scope_filter_with_no_active_agent_drops_targeted() -> None:
    """When ``set_context`` hasn't fired and the API key isn't bound
    to an agent, the safer default for an explicitly-scoped rule is
    NOT to enforce — operators choose targeting deliberately."""
    rule = PolicyRule(
        id=1,
        name="scoped",
        type="deny_regex",
        tenant=None,
        config={},
        agent_ids=("11111111-1111-1111-1111-111111111111",),
    )
    assert _evaluator._scope_filter([rule], agent_id="") == []


def test_active_agent_id_prefers_context_over_config() -> None:
    cfg = _config.EgisaiConfig(
        api_key="x",
        app="test",
        env="dev",
        agent_id="cfg-agent-id",
    )
    _config.set_config(cfg)
    try:
        _context.set_context(agent_id="ctx-agent-id")
        assert _evaluator._active_agent_id() == "ctx-agent-id"
    finally:
        _config._CONFIG = None
        # Reset the contextvar so we don't leak into other tests.
        _context._ctx.set(_context.EgisaiContext())


def test_active_agent_id_falls_back_to_config() -> None:
    cfg = _config.EgisaiConfig(
        api_key="x",
        app="test",
        env="dev",
        agent_id="config-agent-id",
    )
    _config.set_config(cfg)
    try:
        # Make sure no context is set first.
        _context._ctx.set(_context.EgisaiContext())
        assert _evaluator._active_agent_id() == "config-agent-id"
    finally:
        _config._CONFIG = None
