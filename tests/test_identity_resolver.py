"""Unit tests for the Agent Identity v1 resolver (0.17.0).

The resolver walks a 7-tier ladder (Tier 0 → Tier 6) and pushes the
resolved identity onto a ContextVar stack so nested calls inherit.
Each test pins one tier or one invariant; the goal is that adding
a new tier never regresses an older one.

Shape of every test: build a payload, call ``resolve_identity``,
assert ``IdentityRecord.source`` matches the expected tier. The
``fake_backend`` fixture from conftest.py routes the ``ensure_agent``
call through an in-memory mock so no real network is involved.
"""

from __future__ import annotations

from typing import Any

import egisai
from egisai._auto_agent import (
    IdentityRecord,
    _derive_identity_from_system,
    _hash_bundle,
    _name_from_regex,
    current_identity,
    identity_scope,
    push_identity,
    reset_identity,
    resolve_identity,
)


def _init(fake_backend: Any) -> None:
    egisai.init(
        api_key="egis_live_x",
        app="default-app",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )


# ── Tier 0: explicit context manager ───────────────────────────────


def test_tier0_agent_context_manager_pushes_identity(fake_backend: Any) -> None:
    """``with egisai.agent("X"):`` outranks every auto-detection tier."""
    _init(fake_backend)
    with egisai.agent("Triage Specialist"):
        record = resolve_identity({"system": "Different system prompt"})
        assert record is not None
        assert record.source == "explicit"
        assert record.display_name == "Triage Specialist"


def test_tier0_nested_agent_blocks_innermost_wins(fake_backend: Any) -> None:
    """Nested ``with`` blocks form a stack; innermost wins per call."""
    _init(fake_backend)
    with egisai.agent("Outer"):
        with egisai.agent("Inner"):
            assert (current_identity() or IdentityRecord(  # fail clearly
                agent_id=None, display_name="?", identity_key="?",
                identity_hash="?", source="explicit",
            )).display_name == "Inner"
        assert current_identity().display_name == "Outer"  # type: ignore[union-attr]
    assert current_identity() is None


def test_tier0_register_agent_returns_agent_id(fake_backend: Any) -> None:
    """``egisai.register_agent(name)`` pre-creates the dashboard row."""
    _init(fake_backend)
    aid = egisai.register_agent("Pre-Created Agent")
    assert isinstance(aid, str) and len(aid) > 0
    # Calling again with the same name is a dict-lookup cache hit
    # (one ensure call total).
    aid2 = egisai.register_agent("Pre-Created Agent")
    assert aid == aid2
    names = [a["name"] for a in fake_backend.ensured_agents]
    assert names.count("Pre-Created Agent") == 1


# ── Tier 1: stored-prompt ids ───────────────────────────────────────


def test_tier1_openai_prompt_id(fake_backend: Any) -> None:
    """A payload with ``prompt = {"id": "pmpt_…"}`` resolves to Tier 1."""
    _init(fake_backend)
    payload = {
        "prompt": {"id": "pmpt_support_v3", "version": "1"},
    }
    record = resolve_identity(payload)
    assert record is not None
    assert record.source == "stored_prompt:openai"
    assert "pmpt_support_v3" in record.identity_key


def test_tier1_gemini_cached_content(fake_backend: Any) -> None:
    """``cached_content`` strings resolve to gemini Tier 1."""
    _init(fake_backend)
    payload = {"cached_content": "cachedContents/abc-123"}
    record = resolve_identity(payload)
    assert record is not None
    assert record.source == "stored_prompt:gemini"


def test_tier1_takes_precedence_over_system_prompt(fake_backend: Any) -> None:
    """A stored-prompt id ranks higher than a system prompt fingerprint."""
    _init(fake_backend)
    payload = {
        "prompt": {"id": "pmpt_xyz"},
        "system": "You are a different agent.",
    }
    record = resolve_identity(payload)
    assert record is not None
    assert record.source == "stored_prompt:openai"


# ── Tier 3: stack-frame inspection ──────────────────────────────────


def test_tier3_stack_var_agent_name_loose(fake_backend: Any) -> None:
    """In ``loose`` mode, an ``agent_name`` local is picked up."""
    _init(fake_backend)
    agent_name = "Research Analyst"  # noqa: F841 — read by stack walk
    record = resolve_identity(
        {"messages": [{"role": "user", "content": "hi"}]},
        auto_stack_hints="loose",
    )
    assert record is not None
    assert record.source == "stack"
    assert record.display_name == "Research Analyst"


def test_tier3_strict_only_marker(fake_backend: Any) -> None:
    """In ``strict`` mode, only ``__egisai_agent__`` is read."""
    _init(fake_backend)
    agent_name = "Loose Match"  # noqa: F841 — should be ignored
    __egisai_agent__ = "Strict Match"  # noqa: F841 — should be picked
    record = resolve_identity(
        {"messages": [{"role": "user", "content": "hi"}]},
        auto_stack_hints="strict",
    )
    assert record is not None
    assert record.source == "stack"
    assert record.display_name == "Strict Match"


def test_tier3_off_skips_stack(fake_backend: Any) -> None:
    """``off`` disables stack inspection entirely."""
    _init(fake_backend)
    agent_name = "Should Be Ignored"  # noqa: F841
    record = resolve_identity(
        {"messages": [{"role": "user", "content": "hi"}]},
        auto_stack_hints="off",
    )
    # Falls through to Tier 6 (init-time app)
    assert record is not None
    assert record.source == "app"


# ── Tier 5: hash + NER name ─────────────────────────────────────────


def test_tier5_hash_when_no_higher_tier_matches(fake_backend: Any) -> None:
    """A raw chat-style call with a system prompt → Tier 5 hash."""
    _init(fake_backend)
    record = resolve_identity(
        {"system": "You are a Python Developer. Be terse."},
        auto_stack_hints="off",  # disable Tier 3 for this test
    )
    assert record is not None
    assert record.source == "hash"
    # Regex chain still works as a fallback when NER is cold.
    assert record.display_name.lower().startswith(("python", "agent-"))


def test_tier5_same_prompt_same_hash() -> None:
    """Two identical system prompts produce the same identity hash."""
    a = _derive_identity_from_system("You are a Researcher.")
    b = _derive_identity_from_system("You are a Researcher.")
    assert a[0] == b[0]


def test_tier5_different_prompt_different_hash() -> None:
    a = _derive_identity_from_system("You are a Researcher.")
    b = _derive_identity_from_system("You are a Copywriter.")
    assert a[0] != b[0]


def test_tier5_nfkc_normalization_collapses_unicode_variants() -> None:
    """Fullwidth and ASCII identical-meaning strings collide on hash."""
    a = _derive_identity_from_system("You are Test123.")
    b = _derive_identity_from_system("You are Test\uff11\uff12\uff13.")  # fullwidth digits
    assert a[0] == b[0]


# ── Tier 6: app fallback ────────────────────────────────────────────


def test_tier6_app_fallback_when_nothing_else_matches(fake_backend: Any) -> None:
    """Empty payload + no context = fallback to init-time app."""
    _init(fake_backend)
    record = resolve_identity({}, auto_stack_hints="off")
    assert record is not None
    assert record.source == "app"
    assert record.display_name == "default-app"


# ── Cache unification + concurrency ────────────────────────────────


def test_cache_dedupes_repeated_identity(fake_backend: Any) -> None:
    """Resolving the same identity twice = one ensure call."""
    _init(fake_backend)
    p = {"system": "You are a Database Architect."}
    r1 = resolve_identity(p, auto_stack_hints="off")
    r2 = resolve_identity(p, auto_stack_hints="off")
    assert r1 is not None and r2 is not None
    assert r1.agent_id == r2.agent_id
    # Only one HTTP round-trip to ensure_agent.
    ensure_count = sum(
        1 for a in fake_backend.ensured_agents
        if "Architect" in a["name"] or a["name"].startswith("agent-")
    )
    assert ensure_count == 1


def test_identity_hash_shipped_to_backend_on_ensure(fake_backend: Any) -> None:
    """Every ensure call carries identity_hash + identity_source."""
    _init(fake_backend)
    resolve_identity(
        {"system": "You are an analyst."}, auto_stack_hints="off"
    )
    body = next(
        b for b in fake_backend.ensure_requests
        if "analyst" in b.get("name", "").lower()
        or b.get("name", "").startswith("agent-")
    )
    assert body.get("identity_hash"), "identity_hash must be on the wire"
    assert body.get("identity_source") == "hash"
    assert len(body["identity_hash"]) == 64  # sha-256 hex


# ── Identity stack scope ────────────────────────────────────────────


def test_identity_scope_pushes_and_pops() -> None:
    rec = IdentityRecord(
        agent_id="a-uuid",
        display_name="Test",
        identity_key="explicit:Test",
        identity_hash="0" * 64,
        source="explicit",
        push_to_stack=True,
    )
    assert current_identity() is None
    with identity_scope(rec):
        assert current_identity() is rec
    assert current_identity() is None


def test_identity_scope_pops_even_on_exception() -> None:
    rec = IdentityRecord(
        agent_id=None, display_name="x", identity_key="explicit:x",
        identity_hash="1" * 64, source="explicit", push_to_stack=True,
    )
    try:
        with identity_scope(rec):
            assert current_identity() is rec
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert current_identity() is None


def test_push_and_reset_balanced() -> None:
    rec = IdentityRecord(
        agent_id=None, display_name="y", identity_key="explicit:y",
        identity_hash="2" * 64, source="explicit", push_to_stack=True,
    )
    token = push_identity(rec)
    assert current_identity() is rec
    reset_identity(token)
    assert current_identity() is None


# ── Helpers ────────────────────────────────────────────────────────


def test_hash_bundle_orderless_within_collection() -> None:
    """Tools listed in different orders produce the same digest."""
    a = _hash_bundle(("ns", ["tool_a", "tool_b"]))
    b = _hash_bundle(("ns", ["tool_b", "tool_a"]))
    assert a == b


def test_hash_bundle_different_namespace_different_digest() -> None:
    a = _hash_bundle(("openai_agents", "Triage"))
    b = _hash_bundle(("crewai", "Triage"))
    assert a != b


def test_name_from_regex_extracts_specialist() -> None:
    out = _name_from_regex("You are a specialist: Python Developer.")
    assert out == "Python Developer"
