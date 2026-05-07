"""Zero-touch sub-agent detection.

The user's promise: ``import egisai; egisai.init(...)`` is enough. Any
distinct ``system`` prompt observed at call time should auto-register
its own agent on the platform.
"""

from __future__ import annotations

from egisai._auto_agent import _normalize_name, derive_identity


def test_extracts_specialist_role() -> None:
    out = derive_identity(
        {"system": "You are a specialist: Python Developer. Complete the task concisely."},
        messages=None,
    )
    assert out is not None
    digest, name = out
    assert name == "Python Developer"
    assert len(digest) == 64  # sha256 hex


def test_extracts_orchestrator_role_from_first_sentence() -> None:
    out = derive_identity(
        {
            "system": (
                "You are an orchestrator agent.\n"
                "1. Decide if you can handle it yourself...\n"
                "2. Use the delegate_to_agent tool..."
            )
        },
        messages=None,
    )
    assert out is not None
    _, name = out
    # Filler words ("an") get stripped.
    assert name == "orchestrator agent"


def test_extracts_from_openai_style_messages() -> None:
    """OpenAI puts the system message inside `messages` rather than a
    separate kwarg — auto-detection must handle both shapes."""
    out = derive_identity(
        payload={"messages": None},
        messages=[
            {"role": "system", "content": "You are a Copywriter. Be witty."},
            {"role": "user", "content": "Write a tagline for a coffee shop."},
        ],
    )
    assert out is not None
    _, name = out
    assert name == "Copywriter"


def test_no_system_returns_none() -> None:
    out = derive_identity(
        payload={"messages": [{"role": "user", "content": "hello"}]},
        messages=[{"role": "user", "content": "hello"}],
    )
    assert out is None


def test_same_system_yields_same_hash() -> None:
    a = derive_identity({"system": "You are a Researcher."}, None)
    b = derive_identity({"system": "You are a Researcher."}, None)
    assert a is not None and b is not None
    assert a[0] == b[0]
    assert a[1] == b[1]


def test_different_systems_yield_different_hashes() -> None:
    a = derive_identity({"system": "You are a Researcher."}, None)
    b = derive_identity({"system": "You are a Copywriter."}, None)
    assert a is not None and b is not None
    assert a[0] != b[0]


def test_hash_fallback_name_when_no_pattern_matches() -> None:
    out = derive_identity({"system": "Just be helpful."}, None)
    assert out is not None
    digest, name = out
    # Falls back to "agent-<hash>" when the regex chain can't find a name.
    # (or matches the fallback "Just be helpful" via pattern 3)
    assert name  # non-empty


def test_normalize_strips_filler_and_caps_length() -> None:
    assert _normalize_name("a Python Developer") == "Python Developer"
    assert _normalize_name("an Orchestrator Agent") == "Orchestrator Agent"
    long = "x" * 80
    assert len(_normalize_name(long)) <= 60


def test_end_to_end_anthropic_payload_registers_subagent(fake_backend) -> None:
    """When gate_call sees a call with system=…, it registers a fresh
    agent on the platform on first sight."""
    import egisai
    from egisai._patches._common import gate_call

    egisai.init(
        api_key="egis_live_x",
        app="orchestrator",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    # Simulate two distinct sub-agent calls (Anthropic-shaped payload).
    def noop():
        return "ok"

    gate_call(
        source="anthropic",
        target="anthropic.messages.create",
        model="claude-3",
        prompt_text="…",
        stream=False,
        payload={"system": "You are a specialist: Python Developer. Be concise."},
        forward=noop,
    )
    gate_call(
        source="anthropic",
        target="anthropic.messages.create",
        model="claude-3",
        prompt_text="…",
        stream=False,
        payload={"system": "You are a specialist: Copywriter. Be witty."},
        forward=noop,
    )
    # Same system again — should NOT trigger another ensure call.
    gate_call(
        source="anthropic",
        target="anthropic.messages.create",
        model="claude-3",
        prompt_text="…",
        stream=False,
        payload={"system": "You are a specialist: Python Developer. Be concise."},
        forward=noop,
    )

    registered = sorted(a["name"] for a in fake_backend.ensured_agents)
    assert registered == ["Copywriter", "Python Developer"], (
        f"expected exactly two unique sub-agents to register, got {registered}"
    )

    # Every auto-registered agent MUST ship the runtime fingerprint —
    # otherwise the dashboard's Provenance card stays blank for the
    # most common path (system-prompt fingerprinting). Regression
    # guard for the 0.13.2 fix.
    ensure_bodies = fake_backend.ensure_requests
    assert ensure_bodies, "expected at least one /v1/sdk/agents/ensure call"
    for body in ensure_bodies:
        rt = body.get("runtime")
        assert isinstance(rt, dict) and rt, (
            f"runtime fingerprint missing on ensure call: {body!r}"
        )
        # Spot-check the keys the dashboard renders so a key-rename
        # in _runtime.collect_runtime_fingerprint can't silently
        # blank out the Provenance card.
        for key in ("sdk_version", "python", "os", "frameworks"):
            assert key in rt, f"runtime missing {key!r}: {rt!r}"
