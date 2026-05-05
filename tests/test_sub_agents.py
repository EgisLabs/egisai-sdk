"""set_context(agent="…") registers the role with the platform and routes
events under that agent_id."""

from __future__ import annotations


def test_set_context_with_agent_name_calls_ensure(fake_backend) -> None:
    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="orchestrator",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    # First call: should hit the ensure endpoint and cache the id.
    egisai.set_context(agent="Python Developer")

    names = [a["name"] for a in fake_backend.ensured_agents]
    assert names == ["Python Developer"]


def test_set_context_caches_repeat_calls(fake_backend) -> None:
    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="orchestrator",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    egisai.set_context(agent="Python Developer")
    egisai.set_context(agent="Python Developer")
    egisai.set_context(agent="Python Developer")

    names = [a["name"] for a in fake_backend.ensured_agents]
    assert names == ["Python Developer"], (
        "ensure_agent should be called exactly once per unique role name"
    )


def test_event_emits_contextual_agent_id(fake_backend) -> None:
    import egisai
    from egisai._events import build_event

    egisai.init(
        api_key="egis_live_x",
        app="orchestrator",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    # Default: events are tagged with the orchestrator agent_id from handshake.
    ev_default = build_event(source="t", target="x", payload={})
    assert ev_default["agent_id"] == "00000000-0000-0000-0000-000000000002"
    assert ev_default["app"] == "orchestrator"

    # After set_context(agent="…"), the contextual id wins.
    egisai.set_context(agent="Copywriter")
    ev_sub = build_event(source="t", target="x", payload={})
    assert ev_sub["agent_id"] != "00000000-0000-0000-0000-000000000002"
    assert ev_sub["agent_id"].startswith("00000000-0000-0000-0000-")
    assert ev_sub["app"] == "Copywriter"


def test_multiple_roles_each_register(fake_backend) -> None:
    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="orchestrator",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    for role in ["Python Developer", "Copywriter", "Researcher"]:
        egisai.set_context(agent=role)

    registered = sorted(a["name"] for a in fake_backend.ensured_agents)
    assert registered == ["Copywriter", "Python Developer", "Researcher"]
