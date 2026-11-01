"""ETag-driven policy cache."""

from __future__ import annotations


def test_etag_round_trip_avoids_redownload(fake_backend) -> None:
    fake_backend.set_rules([{"id": 1, "name": "x", "type": "deny_regex",
                             "tenant": None, "config": {"pattern": "x"}}],
                            etag='"v1"')

    import egisai

    egisai.init(api_key="egis_live_x", app="a", env="t",
                base_url="http://fake", enable_sse=False)

    from egisai._policy_cache import refresh_now

    # First refresh: server sends 200 with rules.
    # Second refresh: SDK sends If-None-Match -> server 304 -> cache untouched.
    changed = refresh_now()
    assert changed is False  # already in sync from init's initial fetch


def test_cache_updates_when_rules_change(fake_backend) -> None:
    fake_backend.set_rules(
        [{"id": 1, "name": "old", "type": "deny_regex", "tenant": None, "config": {}}],
        etag='"v1"',
    )

    import egisai

    egisai.init(api_key="egis_live_x", app="a", env="t",
                base_url="http://fake", enable_sse=False)

    from egisai._policy_cache import get_rules, refresh_now

    assert [r.name for r in get_rules()] == ["old"]

    fake_backend.set_rules(
        [{"id": 2, "name": "new", "type": "deny_regex", "tenant": None, "config": {}}],
        etag='"v2"',
    )
    assert refresh_now() is True
    assert [r.name for r in get_rules()] == ["new"]


def test_cache_carries_mcp_enforcement_directives(fake_backend) -> None:
    """Stopped MCP servers and disapproved tools ride the policy
    snapshot into the SDK cache, so the inbound gate can refuse them."""
    sid = "AB12CD34-0000-0000-0000-000000000000"
    fake_backend.set_rules(
        [],
        etag='"v1"',
        paused_mcp_server_ids=[sid],
        blocked_mcp_tools=[f"{sid}:delete_repo"],
    )

    import egisai

    egisai.init(api_key="egis_live_x", app="a", env="t",
                base_url="http://fake", enable_sse=False)

    from egisai._policy_cache import (
        get_blocked_mcp_tools,
        get_paused_mcp_server_ids,
    )

    # Server ids are lowercased for case-insensitive matching; the tool
    # name half of the blocked key is preserved verbatim.
    assert sid.lower() in get_paused_mcp_server_ids()
    assert f"{sid.lower()}:delete_repo" in get_blocked_mcp_tools()


def test_cache_clears_mcp_directives_when_resumed(fake_backend) -> None:
    sid = "ab12cd34-0000-0000-0000-000000000000"
    fake_backend.set_rules(
        [], etag='"v1"', paused_mcp_server_ids=[sid],
    )

    import egisai

    egisai.init(api_key="egis_live_x", app="a", env="t",
                base_url="http://fake", enable_sse=False)

    from egisai._policy_cache import get_paused_mcp_server_ids, refresh_now

    assert sid in get_paused_mcp_server_ids()

    # Operator resumes the server: the next snapshot drops the id.
    fake_backend.set_rules([], etag='"v2"', paused_mcp_server_ids=[])
    assert refresh_now() is True
    assert sid not in get_paused_mcp_server_ids()


def test_cache_carries_ungoverned_mcp_servers(fake_backend) -> None:
    """Ungoverned MCP servers ride the snapshot into the SDK cache, so
    the inbound gate can skip policy for them (monitor-only)."""
    sid = "AB12CD34-0000-0000-0000-000000000000"
    fake_backend.set_rules(
        [], etag='"v1"', ungoverned_mcp_server_ids=[sid],
    )

    import egisai

    egisai.init(api_key="egis_live_x", app="a", env="t",
                base_url="http://fake", enable_sse=False)

    from egisai._policy_cache import (
        get_ungoverned_mcp_server_ids,
        refresh_now,
    )

    assert sid.lower() in get_ungoverned_mcp_server_ids()

    # Operator re-governs the server: the next snapshot drops the id.
    fake_backend.set_rules([], etag='"v2"', ungoverned_mcp_server_ids=[])
    assert refresh_now() is True
    assert sid.lower() not in get_ungoverned_mcp_server_ids()
