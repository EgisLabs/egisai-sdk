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
