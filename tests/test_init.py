"""init() handshakes, fetches policies, and starts background workers."""

from __future__ import annotations


def test_init_handshakes_and_caches_policies(fake_backend) -> None:
    fake_backend.set_rules(
        [
            {
                "id": 1,
                "name": "block-foo",
                "type": "deny_regex",
                "tenant": None,
                "config": {"pattern": "foo"},
            }
        ],
        etag='"v1"',
    )

    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="test-app",
        env="test",
        base_url="http://fake",
        enable_sse=False,  # don't try to open a stream in tests
    )

    # Handshake happened
    assert fake_backend.handshake_calls == 1

    # Cache populated
    from egisai._policy_cache import get_etag, get_rules

    rules = get_rules()
    assert len(rules) == 1
    assert rules[0].name == "block-foo"
    assert get_etag() == '"v1"'


def test_init_is_idempotent(fake_backend) -> None:
    import egisai

    egisai.init(api_key="egis_live_x", app="a", env="t", base_url="http://fake", enable_sse=False)
    egisai.init(api_key="egis_live_x", app="a", env="t", base_url="http://fake", enable_sse=False)
    egisai.init(api_key="egis_live_x", app="a", env="t", base_url="http://fake", enable_sse=False)

    # Only one handshake; subsequent calls are ignored.
    assert fake_backend.handshake_calls == 1
