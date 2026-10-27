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


def test_failed_handshake_block_prints_not_active(
    fake_backend, capsys
) -> None:
    """Fail-closed init must not claim '✓ active' after a 401 handshake."""
    import egisai

    fake_backend.handshake_status = 401
    egisai.init(
        api_key="egis_live_bad",
        app="test-app",
        env="test",
        base_url="http://fake",
        on_outage="block",
        enable_sse=False,
        quiet=False,
    )
    out = capsys.readouterr().out
    assert "✓ [egisai] active" not in out
    assert "not active: governed calls will be refused" in out


def test_successful_handshake_still_prints_active(
    fake_backend, capsys
) -> None:
    import egisai

    egisai.init(
        api_key="egis_live_ok",
        app="test-app",
        env="test",
        base_url="http://fake",
        enable_sse=False,
        quiet=False,
    )
    out = capsys.readouterr().out
    assert "✓ [egisai] active" in out
    assert "not active" not in out
