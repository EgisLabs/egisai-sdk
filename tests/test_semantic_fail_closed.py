"""``semantic_on_outage="block"`` makes the semantic guard fail closed.

The default behavior is to fail open on judge outage (preserves
availability). Operators that consider semantic guard their primary
defense for a workload can opt into fail-closed mode at ``init()``.
"""

from __future__ import annotations

import httpx

from egisai.policy.semantic import SemanticBlocker


def _make_blocker(
    handler: callable,  # type: ignore[valid-type]
    *,
    on_outage: str = "allow",
) -> SemanticBlocker:
    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
        on_outage=on_outage,
    )
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return blocker


def _outage_handler(_request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("simulated platform outage")


def _ok_no_match_handler(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"match": False, "intent": "", "confidence": 0.0,
              "tokens_in": 0, "tokens_out": 0},
    )


def test_default_outage_behaviour_is_allow() -> None:
    blocker = _make_blocker(_outage_handler)  # default: on_outage="allow"
    match = blocker.check(
        "Delete all users",
        {"intents": ["delete rows from a database table"]},
    )
    assert match is None  # fail-open


def test_block_on_outage_returns_synthetic_match() -> None:
    blocker = _make_blocker(_outage_handler, on_outage="block")
    match = blocker.check(
        "Delete all users",
        {"intents": ["delete rows from a database table"]},
    )
    assert match is not None
    assert match.intent == "<judge unavailable>"
    assert match.similarity == 0.0


def test_block_on_outage_does_not_affect_normal_match_path() -> None:
    """When the judge IS reachable, ``on_outage='block'`` is irrelevant —
    we behave exactly like the default."""
    blocker = _make_blocker(_ok_no_match_handler, on_outage="block")
    match = blocker.check(
        "What is the capital of France?",
        {"intents": ["delete rows from a database table"]},
    )
    assert match is None


def test_invalid_on_outage_value_raises() -> None:
    import pytest

    with pytest.raises(ValueError):
        SemanticBlocker(
            platform_api_key="egis_live_test",
            platform_base_url="http://fake-platform",
            on_outage="suspend",  # type: ignore[arg-type]
        )
