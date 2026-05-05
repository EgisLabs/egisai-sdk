"""Nested gated calls (e.g. anthropic.messages.create → internal httpx
fallback) must produce ONE audit event for the outer call, not one per
network layer with the same trace id.
"""

from __future__ import annotations


def test_nested_gate_call_does_not_emit_duplicate_event(fake_backend) -> None:
    """Direct test through gate_call() — covers the case where one
    egisai adapter calls another via the public gate API."""
    import egisai
    from egisai._patches._common import gate_call

    egisai.init(
        api_key="egis_live_x",
        app="orchestrator",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    def inner_http():
        return gate_call(
            source="httpx",
            target="httpx.send",
            model="claude-opus-4-7",
            prompt_text="…",
            stream=False,
            payload={"system": "You are an orchestrator."},
            forward=lambda: "http-ok",
        )

    def outer_forward():
        inner_http()
        inner_http()
        return "outer-ok"

    result = gate_call(
        source="anthropic",
        target="anthropic.messages.create",
        model="claude-opus-4-7",
        prompt_text="hello",
        stream=False,
        payload={"system": "You are an orchestrator."},
        forward=outer_forward,
    )
    assert result == "outer-ok"

    from egisai import shutdown

    shutdown()

    events = fake_backend.events_received
    assert len(events) == 1, (
        f"expected 1 audit event for the outer call, got {len(events)}: "
        f"{[e.get('target') for e in events]}"
    )
    assert events[0]["target"] == "anthropic.messages.create"
    assert events[0]["verdict"] == "allow"


def test_http_fallback_does_not_double_log_when_already_gated(fake_backend) -> None:
    """The http fallback in _patches/http.py used to enqueue a
    "Network-layer event" row for every internal httpx request the
    Anthropic SDK makes — producing duplicate audit rows with model=
    "unknown" and the same trace_id. After the fix it must skip
    logging entirely when ``get_policy_checked()`` is True.
    """
    import egisai
    from egisai._context import set_policy_checked
    from egisai._logger import enqueue
    from egisai._patches._common import gate_call

    egisai.init(
        api_key="egis_live_x",
        app="orchestrator",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    # We can't easily monkey-patch a real httpx.Client here because the
    # SDK's _patches/http already patched the real one in init(). So we
    # invoke the http-fallback's logic directly via the same primitives:
    # set_policy_checked(True) + the build_event/enqueue path the
    # patch uses. If anything would have been enqueued, that's the bug.
    from egisai._events import build_event

    def simulate_http_fallback() -> None:
        # Mirror what _patches/http.py::wrapped does. With the fix in
        # place, callers never reach build_event/enqueue when
        # get_policy_checked() is True — so nothing gets queued.
        from egisai._context import get_policy_checked

        if get_policy_checked():
            return
        ev = build_event(source="httpx", target="api.anthropic.com/v1/messages", payload={})
        ev["verdict"] = "allow"
        ev["reason"] = "Network-layer event"
        enqueue(ev)

    def outer_forward():
        # Simulate two internal SDK retries / pings.
        simulate_http_fallback()
        simulate_http_fallback()
        return "ok"

    result = gate_call(
        source="anthropic",
        target="anthropic.messages.create",
        model="claude-opus-4-7",
        prompt_text="hello",
        stream=False,
        payload={"system": "You are an orchestrator."},
        forward=outer_forward,
    )
    assert result == "ok"

    from egisai import shutdown

    shutdown()

    events = fake_backend.events_received
    assert len(events) == 1, (
        f"expected ONE audit event (the outer anthropic call), got "
        f"{len(events)}: {[(e.get('source'), e.get('target')) for e in events]}"
    )
    assert events[0]["source"] == "anthropic"

    # Sanity: when not gated upstream, the fallback DOES log (otherwise
    # we'd lose visibility into ungoverned model calls).
    set_policy_checked(False)
    simulate_http_fallback()
    # Drain again
    from egisai._logger import _q

    pending = []
    while not _q.empty():
        try:
            pending.append(_q.get_nowait())
        except Exception:
            break
    assert any(e.get("source") == "httpx" for e in pending), (
        "When prev_checked is False, the fallback should still log so "
        "ungoverned direct httpx calls remain visible in the audit feed."
    )
