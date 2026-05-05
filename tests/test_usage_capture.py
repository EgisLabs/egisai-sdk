"""Audit events must carry real latency and token counts.

Pre-fix bug: every row in the dashboard's Activity table showed
``0 / 0`` tokens and ``0 ms`` latency because the SDK enqueued the event
*before* calling ``forward()``, and ``build_event`` never populated
those fields anyway. The fix moves the enqueue to after ``forward()``
returns so we can stamp ``latency_ms`` and pull token counts off the
framework's response object.
"""

from __future__ import annotations

import time
from types import SimpleNamespace


def test_event_carries_latency_and_tokens(fake_backend) -> None:
    """End-to-end through gate_call with a fake response that mimics
    OpenAI's ChatCompletion shape."""
    import egisai
    from egisai._patches._common import gate_call
    from egisai._patches.openai import _extract_chat_usage

    egisai.init(
        api_key="egis_live_x",
        app="t",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    fake_response = SimpleNamespace(
        id="chatcmpl-1",
        usage=SimpleNamespace(prompt_tokens=42, completion_tokens=137),
    )

    def slow_forward():
        # Simulate ~50ms of API latency; just enough to land a non-zero
        # latency_ms reading without slowing the test suite down.
        time.sleep(0.05)
        return fake_response

    out = gate_call(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4",
        prompt_text="hello",
        stream=False,
        payload={"messages": [{"role": "user", "content": "hello"}]},
        extract_usage=_extract_chat_usage,
        forward=slow_forward,
    )
    assert out is fake_response

    # Drain the queue to the fake backend.
    from egisai import shutdown

    shutdown()

    events = fake_backend.events_received
    assert len(events) == 1
    ev = events[0]
    assert ev["tokens_in"] == 42
    assert ev["tokens_out"] == 137
    # Latency should be at least the sleep we did, but allow generous
    # headroom for CI jitter.
    assert ev["latency_ms"] >= 40
    assert ev["latency_ms"] < 5000
    # Policy latency is reported alongside model latency. With no
    # rules cached, evaluate() returns immediately — so it's small but
    # still present (>= 0).
    assert "policy_latency_ms" in ev
    assert ev["policy_latency_ms"] >= 0
    assert ev["prompt_chars"] == len("hello")
    assert ev["verdict"] == "allow"


def test_anthropic_extractor_reads_input_output_tokens() -> None:
    """Anthropic's usage shape is ``input_tokens`` / ``output_tokens``,
    NOT ``prompt_tokens`` / ``completion_tokens``."""
    from egisai._patches.anthropic import _extract_message_usage

    response = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=12, output_tokens=34)
    )
    out = _extract_message_usage(response)
    assert out == {"tokens_in": 12, "tokens_out": 34}


def test_google_extractor_reads_usage_metadata() -> None:
    """Gemini wraps usage in ``usage_metadata`` with ``prompt_token_count``
    / ``candidates_token_count``."""
    from egisai._patches.google import _extract_gemini_usage

    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(
            prompt_token_count=10,
            candidates_token_count=20,
            total_token_count=30,
        )
    )
    out = _extract_gemini_usage(response)
    assert out == {"tokens_in": 10, "tokens_out": 20}


def test_extractor_returns_empty_on_missing_usage() -> None:
    """Responses missing usage info shouldn't crash — just return
    no token info, leaving the audit row at zeros."""
    from egisai._patches.anthropic import _extract_message_usage
    from egisai._patches.openai import _extract_chat_usage

    bare = SimpleNamespace(id="x")  # no .usage
    assert _extract_chat_usage(bare) == {}
    assert _extract_message_usage(bare) == {}


def test_dict_shaped_response_also_works() -> None:
    """Legacy v0 / test stubs use plain dicts; extractors must
    handle that too."""
    from egisai._patches.openai import _extract_chat_usage

    response = {"id": "x", "usage": {"prompt_tokens": 5, "completion_tokens": 7}}
    out = _extract_chat_usage(response)
    assert out == {"tokens_in": 5, "tokens_out": 7}


def test_blocked_call_still_logs_zero_latency_and_tokens(fake_backend) -> None:
    """When a policy denies, ``forward()`` never runs — we shouldn't
    leave the call hanging waiting for tokens that will never arrive."""
    import egisai
    from egisai._patches._common import gate_call

    fake_backend.set_rules(
        [
            {
                "id": 1,
                "name": "block-anything",
                "type": "deny_regex",
                "tenant": None,
                "config": {"pattern": ".*", "message": "all blocked"},
            }
        ],
        etag='"sem"',
    )

    egisai.init(
        api_key="egis_live_x",
        app="t",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    forwarded = []

    def forward():
        forwarded.append(True)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=99, completion_tokens=99))

    try:
        gate_call(
            source="openai",
            target="openai.chat.completions.create",
            model="gpt-4",
            prompt_text="anything",
            stream=False,
            payload={"messages": [{"role": "user", "content": "x"}]},
            forward=forward,
        )
    except PermissionError:
        pass

    assert forwarded == [], "forward() must NOT run when the policy denies"

    from egisai import shutdown

    shutdown()

    events = fake_backend.events_received
    assert len(events) == 1
    ev = events[0]
    assert ev["verdict"] == "block"
    # Model never ran → latency_ms stays zero. policy_latency_ms is
    # whatever we spent in evaluate() — small but recorded so the
    # operator can see how much governance overhead a denied call
    # incurred too.
    assert ev["latency_ms"] == 0
    assert "policy_latency_ms" in ev
    assert ev["policy_latency_ms"] >= 0
    assert ev.get("tokens_in") is None or ev["tokens_in"] == 0
