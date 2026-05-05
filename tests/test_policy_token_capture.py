"""Audit events must carry the tokens that policy evaluation consumed.

When ``semantic_guard`` runs its LLM judge, those tokens are real
OpenAI charges that should appear on the audit row alongside the
model-call tokens — same way ``policy_latency_ms`` lives next to
``latency_ms``.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx


def test_judge_usage_lands_on_audit_event(fake_backend) -> None:
    import egisai
    from egisai._evaluator import _get_semantic_blocker
    from egisai._patches._common import gate_call

    # One semantic_guard rule active → judge runs on every gated call.
    fake_backend.set_rules(
        [
            {
                "id": 1,
                "name": "guard-deletion",
                "type": "semantic_guard",
                "tenant": None,
                "config": {"intents": ["delete database rows"]},
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

    # Mock the platform's /v1/sdk/judge endpoint (post-0.7 hybrid).
    # The platform forwards the operator's intent + prompt to OpenAI
    # internally, then returns a verdict + token-usage breakdown the
    # SDK records on this call's policy_tokens_* counters.
    def transport(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "match": False,
                "intent": "",
                "confidence": 0.0,
                "tokens_in": 234,
                "tokens_out": 7,
            },
        )

    blocker = _get_semantic_blocker()
    assert blocker is not None
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(transport))

    # Now run a normal user-side call. The fake "model" returns a
    # response with its own (different) usage block.
    fake_response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=42, completion_tokens=137),
    )

    from egisai._patches.openai import _extract_chat_usage

    out = gate_call(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4",
        prompt_text="What is the capital of France?",
        stream=False,
        payload={"messages": [{"role": "user", "content": "x"}]},
        extract_usage=_extract_chat_usage,
        forward=lambda: fake_response,
    )
    assert out is fake_response

    from egisai import shutdown

    shutdown()

    events = fake_backend.events_received
    assert len(events) == 1
    ev = events[0]

    # Model usage stays its own number…
    assert ev["tokens_in"] == 42
    assert ev["tokens_out"] == 137
    # …and the policy step's token consumption is recorded separately.
    assert ev["policy_tokens_in"] == 234
    assert ev["policy_tokens_out"] == 7


def test_no_semantic_policy_means_zero_policy_tokens(fake_backend) -> None:
    """When no policy makes an LLM call (e.g. only deny_regex active),
    the policy-token fields must stay at 0 — not None, not missing."""
    import egisai
    from egisai._patches._common import gate_call

    fake_backend.set_rules(
        [
            {
                "id": 1,
                "name": "noop",
                "type": "deny_regex",
                "tenant": None,
                "config": {"pattern": "won't match anything ZZZZ"},
            }
        ],
        etag='"r"',
    )

    egisai.init(
        api_key="egis_live_x",
        app="t",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    out = gate_call(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4",
        prompt_text="hello",
        stream=False,
        payload={"messages": [{"role": "user", "content": "hello"}]},
        forward=lambda: SimpleNamespace(),
    )
    assert out is not None

    from egisai import shutdown

    shutdown()

    events = fake_backend.events_received
    assert len(events) == 1
    assert events[0]["policy_tokens_in"] == 0
    assert events[0]["policy_tokens_out"] == 0


def test_policy_tokens_dont_leak_between_calls(fake_backend) -> None:
    """Each call gets a clean accumulator — usage from call N doesn't
    show up on the audit row for call N+1."""
    import egisai
    from egisai._evaluator import _get_semantic_blocker
    from egisai._patches._common import gate_call

    fake_backend.set_rules(
        [
            {
                "id": 1,
                "name": "guard",
                "type": "semantic_guard",
                "tenant": None,
                "config": {"intents": ["something"]},
            }
        ],
        etag='"r"',
    )

    egisai.init(
        api_key="egis_live_x",
        app="t",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    seq = [
        # First call's judge response — 50/3 tokens.
        # Platform-side ``/v1/sdk/judge`` shape (post-0.7 hybrid).
        httpx.Response(
            200,
            json={
                "match": False, "intent": "", "confidence": 0.0,
                "tokens_in": 50, "tokens_out": 3,
            },
        ),
        # Second call — 80/4 tokens.
        httpx.Response(
            200,
            json={
                "match": False, "intent": "", "confidence": 0.0,
                "tokens_in": 80, "tokens_out": 4,
            },
        ),
    ]

    def transport(request: httpx.Request) -> httpx.Response:
        return seq.pop(0)

    blocker = _get_semantic_blocker()
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(transport))

    for _ in range(2):
        gate_call(
            source="openai",
            target="openai.chat.completions.create",
            model="gpt-4",
            prompt_text="anything",
            stream=False,
            payload={"messages": [{"role": "user", "content": "x"}]},
            forward=lambda: SimpleNamespace(),
        )

    from egisai import shutdown

    shutdown()

    events = fake_backend.events_received
    assert len(events) == 2
    # Each call gets ONLY its own judge's tokens — not cumulative.
    assert events[0]["policy_tokens_in"] == 50
    assert events[0]["policy_tokens_out"] == 3
    assert events[1]["policy_tokens_in"] == 80
    assert events[1]["policy_tokens_out"] == 4
