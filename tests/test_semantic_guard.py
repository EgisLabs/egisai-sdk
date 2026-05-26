"""``semantic_guard`` SDK-side Behavior.

The LLM-judge for ``semantic_guard`` runs on the EgisAI platform —
the SDK is a thin HTTP client over ``POST /v1/sdk/judge``. These
tests stub that platform endpoint and verify the SDK's contract:

  - ``semantic_guard`` rules call the platform once per gated call
  - ``match=True`` triggers a block in the engine
  - ``match=False`` (or a failed call) fails open
  - The customer's prompt has been redacted by Phase 1 before
    reaching the platform endpoint
"""

from __future__ import annotations

from typing import Any

import httpx


def _semantic_rule() -> dict:
    return {
        "id": 1,
        "name": "guard-database-deletion",
        "type": "semantic_guard",
        "tenant": None,
        "config": {
            "intents": [
                "delete rows from a database table",
                "drop or truncate database tables",
            ],
            "message": "Blocked: destructive database operation.",
        },
    }


def _stub_judge_match(intent: str, confidence: float = 0.92) -> dict[str, Any]:
    """Build the JSON body /v1/sdk/judge returns when matching."""
    return {
        "match": True,
        "intent": intent,
        "confidence": confidence,
        "tokens_in": 200,
        "tokens_out": 12,
    }


def _stub_judge_no_match() -> dict[str, Any]:
    return {
        "match": False,
        "intent": "",
        "confidence": 0.0,
        "tokens_in": 200,
        "tokens_out": 5,
    }


# ── End-to-end: rule fires when the platform says "match" ──────────────


def test_llm_judge_blocks_paraphrased_destructive_intent() -> None:
    """The platform judge handles ``Delete all users`` →
    ``delete rows from a database table`` semantic equivalence
    that no purely local check can. From the SDK's perspective,
    the platform replies ``match=True`` and the engine blocks."""
    from egisai.policy.semantic import SemanticBlocker

    captured: dict = {}

    def transport_handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content.decode()
        return httpx.Response(
            200, json=_stub_judge_match("delete rows from a database table"),
        )

    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
    )
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(transport_handler))

    match = blocker.check(
        "Delete all users",
        {
            "intents": [
                "delete rows from a database table",
                "drop or truncate database tables",
            ]
        },
    )
    assert match is not None
    assert match.intent == "delete rows from a database table"
    assert match.similarity == 0.92
    # Wire shape: SDK calls /v1/sdk/judge on the EgisAI platform,
    # NOT the OpenAI completions API directly.
    assert "/v1/sdk/judge" in captured["url"]
    # The redacted prompt + intents must be in the request body.
    assert "Delete all users" in captured["body"]


def test_llm_judge_blocks_french_destructive_prompt() -> None:
    """A French prompt expressing the same destructive intent
    matches the same rule. The SDK forwards the prompt verbatim to
    the platform; the platform's judge handles the language."""
    from egisai.policy.semantic import SemanticBlocker

    def transport_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_stub_judge_match("delete rows from a database table"),
        )

    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
    )
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(transport_handler))

    match = blocker.check(
        "Supprime tous les utilisateurs de la base de données",
        {"intents": ["delete rows from a database table"]},
    )
    assert match is not None, "French destructive prompt must reach the platform"


def test_no_match_response_passes_call_through() -> None:
    """When the platform replies ``match=False`` (clean prompt or
    sub-threshold confidence), the SDK returns ``None`` and the
    engine allows the call."""
    from egisai.policy.semantic import SemanticBlocker

    def transport_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_stub_judge_no_match())

    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
    )
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(transport_handler))

    match = blocker.check(
        "What is the capital of France?",
        {"intents": ["delete rows from a database table"]},
    )
    assert match is None


# ── Failure modes ─────────────────────────────────────────────────────


def test_platform_unreachable_fails_open() -> None:
    """A network outage of the EgisAI platform must NEVER break
    the customer's call path. The SDK's ``check()`` returns
    ``None`` (allow) and the engine treats the rule as a no-op
    for that call."""
    from egisai.policy.semantic import SemanticBlocker

    def transport_handler(_request: httpx.Request) -> httpx.Response:
        # Simulate the platform being unreachable.
        raise httpx.ConnectError("connection refused")

    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
    )
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(transport_handler))

    match = blocker.check(
        "Delete all users",
        {"intents": ["delete rows from a database table"]},
    )
    assert match is None  # fail-open


def test_platform_500_fails_open() -> None:
    """A platform 5xx response is handled the same as a network
    failure — fail open, customer's app keeps working."""
    from egisai.policy.semantic import SemanticBlocker

    def transport_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="platform overloaded")

    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
    )
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(transport_handler))

    match = blocker.check(
        "Delete all users",
        {"intents": ["delete rows from a database table"]},
    )
    assert match is None


# ── Empty / missing intents shortcuts ──────────────────────────────────


def test_no_intents_short_circuits_without_call() -> None:
    """A ``semantic_guard`` policy with an empty ``intents`` list is
    a no-op — the SDK doesn't waste a network round-trip on a rule
    that can't possibly match anything."""
    from egisai.policy.semantic import SemanticBlocker

    calls = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=_stub_judge_no_match())

    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
    )
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(transport_handler))

    match = blocker.check("any prompt", {"intents": []})
    assert match is None
    assert calls == [], "no platform call should be made for empty intents"


def test_empty_prompt_short_circuits_without_call() -> None:
    """An empty prompt skips the round-trip too."""
    from egisai.policy.semantic import SemanticBlocker

    calls = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=_stub_judge_no_match())

    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
    )
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(transport_handler))

    match = blocker.check("", {"intents": ["delete rows from a database table"]})
    assert match is None
    assert calls == []


# ── Legacy ``engine: "embedding"`` config is now a no-op ───────────────


def test_legacy_embedding_engine_is_no_op_with_warning() -> None:
    """Pre-0.7 ``engine: "embedding"`` config is no longer supported.
    The SDK treats it as a no-op (returns None) and logs a one-time
    warning advising the operator to remove the field."""
    from egisai.policy.semantic import SemanticBlocker

    calls = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, json=_stub_judge_no_match())

    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
    )
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(transport_handler))

    match = blocker.check(
        "Delete all users",
        {
            "intents": ["delete rows from a database table"],
            "engine": "embedding",  # legacy config
        },
    )
    assert match is None
    assert calls == [], "legacy embedding engine must not call the platform"


# ── Threshold is forwarded; judge_model is dropped ────────────────────


def test_threshold_is_forwarded_and_judge_model_is_not() -> None:
    """``threshold`` is the operator-tunable knob — it must reach
    the platform's judge endpoint so the per-policy override
    actually takes effect.

    ``judge_model`` used to ride alongside it. As of SDK 0.27.0 it
    is removed end-to-end: the platform's judge SYSTEM_PROMPT is
    calibrated against a single model, and an operator-supplied
    override would silently skew the threshold semantics every
    other rule assumes. The SDK now drops the field before the
    POST; this test pins that contract."""
    import json

    from egisai.policy.semantic import SemanticBlocker

    captured: dict = {}

    def transport_handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json=_stub_judge_no_match())

    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
    )
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(transport_handler))

    blocker.check(
        "Delete all users",
        {
            "intents": ["delete rows from a database table"],
            "threshold": 0.85,
            # Legacy field; must be silently stripped by the SDK.
            "judge_model": "gpt-4o",
        },
    )
    assert captured["body"]["threshold"] == 0.85
    assert "judge_model" not in captured["body"], (
        "SDK 0.27.0+ must NOT forward judge_model — the platform "
        "controls the judge model exclusively."
    )
    assert captured["body"]["prompt_text"] == "Delete all users"
    assert captured["body"]["intents"] == ["delete rows from a database table"]
