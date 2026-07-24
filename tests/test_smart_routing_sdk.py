"""Smart Model Routing — SDK-side tests.

Layers under test, bottom-up:

1. ``egisai._routing`` — decision client (dormancy, caching, the
   affirmative-disabled backoff, fail-open) + provider detection +
   the cross-provider payload canonicalizers.
2. Gate plumbing in ``egisai._patches._common`` — ``_prepare_route``
   applies same-provider swaps through the adapter, and
   ``_routed_forward_sync`` reverts + retries on the requested model
   when the routed call fails (a swap must never break a call that
   would have succeeded).
3. End-to-end through the patched OpenAI client against the fake
   backend: the served model is rewritten, the audit event carries
   the requested→served pair, and a dormant org sends zero ``/route``
   requests.
"""

from __future__ import annotations

import sys
import types
from typing import Any

from egisai import _routing
from egisai._patches._common import (
    _prepare_route,
    _routed_forward_sync,
    _stamp_route,
)
from egisai._routing import (
    RoutingAdapter,
    canonicalize_anthropic_messages,
    canonicalize_openai_messages,
    detect_available_providers,
    provider_for_model,
)


def _decision(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "model": "gpt-4o-mini",
        "provider": "openai",
        "direction": "downgrade",
        "reason": "trivial request",
        "projected_savings_usd": 0.01,
        "requested_provider": "openai",
    }
    base.update(overrides)
    return base


# ── Provider detection ───────────────────────────────────────────────


class TestProviderDetection:
    def test_provider_for_model(self) -> None:
        assert provider_for_model("gpt-4o") == "openai"
        assert provider_for_model("o3") == "openai"
        assert provider_for_model("claude-opus-4-8") == "anthropic"
        assert provider_for_model("Claude-Haiku-4-5") == "anthropic"
        assert provider_for_model("gemini-2.5-flash") == "google"
        assert provider_for_model("models/gemini-3.1-pro") == "google"
        # Unknown names default to openai (mirrors the backend).
        assert provider_for_model(None) == "openai"

    def test_requested_provider_always_included(self, monkeypatch) -> None:
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY",
                     "GOOGLE_API_KEY", "GEMINI_API_KEY"):
            monkeypatch.delenv(name, raising=False)
        assert detect_available_providers("anthropic", allow_cross=True) == [
            "anthropic"
        ]

    def test_cross_candidates_require_env_keys(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        assert detect_available_providers("anthropic", allow_cross=True) == [
            "anthropic",
            "openai",
        ]
        # allow_cross=False keeps the floor even with keys present.
        assert detect_available_providers("anthropic", allow_cross=False) == [
            "anthropic"
        ]


# ── Decision client ──────────────────────────────────────────────────


def _maybe_route(**overrides: Any) -> dict[str, Any] | None:
    kwargs: dict[str, Any] = {
        "model": "gpt-4o",
        "prompt_preview": "What does HTTP stand for?",
        "prompt_chars": 26,
        "has_tools": False,
        "agent_id": None,
        "allow_cross": False,
    }
    kwargs.update(overrides)
    return _routing.maybe_route(**kwargs)


class TestMaybeRoute:
    def test_dormant_when_hint_false(self, monkeypatch) -> None:
        calls: list[Any] = []
        monkeypatch.setattr(
            "egisai._backend.route", lambda **kw: calls.append(kw)
        )
        _routing.set_enabled_hint(False)
        assert _maybe_route() is None
        assert calls == []

    def test_decision_returned_and_cached(self, monkeypatch) -> None:
        calls: list[Any] = []

        def fake_route(**kw: Any) -> dict[str, Any]:
            calls.append(kw)
            return {
                "routed": True,
                "model": "gpt-4o-mini",
                "provider": "openai",
                "direction": "downgrade",
                "reason": "trivial",
                "projected_savings_usd": 0.01,
            }

        monkeypatch.setattr("egisai._backend.route", fake_route)
        first = _maybe_route()
        second = _maybe_route()
        assert first is not None
        assert first["model"] == "gpt-4o-mini"
        assert first["requested_provider"] == "openai"
        assert second == first
        assert len(calls) == 1  # cache absorbed the repeat

    def test_keep_answer_cached_as_none(self, monkeypatch) -> None:
        calls: list[Any] = []

        def fake_route(**kw: Any) -> dict[str, Any]:
            calls.append(kw)
            return {"routed": False}

        monkeypatch.setattr("egisai._backend.route", fake_route)
        assert _maybe_route() is None
        assert _maybe_route() is None
        assert len(calls) == 1

    def test_disabled_answer_backs_off(self, monkeypatch) -> None:
        calls: list[Any] = []

        def fake_route(**kw: Any) -> dict[str, Any]:
            calls.append(kw)
            return {"routed": False, "disabled": True}

        monkeypatch.setattr("egisai._backend.route", fake_route)
        assert _maybe_route() is None
        # Backed off — a DIFFERENT prompt still doesn't ask again.
        assert _maybe_route(prompt_preview="another prompt") is None
        assert len(calls) == 1

    def test_invalidate_clears_backoff_and_relearns(self, monkeypatch) -> None:
        calls: list[Any] = []

        def fake_route(**kw: Any) -> dict[str, Any]:
            calls.append(kw)
            return {"routed": False, "disabled": True}

        monkeypatch.setattr("egisai._backend.route", fake_route)
        _routing.set_enabled_hint(False)
        assert _maybe_route() is None
        assert calls == []  # dormant

        # routing.changed SSE → invalidate() → hint flips to "unknown"
        # and the client asks again.
        _routing.invalidate()
        assert _maybe_route() is None
        assert len(calls) == 1

    def test_backend_error_fails_open(self, monkeypatch) -> None:
        def boom(**kw: Any) -> dict[str, Any]:
            raise RuntimeError("network down")

        monkeypatch.setattr("egisai._backend.route", boom)
        assert _maybe_route() is None


# ── Cross-provider canonicalizers ────────────────────────────────────


class TestCanonicalizers:
    def test_openai_plain_messages(self) -> None:
        out = canonicalize_openai_messages(
            [
                {"role": "developer", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ]
        )
        assert out == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]

    def test_openai_rejects_rich_content(self) -> None:
        multimodal = [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "x"}}],
            }
        ]
        assert canonicalize_openai_messages(multimodal) is None
        assert canonicalize_openai_messages([{"role": "tool", "content": "x"}]) is None
        assert canonicalize_openai_messages([]) is None
        # No user turn → nothing to answer; skip translation.
        assert (
            canonicalize_openai_messages(
                [{"role": "system", "content": "hello"}]
            )
            is None
        )

    def test_anthropic_system_and_text_blocks(self) -> None:
        out = canonicalize_anthropic_messages(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part one"},
                        {"type": "text", "text": "part two"},
                    ],
                }
            ],
            system="stay factual",
        )
        assert out == [
            {"role": "system", "content": "stay factual"},
            {"role": "user", "content": "part one\npart two"},
        ]

    def test_anthropic_structured_system_blocks(self) -> None:
        out = canonicalize_anthropic_messages(
            [{"role": "user", "content": "hi"}],
            system=[{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        )
        assert out is not None
        assert out[0] == {"role": "system", "content": "a\nb"}

    def test_anthropic_rejects_non_text_blocks(self) -> None:
        assert (
            canonicalize_anthropic_messages(
                [
                    {
                        "role": "user",
                        "content": [{"type": "image", "source": {}}],
                    }
                ],
                system=None,
            )
            is None
        )
        assert canonicalize_anthropic_messages([], system=None) is None


# ── Gate plumbing ────────────────────────────────────────────────────


class _FakeCall:
    """Stands in for a patch closure's ``kwargs`` dict."""

    def __init__(self, model: str) -> None:
        self.kwargs = {"model": model}

    def adapter(self) -> RoutingAdapter:
        def apply(new_model: str) -> bool:
            self.kwargs["model"] = new_model
            return True

        return RoutingAdapter(apply_same_provider=apply)


class TestGatePlumbing:
    def test_same_provider_swap_applied_and_stamped(self, monkeypatch) -> None:
        call = _FakeCall("gpt-4o")
        monkeypatch.setattr(
            _routing, "maybe_route", lambda **kw: _decision()
        )
        ev: dict[str, Any] = {
            "prompt_preview": "hi",
            "prompt_chars": 2,
            "model": "gpt-4o",
        }
        state = _prepare_route(
            ev=ev,
            model="gpt-4o",
            stream=False,
            payload={},
            routing_adapter=call.adapter(),
        )
        assert state is not None and state["applied"]
        assert call.kwargs["model"] == "gpt-4o-mini"

        _stamp_route(ev, state)
        assert ev["model"] == "gpt-4o-mini"
        assert ev["requested_model"] == "gpt-4o"
        assert ev["requested_provider"] == "openai"
        assert ev["routing_applied"] is True
        assert ev["routing_direction"] == "downgrade"

    def test_cross_decision_without_cross_support_not_applied(
        self, monkeypatch
    ) -> None:
        call = _FakeCall("gpt-4o")
        monkeypatch.setattr(
            _routing,
            "maybe_route",
            lambda **kw: _decision(model="claude-haiku-4-5", provider="anthropic"),
        )
        state = _prepare_route(
            ev={"prompt_preview": "hi", "prompt_chars": 2},
            model="gpt-4o",
            stream=False,
            payload={},
            routing_adapter=call.adapter(),
        )
        assert state is None
        assert call.kwargs["model"] == "gpt-4o"  # untouched

    def test_no_adapter_is_a_noop(self, monkeypatch) -> None:
        called: list[Any] = []
        monkeypatch.setattr(
            _routing, "maybe_route", lambda **kw: called.append(kw)
        )
        assert (
            _prepare_route(
                ev={}, model="gpt-4o", stream=False, payload={},
                routing_adapter=None,
            )
            is None
        )
        assert called == []

    def test_routed_failure_reverts_and_retries(self, monkeypatch) -> None:
        call = _FakeCall("gpt-4o")
        monkeypatch.setattr(
            _routing, "maybe_route", lambda **kw: _decision()
        )
        state = _prepare_route(
            ev={"prompt_preview": "hi", "prompt_chars": 2},
            model="gpt-4o",
            stream=False,
            payload={},
            routing_adapter=call.adapter(),
        )
        assert state is not None
        assert call.kwargs["model"] == "gpt-4o-mini"

        attempts: list[str] = []

        def forward() -> str:
            attempts.append(call.kwargs["model"])
            if call.kwargs["model"] == "gpt-4o-mini":
                raise RuntimeError("routed model unavailable")
            return "ok"

        assert _routed_forward_sync(state, forward) == "ok"
        # First attempt on the routed model, retry on the requested one.
        assert attempts == ["gpt-4o-mini", "gpt-4o"]
        # The revert flipped the applied flag so the gate won't stamp
        # routing fields on the audit event.
        assert state["applied"] is False

    def test_tools_and_streams_stay_same_provider(self, monkeypatch) -> None:
        seen: dict[str, Any] = {}

        def spy(**kw: Any) -> None:
            seen.update(kw)
            return None

        monkeypatch.setattr(_routing, "maybe_route", spy)
        adapter = RoutingAdapter(
            apply_same_provider=lambda m: True,
            build_cross_forward=lambda d: None,
        )
        _prepare_route(
            ev={"prompt_preview": "hi", "prompt_chars": 2},
            model="gpt-4o",
            stream=True,
            payload={"tools": [{"type": "function"}]},
            routing_adapter=adapter,
        )
        assert seen["allow_cross"] is False
        assert seen["has_tools"] is True


# ── End-to-end through the patched OpenAI client ─────────────────────


def _install_fake_openai() -> type:
    fake = types.ModuleType("openai")
    res = types.ModuleType("openai.resources")
    chat = types.ModuleType("openai.resources.chat")
    completions = types.ModuleType("openai.resources.chat.completions")
    responses = types.ModuleType("openai.resources.responses")

    class Completions:
        def create(self, **kwargs):
            return {"id": "real", "kwargs": kwargs}

    class AsyncCompletions:
        async def create(self, **kwargs):
            return {"id": "real-async", "kwargs": kwargs}

    completions.Completions = Completions
    completions.AsyncCompletions = AsyncCompletions
    sys.modules.update(
        {
            "openai": fake,
            "openai.resources": res,
            "openai.resources.chat": chat,
            "openai.resources.chat.completions": completions,
            "openai.resources.responses": responses,
        }
    )
    return Completions


def test_e2e_same_provider_swap_and_audit(fake_backend) -> None:
    Completions = _install_fake_openai()
    fake_backend.features = {"smart_model_routing": True}
    fake_backend.route_response = {
        "routed": True,
        "model": "gpt-4o-mini",
        "provider": "openai",
        "direction": "downgrade",
        "reason": "trivial request",
        "projected_savings_usd": 0.01,
    }

    import egisai

    egisai.init(
        api_key="egis_live_x", app="a", env="t",
        base_url="http://fake", enable_sse=False,
    )

    out = Completions().create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What does HTTP stand for?"}],
    )
    # The upstream call ran on the routed model.
    assert out["kwargs"]["model"] == "gpt-4o-mini"

    # The decision request carried the audit preview, never raw text
    # beyond it, and the requested model.
    assert len(fake_backend.route_requests) == 1
    req = fake_backend.route_requests[0]
    assert req["model"] == "gpt-4o"
    assert "HTTP" in req["prompt_preview"]

    egisai.shutdown()  # flush the audit event
    routed_events = [
        e for e in fake_backend.events_received if e.get("routing_applied")
    ]
    assert routed_events, "no routed audit event was flushed"
    ev = routed_events[0]
    assert ev["model"] == "gpt-4o-mini"
    assert ev["requested_model"] == "gpt-4o"
    assert ev["routing_direction"] == "downgrade"


def test_e2e_dormant_org_sends_zero_route_calls(fake_backend) -> None:
    Completions = _install_fake_openai()
    fake_backend.features = {}  # no entitlement

    import egisai

    egisai.init(
        api_key="egis_live_x", app="a", env="t",
        base_url="http://fake", enable_sse=False,
    )

    out = Completions().create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )
    assert out["kwargs"]["model"] == "gpt-4o"
    assert fake_backend.route_requests == []
