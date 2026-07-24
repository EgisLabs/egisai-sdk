"""Smart Model Routing — ``ClaudeSDKClient`` per-turn sessions.

The CLI subprocess boots at ``connect()`` with ``--model`` frozen, so
client sessions can't use the module-level ``query()``'s trick of
rewriting ``options.model`` up front. Instead the wrapper pins the
routed model on the LIVE session via the ``set_model`` control request
(claude-agent-sdk ≥ 0.1.x) right before each governed turn's prompt
goes over stdio — and restores the user's configured model on the next
turn when the engine stops routing.

These tests pin that contract end-to-end against the FakeBackend:

* a routed decision calls ``set_model`` with the target and stamps the
  turn's audit event (``requested → served`` + direction/reason);
* "keep the requested model" answers leave the session untouched;
* a later unrouted turn RESTORES the configured model (``set_model``
  persists session-wide; decisions are per-turn);
* ``set_model`` raising, or missing entirely (older SDKs), fails open
  — the user's call must never break because routing had a bad day.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import AsyncIterator
from typing import Any

import pytest

# ── Fake claude_agent_sdk (class names must match upstream — the
#    patch duck-types on ``type(message).__name__``) ─────────────────


class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class AssistantMessage:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class ResultMessage:
    def __init__(self) -> None:
        self.usage = {"input_tokens": 100, "output_tokens": 200}
        self.total_cost_usd = 0.01
        self.is_error = False


class _Options:
    def __init__(self, *, model: str | None = "claude-opus-4-8") -> None:
        self.system_prompt = "You are the Routing Repro Agent."
        self.allowed_tools: list[str] = []
        self.permission_mode = "auto"
        self.model = model
        self.mcp_servers: dict[str, Any] = {}


def _make_client_cls(*, with_set_model: bool = True) -> type:
    class _Client:
        def __init__(self, options: Any = None) -> None:
            self.options = options
            self.set_model_calls: list[str | None] = []

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

        async def query(self, prompt: Any, session_id: str = "default") -> None:
            return None

        async def receive_messages(self) -> AsyncIterator[Any]:
            yield AssistantMessage([TextBlock("All done.")])
            yield ResultMessage()

        async def receive_response(self) -> AsyncIterator[Any]:
            async for msg in self.receive_messages():
                yield msg
                if isinstance(msg, ResultMessage):
                    return

    if with_set_model:

        async def set_model(self: Any, model: str | None = None) -> None:
            self.set_model_calls.append(model)

        _Client.set_model = set_model  # type: ignore[attr-defined]

    return _Client


@pytest.fixture
def fake_claude_client(fake_backend: Any):
    """Yield ``(fake_backend, ClientClass, boot)``.

    Each test scripts ``fake_backend`` (features / route_response)
    FIRST, then calls ``boot()`` — which runs ``egisai.init`` (the
    handshake reads the scripted features), installs the fake
    ``claude_agent_sdk`` module, and applies the patch to it. Init
    runs before the module lands in ``sys.modules`` so the explicit
    ``apply()`` is the first (and only) wrap — same ordering as the
    governance test fixture.
    """
    client_cls = _make_client_cls()

    def _boot() -> None:
        import egisai

        egisai.init(
            api_key="egis_live_x", app="a", env="t",
            base_url="http://fake", enable_sse=False,
        )

        mod = types.ModuleType("claude_agent_sdk")
        mod.ClaudeSDKClient = client_cls
        mod.AssistantMessage = AssistantMessage
        mod.TextBlock = TextBlock
        mod.ResultMessage = ResultMessage
        sys.modules["claude_agent_sdk"] = mod

        from egisai._patches import claude_agent_sdk as patch

        assert patch.apply() is True

    yield fake_backend, client_cls, _boot
    sys.modules.pop("claude_agent_sdk", None)


def _routed_response(model: str = "claude-haiku-4-5") -> dict[str, Any]:
    return {
        "routed": True,
        "model": model,
        "provider": "anthropic",
        "direction": "downgrade",
        "reason": "simple request; lighter model suffices",
        "projected_savings_usd": 0.02,
    }


async def _one_turn(client: Any, prompt: str) -> None:
    await client.query(prompt)
    async for _ in client.receive_response():
        pass


def _flush_events() -> None:
    import egisai

    egisai.shutdown()


def test_routed_turn_pins_model_and_stamps_audit(fake_claude_client) -> None:
    fake_backend, client_cls, boot = fake_claude_client
    fake_backend.features = {"smart_model_routing": True}
    fake_backend.route_response = _routed_response()
    boot()

    client = client_cls(options=_Options())

    async def run() -> None:
        async with client:
            await _one_turn(client, "Summarize the Q3 media plan status.")

    asyncio.run(run())

    # The live session was pinned to the routed model via set_model.
    assert client.set_model_calls == ["claude-haiku-4-5"]

    # The decision request carried the requested model + audit preview.
    assert len(fake_backend.route_requests) == 1
    req = fake_backend.route_requests[0]
    assert req["model"] == "claude-opus-4-8"
    assert "media plan" in req["prompt_preview"]

    _flush_events()
    routed = [
        e for e in fake_backend.events_received if e.get("routing_applied")
    ]
    assert routed, "no routed audit event was flushed"
    ev = routed[-1]
    assert ev["model"] == "claude-haiku-4-5"
    assert ev["requested_model"] == "claude-opus-4-8"
    assert ev["requested_provider"] == "anthropic"
    assert ev["routing_direction"] == "downgrade"


def test_unrouted_turn_leaves_session_untouched(fake_claude_client) -> None:
    fake_backend, client_cls, boot = fake_claude_client
    fake_backend.features = {"smart_model_routing": True}
    fake_backend.route_response = {"routed": False}
    boot()

    client = client_cls(options=_Options())

    async def run() -> None:
        async with client:
            await _one_turn(client, "Reconcile the spend ledger.")

    asyncio.run(run())

    assert client.set_model_calls == []
    _flush_events()
    assert not any(
        e.get("routing_applied") for e in fake_backend.events_received
    )


def test_next_unrouted_turn_restores_configured_model(
    fake_claude_client,
) -> None:
    """``set_model`` persists session-wide; a decision is per-turn.
    When the engine stops routing, the wrapper must switch the session
    back to the model the user configured."""
    fake_backend, client_cls, boot = fake_claude_client
    fake_backend.features = {"smart_model_routing": True}
    fake_backend.route_response = _routed_response()
    boot()

    client = client_cls(options=_Options(model="claude-opus-4-8"))

    async def run() -> None:
        async with client:
            await _one_turn(client, "Draft the campaign concept brief.")
            # Distinct prompt ⇒ distinct decision-cache key ⇒ the SDK
            # consults the platform again and now hears "keep".
            fake_backend.route_response = {"routed": False}
            await _one_turn(client, "Now check budget pacing breaks.")

    asyncio.run(run())

    assert client.set_model_calls == ["claude-haiku-4-5", "claude-opus-4-8"]

    _flush_events()
    # Terminal model_call rows only (a turn also ships a provisional
    # row with the same seq and no usage — see the governance tests).
    routed_flags = [
        bool(e.get("routing_applied"))
        for e in fake_backend.events_received
        if e.get("kind") == "run.step"
        and e.get("step_kind") == "model_call"
        and e.get("tokens_in") is not None
    ]
    assert routed_flags == [True, False]


def test_set_model_failure_fails_open(fake_claude_client) -> None:
    fake_backend, client_cls, boot = fake_claude_client
    fake_backend.features = {"smart_model_routing": True}
    fake_backend.route_response = _routed_response()
    boot()

    async def broken_set_model(self: Any, model: str | None = None) -> None:
        raise RuntimeError("control channel down")

    client_cls.set_model = broken_set_model  # type: ignore[attr-defined]
    client = client_cls(options=_Options())

    seen: list[Any] = []

    async def run() -> None:
        async with client:
            await client.query("Screen this ad copy for claims.")
            async for msg in client.receive_response():
                seen.append(msg)

    asyncio.run(run())

    # The turn completed despite the routing failure…
    assert any(isinstance(m, ResultMessage) for m in seen)
    # …and no routing stamps were shipped (the swap never happened).
    _flush_events()
    assert not any(
        e.get("routing_applied") for e in fake_backend.events_received
    )


def test_old_sdk_without_set_model_skips_routing(fake_claude_client) -> None:
    """Older claude-agent-sdk releases have no ``set_model`` control
    request — routing must stay fully dormant (zero /route calls)."""
    fake_backend, client_cls, boot = fake_claude_client
    fake_backend.features = {"smart_model_routing": True}
    fake_backend.route_response = _routed_response()
    boot()

    delattr(client_cls, "set_model")
    client = client_cls(options=_Options())

    async def run() -> None:
        async with client:
            await _one_turn(client, "Route this brief to a practice.")

    asyncio.run(run())

    assert fake_backend.route_requests == []
    _flush_events()
    assert not any(
        e.get("routing_applied") for e in fake_backend.events_received
    )


def test_unentitled_org_sends_zero_route_calls(fake_claude_client) -> None:
    fake_backend, client_cls, boot = fake_claude_client
    fake_backend.features = {}  # no smart_model_routing entitlement
    fake_backend.route_response = _routed_response()
    boot()

    client = client_cls(options=_Options())

    async def run() -> None:
        async with client:
            await _one_turn(client, "Build the audience segments.")

    asyncio.run(run())

    assert client.set_model_calls == []
    assert fake_backend.route_requests == []
