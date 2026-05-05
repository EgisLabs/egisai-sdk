"""Lazy ``app`` registration — the fix for ghost agents.

Until r0.6 the SDK's handshake auto-created an Agent named after
``egisai.init(app=...)`` whether or not any actual call ever flowed
through it. For multi-agent apps (every call carries a system
prompt → every call attributes to a sub-agent fingerprint, never
to the ``app`` agent) that produced a permanent "Last Run: Never"
ghost on the dashboard.

The fix has two halves we test here:

1. **Backend handshake** (simulated by the fake) returns
   ``agent_id = None`` when the API key isn't bound to a specific
   agent. The SDK accepts that and stores ``cfg.agent_id = None``.
2. **SDK ``_attribute_event``** lazily registers the ``app`` agent
   only when a call actually falls through to needing it (no
   ``set_context``, no system prompt, ``cfg.agent_id`` is None).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

# ── Shared: fake backend that mimics an UNBOUND API key ────────────


class _UnboundBackend:
    """Like the conftest ``FakeBackend`` but the handshake response
    has ``agent_id = None`` — what the real backend now returns
    when the API key isn't bound to a specific agent."""

    def __init__(self) -> None:
        self.events_received: list[dict[str, Any]] = []
        self.ensured_agents: list[dict[str, Any]] = []
        self.handshake_calls = 0
        self._next_agent_serial = 100

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/v1/sdk/handshake"):
            self.handshake_calls += 1
            return httpx.Response(
                200,
                json={
                    "org_id": "00000000-0000-0000-0000-000000000001",
                    # No bound agent — SDK lazy-registers later.
                    "agent_id": None,
                    "agent_name": None,
                    "api_key_name": "test-key",
                    "policy_etag": '"empty"',
                    "server_time": "2026-01-01T00:00:00Z",
                },
            )
        if path.endswith("/v1/sdk/policies"):
            return httpx.Response(
                200, json={"etag": '"empty"', "rules": []}
            )
        if path.endswith("/v1/sdk/agents/ensure"):
            import json

            body = json.loads(request.content.decode())
            name = body["name"]
            existing = next(
                (a for a in self.ensured_agents if a["name"] == name), None
            )
            if existing is not None:
                return httpx.Response(200, json={**existing, "created": False})
            self._next_agent_serial += 1
            agent = {
                "id": f"00000000-0000-0000-0000-{self._next_agent_serial:012d}",
                "name": name,
                "description": body.get("description") or "",
                "created_at": "2026-01-01T00:00:00Z",
            }
            self.ensured_agents.append(agent)
            return httpx.Response(200, json={**agent, "created": True})
        if path.endswith("/v1/sdk/events"):
            import json

            self.events_received.extend(
                json.loads(request.content.decode())["events"]
            )
            return httpx.Response(204)
        if path.endswith("/v1/sdk/stream"):
            return httpx.Response(200, content=b"event: ready\ndata: {}\n\n")
        return httpx.Response(404)


@pytest.fixture
def unbound_backend(monkeypatch: pytest.MonkeyPatch):
    """Install ``_UnboundBackend`` as the SDK's HTTP transport.

    Mirrors the conftest ``fake_backend`` pattern: patches
    ``egisai._backend.get_client`` to return an ``httpx.Client``
    wired to a ``MockTransport``. The ``reset_sdk`` autouse fixture
    in conftest already wipes config / caches between tests, so we
    don't need to touch ``sys.modules`` (doing so would leave
    dangling imports in other test modules).
    """
    backend = _UnboundBackend()

    transport = httpx.MockTransport(backend.handle)

    from egisai import _backend as backend_mod

    def patched_get_client() -> httpx.Client:
        if backend_mod._client is None:
            from egisai._config import get_config

            cfg = get_config()
            backend_mod._client = httpx.Client(
                base_url=cfg.base_url.rstrip("/"),
                timeout=cfg.timeout_seconds,
                headers={
                    "Authorization": f"Bearer {cfg.api_key}",
                    "User-Agent": f"egisai-sdk/{cfg.sdk_version}",
                },
                transport=transport,
            )
        return backend_mod._client

    monkeypatch.setattr(backend_mod, "get_client", patched_get_client)

    yield backend

    if backend_mod._client is not None:
        backend_mod._client.close()
        backend_mod._client = None


# ── Tests ──────────────────────────────────────────────────────────


def test_unbound_handshake_does_not_create_app_agent(
    unbound_backend: _UnboundBackend,
) -> None:
    # Calling init() with an unbound key should NOT create the app
    # agent on the platform — that's the whole point of the fix.
    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="my-test-agents",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    # Handshake fired, but no ensure_agent call yet.
    assert unbound_backend.handshake_calls == 1
    assert unbound_backend.ensured_agents == []


def test_lazy_app_registration_fires_on_first_no_system_prompt_call(
    unbound_backend: _UnboundBackend,
) -> None:
    # First gated call with no system prompt → the SDK should
    # lazy-register the ``app`` agent.
    import egisai
    from egisai._patches._common import _attribute_event

    egisai.init(
        api_key="egis_live_x",
        app="my-test-agents",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    # Build an event manually — we don't want to set up the full
    # framework patch path for this unit test. The attribution
    # function is what we're exercising.
    ev: dict[str, Any] = {}
    payload = {
        "model": "gpt-4o-mini",
        # No "system" field on the payload AND no system message —
        # this is the fallback case.
        "messages": [{"role": "user", "content": "hi"}],
    }
    _attribute_event(ev, payload)

    # ensure_agent fired exactly once with our app name.
    assert len(unbound_backend.ensured_agents) == 1
    registered = unbound_backend.ensured_agents[0]
    assert registered["name"] == "my-test-agents"
    # Event got tagged with the freshly-registered id.
    assert ev["agent_id"] == registered["id"]
    assert ev["app"] == "my-test-agents"


def test_lazy_registration_caches_after_first_call(
    unbound_backend: _UnboundBackend,
) -> None:
    # Second + third calls reuse the cached id — no extra
    # ensure_agent round trips.
    import egisai
    from egisai._patches._common import _attribute_event

    egisai.init(
        api_key="egis_live_x",
        app="my-test-agents",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    payload = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    for _ in range(3):
        _attribute_event({}, payload)

    assert len(unbound_backend.ensured_agents) == 1
    assert unbound_backend.ensured_agents[0]["name"] == "my-test-agents"


def test_system_prompt_call_does_not_lazy_register_app(
    unbound_backend: _UnboundBackend,
) -> None:
    # When the call has a system prompt, it gets a sub-agent — the
    # ``app`` lazy registration is skipped entirely. This is the
    # multi-agent scenario the fix was designed for.
    import egisai
    from egisai._patches._common import _attribute_event

    egisai.init(
        api_key="egis_live_x",
        app="my-test-agents",  # this should NEVER be registered in a multi-agent script
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    payload = {
        "model": "gpt-4o-mini",
        "system": "You are a Python Developer.",
        "messages": [{"role": "user", "content": "write a script"}],
    }
    _attribute_event({}, payload)

    # The sub-agent ("Python Developer") was registered, but NOT
    # the ``app`` agent ("my-test-agents") — exactly the behaviour
    # the user reported they wanted.
    names = [a["name"] for a in unbound_backend.ensured_agents]
    assert "Python Developer" in names
    assert "my-test-agents" not in names
