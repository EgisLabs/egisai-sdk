"""Test fixtures: a fake-backend httpx mock + reset of the SDK between tests.

Every test resets the global SDK config + cache so they don't leak.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest


@pytest.fixture(autouse=True)
def reset_sdk() -> Iterator[None]:
    """Wipe the SDK between tests so they're independent."""
    # Make sure modules are loaded
    from egisai import _config, _logger, _policy_cache, _refresher  # noqa: F401

    # Stop background workers and reset state.
    try:
        from egisai import shutdown
        shutdown()
    except Exception:
        pass

    _config._CONFIG = None
    _policy_cache.clear()

    from egisai import _auto_agent, _context

    _context._agent_id_cache.clear()
    _context._ctx.set(_context.EgisaiContext())
    _auto_agent._id_cache.clear()
    # Identity v1: clear unified cache + identity stack so a test
    # that pushed a framework identity doesn't leak it into the next
    # test's resolver.
    _auto_agent._identity_cache.clear()
    _auto_agent._identity_stack.set(())

    # 0.18.0 — clear any open Run that a previous test forgot to close.
    from egisai import _run

    _run.reset_for_tests()

    # The init module also caches whether it's been called via _CONFIG; the
    # logger module's queue persists between tests, so we drain it.
    while not _logger._q.empty():
        try:
            _logger._q.get_nowait()
        except Exception:
            break

    yield

    try:
        from egisai import shutdown
        shutdown()
    except Exception:
        pass


@pytest.fixture
def fake_backend(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeBackend]:
    """A drop-in fake of /v1/sdk/* endpoints — no real network."""
    fb = FakeBackend()

    def transport_handler(request: httpx.Request) -> httpx.Response:
        return fb.handle(request)

    transport = httpx.MockTransport(transport_handler)

    # Patch egisai._backend.get_client so it returns a client wired to our transport.
    from egisai import _backend

    def patched_get_client() -> httpx.Client:
        if _backend._client is None:
            from egisai._config import get_config
            cfg = get_config()
            _backend._client = httpx.Client(
                base_url=cfg.base_url.rstrip("/"),
                timeout=cfg.timeout_seconds,
                headers={
                    "Authorization": f"Bearer {cfg.api_key}",
                    "User-Agent": f"egisai-sdk/{cfg.sdk_version}",
                },
                transport=transport,
            )
        return _backend._client

    monkeypatch.setattr(_backend, "get_client", patched_get_client)

    yield fb

    # Cleanup the test client
    if _backend._client is not None:
        _backend._client.close()
        _backend._client = None


class FakeBackend:
    """Minimal in-memory backend for SDK tests."""

    def __init__(self) -> None:
        self.events_received: list[dict[str, Any]] = []
        self.rules: list[dict[str, Any]] = []
        self.etag: str = '"empty"'
        self.handshake_calls = 0
        self.ensured_agents: list[dict[str, Any]] = []
        # Raw request bodies hitting /v1/sdk/agents/ensure — useful
        # for tests that need to assert what the SDK actually shipped
        # (e.g. that runtime fingerprint was included). Indexed in
        # call order; one entry per HTTP request.
        self.ensure_requests: list[dict[str, Any]] = []
        self.handshake_requests: list[dict[str, Any]] = []
        # Startup-warning telemetry — captured per request so tests
        # can pin both the payload shape and the no-payload-on-
        # privacy-exit cases without standing up real HTTP.
        self.startup_warnings: list[dict[str, Any]] = []
        self._next_agent_serial = 100

    def set_rules(self, rules: list[dict[str, Any]], etag: str = '"new"') -> None:
        self.rules = rules
        self.etag = etag

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/v1/sdk/handshake"):
            self.handshake_calls += 1
            try:
                import json

                if request.content:
                    self.handshake_requests.append(
                        json.loads(request.content.decode())
                    )
            except Exception:
                pass
            return httpx.Response(
                200,
                json={
                    "org_id": "00000000-0000-0000-0000-000000000001",
                    "agent_id": "00000000-0000-0000-0000-000000000002",
                    "agent_name": "test-agent",
                    "api_key_name": "test-key",
                    "policy_etag": self.etag,
                    "server_time": "2026-01-01T00:00:00Z",
                },
            )
        if path.endswith("/v1/sdk/policies"):
            if request.headers.get("if-none-match") == self.etag:
                return httpx.Response(304)
            return httpx.Response(200, json={"etag": self.etag, "rules": self.rules})
        if path.endswith("/v1/sdk/agents/ensure"):
            import json

            body = json.loads(request.content.decode())
            self.ensure_requests.append(body)
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

            self.events_received.extend(json.loads(request.content.decode())["events"])
            return httpx.Response(204)
        if path.endswith("/v1/sdk/telemetry/startup-warning"):
            import json

            try:
                self.startup_warnings.append(
                    json.loads(request.content.decode())
                )
            except Exception:
                self.startup_warnings.append({})
            return httpx.Response(204)
        if path.endswith("/v1/sdk/stream"):
            # We don't actually run SSE in unit tests — just return a non-streaming 200.
            return httpx.Response(200, content=b"event: ready\ndata: {}\n\n")
        return httpx.Response(404)
