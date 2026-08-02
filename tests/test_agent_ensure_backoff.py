"""Agent registration must not stall the customer's call path.

``ensure_agent`` is the one backend hop that runs *inline* on a model
call — the first time each agent identity is seen. Everything about
that hop is a latency budget question, and the failure shape that
hurts is not "connection refused" (microseconds) but a black-holed
backend that accepts the connection and never answers.

Three guards are pinned here:

1. **Negative caching.** A failed ensure arms a short backoff for that
   identity, so an outage costs one attempt per window instead of one
   attempt per call. Without it, every call re-attempted the hop and
   paid the full timeout, serialized behind the identity lock.
2. **A tight, separate timeout.** Registration gets ~2 s, not the
   shared 10 s client budget.
3. **A capped ``Retry-After``.** A rate-limited hot-path hop is
   skipped, never slept through; and no caller anywhere can be made
   to sleep for minutes by a server-supplied header.

The customer-visible invariant behind all three: a governed call
against an unreachable backend still returns, and still gets governed.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx
import pytest

from egisai import _auto_agent, _backend, _config


def _cfg(**overrides: Any) -> _config.EgisaiConfig:
    base: dict[str, Any] = {
        "api_key": "egis_test_key",
        "app": "ensure-backoff-tests",
        "env": "test",
        "base_url": "https://app.egisai.co",
    }
    base.update(overrides)
    return _config.EgisaiConfig(**base)


class _CountingBackend:
    """Counts ``/v1/sdk/agents/ensure`` hits and scripts the answer."""

    def __init__(self, response: httpx.Response | Exception) -> None:
        self.response = response
        self.ensure_calls = 0
        self.timeouts: list[Any] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/v1/sdk/agents/ensure"):
            self.ensure_calls += 1
            self.timeouts.append(
                request.extensions.get("timeout", {}).get("connect")
            )
            if isinstance(self.response, Exception):
                raise self.response
            return self.response
        return httpx.Response(404)


@pytest.fixture
def counting_backend(monkeypatch: pytest.MonkeyPatch):
    """Install a scripted transport, like conftest's ``fake_backend``."""
    holder: dict[str, _CountingBackend] = {}

    def install(response: httpx.Response | Exception) -> _CountingBackend:
        backend = _CountingBackend(response)
        holder["backend"] = backend
        transport = httpx.MockTransport(backend.handle)

        def patched_get_client() -> httpx.Client:
            if _backend._client is None:
                cfg = _config.get_config()
                _backend._client = httpx.Client(
                    base_url=cfg.base_url.rstrip("/"),
                    timeout=cfg.timeout_seconds,
                    transport=transport,
                )
            return _backend._client

        monkeypatch.setattr(_backend, "get_client", patched_get_client)
        return backend

    yield install

    if _backend._client is not None:
        _backend._client.close()
        _backend._client = None


def _ensure(key: str = "hash:abc") -> str | None:
    return _auto_agent._ensure_agent_id(
        display_name="Support Bot",
        identity_key=key,
        identity_hash="abc123",
        source="hash",
    )


# ── Negative caching ────────────────────────────────────────────────


def test_failed_ensure_is_backed_off(counting_backend) -> None:
    """One attempt per window, not one per call."""
    _config.set_config(_cfg())
    backend = counting_backend(httpx.ConnectError("refused"))

    for _ in range(5):
        assert _ensure() is None

    assert backend.ensure_calls == 1


def test_backoff_expires_and_one_call_retries(counting_backend) -> None:
    _config.set_config(_cfg())
    backend = counting_backend(httpx.ConnectError("refused"))

    assert _ensure() is None
    assert backend.ensure_calls == 1

    # Expire the window without sleeping through it.
    _auto_agent._ensure_backoff["hash:abc"] = time.monotonic() - 0.01

    assert _ensure() is None
    assert backend.ensure_calls == 2


def test_backoff_is_per_identity(counting_backend) -> None:
    """One unregisterable agent must not mask every other agent."""
    _config.set_config(_cfg())
    backend = counting_backend(httpx.ConnectError("refused"))

    _ensure("hash:one")
    _ensure("hash:one")
    _ensure("hash:two")

    assert backend.ensure_calls == 2


def test_non_2xx_also_arms_the_backoff(counting_backend) -> None:
    """A 500 from the backend is as unhelpful as no answer at all."""
    _config.set_config(_cfg())
    backend = counting_backend(httpx.Response(500, json={}))

    _ensure()
    _ensure()

    assert backend.ensure_calls == 1


def test_2xx_without_an_id_arms_the_backoff(counting_backend) -> None:
    """A malformed success would otherwise loop forever on every call."""
    _config.set_config(_cfg())
    backend = counting_backend(httpx.Response(200, json={"created": True}))

    _ensure()
    _ensure()

    assert backend.ensure_calls == 1


def test_success_caches_and_clears_the_backoff(counting_backend) -> None:
    _config.set_config(_cfg())
    backend = counting_backend(httpx.ConnectError("refused"))
    assert _ensure() is None
    assert "hash:abc" in _auto_agent._ensure_backoff

    # Backend recovers; expire the window so the retry goes out.
    _auto_agent._ensure_backoff["hash:abc"] = time.monotonic() - 0.01
    backend.response = httpx.Response(
        200, json={"id": "agent-1", "created": True}
    )

    assert _ensure() == "agent-1"
    assert "hash:abc" not in _auto_agent._ensure_backoff
    # Steady state is a dict lookup — no further round-trips.
    calls_after_success = backend.ensure_calls
    assert _ensure() == "agent-1"
    assert backend.ensure_calls == calls_after_success


def test_zero_backoff_disables_negative_caching(
    counting_backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Escape hatch for anyone who wants the old retry-every-call
    behavior (e.g. a short-lived job that must register or bust)."""
    monkeypatch.setenv(_auto_agent._ENSURE_BACKOFF_ENV, "0")
    _config.set_config(_cfg())
    backend = counting_backend(httpx.ConnectError("refused"))

    _ensure()
    _ensure()

    assert backend.ensure_calls == 2


def test_backoff_map_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_auto_agent, "_ENSURE_BACKOFF_MAX_ENTRIES", 10)
    for i in range(50):
        _auto_agent._note_ensure_failure(f"hash:{i}")
    assert len(_auto_agent._ensure_backoff) <= 10


def test_concurrent_callers_make_one_attempt(counting_backend) -> None:
    """The lock-holder's failure must be visible to everyone queued.

    Before the in-lock re-check, each of N threads that piled up on
    ``_identity_lock`` went on to pay its own full timeout in turn —
    the exact amplification this guards against.
    """
    _config.set_config(_cfg())
    backend = counting_backend(httpx.ConnectError("refused"))
    start = threading.Barrier(8)

    def worker() -> None:
        start.wait()
        _ensure()

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not any(t.is_alive() for t in threads)
    assert backend.ensure_calls == 1


# ── Timeout budget ──────────────────────────────────────────────────


def test_ensure_uses_its_own_short_timeout(counting_backend) -> None:
    """Not the shared 10 s client budget — this hop is on the hot path."""
    _config.set_config(_cfg(timeout_seconds=10.0))
    backend = counting_backend(httpx.Response(200, json={"id": "a", "created": True}))

    _ensure()

    assert backend.timeouts == [_backend.ENSURE_AGENT_TIMEOUT_S]


def test_ensure_timeout_is_tunable(
    counting_backend, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_backend.ENSURE_AGENT_TIMEOUT_ENV, "0.5")
    _config.set_config(_cfg())
    backend = counting_backend(httpx.Response(200, json={"id": "a", "created": True}))

    _ensure()

    assert backend.timeouts == [0.5]


# ── Retry-After clamping ────────────────────────────────────────────


def test_retry_after_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """A server-supplied ``Retry-After`` can't stall a thread for minutes."""
    slept: list[float] = []
    monkeypatch.setattr(_backend.time, "sleep", slept.append)
    calls: list[int] = []

    def fn() -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "600"})
        return httpx.Response(200)

    r = _backend._retry_on_429("test", fn)

    assert r.status_code == 200
    assert slept == [_backend.RETRY_AFTER_CAP_S]


def test_retry_after_cap_is_tunable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_backend.RETRY_AFTER_CAP_ENV, "0.25")
    slept: list[float] = []
    monkeypatch.setattr(_backend.time, "sleep", slept.append)

    _backend._retry_on_429(
        "test", lambda: httpx.Response(429, headers={"Retry-After": "600"})
    )

    assert slept == [0.25] * _backend.RETRY_429_MAX


def test_small_retry_after_is_honoured_verbatim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(_backend.time, "sleep", slept.append)
    calls: list[int] = []

    def fn() -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "1"})
        return httpx.Response(200)

    _backend._retry_on_429("test", fn)

    assert slept == [1.0]


def test_ensure_agent_does_not_retry_a_429(counting_backend) -> None:
    """A rate-limited registration is skipped, not slept through.

    The identity retries after the caller-side backoff instead, so the
    hot path pays one round-trip at most.
    """
    _config.set_config(_cfg())
    backend = counting_backend(
        httpx.Response(429, headers={"Retry-After": "60"}, json={})
    )

    started = time.monotonic()
    assert _ensure() is None
    elapsed = time.monotonic() - started

    assert backend.ensure_calls == 1
    assert elapsed < 1.0, "the hot path must not sleep on a 429"


# ── End to end: an unreachable backend never breaks a call ──────────


def test_governed_call_survives_an_unreachable_backend(
    counting_backend,
) -> None:
    openai = pytest.importorskip("openai")
    from egisai._patches import openai as patch_openai

    assert patch_openai.apply()
    _config.set_config(_cfg())
    backend = counting_backend(httpx.ConnectError("refused"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    client = openai.OpenAI(
        api_key="sk-provider-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    for _ in range(8):
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a support agent."},
                {"role": "user", "content": "hello"},
            ],
        )
        assert resp.choices[0].message.content == "hi"

    # The point: attempts scale with the number of distinct identities
    # the resolver tried (here the prompt-derived agent plus the
    # ``app`` fallback), never with the number of calls. Eight calls
    # would have cost eight doomed round-trips before the backoff.
    assert backend.ensure_calls == 2
