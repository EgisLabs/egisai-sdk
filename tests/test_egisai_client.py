"""``egisai.Client`` / ``egisai.AsyncClient`` — the Gateway-first client.

The contract under test: ``import egisai`` is the only import the
customer writes; the client always talks to the Gateway with both
keys in the right headers; ``init()`` is optional but, when active,
per-call context (``set_context`` → ``X-Egis-Agent``) rides along
without the local gate ever running (no double governance).
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

pytest.importorskip("openai")

import egisai  # noqa: E402
from egisai import _config, _context  # noqa: E402
from egisai._patches import openai as patch_openai  # noqa: E402


def _completion_json() -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1,
        "model": "claude-sonnet-4-5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _transport(seen: dict[str, Any]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["egis_key"] = request.headers.get("X-Egis-Api-Key")
        seen["agent"] = request.headers.get("X-Egis-Agent")
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json=_completion_json())

    return httpx.MockTransport(handler)


def test_client_talks_to_the_gateway_without_init() -> None:
    """No ``egisai.init()`` required — keys are carried by the client."""
    seen: dict[str, Any] = {}
    client = egisai.Client(
        api_key="egis_live_key",
        provider_key="sk-ant-test",
        http_client=httpx.Client(transport=_transport(seen)),
    )
    resp = client.chat.completions.create(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    assert resp.choices[0].message.content == "hi"
    assert seen["url"] == "https://app.egisai.co/v1/chat/completions"
    assert seen["egis_key"] == "egis_live_key"
    assert seen["auth"] == "Bearer sk-ant-test"


def test_client_agent_kwarg_names_every_call() -> None:
    seen: dict[str, Any] = {}
    client = egisai.Client(
        api_key="egis_live_key",
        provider_key="sk-test",
        agent="support-triage-bot",
        http_client=httpx.Client(transport=_transport(seen)),
    )
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    assert seen["agent"] == "support-triage-bot"


def test_client_key_resolution_falls_back_to_init() -> None:
    _config.set_config(
        _config.EgisaiConfig(
            api_key="egis_from_init",
            app="t",
            env="test",
            base_url="https://eu.egisai.co",
        )
    )
    seen: dict[str, Any] = {}
    client = egisai.Client(
        provider_key="sk-test",
        http_client=httpx.Client(transport=_transport(seen)),
    )
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    # Both the key AND the regional base_url come from init's config.
    assert seen["egis_key"] == "egis_from_init"
    assert seen["url"] == "https://eu.egisai.co/v1/chat/completions"


def test_client_without_any_key_raises() -> None:
    with pytest.raises(RuntimeError, match="egis_live_"):
        egisai.Client(provider_key="sk-test")


def test_client_vault_mode_sends_sentinel_not_provider_key(monkeypatch) -> None:
    """No provider_key + no OPENAI_API_KEY → BYOK vault mode.

    The Client still authenticates with the Egis key, but Authorization
    carries an Egis-namespaced placeholder the Gateway resolves against
    the org's stored provider keys — never a real provider credential.
    """
    from egisai._client import _VAULT_SENTINEL

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(_config, "_CONFIG", None)
    seen: dict[str, Any] = {}
    client = egisai.Client(
        api_key="egis_live_key",
        http_client=httpx.Client(transport=_transport(seen)),
    )
    client.chat.completions.create(
        model="claude-sonnet-4-5",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    assert seen["egis_key"] == "egis_live_key"
    assert seen["auth"] == f"Bearer {_VAULT_SENTINEL}"


def test_set_context_rides_along_when_init_is_active() -> None:
    """With the openai patch applied and config present, per-call
    context becomes ``X-Egis-Agent`` — even though ``gateway_mode``
    is off — because the client points at the Gateway. The client's
    own key must NOT be clobbered by the per-call injection."""
    assert patch_openai.apply()
    _config.set_config(
        _config.EgisaiConfig(
            api_key="egis_init_key",
            app="t",
            env="test",
            base_url="https://app.egisai.co",
            gateway_mode=False,
        )
    )
    _context._ctx.set(_context.EgisaiContext(agent_name="Contextual"))
    seen: dict[str, Any] = {}
    client = egisai.Client(
        api_key="egis_client_key",
        provider_key="sk-test",
        http_client=httpx.Client(transport=_transport(seen)),
    )
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    assert seen["agent"] == "Contextual"
    assert seen["egis_key"] == "egis_client_key"
    assert seen["url"] == "https://app.egisai.co/v1/chat/completions"


@pytest.mark.asyncio
async def test_async_client_parity() -> None:
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["egis_key"] = request.headers.get("X-Egis-Api-Key")
        return httpx.Response(200, json=_completion_json())

    client = egisai.AsyncClient(
        api_key="egis_live_key",
        provider_key="sk-test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    assert resp.choices[0].message.content == "hi"
    assert seen["url"] == "https://app.egisai.co/v1/chat/completions"
    assert seen["egis_key"] == "egis_live_key"
