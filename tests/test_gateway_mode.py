"""Gateway mode (``init(gateway=True)``) — reroute + header contract.

The mode's promise: the customer's calling convention and the SDK's
context API (``set_context`` / ``with egisai.agent(...)``) keep
working exactly as in local mode, but chat-completions calls travel
through the platform's inline Gateway, which evaluates + audits
server-side. These tests pin:

* config plumbing (``enabled`` / ``gateway_base_url``),
* header injection precedence (explicit identity → ``X-Egis-Agent``;
  caller-supplied headers win),
* the reroute itself against a real ``openai`` client with a mock
  transport — URL, headers, and provider-key passthrough,
* the fail-open fallback for Azure-flavoured clients.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

openai = pytest.importorskip("openai")

from egisai import _config, _context, _gateway  # noqa: E402
from egisai._patches import openai as patch_openai  # noqa: E402


def _cfg(**overrides: Any) -> _config.EgisaiConfig:
    base: dict[str, Any] = {
        "api_key": "egis_test_key",
        "app": "gateway-tests",
        "env": "test",
        "base_url": "https://app.egisai.co",
        "gateway_mode": True,
    }
    base.update(overrides)
    return _config.EgisaiConfig(**base)


def _completion_json() -> dict[str, Any]:
    return {
        "id": "chatcmpl-test",
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
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


# ── Config plumbing ──────────────────────────────────────────────────


def test_disabled_without_config() -> None:
    assert _gateway.enabled() is False


def test_disabled_when_flag_off() -> None:
    _config.set_config(_cfg(gateway_mode=False))
    assert _gateway.enabled() is False


def test_gateway_base_url_appends_v1() -> None:
    _config.set_config(_cfg(base_url="https://app.egisai.co/"))
    assert _gateway.gateway_base_url() == "https://app.egisai.co/v1"


# ── Header injection ─────────────────────────────────────────────────


def test_inject_headers_carries_api_key() -> None:
    _config.set_config(_cfg())
    kwargs: dict[str, Any] = {}
    _gateway.inject_headers(kwargs)
    assert kwargs["extra_headers"]["X-Egis-Api-Key"] == "egis_test_key"
    assert "X-Egis-Agent" not in kwargs["extra_headers"]


def test_explicit_context_agent_becomes_header() -> None:
    """``set_context(agent=…)`` must survive the mode switch — the
    context var is read directly (no network registration needed for
    the header to ship)."""
    _config.set_config(_cfg())
    _context._ctx.set(_context.EgisaiContext(agent_name="Triage"))
    kwargs: dict[str, Any] = {}
    _gateway.inject_headers(kwargs)
    assert kwargs["extra_headers"]["X-Egis-Agent"] == "Triage"


def test_caller_supplied_headers_win() -> None:
    _config.set_config(_cfg())
    _context._ctx.set(_context.EgisaiContext(agent_name="Triage"))
    kwargs: dict[str, Any] = {
        "extra_headers": {"X-Egis-Agent": "Override", "X-Custom": "1"}
    }
    _gateway.inject_headers(kwargs)
    assert kwargs["extra_headers"]["X-Egis-Agent"] == "Override"
    assert kwargs["extra_headers"]["X-Custom"] == "1"
    assert kwargs["extra_headers"]["X-Egis-Api-Key"] == "egis_test_key"


def test_inject_headers_carries_context_fields() -> None:
    """Every ``set_context`` field ships as its ``X-Egis-*`` header so
    gateway-audited runs get the same Context section as SDK-audited
    runs."""
    _config.set_config(_cfg())
    _context._ctx.set(
        _context.EgisaiContext(
            user_id="support-ops-1042",
            user_role="support_engineer",
            session_id="sess-abc123",
            workflow_id="wf-refund-1",
            end_user_id="a" * 64,  # pre-hashed per the docs
        )
    )
    kwargs: dict[str, Any] = {}
    _gateway.inject_headers(kwargs)
    headers = kwargs["extra_headers"]
    assert headers["X-Egis-User"] == "support-ops-1042"
    assert headers["X-Egis-User-Role"] == "support_engineer"
    assert headers["X-Egis-Session"] == "sess-abc123"
    assert headers["X-Egis-Workflow"] == "wf-refund-1"
    assert headers["X-Egis-End-User"] == "a" * 64


def test_unset_context_fields_send_no_headers() -> None:
    """No context ⇒ no headers — the Gateway must be able to tell
    "never set" apart from "set to empty"."""
    _config.set_config(_cfg())
    _context._ctx.set(_context.EgisaiContext())
    kwargs: dict[str, Any] = {}
    _gateway.inject_headers(kwargs)
    for header in (
        "X-Egis-User",
        "X-Egis-User-Role",
        "X-Egis-Session",
        "X-Egis-Workflow",
        "X-Egis-End-User",
    ):
        assert header not in kwargs["extra_headers"]


def test_context_values_are_percent_encoded_header_safe() -> None:
    """Unicode / control chars / literal ``%`` must reach the wire as
    pure printable ASCII — a non-latin-1 header value raises inside
    the customer's call in httpx, which would violate fail-open."""
    _config.set_config(_cfg())
    _context._ctx.set(
        _context.EgisaiContext(
            user_id="maría@empresa.es",
            session_id="sess 100%\nready",
        )
    )
    kwargs: dict[str, Any] = {}
    _gateway.inject_headers(kwargs)
    headers = kwargs["extra_headers"]
    # Lossless round-trip via percent-decoding.
    from urllib.parse import unquote

    assert unquote(headers["X-Egis-User"]) == "maría@empresa.es"
    assert unquote(headers["X-Egis-Session"]) == "sess 100%\nready"
    # Wire-safety: printable ASCII only, no CR/LF smuggling.
    for value in (headers["X-Egis-User"], headers["X-Egis-Session"]):
        value.encode("ascii")  # raises if any non-ASCII survived
        assert "\n" not in value and "\r" not in value


def test_context_values_are_capped_before_encoding() -> None:
    """Caps mirror the backend's column widths (user_role → 64)."""
    _config.set_config(_cfg())
    _context._ctx.set(
        _context.EgisaiContext(user_role="r" * 500, session_id="s" * 500)
    )
    kwargs: dict[str, Any] = {}
    _gateway.inject_headers(kwargs)
    assert kwargs["extra_headers"]["X-Egis-User-Role"] == "r" * 64
    assert kwargs["extra_headers"]["X-Egis-Session"] == "s" * 255


def test_caller_supplied_context_headers_win() -> None:
    """Same conflict rule as ``X-Egis-Agent`` — a customer-set header
    is intentional and must not be clobbered by ``set_context``."""
    _config.set_config(_cfg())
    _context._ctx.set(_context.EgisaiContext(session_id="from-context"))
    kwargs: dict[str, Any] = {
        "extra_headers": {"X-Egis-Session": "hand-set"}
    }
    _gateway.inject_headers(kwargs)
    assert kwargs["extra_headers"]["X-Egis-Session"] == "hand-set"


def test_pushed_agent_block_wins_over_set_context() -> None:
    """Tier-0 precedence carries over: ``with egisai.agent(...)``
    (the pushed identity stack) beats ``set_context``."""
    from egisai._auto_agent import IdentityRecord, push_identity, reset_identity

    _config.set_config(_cfg())
    _context._ctx.set(_context.EgisaiContext(agent_name="Outer"))
    token = push_identity(
        IdentityRecord(
            agent_id=None,
            display_name="Inner",
            identity_key="explicit:Inner",
            identity_hash="h",
            source="explicit",
        )
    )
    try:
        kwargs: dict[str, Any] = {}
        _gateway.inject_headers(kwargs)
        assert kwargs["extra_headers"]["X-Egis-Agent"] == "Inner"
    finally:
        reset_identity(token)


# ── The reroute, end to end against a mock transport ────────────────


def _mock_client(seen: dict[str, Any]) -> openai.OpenAI:
    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["egis_key"] = request.headers.get("X-Egis-Api-Key")
        seen["agent"] = request.headers.get("X-Egis-Agent")
        seen["auth"] = request.headers.get("Authorization")
        seen["session"] = request.headers.get("X-Egis-Session")
        seen["end_user"] = request.headers.get("X-Egis-End-User")
        return httpx.Response(200, json=_completion_json())

    return openai.OpenAI(
        api_key="sk-provider-key",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def test_call_is_rerouted_to_the_gateway() -> None:
    assert patch_openai.apply()
    _config.set_config(_cfg())
    _context._ctx.set(
        _context.EgisaiContext(
            agent_name="Triage",
            session_id="sess-wire-1",
            end_user_id="b" * 64,
        )
    )
    seen: dict[str, Any] = {}
    client = _mock_client(seen)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    assert resp.choices[0].message.content == "hi"
    # The request went to <base_url>/v1/…, not api.openai.com.
    assert seen["url"] == "https://app.egisai.co/v1/chat/completions"
    # Egis key + explicit identity + request context on the wire; the
    # provider key is untouched in Authorization for the gateway to
    # forward.
    assert seen["egis_key"] == "egis_test_key"
    assert seen["agent"] == "Triage"
    assert seen["session"] == "sess-wire-1"
    assert seen["end_user"] == "b" * 64
    assert seen["auth"] == "Bearer sk-provider-key"


def test_local_mode_does_not_reroute() -> None:
    """With ``gateway_mode=False`` the exact same call must reach the
    client's own base URL (the normal in-process governance path)."""
    assert patch_openai.apply()
    _config.set_config(_cfg(gateway_mode=False))
    seen: dict[str, Any] = {}
    client = _mock_client(seen)

    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    assert seen["url"].startswith("https://api.openai.com/")
    assert seen["egis_key"] is None


def test_client_already_pointed_at_gateway_is_not_double_hopped() -> None:
    """A customer who set ``base_url`` to the gateway themselves gets
    headers injected but no second client copy / no URL change."""
    assert patch_openai.apply()
    _config.set_config(_cfg())
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["egis_key"] = request.headers.get("X-Egis-Api-Key")
        return httpx.Response(200, json=_completion_json())

    client = openai.OpenAI(
        api_key="sk-provider-key",
        base_url="https://app.egisai.co/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    assert seen["url"] == "https://app.egisai.co/v1/chat/completions"
    assert seen["egis_key"] == "egis_test_key"


def test_azure_clients_raise_reroute_unavailable() -> None:
    """Azure's deployment-based URLs are incompatible with the
    passthrough contract; the reroute must refuse so the patch falls
    back to in-process governance."""
    _config.set_config(_cfg())

    class AzureOpenAIFake:
        base_url = "https://myorg.openai.azure.com/"

    class Resource:
        _client = AzureOpenAIFake()

    with pytest.raises(_gateway.RerouteUnavailable):
        _gateway._gateway_resource(Resource())


@pytest.mark.asyncio
async def test_async_call_is_rerouted_too() -> None:
    assert patch_openai.apply()
    _config.set_config(_cfg())
    seen: dict[str, Any] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["egis_key"] = request.headers.get("X-Egis-Api-Key")
        return httpx.Response(200, json=_completion_json())

    client = openai.AsyncOpenAI(
        api_key="sk-provider-key",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )
    assert resp.choices[0].message.content == "hi"
    assert seen["url"] == "https://app.egisai.co/v1/chat/completions"
    assert seen["egis_key"] == "egis_test_key"
