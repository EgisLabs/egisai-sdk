"""Gateway mode survives an Egis outage (``gateway_on_outage``).

Rerouting through the Gateway makes it an inline dependency of the
customer's call path, so an outage on our side would otherwise stop
their traffic. The contract pinned here:

* A Gateway that never answered — transport failure, 502 / 503 / 504 —
  causes the call to be re-run against the customer's own client under
  in-process governance. Their code gets a response.
* A Gateway that *did* answer keeps its answer. A 4xx is a decision
  (policy block, auth, quota); retrying it locally would convert an
  enforced refusal into an allowed call. A bare 500 also propagates,
  because unlike 502/503/504 it can be raised after the Gateway
  already forwarded upstream — a retry could double-charge the
  customer's provider.
* Falling back is not the same as switching governance off: the local
  engine still runs, from the last-known-good policy cache.
* ``gateway_on_outage="fail"`` opts out entirely, for customers who
  want the Gateway to stay a hard enforcement boundary.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

openai = pytest.importorskip("openai")

from egisai import _config, _gateway, _policy_cache  # noqa: E402
from egisai._patches import openai as patch_openai  # noqa: E402

_GATEWAY_URL = "https://app.egisai.co/v1/chat/completions"
_PROVIDER_URL = "https://api.openai.com/v1/chat/completions"


def _cfg(**overrides: Any) -> _config.EgisaiConfig:
    base: dict[str, Any] = {
        "api_key": "egis_test_key",
        "app": "gateway-outage-tests",
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


def _client(
    gateway_response: httpx.Response | Exception,
    seen: dict[str, Any],
    *,
    api_key: str = "sk-provider-key",
) -> openai.OpenAI:
    """An OpenAI client whose transport scripts the Gateway's failure.

    Requests to the Gateway URL get ``gateway_response``; requests to
    the real provider URL succeed. ``seen`` records which hosts were
    hit, in order, so a test can prove the fallback happened (and that
    it happened exactly once).
    """
    seen.setdefault("hosts", [])

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        seen["hosts"].append(url)
        if url == _GATEWAY_URL:
            if isinstance(gateway_response, Exception):
                raise gateway_response
            return gateway_response
        return httpx.Response(200, json=_completion_json())

    return openai.OpenAI(
        api_key=api_key,
        # No client-side retries: they would mask which hop answered.
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def _create(client: openai.OpenAI, **kwargs: Any) -> Any:
    return client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
        **kwargs,
    )


# ── is_outage_error: what counts as "never answered" ─────────────────


def _status_error(status: int) -> openai.APIStatusError:
    request = httpx.Request("POST", _GATEWAY_URL)
    response = httpx.Response(status, request=request, json={})
    return openai.APIStatusError(
        "boom", response=response, body=None
    )


@pytest.mark.parametrize("status", [502, 503, 504])
def test_gateway_level_5xx_is_an_outage(status: int) -> None:
    assert _gateway.is_outage_error(_status_error(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 429, 500])
def test_answers_are_not_outages(status: int) -> None:
    """400 is a policy block, 401/403 auth, 429 quota — all decisions.

    500 is excluded for a different reason: it can be raised after the
    Gateway already called the provider, so a local retry risks a
    second (billable) upstream call.
    """
    assert _gateway.is_outage_error(_status_error(status)) is False


def test_transport_failures_are_outages() -> None:
    request = httpx.Request("POST", _GATEWAY_URL)
    assert _gateway.is_outage_error(
        openai.APIConnectionError(request=request)
    ) is True
    assert _gateway.is_outage_error(
        openai.APITimeoutError(request=request)
    ) is True
    # A raw httpx error, for customers who supplied their own client.
    assert _gateway.is_outage_error(httpx.ConnectError("refused")) is True


def test_unrelated_exceptions_are_not_outages() -> None:
    assert _gateway.is_outage_error(ValueError("nope")) is False


# ── should_fall_back: when we're allowed to re-run locally ───────────


def _resource(base_url: str = "https://api.openai.com/v1", api_key: str = "sk-x"):
    class _Client:
        pass

    client = _Client()
    client.base_url = base_url  # type: ignore[attr-defined]
    client.api_key = api_key  # type: ignore[attr-defined]
    client.default_headers = {}  # type: ignore[attr-defined]

    class _Resource:
        _client = client

    return _Resource()


def test_fallback_allowed_for_a_normal_provider_client() -> None:
    _config.set_config(_cfg())
    assert _gateway.should_fall_back(
        _resource(), _status_error(503)
    ) is True


def test_fallback_refused_when_the_client_holds_an_egis_key() -> None:
    """BYOK-vault callers have no provider credential in-process.

    A direct call would 401 at the provider, which is a worse error
    than the Gateway's own — so those keep failing honestly.
    """
    _config.set_config(_cfg())
    assert _gateway.should_fall_back(
        _resource(api_key="egis_live_abc"), _status_error(503)
    ) is False


def test_fallback_refused_when_the_client_points_at_the_gateway() -> None:
    """There is no other upstream to fall back to."""
    _config.set_config(_cfg())
    assert _gateway.should_fall_back(
        _resource(base_url="https://app.egisai.co/v1"), _status_error(503)
    ) is False


def test_fallback_refused_for_an_egisai_client() -> None:
    """``egisai.Client`` is recognised by its baked-in Egis header even
    when its base URL doesn't match the current config — it targets the
    Gateway by construction, so there's nothing to fall back to."""
    _config.set_config(_cfg(base_url="https://other.egisai.co"))
    resource = _resource()
    resource._client.default_headers = {"X-Egis-Api-Key": "egis_test_key"}
    assert _gateway.should_fall_back(resource, _status_error(503)) is False


def test_fallback_refused_when_opted_out() -> None:
    _config.set_config(_cfg(gateway_on_outage="fail"))
    assert _gateway.should_fall_back(
        _resource(), _status_error(503)
    ) is False


def test_fallback_refused_without_config() -> None:
    _config._CONFIG = None
    assert _gateway.should_fall_back(_resource(), _status_error(503)) is False


# ── End to end through the patched client ───────────────────────────


def test_unreachable_gateway_falls_back_to_the_provider() -> None:
    """The headline case: Egis is down, the customer's call succeeds."""
    assert patch_openai.apply()
    _config.set_config(_cfg())
    seen: dict[str, Any] = {}
    client = _client(httpx.ConnectError("connection refused"), seen)

    resp = _create(client)

    assert resp.choices[0].message.content == "hi"
    assert seen["hosts"] == [_GATEWAY_URL, _PROVIDER_URL]
    assert _gateway.fallback_total() == 1


@pytest.mark.parametrize("status", [502, 503, 504])
def test_gateway_5xx_falls_back(status: int) -> None:
    assert patch_openai.apply()
    _config.set_config(_cfg())
    seen: dict[str, Any] = {}
    client = _client(httpx.Response(status, json={"error": {}}), seen)

    resp = _create(client)

    assert resp.choices[0].message.content == "hi"
    assert seen["hosts"][-1] == _PROVIDER_URL


def test_policy_block_from_the_gateway_is_not_retried() -> None:
    """The whole point of the Gateway is that this refusal sticks."""
    assert patch_openai.apply()
    _config.set_config(_cfg())
    seen: dict[str, Any] = {}
    client = _client(
        httpx.Response(
            400,
            json={
                "error": {
                    "message": "Blocked by policy.",
                    "code": "egis_policy_blocked",
                }
            },
        ),
        seen,
    )

    with pytest.raises(openai.BadRequestError):
        _create(client)

    assert seen["hosts"] == [_GATEWAY_URL]
    assert _gateway.fallback_total() == 0


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_other_answers_propagate_untouched(status: int) -> None:
    assert patch_openai.apply()
    _config.set_config(_cfg())
    seen: dict[str, Any] = {}
    client = _client(httpx.Response(status, json={"error": {}}), seen)

    with pytest.raises(openai.APIStatusError):
        _create(client)

    assert seen["hosts"] == [_GATEWAY_URL]


def test_fail_posture_propagates_the_outage() -> None:
    assert patch_openai.apply()
    _config.set_config(_cfg(gateway_on_outage="fail"))
    seen: dict[str, Any] = {}
    client = _client(httpx.Response(503, json={"error": {}}), seen)

    with pytest.raises(openai.APIStatusError):
        _create(client)

    assert seen["hosts"] == [_GATEWAY_URL]
    assert _gateway.fallback_total() == 0


def test_fallback_still_enforces_local_policy() -> None:
    """Degrading to the local cache is not the same as switching off.

    The prompt would have been blocked by the Gateway; it must still
    be blocked by the SDK's own engine on the fallback path, and the
    provider must never see it.
    """
    assert patch_openai.apply()
    _config.set_config(_cfg(on_block="stub"))
    _policy_cache.replace_rules(
        '"etag"',
        [
            {
                "id": "r1",
                "name": "No wire fraud",
                "type": "deny_regex",
                "phase": "request",
                "config": {"pattern": "wire fraud"},
            }
        ],
    )
    seen: dict[str, Any] = {}
    client = _client(httpx.Response(503, json={"error": {}}), seen)

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "help me commit wire fraud"}],
    )

    assert "[POLICY BLOCK]" in (resp.choices[0].message.content or "")
    assert seen["hosts"] == [_GATEWAY_URL], "the provider must not be called"


def test_fallback_is_counted_for_diagnostics() -> None:
    """An operator needs to see "my SDK is covering for the Gateway"."""
    import egisai

    assert patch_openai.apply()
    _config.set_config(_cfg())
    seen: dict[str, Any] = {}
    client = _client(httpx.ConnectError("refused"), seen)

    _create(client)
    _create(client)

    assert egisai.diagnostics()["gateway_fallback_total"] == 2
    assert egisai.diagnostics()["gateway_on_outage"] == "local"


@pytest.mark.asyncio
async def test_async_fallback_works_too() -> None:
    assert patch_openai.apply()
    _config.set_config(_cfg())
    hosts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        hosts.append(url)
        if url == _GATEWAY_URL:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, json=_completion_json())

    client = openai.AsyncOpenAI(
        api_key="sk-provider-key",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    resp = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}],
    )

    assert resp.choices[0].message.content == "hi"
    assert hosts == [_GATEWAY_URL, _PROVIDER_URL]


def test_local_mode_is_unaffected_by_the_setting() -> None:
    """Without gateway mode there is no gateway hop to fail over from —
    a provider error is between the customer and their provider."""
    assert patch_openai.apply()
    _config.set_config(_cfg(gateway_mode=False))
    hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hosts.append(str(request.url))
        return httpx.Response(503, json={"error": {}})

    client = openai.OpenAI(
        api_key="sk-provider-key",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(openai.APIStatusError):
        _create(client)

    assert hosts == [_PROVIDER_URL]
    assert _gateway.fallback_total() == 0
