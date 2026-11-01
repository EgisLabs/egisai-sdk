"""The on-prem provider-denied refusal must be a status the OpenAI SDK
does NOT auto-retry.

The on-prem gateway refuses a disallowed public AI provider with
``egis_onprem_upstream_denied``. That verdict is terminal — retrying
cannot change it — so the gateway returns **403**, not a 5xx. This test
pins the behaviour that motivated the status choice: the stock OpenAI
SDK retries 5xx (it raises ``InternalServerError`` and dials again,
burning the agent's budget on a decision that will never flip) but
surfaces a 403 immediately as ``PermissionDeniedError`` after a single
attempt.

Guarded by ``importorskip`` so environments without the optional
``openai`` extra simply skip it.
"""

from __future__ import annotations

import pytest

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")


_DENIED_ENVELOPE = {
    "error": {
        "message": (
            "On-prem gateway refuses public upstream for provider 'openai' "
            "(model 'gpt-4o-mini'): this install is private-only and the "
            "'openai' public AI provider is not allowed. An owner/admin can "
            "allow it in Settings → Organization → LLM providers, or set a "
            "private base_url_override for a self-hosted endpoint."
        ),
        "type": "invalid_request_error",
        "code": "egis_onprem_upstream_denied",
    }
}


def _counting_client(status: int) -> tuple[openai.OpenAI, list[int]]:
    """An OpenAI client whose transport records every attempt.

    ``max_retries`` is set high so that *if* the SDK considered the
    status retryable we would see more than one attempt.
    """
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(status, json=_DENIED_ENVELOPE)

    client = openai.OpenAI(
        api_key="sk-provider-key",
        base_url="https://onprem.egis.local/v1",
        max_retries=4,
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return client, attempts


def test_openai_sdk_does_not_retry_the_403_denial() -> None:
    client, attempts = _counting_client(403)
    with pytest.raises(openai.PermissionDeniedError) as exc:
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    # Exactly ONE dial — the SDK did not auto-retry the denial.
    assert len(attempts) == 1
    assert exc.value.status_code == 403
    # The machine code rides through untouched for client branching.
    assert exc.value.body["code"] == "egis_onprem_upstream_denied"


def test_openai_sdk_would_retry_a_503_control() -> None:
    """Control: the SAME body at 503 IS retried, which is exactly the
    budget-burning behaviour the 403 choice avoids."""
    client, attempts = _counting_client(503)
    with pytest.raises(openai.InternalServerError):
        client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hi"}],
        )
    # Initial attempt + at least one retry.
    assert len(attempts) > 1
