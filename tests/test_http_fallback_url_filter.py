"""HTTP fallback must only log actual model-call endpoints.

Regression coverage for the OpenAI Agents tracing leak:

The ``openai-agents`` SDK uploads its trace data via
``httpx.Client`` to ``https://api.openai.com/v1/traces/ingest``
*after* ``Runner.run`` returns. The pre-fix host-only URL match
(``"api.openai.com" in url``) treated those tracing uploads as a
new ungoverned model call, enqueuing a phantom
``model="unknown"``, ``verdict="allow"`` audit row attributed to
the app instead of the agent. On the dashboard the operator saw
two run rows per agent invocation: the legitimate one (correctly
attributed, correctly verdict-stamped) and the phantom one
(useless, misleading).

The fix is in ``_patches/http._looks_like_model_call``: a URL
must contain BOTH a known LLM-provider host AND a known
model-call path token (``/chat/completions``, ``/responses``,
``/messages``, etc.) for the fallback to log it. Tracing,
file-upload, audio, and image-generation endpoints all share the
host but never share the path, so they're now silently ignored
exactly as the operator expects.

We test the predicate directly because it's the actual policy
choice the patch encodes — wiring it through real ``httpx``
would test ``httpx`` more than the predicate.
"""

from __future__ import annotations

import pytest

from egisai._patches.http import _looks_like_model_call


@pytest.mark.parametrize(
    "url",
    [
        # OpenAI Chat Completions
        "https://api.openai.com/v1/chat/completions",
        # OpenAI Responses
        "https://api.openai.com/v1/responses",
        # OpenAI legacy completions
        "https://api.openai.com/v1/completions",
        # OpenAI embeddings
        "https://api.openai.com/v1/embeddings",
        # Anthropic
        "https://api.anthropic.com/v1/messages",
        "https://api.anthropic.com/v1/complete",
        # Google Gemini
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:streamGenerateContent",
        "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent",
        # Azure OpenAI
        "https://contoso.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-08-01-preview",
        # Together / Groq / Cohere / Mistral
        "https://api.together.xyz/v1/chat/completions",
        "https://api.groq.com/openai/v1/chat/completions",
        "https://api.cohere.com/v1/chat",
        "https://api.cohere.com/v1/generate",
        "https://api.mistral.ai/v1/chat/completions",
    ],
)
def test_real_model_call_urls_are_matched(url: str) -> None:
    assert _looks_like_model_call(url), (
        f"URL should be recognised as a model call: {url!r}"
    )


@pytest.mark.parametrize(
    "url",
    [
        # The exact OpenAI Agents tracing upload that produced the
        # phantom dashboard row before the fix. THIS is the one
        # the regression test exists for.
        "https://api.openai.com/v1/traces/ingest",
        # Other non-model endpoints on the same hosts that should
        # also stay quiet — Assistants API, Files, Audio, Images,
        # Moderations, Models listing, Organisation admin, …
        "https://api.openai.com/v1/threads",
        "https://api.openai.com/v1/threads/thread_abc/runs",
        "https://api.openai.com/v1/files",
        "https://api.openai.com/v1/audio/transcriptions",
        "https://api.openai.com/v1/audio/speech",
        "https://api.openai.com/v1/images/generations",
        "https://api.openai.com/v1/moderations",
        "https://api.openai.com/v1/models",
        "https://api.openai.com/v1/organization/audit_logs",
        # Anthropic non-message endpoints (model listing,
        # workspace management, …).
        "https://api.anthropic.com/v1/models",
        "https://api.anthropic.com/v1/organizations/abc/workspaces",
        # Google non-generation endpoints.
        "https://generativelanguage.googleapis.com/v1beta/models",
        "https://generativelanguage.googleapis.com/v1beta/files",
        # Together / Cohere / Mistral non-LLM endpoints.
        "https://api.together.xyz/v1/files",
        "https://api.cohere.com/v1/embed",  # cohere embed has its own endpoint we deliberately don't catch via fallback
        "https://api.mistral.ai/v1/files",
        # Completely unrelated hosts — never logged regardless.
        "https://example.com/v1/chat/completions",
        "https://internal-tool.company.local/api/proxy",
    ],
)
def test_non_model_urls_are_ignored(url: str) -> None:
    assert not _looks_like_model_call(url), (
        f"URL should NOT be treated as a model call: {url!r}"
    )


def test_openai_agents_tracing_endpoint_does_not_leak_through_httpx_patch(
    fake_backend,
) -> None:
    """End-to-end: simulate the exact ``Runner.run`` → tracing-export
    sequence that produced the phantom run.

    We can't import ``openai-agents`` at test time (it's not a hard
    dep), but we can mirror the steps the SDK takes:

      1. A real gated ``Runner.run`` opens + closes a Run with a
         single blocked model_call step.
      2. *After* the Run closes (so ``policy_checked`` is back to
         False), the tracing exporter POSTs to
         ``api.openai.com/v1/traces/ingest`` with its own
         ``httpx.Client``.

    Pre-fix, step 2 enqueued an extra ``model=unknown`` /
    ``verdict=allow`` legacy event the dashboard rendered as a
    second run. Post-fix, step 2 is silently ignored because
    ``/v1/traces/ingest`` doesn't carry a known model-call path
    token.
    """
    import egisai
    from egisai._logger import _q
    from egisai._patches._common import gate_call

    egisai.init(
        api_key="egis_live_x",
        app="orchestrator",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    # Step 1 — a gated call that the policy immediately blocks. We
    # don't need a real openai-agents Runner here; what matters is
    # that the gate ran end-to-end and ``set_policy_checked`` is back
    # to False by the time step 2 fires (mirroring the post-Runner.run
    # state where the tracing exporter actually runs).
    def forward() -> str:  # pragma: no cover — never invoked in this test
        return "ok"

    gate_call(
        source="openai",
        target="openai.responses.create",
        model="gpt-4o",
        prompt_text="hi",
        stream=False,
        payload={"input": "hi"},
        forward=forward,
    )

    # Step 2 — emulate the tracing exporter's httpx call by invoking
    # the patched ``httpx.Client.request`` directly. The patch
    # short-circuits before ``orig(...)`` runs when the URL doesn't
    # match, so we don't actually need a live HTTP server. We DO need
    # the real wrapped ``httpx.Client.request`` (the SDK's init()
    # patched it earlier in this process), so import it from httpx
    # at use time.
    import httpx

    class _StubResponse:
        status_code = 200

        def raise_for_status(self) -> None:  # noqa: D401
            return None

    # Replace the underlying ``orig`` call with a stub that doesn't
    # touch the network. We can't easily monkey-patch the captured
    # ``orig`` closure, so we substitute the entire ``httpx.Client``
    # transport for this one call by going through ``Client.request``
    # via our wrapper but with a no-op transport.
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(204, request=request)
        )
    )
    try:
        client.request(
            "POST",
            "https://api.openai.com/v1/traces/ingest",
            json={"data": [{"object": "trace", "id": "trace_123"}]},
        )
    finally:
        client.close()

    # Drain queued events and assert no httpx Network-layer event
    # leaked through for the tracing endpoint.
    drained: list[dict] = []
    while not _q.empty():
        try:
            drained.append(_q.get_nowait())
        except Exception:
            break

    leak = [
        e
        for e in drained
        if e.get("source") == "httpx"
        and "traces/ingest" in str(e.get("target") or "")
    ]
    assert leak == [], (
        "OpenAI Agents tracing upload must NOT generate a phantom "
        f"audit row, got: {leak!r}"
    )
