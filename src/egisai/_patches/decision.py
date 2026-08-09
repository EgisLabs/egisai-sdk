"""Tell an egress node that this call was already governed.

An organization can run both control points at once, and increasingly
will: the SDK inside the services their own team wrote, an egress node
in front of everything else. Without a signal between them, a single
model call made by an SDK-instrumented service passes through the node
too — and gets evaluated twice and written to the audit log twice. Two
rows for one call is the kind of detail that makes a compliance export
impossible to defend.

So the SDK stamps one header on calls it has already decided, and the
node steps aside when it sees it. The rules that keep this from being a
bypass:

**It is only ever stamped on a governed call.** The header is written
inside the window where ``policy_checked`` is true, which is the same
window the in-process decision was made in. A call the SDK merely
observed does not get it, and the node governs that one.

**It only goes to hosts we already govern.** The same host list the
node uses. This never appears on a request to a customer's own API.

**It carries no authority.** The value is a marker, not a token. A node
that trusts it is trusting traffic from inside its own network that was
already going to be forwarded; the header cannot make the node do
anything it would not otherwise do, only skip work that was already
done closer to the code.

With no node in the path — the overwhelmingly common case — this adds
one short header to a request and changes nothing else.
"""

from __future__ import annotations

import logging
from typing import Any

from egisai._context import get_policy_checked
from egisai._patches import has_module

LOGGER = logging.getLogger("egisai.patches.decision")

HEADER = "X-Egis-Decision"

#: Deliberately opaque and deliberately constant. A verdict here would
#: invite somebody to trust it, and the node has no way to tell a real
#: one from a forged one.
VALUE = "governed"

#: The model vendors an egress node inspects. Kept in step with
#: ``egress_service.DEFAULT_GOVERNED_HOSTS`` on the backend; a host
#: missing here costs a duplicate row, never a missed decision.
_HOSTS: frozenset[str] = frozenset(
    {
        "api.openai.com",
        "api.anthropic.com",
        "generativelanguage.googleapis.com",
        "api.mistral.ai",
        "api.cohere.ai",
        "api.cohere.com",
        "api.groq.com",
        "api.together.xyz",
        "api.deepseek.com",
        "api.x.ai",
        "api.perplexity.ai",
        "openrouter.ai",
        "api.voyageai.com",
        "api.fireworks.ai",
        "api.replicate.com",
    }
)


def _governed_host(host: str) -> bool:
    lowered = (host or "").lower()
    if lowered in _HOSTS:
        return True
    if lowered.endswith(".azure.com") or lowered.endswith(
        ".openai.azure.com"
    ):
        return True
    return lowered.endswith(".amazonaws.com") and "bedrock" in lowered


def _stamp(request: Any) -> None:
    """Add the header in place. Never raises, never overwrites."""
    try:
        if not get_policy_checked():
            return
        host = getattr(getattr(request, "url", None), "host", "") or ""
        if not _governed_host(str(host)):
            return
        headers = request.headers
        if HEADER not in headers:
            headers[HEADER] = VALUE
    except Exception:  # noqa: BLE001
        LOGGER.debug("could not stamp the decision header", exc_info=True)


def apply() -> bool:
    """Patch ``httpx``'s send path, which every provider SDK sits on.

    ``send`` rather than ``request``: the OpenAI and Anthropic clients
    build a ``Request`` themselves and hand it straight to ``send``, so
    a wrapper on ``request`` would never see their traffic — which is
    most of it.
    """
    if not has_module("httpx"):
        return False
    try:
        import httpx  # type: ignore
    except Exception:  # noqa: BLE001
        return False

    patched = False

    if not getattr(httpx.Client.send, "__egisai_wrapped__", False):
        orig_sync = httpx.Client.send

        def sync_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
            _stamp(request)
            return orig_sync(self, request, **kwargs)

        sync_send.__egisai_wrapped__ = True  # type: ignore[attr-defined]
        httpx.Client.send = sync_send  # type: ignore[assignment]
        patched = True

    if not getattr(httpx.AsyncClient.send, "__egisai_wrapped__", False):
        orig_async = httpx.AsyncClient.send

        async def async_send(self, request, **kwargs):  # type: ignore[no-untyped-def]
            _stamp(request)
            return await orig_async(self, request, **kwargs)

        async_send.__egisai_wrapped__ = True  # type: ignore[attr-defined]
        httpx.AsyncClient.send = async_send  # type: ignore[assignment]
        patched = True

    return patched
