"""HTTP fallback patcher for ``httpx`` and ``requests``.

Logs model-API-shaped traffic from clients we don't have a
dedicated adapter for. URLs are matched against a known-host
allow-list so unrelated HTTP traffic is left alone. When a
dedicated adapter has already gated the call, the fallback does
nothing.
"""

from __future__ import annotations

import logging
import threading

from egisai._context import get_policy_checked
from egisai._events import build_event
from egisai._logger import enqueue
from egisai._patches import has_module
from egisai._patches._common import _attribute_event

LOGGER = logging.getLogger("egisai.patches.http")

# Hostnames of the org's registered custom (self-hosted) model
# endpoints, delivered on the policy handshake (``custom_model_hosts``
# in ``/v1/sdk/policies``) and pushed here by ``_policy_cache`` on
# every refresh. A raw httpx/requests call to one of these hosts on an
# inference-shaped path is AUDITED exactly like the built-in public
# providers — the Case-4 analog so a private/self-hosted model called
# over a transport we don't have a dedicated adapter for still shows up
# on the dashboard. Audit-only, fail-open; never blocks the call.
#
# Guarded by a lock because the refresher thread writes while the
# customer's model-call thread reads. Reads copy the frozenset once
# per call so matching never holds the lock across the URL scan.
_custom_hosts_lock = threading.Lock()
_custom_hosts: frozenset[str] = frozenset()

# Inference-shaped path tokens accepted for a *custom* host. Broader
# than any single public provider because self-hosted servers expose a
# grab-bag of conventions (vLLM / TGI / Ollama / LM Studio …), but we
# still require host+path so unrelated traffic to the same host (health
# checks, metrics scrapes) isn't logged as a phantom model call.
_CUSTOM_PATH_TOKENS: tuple[str, ...] = (
    "/chat/completions",
    "/completions",
    "/responses",
    "/embeddings",
    "/messages",
    "/generate",
    "/api/chat",
    "/api/generate",
)


def set_custom_model_hosts(hosts: list[str] | None) -> None:
    """Replace the audited custom-model host set. Thread-safe, never raises.

    Called by ``_policy_cache`` on every successful policy refresh with
    the org's ``custom_model_hosts``. ``None`` / empty clears the set
    (the org has no custom models) — the built-in public host list is
    unaffected either way.
    """
    try:
        normalized = frozenset(
            h.strip().lower() for h in (hosts or []) if h and h.strip()
        )
    except Exception:  # noqa: BLE001
        return
    global _custom_hosts
    with _custom_hosts_lock:
        _custom_hosts = normalized


def _custom_hosts_snapshot() -> frozenset[str]:
    with _custom_hosts_lock:
        return _custom_hosts

# (host_token, path_token) pairs — BOTH substrings must appear in
# the URL for it to count as a model call. The earlier host-only
# match was too permissive: every framework-side bookkeeping HTTP
# request that *happens* to share a host with the LLM provider got
# logged as a phantom audit event. The canonical offender is the
# OpenAI Agents SDK's own tracing exporter, which POSTs to
# ``https://api.openai.com/v1/traces/ingest`` after each
# ``Runner.run`` returns. With the host-only match those uploads
# surfaced on the dashboard as ghost ``model="unknown"`` /
# ``verdict="allow"`` rows attributed to the app instead of the
# agent, *in addition to* the real run that already had its own
# verdict — cosmetically wrong and operationally misleading.
#
# We list every model-call path we know about per provider so a
# new endpoint (e.g. OpenAI ships ``/v1/edit``) requires an
# explicit add — false-negative-by-default is the right posture
# for an audit fallback. Dedicated patches (``_patches.openai``,
# ``_patches.anthropic``, …) cover the in-band path; this fallback
# only matters when the customer used a transport we don't have an
# adapter for.
_MODEL_CALL_URL_TOKENS: tuple[tuple[str, str], ...] = (
    # OpenAI Chat / Completions / Responses / Embeddings
    ("api.openai.com", "/chat/completions"),
    ("api.openai.com", "/completions"),
    ("api.openai.com", "/responses"),
    ("api.openai.com", "/embeddings"),
    # Anthropic Messages / legacy Complete
    ("api.anthropic.com", "/messages"),
    ("api.anthropic.com", "/complete"),
    # Google Generative Language (Gemini)
    ("generativelanguage.googleapis.com", ":generatecontent"),
    ("generativelanguage.googleapis.com", ":streamgeneratecontent"),
    ("generativelanguage.googleapis.com", ":embedcontent"),
    # Azure OpenAI
    (".azure.com", "/openai/deployments"),
    # Together AI
    ("api.together.xyz", "/chat/completions"),
    ("api.together.xyz", "/completions"),
    # Groq (OpenAI-compatible)
    ("api.groq.com", "/chat/completions"),
    ("api.groq.com", "/completions"),
    # Cohere
    ("api.cohere.com", "/chat"),
    ("api.cohere.com", "/generate"),
    # Mistral
    ("api.mistral.ai", "/chat/completions"),
    ("api.mistral.ai", "/completions"),
)


def _looks_like_model_call(url: str) -> bool:
    """Return True only when the URL targets a known model-call endpoint.

    Hostname alone is not enough — see the docstring on
    ``_MODEL_CALL_URL_TOKENS`` for the OpenAI Agents tracing
    incident that motivated the host+path requirement.
    """
    u = url.lower()
    if any(host in u and path in u for host, path in _MODEL_CALL_URL_TOKENS):
        return True
    # Org-registered custom (self-hosted) model hosts — same host+path
    # discipline, broader path vocabulary (see ``_CUSTOM_PATH_TOKENS``).
    custom = _custom_hosts_snapshot()
    if custom and any(
        host in u and path in u
        for host in custom
        for path in _CUSTOM_PATH_TOKENS
    ):
        return True
    return False


def _patch_requests() -> bool:
    if not has_module("requests"):
        return False
    try:
        from requests.sessions import Session  # type: ignore
    except Exception:  # noqa: BLE001
        return False
    if getattr(Session.request, "__egisai_wrapped__", False):
        return True

    orig = Session.request

    def wrapped(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
        try:
            if get_policy_checked():
                return orig(self, method, url, **kwargs)
            if _looks_like_model_call(str(url)):
                payload = {"method": method, "json": kwargs.get("json")}
                ev = build_event(
                    source="requests",
                    target=str(url),
                    payload=payload,
                )
                _attribute_event(ev, payload.get("json"))
                ev["verdict"] = "allow"
                ev["reason"] = "Network-layer event"
                enqueue(ev)
        except Exception:  # noqa: BLE001
            pass
        return orig(self, method, url, **kwargs)

    wrapped.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    Session.request = wrapped  # type: ignore[assignment]
    return True


def _patch_httpx() -> bool:
    if not has_module("httpx"):
        return False
    try:
        import httpx  # type: ignore
    except Exception:  # noqa: BLE001
        return False
    if getattr(httpx.Client.request, "__egisai_wrapped__", False):
        return True

    orig = httpx.Client.request

    def wrapped(self, method, url, **kwargs):  # type: ignore[no-untyped-def]
        try:
            if get_policy_checked():
                return orig(self, method, url, **kwargs)
            url_s = str(url)
            if _looks_like_model_call(url_s):
                payload = {"method": method, "json": kwargs.get("json")}
                ev = build_event(
                    source="httpx",
                    target=url_s,
                    payload=payload,
                )
                _attribute_event(ev, payload.get("json"))
                ev["verdict"] = "allow"
                ev["reason"] = "Network-layer event"
                enqueue(ev)
        except Exception:  # noqa: BLE001
            pass
        return orig(self, method, url, **kwargs)

    wrapped.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    httpx.Client.request = wrapped  # type: ignore[assignment]
    return True


def apply() -> bool:
    a = _patch_requests()
    b = _patch_httpx()
    return a or b
