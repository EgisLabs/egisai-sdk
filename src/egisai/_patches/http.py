"""HTTP fallback patcher for ``httpx`` and ``requests``.

Logs model-API-shaped traffic from clients we don't have a
dedicated adapter for. URLs are matched against a known-host
allow-list so unrelated HTTP traffic is left alone. When a
dedicated adapter has already gated the call, the fallback does
nothing.
"""

from __future__ import annotations

import logging

from egisai._context import get_policy_checked
from egisai._events import build_event
from egisai._logger import enqueue
from egisai._patches import has_module

LOGGER = logging.getLogger("egisai.patches.http")

_MODEL_HOST_TOKENS = (
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    ".azure.com/openai",
    "api.together.xyz",
    "api.groq.com",
    "api.cohere.com",
    "api.mistral.ai",
)


def _looks_like_model_call(url: str) -> bool:
    u = url.lower()
    return any(host in u for host in _MODEL_HOST_TOKENS)


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
                ev = build_event(
                    source="requests",
                    target=str(url),
                    payload={"method": method, "json": kwargs.get("json")},
                )
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
                ev = build_event(
                    source="httpx",
                    target=url_s,
                    payload={"method": method, "json": kwargs.get("json")},
                )
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
