"""``egisai.Client`` — a first-class client for the inline Gateway.

The user-facing contract puts Egis first::

    import egisai

    client = egisai.Client(
        api_key="egis_live_…",       # your Egis key
        provider_key="sk-ant-…",     # forwarded to the provider untouched
    )
    response = client.chat.completions.create(
        model="claude-sonnet-4-5",   # model name picks the provider
        messages=[{"role": "user", "content": "Hello!"}],
    )

``provider_key`` is optional. Omit it to use **BYOK vault mode**:
store your provider keys once on the dashboard's Gateway page and the
Client sends only your Egis key — the Gateway forwards the stored key
for the request's provider. This is the same capability that lets
header-less platforms (Cursor, n8n) connect with just the Egis key::

    client = egisai.Client(api_key="egis_live_…")   # keys from the vault

No provider import, no ``base_url`` wiring, no header plumbing — the
client always talks to the platform's Gateway, which evaluates
policies, sanitizes/blocks inline, routes to the right provider from
the model name, and writes the audit row server-side.

Under the hood the call surface (``.chat.completions.create``,
streaming, etc.) is delegated to the ``openai`` package configured
for the Gateway — it is the Gateway's wire format, an implementation
detail the customer never sees. The dependency ships via the
``egisai[openai]`` extra; constructing a Client without it raises a
clear install hint.

``egisai.init()`` is NOT required — the Client carries its own keys.
When ``init()`` *has* run, the openai patch recognises gateway-bound
clients and adds per-call context (``egisai.set_context(agent=…)`` /
``with egisai.agent(…):`` → ``X-Egis-Agent``, plus the other
``set_context`` fields as ``X-Egis-User`` / ``X-Egis-User-Role`` /
``X-Egis-Session`` / ``X-Egis-Workflow`` / ``X-Egis-End-User``)
without ever running the local gate, so nothing is governed twice.
"""

from __future__ import annotations

import os
from typing import Any

from egisai._config import get_config_optional

_INSTALL_HINT = (
    "egisai.Client requires the 'openai' package (the Gateway's wire "
    "format). Install it with: pip install 'egisai[openai]'"
)

#: BYOK vault sentinel. When no provider key is configured anywhere,
#: the OpenAI transport still needs *some* ``api_key`` to construct —
#: it always emits it as ``Authorization: Bearer <key>``. This
#: Egis-namespaced placeholder is recognised by the Gateway as "not a
#: provider key" (it starts with ``egis_``), so the Gateway resolves
#: the real provider key from the org's stored BYOK vault, keyed by
#: the request model's provider. It is never a real credential and is
#: never sent to a provider.
_VAULT_SENTINEL = "egis_vault_no_provider_key"


def _resolve_egis_key(api_key: str | None) -> str:
    if api_key:
        return api_key
    cfg = get_config_optional()
    if cfg is not None and cfg.api_key:
        return cfg.api_key
    env_key = os.getenv("EGISAI_API_KEY", "")
    if env_key:
        return env_key
    raise RuntimeError(
        "egisai.Client requires `api_key` (your egis_live_… key), the "
        "EGISAI_API_KEY env var, or a prior egisai.init(api_key=…)."
    )


def _resolve_gateway_url(base_url: str | None) -> str:
    root = base_url
    if not root:
        cfg = get_config_optional()
        root = cfg.base_url if cfg is not None else None
    if not root:
        root = os.getenv("EGISAI_BASE_URL") or "https://app.egisai.co"
    return root.rstrip("/") + "/v1"


def _build_inner(
    *,
    is_async: bool,
    api_key: str | None,
    provider_key: str | None,
    agent: str | None,
    base_url: str | None,
    openai_kwargs: dict[str, Any],
) -> Any:
    try:
        # Optional extra (egisai[openai]) — absent in a base install, so
        # mypy in a clean environment can't resolve it. Same ignore as
        # the lazy import in _patches/openai.py.
        import openai  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover — exercised via CI matrix
        raise ImportError(_INSTALL_HINT) from exc

    headers: dict[str, str] = {"X-Egis-Api-Key": _resolve_egis_key(api_key)}
    if agent:
        headers["X-Egis-Agent"] = agent
    extra_headers = openai_kwargs.pop("default_headers", None)
    if isinstance(extra_headers, dict):
        headers.update({str(k): v for k, v in extra_headers.items()})

    if provider_key is not None:
        openai_kwargs["api_key"] = provider_key
    elif "api_key" not in openai_kwargs and not os.getenv("OPENAI_API_KEY"):
        # BYOK vault mode: no provider key configured anywhere. Hand
        # the transport a placeholder the Gateway resolves against the
        # org's stored provider keys. (If OPENAI_API_KEY *is* set, the
        # transport uses it — legacy passthrough for the default
        # provider — so we don't override that here.)
        openai_kwargs["api_key"] = _VAULT_SENTINEL

    cls = openai.AsyncOpenAI if is_async else openai.OpenAI
    return cls(
        base_url=_resolve_gateway_url(base_url),
        default_headers=headers,
        **openai_kwargs,
    )


class Client:
    """Synchronous Gateway client. See the module docstring.

    Parameters
    ----------
    api_key
        Your Egis key (``egis_live_…``). Falls back to a prior
        ``egisai.init()``'s key, then the ``EGISAI_API_KEY`` env var.
    provider_key
        The upstream provider's key (``sk-…`` / ``sk-ant-…`` / …),
        forwarded untouched in ``Authorization``; never stored or
        logged by the platform. **Optional** — omit it to use BYOK
        vault mode, where the Gateway supplies the provider key from
        the org's stored, encrypted keys (configured on the Gateway
        page). Falls back to ``OPENAI_API_KEY`` when set, for legacy
        default-provider passthrough.
    agent
        Optional explicit agent name for every call from this client
        (the ``X-Egis-Agent`` header). Per-call context set via
        ``egisai.set_context`` / ``with egisai.agent(…):`` still wins
        when ``egisai.init()`` is active.
    base_url
        Platform URL override for self-hosted / regional installs.
        Defaults like ``init()``: ``EGISAI_BASE_URL`` or the hosted
        control plane.
    **openai_kwargs
        Passed through to the underlying transport (``timeout``,
        ``max_retries``, ``http_client``, …).
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        provider_key: str | None = None,
        agent: str | None = None,
        base_url: str | None = None,
        **openai_kwargs: Any,
    ) -> None:
        self._inner = _build_inner(
            is_async=False,
            api_key=api_key,
            provider_key=provider_key,
            agent=agent,
            base_url=base_url,
            openai_kwargs=openai_kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return f"egisai.Client(base_url={str(self._inner.base_url)!r})"


class AsyncClient:
    """Asynchronous sibling of :class:`Client` — same parameters."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        provider_key: str | None = None,
        agent: str | None = None,
        base_url: str | None = None,
        **openai_kwargs: Any,
    ) -> None:
        self._inner = _build_inner(
            is_async=True,
            api_key=api_key,
            provider_key=provider_key,
            agent=agent,
            base_url=base_url,
            openai_kwargs=openai_kwargs,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return f"egisai.AsyncClient(base_url={str(self._inner.base_url)!r})"
