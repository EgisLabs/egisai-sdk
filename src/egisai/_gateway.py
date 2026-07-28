"""Gateway mode — route governed calls through the inline Gateway.

Activated with ``egisai.init(gateway=True)`` (or ``EGISAI_GATEWAY=1``).
In this mode the OpenAI chat-completions patch stops evaluating
policies in-process and instead reroutes each call to the platform's
OpenAI-compatible Gateway (``<base_url>/v1/chat/completions``), which
runs the exact same engine server-side and writes the audit row
itself. The SDK's job shrinks to what only it can do:

* keep the customer's calling convention untouched (they still call
  ``client.chat.completions.create(...)`` on their own client), and
* carry the SDK's identity context over the wire — an explicit
  ``egisai.set_context(agent=…)`` / ``with egisai.agent(…)`` becomes
  the ``X-Egis-Agent`` header, preserving the "explicit wins over
  auto-detection" precedence. Without an explicit identity no header
  is sent and the Gateway fingerprints the system prompt server-side
  with the same algorithm the SDK uses locally.
* carry the rest of the request context the same way — the
  ``set_context`` fields ``user_id`` / ``user_role`` / ``session_id``
  / ``workflow_id`` / ``end_user_id`` ship as ``X-Egis-User`` /
  ``X-Egis-User-Role`` / ``X-Egis-Session`` / ``X-Egis-Workflow`` /
  ``X-Egis-End-User`` so gateway-audited runs show the same Context
  section as SDK-audited runs. Values are percent-encoded (RFC 3986,
  UTF-8) before hitting the wire: HTTP header values must be
  latin-1-safe or the transport raises *inside the customer's call*,
  which would violate fail-open. The Gateway decodes on intake.
  ``end_user_id`` follows the documented convention of being a hash
  already; the backend re-hashes on intake regardless, so the raw
  value never persists.

Scope (v1):

* Only ``chat.completions.create`` is rerouted — that is the surface
  the Gateway implements. The Responses API, embeddings, and every
  non-OpenAI provider keep the normal in-process governance path, so
  nothing silently loses coverage.
* Azure OpenAI clients are never rerouted (their URL scheme is
  deployment-based and incompatible with the passthrough contract).
* Fail open: if the reroute cannot even be constructed, the call
  falls back to the local-governance path — policies still enforce
  from the local cache, per the SDK's availability philosophy.

Outage fallback (``init(gateway_on_outage=…)``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rerouting makes the Gateway an inline dependency of the customer's
call path, so an outage on our side would otherwise stop their
traffic. :func:`should_fall_back` decides whether the SDK may re-run
the call locally instead. It says yes only when *all* of these hold:

* ``gateway_on_outage="local"`` (the default),
* the error means "we never got an answer from Egis" — a transport
  failure, or HTTP 502 / 503 / 504 (see :func:`is_outage_error`),
* the customer's own client can still reach the provider directly:
  it isn't pointed at the Gateway itself and its ``api_key`` is a
  real provider key, not an ``egis_`` key (BYOK-vault callers hold
  no upstream credential locally — see
  :func:`_client_can_reach_provider`).

Everything else propagates. In particular a 4xx is an *answer*: a
policy block, an auth/entitlement refusal, or a rate limit. Retrying
those locally would convert an enforced decision into an allowed
call. A bare 500 also propagates — unlike 502/503/504 it can be
raised after the Gateway already forwarded upstream, so retrying
risks double-charging the customer's provider.

Falling back does not disable governance: the local path runs the
same engine against the last-known-good policy cache, with PII
detection fully local. It does mean the audit row is written by the
SDK rather than the Gateway, and that a policy created during the
outage isn't visible yet. Both are logged loudly.
"""

from __future__ import annotations

import logging
import threading
from typing import Any
from urllib.parse import quote

import httpx

from egisai._config import get_config_optional

LOGGER = logging.getLogger("egisai.gateway")

_API_KEY_HEADER = "X-Egis-Api-Key"
_AGENT_HEADER = "X-Egis-Agent"

# ``set_context`` fields carried over the gateway wire, with the raw
# character budget each one gets before encoding. The caps mirror the
# backend's ``runs`` column widths (``_RUN_CONTEXT_LIMITS`` in
# ``routers/sdk.py``) so the gateway and SDK ingest paths truncate
# identically.
_CONTEXT_HEADER_FIELDS: tuple[tuple[str, str, int], ...] = (
    ("user_id", "X-Egis-User", 255),
    ("user_role", "X-Egis-User-Role", 64),
    ("session_id", "X-Egis-Session", 255),
    ("workflow_id", "X-Egis-Workflow", 255),
    ("end_user_id", "X-Egis-End-User", 255),
)

# Characters left un-escaped on top of RFC 3986 unreserved
# (letters, digits, ``-._~``). Chosen so common id shapes — emails,
# UUIDs, ``sess-…``/``wf_…`` slugs, ISO timestamps, paths — cross the
# wire byte-identical and stay human-readable in server logs. ``%``
# is always escaped, which is what makes the encoding lossless.
_HEADER_VALUE_SAFE = "@:/+=,"


def _encode_context_value(raw: str, max_chars: int) -> str | None:
    """Trim, cap, and percent-encode one context value for a header.

    Returns ``None`` for blank input so the header is simply omitted.
    Truncation happens BEFORE encoding — the cap is a data budget
    (matching the backend column), not a wire-bytes budget.
    """
    trimmed = raw.strip()[:max_chars]
    if not trimmed:
        return None
    return quote(trimmed, safe=_HEADER_VALUE_SAFE)


class RerouteUnavailable(Exception):
    """Raised when the gateway reroute cannot be constructed.

    The caller (the openai patch) catches this and falls back to the
    normal local-governance path — never the customer.
    """


def enabled() -> bool:
    cfg = get_config_optional()
    return cfg is not None and cfg.gateway_mode


def points_at_gateway(resource: Any) -> bool:
    """Is this resource's client already targeting the Gateway?

    True for ``egisai.Client`` / ``egisai.AsyncClient`` (which always
    do) and for hand-configured OpenAI clients whose ``base_url`` is
    the platform's ``/v1``. Used by the openai patch to skip the
    local gate for such calls even when ``gateway_mode`` is off —
    the Gateway governs them server-side, and running the local
    engine too would evaluate and audit everything twice.

    Recognition is two-pronged: base-URL match against the current
    config, or an ``X-Egis-Api-Key`` baked into the client's default
    headers. The header prong keeps ``egisai.Client`` on the gateway
    path even when ``init()`` never ran (the client carries its own
    keys) or when it ran with a different ``base_url``.
    """
    client = getattr(resource, "_client", None)
    if client is None:
        return False
    if _client_carries_egis_key(resource):
        return True
    cfg = get_config_optional()
    if cfg is None:
        return False
    try:
        current = str(getattr(client, "base_url", "")).rstrip("/")
    except Exception:  # noqa: BLE001
        return False
    return current == cfg.base_url.rstrip("/") + "/v1"


def should_carry(resource: Any) -> bool:
    """Should the patch hand this chat call to the Gateway path?"""
    return enabled() or points_at_gateway(resource)


# ── Outage fallback ─────────────────────────────────────────────────

# HTTP statuses that mean "the request never reached a Gateway
# handler, or the handler declared itself unavailable". 503 is what
# the Gateway returns when it can't load policy (see the backend's
# ``GatewayDegradedError``); 502 / 504 come from the load balancer in
# front of it. All three are safe to re-run: no upstream provider
# call was made under them.
_OUTAGE_STATUS_CODES = frozenset({502, 503, 504})

# ``openai`` exception class names that mean "no HTTP response".
# Matched by name across the MRO so this module stays importable
# without ``openai`` installed, and so it keeps working if upstream
# re-parents the classes. ``APITimeoutError`` subclasses
# ``APIConnectionError`` today; both are listed for clarity.
_OUTAGE_EXC_NAMES = frozenset(
    {"APIConnectionError", "APITimeoutError"}
)

_fallback_lock = threading.Lock()
_fallback_total = 0


def fallback_total() -> int:
    """How many calls have fallen back to local governance.

    Surfaced by :func:`egisai.diagnostics` so an operator can tell
    "the Gateway is degraded and my SDK is covering for it" apart
    from "everything is fine".
    """
    with _fallback_lock:
        return _fallback_total


def reset_fallback_total() -> None:
    """Reset the fallback counter. Used by tests."""
    global _fallback_total
    with _fallback_lock:
        _fallback_total = 0


def is_outage_error(exc: BaseException) -> bool:
    """Does ``exc`` mean the Gateway never answered this call?

    See the module docstring for why 4xx and a bare 500 are excluded.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status in _OUTAGE_STATUS_CODES
    for klass in type(exc).__mro__:
        if klass.__name__ in _OUTAGE_EXC_NAMES:
            return True
    # Raw httpx transport errors reach us when the customer supplied
    # their own ``http_client`` and the failure escaped before the
    # openai layer could wrap it.
    return isinstance(exc, httpx.TransportError)


def _client_can_reach_provider(resource: Any) -> bool:
    """Can this resource's client still call the provider directly?

    Two disqualifiers:

    * The client is already pointed at the Gateway (``egisai.Client``
      or a hand-set ``base_url``). There is no other upstream to fall
      back to — the reroute wasn't what put us here.
    * The client's key is an Egis key (BYOK-vault / Bearer mode). The
      provider key lives in our encrypted vault, not in the process,
      so a direct call would 401 at the provider. Failing with the
      Gateway's own error is more honest than that.
    """
    if points_at_gateway(resource):
        return False
    try:
        api_key = getattr(resource._client, "api_key", None)  # noqa: SLF001
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(api_key, str) or not api_key.strip():
        return False
    return not api_key.startswith("egis_")


def should_fall_back(resource: Any, exc: BaseException) -> bool:
    """Should this failed gateway call be re-run under local governance?"""
    cfg = get_config_optional()
    if cfg is None or cfg.gateway_on_outage != "local":
        return False
    if not is_outage_error(exc):
        return False
    return _client_can_reach_provider(resource)


def note_fallback(exc: BaseException) -> None:
    """Count + announce one outage fallback.

    Loud by design (``sdk-design-philosophy.mdc`` rule 5): the
    customer's call is being served, but it is being governed by the
    local cache instead of the Gateway, and the audit row for it now
    comes from the SDK. An operator must be able to find that in the
    logs without turning on debug logging.
    """
    global _fallback_total
    with _fallback_lock:
        _fallback_total += 1
        total = _fallback_total
    status = getattr(exc, "status_code", None)
    LOGGER.warning(
        "[egisai] gateway unreachable (%s%s) — governing this call "
        "in-process from the local policy cache instead "
        "(gateway_on_outage='local', fallbacks=%d)",
        exc.__class__.__name__,
        f" HTTP {status}" if isinstance(status, int) else "",
        total,
    )


def gateway_base_url() -> str:
    """The OpenAI-compatible base URL served by the platform."""
    cfg = get_config_optional()
    if cfg is None:  # pragma: no cover — callers check ``enabled()`` first
        raise RerouteUnavailable("egisai not initialized")
    return cfg.base_url.rstrip("/") + "/v1"


def _explicit_agent_name() -> str | None:
    """The explicit identity to ship as ``X-Egis-Agent``, if any.

    Precedence mirrors the local resolver's Tier 0: a pushed
    ``with egisai.agent(...)`` block wins over ``set_context``;
    auto-detection tiers are deliberately NOT consulted — the
    Gateway runs the same system-prompt fingerprint server-side, so
    shipping a locally-derived name would only duplicate it.
    """
    try:
        from egisai._auto_agent import current_identity

        pushed = current_identity()
        if pushed is not None and pushed.display_name:
            return str(pushed.display_name)
        from egisai._context import get_context

        name = get_context().agent_name
        return str(name) if name else None
    except Exception:  # noqa: BLE001
        return None


def _context_headers() -> dict[str, str]:
    """The ``set_context`` fields as wire-ready headers.

    Fail-open like every hook on the hot path: any surprise in the
    context read or encoding returns what was collected so far — a
    missing context header degrades the audit row's metadata, never
    the customer's call.
    """
    headers: dict[str, str] = {}
    try:
        from egisai._context import get_context

        ctx = get_context()
        for field, header, max_chars in _CONTEXT_HEADER_FIELDS:
            raw = getattr(ctx, field, None)
            if isinstance(raw, str):
                encoded = _encode_context_value(raw, max_chars)
                if encoded:
                    headers[header] = encoded
    except Exception:  # noqa: BLE001
        LOGGER.debug("could not collect context headers", exc_info=True)
    return headers


def inject_headers(kwargs: dict[str, Any], *, include_api_key: bool = True) -> None:
    """Merge the Egis headers into the call's ``extra_headers``.

    Carries the API key (optional), the explicit agent identity, and
    the ``set_context`` request-context fields (percent-encoded; see
    the module docstring for the wire contract).

    Caller-provided headers win on conflict — if the customer set
    their own ``X-Egis-Agent`` we assume they meant it.
    ``include_api_key=False`` skips the key header for clients that
    already carry their own (``egisai.Client`` sets it at
    construction; per-call injection must not clobber it).
    """
    cfg = get_config_optional()
    if cfg is None:  # pragma: no cover — callers check ``enabled()`` first
        raise RerouteUnavailable("egisai not initialized")
    headers: dict[str, str] = {}
    if include_api_key:
        headers[_API_KEY_HEADER] = cfg.api_key
    agent_name = _explicit_agent_name()
    if agent_name:
        headers[_AGENT_HEADER] = agent_name
    headers.update(_context_headers())
    existing = kwargs.get("extra_headers")
    if isinstance(existing, dict):
        headers.update({str(k): v for k, v in existing.items()})
    if headers:
        kwargs["extra_headers"] = headers


def _gateway_resource(resource: Any) -> Any:
    """A ``chat.completions`` resource whose client targets the gateway.

    Built via the client's own ``copy()`` so every transport option
    (timeout, proxies, retries) carries over; only ``base_url``
    changes. The customer's ``Authorization`` (their provider key)
    rides along untouched. When the caller already pointed their
    client at the gateway, the resource is reused as-is — no second
    hop, headers still injected.

    Raises :class:`RerouteUnavailable` for Azure-flavoured clients
    (deployment-based URLs are incompatible with the passthrough
    contract) and for any construction failure — the patch then
    falls back to the local-governance path.
    """
    try:
        client = resource._client  # noqa: SLF001 — upstream-stable attribute
        if "azure" in type(client).__name__.lower():
            raise RerouteUnavailable("Azure clients are not rerouted")
        if str(getattr(client, "base_url", "")).rstrip("/") == gateway_base_url():
            return resource
        gw_client = client.copy(base_url=gateway_base_url())
        return gw_client.chat.completions
    except RerouteUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RerouteUnavailable(str(exc)) from exc


def _client_carries_egis_key(resource: Any) -> bool:
    """True when the client already has ``X-Egis-Api-Key`` baked into
    its default headers (``egisai.Client`` / manual configuration)."""
    try:
        headers = resource._client.default_headers  # noqa: SLF001
        return _API_KEY_HEADER in headers
    except Exception:  # noqa: BLE001
        return False


def forward_chat(
    resource: Any, orig: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Reroute a sync ``chat.completions.create`` through the gateway.

    Calls ``orig`` (the pre-patch upstream method) directly on the
    gateway-pointed resource so the local gate never runs — the
    Gateway evaluates, enforces, and audits server-side.

    Config-less processes (an ``egisai.Client`` without ``init()``)
    pass straight through: the client already carries its keys and
    there is no context to inject.
    """
    if get_config_optional() is None:
        return orig(resource, *args, **kwargs)
    gw_resource = _gateway_resource(resource)
    call_kwargs = dict(kwargs)
    inject_headers(
        call_kwargs, include_api_key=not _client_carries_egis_key(gw_resource)
    )
    return orig(gw_resource, *args, **call_kwargs)


async def forward_chat_async(
    resource: Any, orig: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> Any:
    """Async sibling of :func:`forward_chat`."""
    if get_config_optional() is None:
        return await orig(resource, *args, **kwargs)
    gw_resource = _gateway_resource(resource)
    call_kwargs = dict(kwargs)
    inject_headers(
        call_kwargs, include_api_key=not _client_carries_egis_key(gw_resource)
    )
    return await orig(gw_resource, *args, **call_kwargs)
