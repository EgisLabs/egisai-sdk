"""HTTP client for the platform's SDK endpoints.

Single shared ``httpx.Client`` used by all background workers so
handshake / refresh / flush share the same TCP keep-alive connection.

Error messages carry the operation name and HTTP status only —
upstream response bodies are never echoed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx

from egisai._config import get_config

LOGGER = logging.getLogger("egisai.backend")

# Auto-retry budget for HTTP 429. After this many attempts the
# 429 is surfaced to the caller.
RETRY_429_MAX = 3
RETRY_429_FALLBACK_SLEEP_S = 1.0


class BackendError(Exception):
    """Backend returned an error or could not be reached."""


T = TypeVar("T")


def _retry_on_429(
    op: str,
    fn: Callable[[], httpx.Response],
) -> httpx.Response:
    """Execute ``fn`` and transparently retry on HTTP 429.

    Honours ``Retry-After`` (delta-seconds) and falls back to a
    constant sleep otherwise.
    """
    last: httpx.Response | None = None
    for attempt in range(RETRY_429_MAX + 1):
        last = fn()
        if last.status_code != 429:
            return last
        if attempt >= RETRY_429_MAX:
            break
        retry_after_raw = last.headers.get("Retry-After")
        delay = RETRY_429_FALLBACK_SLEEP_S
        if retry_after_raw:
            try:
                delay = max(0.1, float(retry_after_raw))
            except ValueError:
                pass
        LOGGER.info(
            "%s rate-limited (HTTP 429) — retrying in %.1fs (attempt %d/%d)",
            op, delay, attempt + 1, RETRY_429_MAX,
        )
        time.sleep(delay)
    return last  # type: ignore[return-value]


_client: httpx.Client | None = None


def get_client() -> httpx.Client:
    global _client
    if _client is None:
        cfg = get_config()
        _client = httpx.Client(
            base_url=cfg.base_url.rstrip("/"),
            timeout=cfg.timeout_seconds,
            headers={
                "Authorization": f"Bearer {cfg.api_key}",
                "User-Agent": f"egisai-sdk/{cfg.sdk_version}",
            },
        )
    return _client


def close_client() -> None:
    global _client
    if _client is not None:
        try:
            _client.close()
        finally:
            _client = None


def _http_error(*, op: str, status: int) -> BackendError:
    """Build a ``BackendError`` with operation name + HTTP status only."""
    return BackendError(f"{op} failed (HTTP {status})")


def handshake(
    *,
    app: str,
    env: str,
    sdk_version: str,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the API key and (optionally) stamp runtime.

    ``runtime`` (added in 0.13.1) is the same platform-side fingerprint
    blob shipped by :func:`ensure_agent`. Sending it on handshake
    populates the Provenance card for an API-key-bound agent on first
    contact, without waiting for a sub-agent ``set_context`` call.
    Older backends ignore the field.
    """
    payload: dict[str, Any] = {
        "app": app,
        "env": env,
        "sdk_version": sdk_version,
    }
    if runtime:
        payload["runtime"] = runtime
    r = _retry_on_429(
        "handshake",
        lambda: get_client().post("/v1/sdk/handshake", json=payload),
    )
    if r.status_code != 200:
        raise _http_error(op="handshake", status=r.status_code)
    return r.json()


def fetch_policies(
    etag: str | None = None,
) -> tuple[str | None, list[dict] | None, list[str] | None, list[str] | None]:
    """Pull the per-org policy + paused/ungoverned-agent snapshot.

    Returns ``(new_etag, rules, paused_agent_ids, ungoverned_agent_ids)``.

    * ``rules`` is ``None`` on a 304 (cache still fresh — the
      caller leaves its current rule list AND its current
      paused / ungoverned agent-set caches untouched).
    * On 200 ``rules`` is the freshly-fetched rule list,
      ``paused_agent_ids`` is the freshly-fetched set of paused
      agent UUIDs, and ``ungoverned_agent_ids`` is the set of
      agents whose policy enforcement an operator turned off
      (monitor-only mode). All UUIDs are lower-case canonical
      8-4-4-4-12 form. Older backends that don't ship a field
      return an empty list — the SDK then treats the org as
      having no paused / no ungoverned agents, which matches
      their pre-rollout Behavior (and is the safe, enforcing
      direction for the ungoverned set).

    The tuple-return wire shape is intentional: callers (the
    in-process ``_policy_cache``) want every piece of state to
    update atomically, in lockstep with the same ETag, so a
    well-timed pause / ungovern never lands inconsistently
    against a just-fetched rule set.
    """
    headers: dict[str, str] = {}
    if etag:
        headers["If-None-Match"] = etag
    r = _retry_on_429(
        "fetch_policies",
        lambda: get_client().get("/v1/sdk/policies", headers=headers),
    )
    if r.status_code == 304:
        return etag, None, None, None
    if r.status_code != 200:
        raise _http_error(op="fetch_policies", status=r.status_code)
    body = r.json()
    raw_paused = body.get("paused_agent_ids") or []
    paused: list[str] = [
        str(a).strip().lower() for a in raw_paused if a
    ]
    raw_ungoverned = body.get("ungoverned_agent_ids") or []
    ungoverned: list[str] = [
        str(a).strip().lower() for a in raw_ungoverned if a
    ]
    return body.get("etag"), body.get("rules", []), paused, ungoverned


def ensure_agent(
    *,
    name: str,
    description: str | None = None,
    runtime: dict[str, Any] | None = None,
    identity_hash: str | None = None,
    identity_source: str | None = None,
    system_prompt_excerpt: str | None = None,
) -> dict[str, Any]:
    """Find-or-create an agent in the caller's org by name. Idempotent.

    ``runtime`` (added in 0.13.0) is the platform-side fingerprint
    blob produced by :func:`egisai._runtime.collect_runtime_fingerprint`.
    The backend stamps it onto the agent's Provenance card and uses
    deltas to spot ``runtime_change`` anomalies. Sending it is
    optional; older backends ignore unknown keys.

    ``identity_hash`` + ``identity_source`` (added in 0.17.0) are the
    SDK-computed composite-fingerprint hash + provenance tag from
    :mod:`egisai._auto_agent`. The backend dedups by
    ``(org_id, identity_hash)`` first, then falls back to
    ``(org_id, name_normalized)`` for legacy SDKs. Backends < 0.36
    ignore both fields silently.

    ``system_prompt_excerpt`` is a PII-sanitised, truncated excerpt of
    the agent's system prompt (already scrubbed by the SDK's PII
    engine — see :func:`egisai._auto_agent._sanitized_excerpt`). When
    present, the backend uses it transiently to generate a human
    description + business function in the background; it is never
    persisted or logged server-side. Omitted when ``auto_describe`` is
    off or the agent has no system prompt. Older backends ignore it.
    """
    payload: dict[str, Any] = {"name": name}
    if description:
        payload["description"] = description
    if runtime:
        payload["runtime"] = runtime
    if identity_hash:
        payload["identity_hash"] = identity_hash
    if identity_source:
        payload["identity_source"] = identity_source
    if system_prompt_excerpt:
        payload["system_prompt_excerpt"] = system_prompt_excerpt
    # DEBUG breadcrumb so a developer staring at an empty Provenance
    # card on the dashboard can confirm "yes, the SDK actually shipped
    # the fingerprint" without reaching for tcpdump. Off by default;
    # opt-in via the standard logging config (set ``egisai.backend``
    # to DEBUG) or the ``EGISAI_DEBUG=1`` env var honoured elsewhere
    # in the SDK.
    if LOGGER.isEnabledFor(logging.DEBUG):
        rt_keys = sorted(runtime.keys()) if runtime else []
        LOGGER.debug(
            "ensure_agent name=%r description=%s runtime_keys=%s",
            name,
            "set" if description else "none",
            rt_keys,
        )
    r = _retry_on_429(
        "ensure_agent",
        lambda: get_client().post("/v1/sdk/agents/ensure", json=payload),
    )
    if r.status_code not in (200, 201):
        raise _http_error(op="ensure_agent", status=r.status_code)
    return r.json()


def ensure_mcp_server(
    *,
    name: str,
    description: str | None = None,
    transport: str | None = None,
    server_url: str | None = None,
    identity_hash: str | None = None,
    identity_source: str | None = None,
    runtime: dict[str, Any] | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Find-or-create an MCP server in the caller's org by name/identity.

    Part of the MCP Servers add-on. Mirrors :func:`ensure_agent`: the
    backend dedups by ``(org_id, identity_hash)`` first, then by
    ``(org_id, name_normalized)``. ``tools`` is the inventory the SDK
    discovered (each ``{name, description, schema_hash}``); the backend
    upserts ``mcp_server_tools`` rows from it. Older/non-add-on
    backends 404/400 this route — the caller treats any non-2xx as
    "registration unavailable" and fails open.
    """
    payload: dict[str, Any] = {"name": name}
    if description:
        payload["description"] = description
    if transport:
        payload["transport"] = transport
    if server_url:
        payload["server_url"] = server_url
    if identity_hash:
        payload["identity_hash"] = identity_hash
    if identity_source:
        payload["identity_source"] = identity_source
    if runtime:
        payload["runtime"] = runtime
    if tools:
        payload["tools"] = tools
    r = _retry_on_429(
        "ensure_mcp_server",
        lambda: get_client().post("/v1/sdk/mcp-servers/ensure", json=payload),
    )
    if r.status_code not in (200, 201):
        raise _http_error(op="ensure_mcp_server", status=r.status_code)
    return r.json()


def report_agent_access(
    *,
    agent_id: str,
    items: list[dict[str, Any]],
    bundle_hash: str,
) -> None:
    """Ship an agent's declared access inventory. Fire-and-forget.

    Backs the dashboard's per-agent "Access" tab. ``items`` is the
    metadata-only bundle built by :func:`egisai._access.extract_access_items`
    (tool names, PII-sanitized descriptions, schema hashes, parameter
    names — never schemas or arguments). ``bundle_hash`` lets the
    backend skip a no-op sync cheaply. Older backends 404 this route;
    the caller treats any non-2xx as "reporting unavailable" and
    fails open.
    """
    r = _retry_on_429(
        "report_agent_access",
        lambda: get_client().post(
            "/v1/sdk/agents/access",
            json={
                "agent_id": agent_id,
                "bundle_hash": bundle_hash,
                "items": items,
            },
        ),
    )
    if r.status_code not in (200, 201):
        raise _http_error(op="report_agent_access", status=r.status_code)


def post_events(events: list[dict[str, Any]]) -> None:
    if not events:
        return
    try:
        r = _retry_on_429(
            "post_events",
            lambda: get_client().post("/v1/sdk/events", json={"events": events}),
        )
        if r.status_code >= 400:
            LOGGER.warning(
                "egisai event flush failed: HTTP %s (batch_size=%d)",
                r.status_code,
                len(events),
            )
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning(
            "egisai event flush errored: %s",
            exc.__class__.__name__,
        )


# ── SDK health telemetry ──────────────────────────────────────────
#
# One-shot fire-and-forget POST used when the SDK detects an init-
# time problem the operator should know about — e.g. the PII NER
# analyzer fails to load because of a missing transitive dep. The
# SDK still falls open into its regex-fallback path and the user's
# call site keeps working; this hop just surfaces the warning on
# the operator's dashboard so they find out *before* the next
# customer pings them about it.
#
# Privacy-side contract (per security-and-compliance.mdc):
#
#  - Payload carries operator-controlled diagnostic data only:
#    a machine-readable ``code`` (e.g. ``pii_ner_loader_failed``),
#    the exception class name, a sanitized one-line error message,
#    and platform fingerprint bits (SDK version, Python version,
#    OS family).
#  - The error message is **scrubbed** for obvious filesystem
#    paths (``/Users/<name>/`` → ``/Users/<redacted>/``) and
#    truncated before transmission. We never ship the exception's
#    traceback, locals, or repr of any in-process object.
#  - No prompt text, response text, API key, agent name, agent ID,
#    or customer-identifying value ever reaches this endpoint. The
#    payload is, by design, the same shape we'd be comfortable
#    surfacing on a public status page.
#
# Reliability contract:
#
#  - Fire-and-forget: catches every exception and never raises.
#    A backend outage MUST NOT delay ``egisai.init()`` or break
#    the user's first model call.
#  - No retries: the warning fires once per process per code.
#    Re-emitting on every restart would inflate dashboard counts
#    and bury new signals under repeats.
#  - Short timeout (3 s) so a slow / unreachable backend can't
#    stall the PII loader's daemon thread for the default 10 s.


def _sanitize_telemetry_string(raw: str, *, max_chars: int = 256) -> str:
    """Scrub obvious filesystem paths and truncate.

    The exception message comes from upstream code (spaCy, Presidio,
    pip, …) and 99% of the time it's a short class-of-error string
    like ``"No module named 'click'"``. The remaining 1% — file-
    backed errors — can legitimately embed ``/Users/<operator>/…``
    or ``/home/<operator>/…`` paths that we treat as PII for the
    purposes of telemetry. A small regex scrub keeps the operator's
    home-dir layout off our dashboards without losing the signal of
    *which* file class blew up. Truncation caps the field for the
    database column and prevents a tracebackish dump from clogging
    the UI.
    """
    import re

    s = re.sub(r"(/Users/|/home/)[^/\s'\"]+", r"\1<redacted>", raw)
    s = re.sub(
        r"([Cc]:[\\/]Users[\\/])[^\\/\s'\"]+",
        r"\1<redacted>",
        s,
    )
    return s[:max_chars]


def post_startup_warning(code: str, exc: BaseException) -> None:
    """Best-effort POST to surface an SDK init-time warning on the dashboard.

    ``code`` is a stable machine identifier (e.g.
    ``"pii_ner_loader_failed"``); ``exc`` is the exception the
    caller already logged. Both are encoded into a tiny JSON blob,
    POSTed to ``/v1/sdk/telemetry/startup-warning``, and forgotten.
    Every failure mode (no client, no network, 4xx, 5xx, slow
    backend, malformed exception) is swallowed — the function never
    raises.
    """
    try:
        from egisai._config import get_config_optional
        from egisai._runtime import collect_runtime_fingerprint

        cfg = get_config_optional()
        if cfg is None:
            return
        rt = collect_runtime_fingerprint(sdk_version=cfg.sdk_version)
        payload: dict[str, Any] = {
            "code": code,
            "error_class": exc.__class__.__name__,
            "error_message": _sanitize_telemetry_string(str(exc)),
            "sdk_version": cfg.sdk_version,
            "python_version": rt.get("python"),
            "os": rt.get("os"),
        }
        r = get_client().post(
            "/v1/sdk/telemetry/startup-warning",
            json=payload,
            timeout=3.0,
        )
        if r.status_code >= 400 and LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug("startup-warning POST got HTTP %s", r.status_code)
    except Exception as exc2:  # noqa: BLE001
        if LOGGER.isEnabledFor(logging.DEBUG):
            LOGGER.debug(
                "startup-warning POST errored: %s",
                exc2.__class__.__name__,
            )
