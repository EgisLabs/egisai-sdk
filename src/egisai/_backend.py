"""HTTP client for the platform's SDK endpoints.

Single shared ``httpx.Client`` used by all background workers so
handshake / refresh / flush share the same TCP keep-alive connection.

Error messages carry the operation name and HTTP status only —
upstream response bodies are never echoed.
"""

from __future__ import annotations

import logging
import os
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

# Ceiling on the ``Retry-After`` value we are willing to honour, in
# seconds. The header is operator-controlled on our side, but the
# retry ``time.sleep`` happens wherever the caller runs — and some
# callers (``ensure_agent``) sit on the customer's model-call hot
# path. Without a cap, a backend answering ``Retry-After: 300``
# would stall a customer's ``client.chat.completions.create(...)``
# for five minutes per attempt, which is a far worse outcome than
# skipping the operation and retrying on the next call.
#
# ``egisai.policy.semantic`` has carried the same clamp for its
# judge round-trip since 0.11; this is the same idea applied to the
# shared client. Tunable so an operator running a self-hosted
# backend with long maintenance windows can widen it.
RETRY_AFTER_CAP_S = 5.0
RETRY_AFTER_CAP_ENV = "EGISAI_RETRY_AFTER_MAX_SECS"

# Wall-clock budget for ``ensure_agent``. Deliberately much tighter
# than the shared ``timeout_seconds`` (10 s) because this is the one
# backend round-trip that runs *inline* on the customer's model call,
# the first time each agent identity is seen. A slow or black-holed
# backend must cost the call a couple of seconds at most; the SDK
# then proceeds unattributed and retries after the caller-side
# backoff (see ``egisai._auto_agent._ensure_agent_id``).
ENSURE_AGENT_TIMEOUT_S = 2.0
ENSURE_AGENT_TIMEOUT_ENV = "EGISAI_AGENT_ENSURE_TIMEOUT_SECS"


def _env_float(name: str, default: float, *, lo: float) -> float:
    """Read a float env var, clamped at ``lo``. Never raises."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return max(lo, float(raw.strip()))
    except ValueError:
        LOGGER.debug("ignoring unparseable %s=%r", name, raw)
        return default


def retry_after_cap_secs() -> float:
    """Current ceiling for honoured ``Retry-After`` values."""
    return _env_float(RETRY_AFTER_CAP_ENV, RETRY_AFTER_CAP_S, lo=0.1)


def ensure_agent_timeout_secs() -> float:
    """Current wall-clock budget for the inline ``ensure_agent`` hop."""
    return _env_float(ENSURE_AGENT_TIMEOUT_ENV, ENSURE_AGENT_TIMEOUT_S, lo=0.1)


class BackendError(Exception):
    """Backend returned an error or could not be reached."""


T = TypeVar("T")


def _retry_on_429(
    op: str,
    fn: Callable[[], httpx.Response],
    *,
    max_attempts: int = RETRY_429_MAX,
) -> httpx.Response:
    """Execute ``fn`` and transparently retry on HTTP 429.

    Honours ``Retry-After`` (delta-seconds), clamped to
    :func:`retry_after_cap_secs` so a large server-supplied value can
    never stall the calling thread for minutes, and falls back to a
    constant sleep when the header is absent or unparseable.

    ``max_attempts=0`` disables retrying entirely — the 429 is handed
    straight back. Hot-path callers use that: a rate-limited
    best-effort operation is better skipped than slept through.
    """
    cap = retry_after_cap_secs()
    last: httpx.Response | None = None
    for attempt in range(max_attempts + 1):
        last = fn()
        if last.status_code != 429:
            return last
        if attempt >= max_attempts:
            break
        retry_after_raw = last.headers.get("Retry-After")
        delay = RETRY_429_FALLBACK_SLEEP_S
        if retry_after_raw:
            try:
                delay = max(0.1, float(retry_after_raw))
            except ValueError:
                pass
        delay = min(delay, cap)
        LOGGER.info(
            "%s rate-limited (HTTP 429) — retrying in %.1fs (attempt %d/%d)",
            op, delay, attempt + 1, max_attempts,
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


def fetch_usage() -> dict[str, Any] | None:
    """Pull the per-org usage snapshot for ``rate_limit`` / ``budget_limit``.

    Returns the parsed JSON payload, or ``None`` on any non-200 —
    including 404 from backends that pre-date ``/v1/sdk/usage``.
    Callers (the ``egisai.policy.limits`` sync worker) treat ``None``
    as "keep the previous snapshot" so limit enforcement degrades to
    local-only counting instead of erroring. No 429-retry wrapper:
    the worker retries on its own schedule anyway, and a retry storm
    from many SDK processes would defeat the endpoint's purpose.
    """
    try:
        r = get_client().get("/v1/sdk/usage")
    except Exception:  # noqa: BLE001
        return None
    if r.status_code != 200:
        return None
    try:
        body = r.json()
    except Exception:  # noqa: BLE001
        return None
    return body if isinstance(body, dict) else None


def ensure_agent(
    *,
    name: str,
    description: str | None = None,
    runtime: dict[str, Any] | None = None,
    identity_hash: str | None = None,
    identity_source: str | None = None,
    system_prompt_excerpt: str | None = None,
    identity_version: str | None = None,
    identity_hash_legacy: str | None = None,
    identity_simhash: str | None = None,
    tool_bundle_hash: str | None = None,
    model: str | None = None,
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

    Identity v2 fields (added in 0.44.0; older backends ignore them):

    * ``identity_version`` — hash-recipe version ("2") so the backend
      knows how ``identity_hash`` was computed.
    * ``identity_hash_legacy`` — the v1 hash for the same identity;
      lets the backend re-stamp a pre-v2 agent row in place instead
      of forking a duplicate on SDK upgrade.
    * ``identity_simhash`` — 16-hex 64-bit SimHash of the canonical
      prompt (non-reversible; privacy contract unchanged).
    * ``tool_bundle_hash`` — SHA-256 of the sorted tool-name set;
      corroborates prompt-evolution reconciliation server-side.
    * ``model`` — the model id observed on the registering call.
      Observed *metadata* only (models_seen histogram) — never part
      of any identity hash.

    Availability contract: this is the only backend hop that runs
    inline on the customer's model call, so it gets its own tight
    timeout (:func:`ensure_agent_timeout_secs`) and no 429 retry
    loop. A rate-limited or unreachable backend surfaces as a
    ``BackendError`` immediately; the caller treats registration as
    unavailable, proceeds unattributed, and backs off before trying
    again.
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
    if identity_version:
        payload["identity_version"] = identity_version
    if identity_hash_legacy:
        payload["identity_hash_legacy"] = identity_hash_legacy
    if identity_simhash:
        payload["identity_simhash"] = identity_simhash
    if tool_bundle_hash:
        payload["tool_bundle_hash"] = tool_bundle_hash
    if model:
        payload["model"] = model
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
        lambda: get_client().post(
            "/v1/sdk/agents/ensure",
            json=payload,
            timeout=ensure_agent_timeout_secs(),
        ),
        max_attempts=0,
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


def route(
    *,
    model: str,
    prompt_preview: str,
    prompt_chars: int,
    has_tools: bool,
    available_providers: list[str],
    agent_id: str | None,
    has_images: bool = False,
    max_output_tokens: int = 0,
    uses_prompt_caching: bool = False,
    timeout_s: float = 3.0,
) -> dict[str, Any] | None:
    """Ask the platform for a Smart Model Routing decision. Never raises.

    Hot-path call — no 429 retry loop (a rate-limited decision is just
    skipped), short explicit timeout, and every failure mode returns
    ``None`` so the caller keeps the requested model (fail-open).
    ``prompt_preview`` MUST be the post-sanitization, label-redacted
    audit preview — never raw text.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt_preview": prompt_preview,
        "prompt_chars": prompt_chars,
        "has_tools": has_tools,
        "has_images": has_images,
        "max_output_tokens": max_output_tokens,
        "uses_prompt_caching": uses_prompt_caching,
        "available_providers": available_providers,
    }
    if agent_id:
        payload["agent_id"] = agent_id
    try:
        r = get_client().post("/v1/sdk/route", json=payload, timeout=timeout_s)
        if r.status_code != 200:
            return None
        body = r.json()
        return body if isinstance(body, dict) else None
    except Exception:  # noqa: BLE001
        LOGGER.debug("route decision request failed", exc_info=True)
        return None


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
