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
) -> tuple[str | None, list[dict] | None, list[str] | None]:
    """Pull the per-org policy + paused-agent snapshot.

    Returns ``(new_etag, rules, paused_agent_ids)``.

    * ``rules`` is ``None`` on a 304 (cache still fresh — the
      caller leaves its current rule list AND its current
      ``paused_agent_ids`` cache untouched).
    * On 200 ``rules`` is the freshly-fetched rule list and
      ``paused_agent_ids`` is the freshly-fetched set of paused
      agent UUIDs (lower-case canonical 8-4-4-4-12 form).
      Older backends that don't ship the field return an empty
      list — the SDK then treats the org as having no paused
      agents, which matches their pre-rollout behaviour.

    The triple-return wire shape is intentional: callers (the
    in-process ``_policy_cache``) want both pieces of state to
    update atomically, in lockstep with the same ETag, so a
    well-timed pause never lands inconsistently against a
    just-fetched rule set.
    """
    headers: dict[str, str] = {}
    if etag:
        headers["If-None-Match"] = etag
    r = _retry_on_429(
        "fetch_policies",
        lambda: get_client().get("/v1/sdk/policies", headers=headers),
    )
    if r.status_code == 304:
        return etag, None, None
    if r.status_code != 200:
        raise _http_error(op="fetch_policies", status=r.status_code)
    body = r.json()
    raw_paused = body.get("paused_agent_ids") or []
    paused: list[str] = [
        str(a).strip().lower() for a in raw_paused if a
    ]
    return body.get("etag"), body.get("rules", []), paused


def ensure_agent(
    *,
    name: str,
    description: str | None = None,
    runtime: dict[str, Any] | None = None,
    identity_hash: str | None = None,
    identity_source: str | None = None,
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
