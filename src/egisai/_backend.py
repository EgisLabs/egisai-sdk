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


def handshake(*, app: str, env: str, sdk_version: str) -> dict[str, Any]:
    r = _retry_on_429(
        "handshake",
        lambda: get_client().post(
            "/v1/sdk/handshake",
            json={"app": app, "env": env, "sdk_version": sdk_version},
        ),
    )
    if r.status_code != 200:
        raise _http_error(op="handshake", status=r.status_code)
    return r.json()


def fetch_policies(etag: str | None = None) -> tuple[str | None, list[dict] | None]:
    """Returns ``(new_etag, rules)``. ``rules`` is ``None`` on 304."""
    headers: dict[str, str] = {}
    if etag:
        headers["If-None-Match"] = etag
    r = _retry_on_429(
        "fetch_policies",
        lambda: get_client().get("/v1/sdk/policies", headers=headers),
    )
    if r.status_code == 304:
        return etag, None
    if r.status_code != 200:
        raise _http_error(op="fetch_policies", status=r.status_code)
    body = r.json()
    return body.get("etag"), body.get("rules", [])


def ensure_agent(
    *,
    name: str,
    description: str | None = None,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Find-or-create an agent in the caller's org by name. Idempotent.

    ``runtime`` (added in 0.13.0) is the platform-side fingerprint
    blob produced by :func:`egisai._runtime.collect_runtime_fingerprint`.
    The backend stamps it onto the agent's Provenance card and uses
    deltas to spot ``runtime_change`` anomalies. Sending it is
    optional; older backends ignore unknown keys.
    """
    payload: dict[str, Any] = {"name": name}
    if description:
        payload["description"] = description
    if runtime:
        payload["runtime"] = runtime
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
