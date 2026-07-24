"""Live policy refresher — keeps the local cache in sync.

Runs in a daemon thread. Prefers a server-sent-event stream and
falls back to ETag polling when SSE is disabled or disconnects.
"""

from __future__ import annotations

import logging
import threading
import time

import httpx

from egisai._backend import get_client
from egisai._config import get_config
from egisai._policy_cache import refresh_now

LOGGER = logging.getLogger("egisai.refresher")

_thread: threading.Thread | None = None
_stop_event = threading.Event()


def _poll_loop() -> None:
    cfg = get_config()
    while not _stop_event.is_set():
        try:
            refresh_now()
        except Exception:  # noqa: BLE001
            LOGGER.debug("policy poll failed", exc_info=True)
        if _stop_event.wait(timeout=cfg.refresh_interval_seconds):
            return


def _sse_listen_loop() -> None:
    """Stream policy-change events; reconnect with backoff on failure."""
    cfg = get_config()
    backoff = 1.0
    while not _stop_event.is_set():
        try:
            with get_client().stream(
                "GET", "/v1/sdk/stream", timeout=httpx.Timeout(60.0, read=None)
            ) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"SSE handshake failed: HTTP {r.status_code}")
                LOGGER.debug("egisai SSE connected")
                backoff = 1.0
                event_name = ""
                for line in r.iter_lines():
                    if _stop_event.is_set():
                        return
                    if not line:
                        event_name = ""
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        event_name = line[6:].strip()
                        continue
                    if line.startswith("data:"):
                        data = line[5:].strip()
                        _handle_sse(event_name, data)
        except Exception:  # noqa: BLE001
            LOGGER.debug("egisai SSE disconnected; retrying", exc_info=True)
        try:
            refresh_now()
        except Exception:  # noqa: BLE001
            pass
        if _stop_event.wait(timeout=backoff):
            return
        backoff = min(backoff * 2, 30.0)
        if _stop_event.wait(timeout=cfg.refresh_interval_seconds):
            return


# Event-name prefixes the SDK reacts to. Both ``policy.*`` (rule
# create / update / delete / toggle) and ``agent.*`` (operator
# pause / resume) bump the same shared cache via the ETag-aware
# ``/v1/sdk/policies`` round-trip — so one event filter covers
# both kinds of state mutation. Adding a new prefix here is the
# extension point for future "things the SDK should re-fetch on"
# (e.g. an org-wide ``settings.*`` topic).
_REFRESH_EVENT_PREFIXES: tuple[str, ...] = ("policy.", "agent.")


def _handle_sse(event_name: str, data: str) -> None:
    """React to one server-sent event.

    Both ``policy.*`` and ``agent.*`` events trigger a refresh;
    the data payload is treated as an opaque trigger so a forged
    event body cannot poison the SDK's cache. The actual
    snapshot — rules + paused-agent set — is pulled by
    ``refresh_now()`` so we always go through the cache-aware
    ETag path on the server (one no-op 304 round-trip on a
    duplicate event, no payload validation here).

    ``routing.*`` events (the Model Center master switch or a
    per-agent override flipping) drop the Smart Model Routing
    client's caches instead — the next governed call re-asks
    ``/v1/sdk/route``, whose authoritative answer reflects the
    flip. Same opaque-trigger posture: the event body is never
    trusted as state.
    """
    if event_name.startswith("routing."):
        try:
            from egisai import _routing

            _routing.invalidate()
        except Exception:  # noqa: BLE001
            LOGGER.debug("routing cache invalidation failed", exc_info=True)
        return
    if not any(event_name.startswith(p) for p in _REFRESH_EVENT_PREFIXES):
        return
    _ = data  # opaque trigger; refresh_now() handles cache validation
    try:
        refresh_now()
    except Exception:  # noqa: BLE001
        LOGGER.debug("refresh after SSE event failed", exc_info=True)


def start_worker() -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    cfg = get_config()
    _stop_event.clear()
    target = _sse_listen_loop if cfg.enable_sse else _poll_loop
    _thread = threading.Thread(target=target, name="egisai-refresh", daemon=True)
    _thread.start()


def stop_worker(timeout: float = 2.0) -> None:
    if _thread is None:
        return
    _stop_event.set()
    if _thread.is_alive():
        _thread.join(timeout=timeout)


def now() -> float:
    return time.monotonic()
