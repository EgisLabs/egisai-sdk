"""Async, batched, fire-and-forget event logger.

Events are queued in memory and drained by a daemon thread on a
periodic + batch-size-driven schedule. Flushing never blocks the
request path; the queue is bounded so a platform outage can't grow
memory unboundedly.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Optional

from egisai._backend import post_events
from egisai._config import get_config

LOGGER = logging.getLogger("egisai.logger")

_QUEUE_MAX = 5000

_q: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=_QUEUE_MAX)
_thread: Optional[threading.Thread] = None
_stop_event = threading.Event()


def enqueue(event: dict[str, Any]) -> None:
    """Drop an event onto the flush queue. Never blocks the caller.

    Strips ``payload`` and any underscore-prefixed keys before
    queueing — only previewable, audit-safe fields ship.
    """
    safe = {
        k: v for k, v in event.items()
        if k != "payload" and not (isinstance(k, str) and k.startswith("_"))
    }
    try:
        _q.put_nowait(safe)
    except queue.Full:
        try:
            _q.get_nowait()
            _q.put_nowait(safe)
        except queue.Empty:
            pass


def _drain(max_items: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    while len(out) < max_items:
        try:
            out.append(_q.get_nowait())
        except queue.Empty:
            break
    return out


def _flush_loop() -> None:
    cfg = get_config()
    interval = cfg.flush_interval_seconds
    batch_size = cfg.flush_batch_size
    while not _stop_event.is_set():
        try:
            first = _q.get(timeout=interval)
        except queue.Empty:
            continue
        batch = [first] + _drain(max_items=batch_size - 1)
        try:
            post_events(batch)
        except Exception:  # noqa: BLE001
            LOGGER.warning("egisai flush worker error", exc_info=True)


def start_worker() -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(
        target=_flush_loop,
        name="egisai-flush",
        daemon=True,
    )
    _thread.start()


def stop_worker(timeout: float = 2.0) -> None:
    """Drain remaining events and stop. Safe to call multiple times."""
    if _thread is None:
        return
    try:
        remaining = _drain(max_items=_QUEUE_MAX)
        if remaining:
            post_events(remaining)
    except Exception:  # noqa: BLE001
        pass
    _stop_event.set()
    if _thread.is_alive():
        _thread.join(timeout=timeout)


def queue_size() -> int:
    return _q.qsize()


def now_monotonic() -> float:
    return time.monotonic()
