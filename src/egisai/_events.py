"""Event construction — the canonical shape sent to the audit log."""

from __future__ import annotations

import time
import uuid
from typing import Any

from egisai._config import get_config
from egisai._context import ensure_trace_id, get_context


def now_iso() -> str:
    """UTC, ISO 8601 with seconds precision."""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def safe_preview(payload: object, max_len: int = 280) -> str:
    """Bounded human-friendly preview for audit logs."""
    try:
        s = repr(payload)
    except Exception:
        s = "<unrepr-able>"
    if len(s) > max_len:
        return s[: max_len - 3] + "..."
    return s


def build_event(
    *,
    source: str,
    target: str,
    payload: Any,
    model: str | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    """Construct a base event for governance + logging.

    `source` is the integration name (openai/anthropic/genai/httpx/...).
    `target` is the conceptual operation (e.g. "openai.chat.completions.create").
    """
    cfg = get_config()
    ctx = get_context()
    trace = ensure_trace_id()
    return {
        "event_id": uuid.uuid4().hex,
        "trace_id": trace,
        "timestamp": now_iso(),
        "app": ctx.agent_name or cfg.app,
        "env": cfg.env,
        "org_id": cfg.org_id,
        "agent_id": ctx.agent_id or cfg.agent_id,
        "user_id": ctx.user_id,
        "user_role": ctx.user_role,
        "session_id": ctx.session_id,
        "workflow_id": ctx.workflow_id,
        # Opaque end-user id used by the platform for per-end-user
        # behavioral roll-ups. The wire shape is post-hash where
        # callers follow the docs; the backend hashes again on
        # intake so a paste of a real customer-id never persists raw.
        "end_user_id": ctx.end_user_id,
        "source": source,
        "target": target,
        "model": model,
        "stream": stream,
        "payload": payload,
        "payload_preview": safe_preview(payload),
    }
