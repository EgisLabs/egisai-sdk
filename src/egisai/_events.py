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
    """Bounded, label-redacted preview for audit logs.

    Privacy contract (``security-and-compliance.mdc`` §1, §5):
    this preview is shipped to the audit log on every framework
    patch and on every code path — allow, sanitize, AND **block**
    (the block path doesn't re-set ``payload_preview`` after build
    time, so what we produce here is what auditors see).
    Therefore we MUST push the rendered string through
    ``label_redact`` BEFORE truncating, so a payload that carries
    a raw SSN / credit card / IBAN never leaves the SDK boundary
    even if a policy didn't flag it (e.g. an allow-path call with
    incidentally PII-bearing text, or an input-side block whose
    payload preview was set at build time).

    ``label_redact`` is regex+checksum-only (no Presidio dependency
    at this seam, so we can't accidentally trigger the analyzer's
    cold-start) and replaces SSN / credit card / IBAN / email /
    phone / API-key matches with the ``<TYPE>`` token. That is the
    same redaction the dashboard renders next to verdict pills, so
    a single preview field reads identically pre- and post-shipping.
    """
    try:
        s = repr(payload)
    except Exception:
        s = "<unrepr-able>"
    # Local import — ``egisai.policy`` is the heavier module that
    # imports the rule engine + PII scanners. We deliberately do
    # NOT pull that into this file's top-level import graph so
    # circular-import risk stays zero (``_events.py`` is imported
    # very early during ``init()``, before ``policy`` is wired).
    try:
        from egisai.policy import label_redact

        s = label_redact(s)
    except Exception:  # noqa: BLE001
        # Fail-open: if redaction errored for any reason, fall
        # back to the raw repr — but truncate to ``max_len`` so
        # an inadvertent leak is still bounded in size. We don't
        # WIDEN the leak surface by raising here.
        pass
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
