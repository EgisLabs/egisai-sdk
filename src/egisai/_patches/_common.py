"""Shared gate logic used by every framework patch.

For each governed call:

1. Build an audit event from the captured kwargs.
2. Auto-detect the agent identity from the system prompt
   (``set_context(agent=…)`` wins).
3. Evaluate the cached policies.
4. Allow → forward to the original function, time it, capture token
   usage. Sanitize → mask in place, then forward. Block → raise
   ``PermissionError`` or return the framework-specific stub.
5. Enqueue the event for async flushing.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from egisai._auto_agent import derive_identity, resolve_agent_id
from egisai._config import get_config
from egisai._context import (
    ensure_trace_id,
    get_context,
    get_policy_checked,
    get_policy_usage,
    get_source,
    reset_policy_usage,
    reset_trace,
    set_policy_checked,
    set_source,
)
from egisai._evaluator import (
    InputCall,
    evaluate,
    extract_payload_text,
    mutate_prompt_text,
)
from egisai._events import build_event, safe_preview
from egisai._logger import enqueue
from egisai.policy import MatchedPolicyRecord, PolicyDecision, label_redact


def _serialize_matched_policies(
    decision: PolicyDecision,
) -> list[dict[str, Any]]:
    """Convert ``decision.matched_policies`` to JSON-friendly dicts."""
    out: list[dict[str, Any]] = []
    for r in decision.matched_policies:
        out.append(
            {
                "name": r.name,
                "type": r.type,
                "verdict": r.verdict,
                "reason_code": r.reason_code,
                "message": r.message,
                "sanitize_kinds": list(r.sanitize_kinds),
                "sanitize_mask_char": r.sanitize_mask_char,
            }
        )
    if not out and decision.matched_policy and decision.verdict in (
        "block",
        "sanitize",
    ):
        out.append(
            {
                "name": decision.matched_policy,
                "type": "",
                "verdict": decision.verdict,
                "reason_code": decision.reason_code or "",
                "message": decision.message or "",
                "sanitize_kinds": list(decision.sanitize_kinds),
                "sanitize_mask_char": decision.sanitize_mask_char,
            }
        )
    return out
from egisai.policy.pii import Sanitization, sanitize as pii_sanitize

_MAX_PREVIEW_LEN = 2048


def _safe_text_preview(text: str | None) -> str | None:
    """Label-redact + truncate.

    Any string returned by this function is safe to persist in the
    audit log even if it originally contained raw PII.
    """
    if not text:
        return text
    redacted = label_redact(text)
    if len(redacted) <= _MAX_PREVIEW_LEN:
        return redacted
    return redacted[: _MAX_PREVIEW_LEN - 3] + "..."

LOGGER = logging.getLogger("egisai.patches")

# A function that, given the framework-specific response object, returns
# a small dict with ``tokens_in``, ``tokens_out``, and (optionally)
# ``cost_usd``. Returning an empty dict is fine — the audit row just
# stays with zeros for that call.
ExtractUsage = Callable[[Any], dict[str, Any]]


def _stamp_usage(ev: dict, response: Any, extract_usage: Optional[ExtractUsage]) -> None:
    """Copy ``tokens_in`` / ``tokens_out`` / ``cost_usd`` onto the event."""
    if extract_usage is None or response is None:
        return
    try:
        usage = extract_usage(response) or {}
    except Exception:  # noqa: BLE001
        LOGGER.debug("usage extractor failed for %s", ev.get("source"), exc_info=True)
        return
    if "tokens_in" in usage and usage["tokens_in"] is not None:
        ev["tokens_in"] = int(usage["tokens_in"])
    if "tokens_out" in usage and usage["tokens_out"] is not None:
        ev["tokens_out"] = int(usage["tokens_out"])
    if usage.get("cost_usd") is not None:
        ev["cost_usd"] = float(usage["cost_usd"])


def _apply_sanitization(
    *, decision: PolicyDecision, payload: Any, ev: dict
) -> None:
    """Mask PII in ``payload`` and stamp the audit event with what we did.

    Mutates the payload in place so the upstream framework
    serializes the masked text. The audit event records the count
    and mask shape per kind, never the original value.
    """
    aggregated: dict[str, Sanitization] = {}

    def _transform(text: str) -> str:
        new_text, records = pii_sanitize(
            text,
            kinds=decision.sanitize_kinds or None,
            mask_char=decision.sanitize_mask_char,
        )
        for rec in records:
            existing = aggregated.get(rec.kind)
            if existing is None:
                aggregated[rec.kind] = rec
            else:
                aggregated[rec.kind] = Sanitization(
                    kind=rec.kind,
                    count=existing.count + rec.count,
                    pattern=existing.pattern,
                )
        return new_text

    before_text = ev.get("_prompt_text_original") or extract_payload_text(payload)
    ev["prompt_preview_before"] = _safe_text_preview(before_text)

    mutate_prompt_text(payload, _transform)

    ev["payload_preview"] = safe_preview(payload)

    after_text = extract_payload_text(payload)
    ev["prompt_preview"] = _safe_text_preview(after_text)

    ev["sanitizations"] = [
        {"kind": s.kind, "count": s.count, "pattern": s.pattern}
        for s in aggregated.values()
    ]


def _attribute_event(ev: dict, payload: Any) -> None:
    """Attribute the event to the right agent identity.

    Resolution order (first match wins):

    1. ``set_context(agent="…")`` (already on ``ctx.agent_id``).
    2. System-prompt fingerprint — auto-register a sub-agent.
    3. Lazy registration of the init-time ``app`` name.
    """
    ctx = get_context()
    if ctx.agent_id:
        return

    messages = payload.get("messages") if isinstance(payload, dict) else None
    identity = derive_identity(payload, messages)
    if identity is not None:
        identity_hash, display_name = identity
        agent_id = resolve_agent_id(identity_hash, display_name)
        if agent_id:
            ev["agent_id"] = agent_id
            ev["app"] = display_name
        return

    cfg = get_config()
    if cfg.agent_id:
        return
    if not cfg.app:
        return
    synthetic_hash = "__app__:" + cfg.app
    agent_id = resolve_agent_id(synthetic_hash, cfg.app)
    if agent_id:
        ev["agent_id"] = agent_id
        ev["app"] = cfg.app


def gate_call(
    *,
    source: str,
    target: str,
    model: str,
    prompt_text: str,
    stream: bool,
    payload: Any,
    stub_factory: Optional[Callable[[PolicyDecision, str, str], Any]] = None,
    extract_usage: Optional[ExtractUsage] = None,
    forward: Callable[[], Any],
) -> Any:
    """Run the gate around a single model call.

    ``forward`` is a zero-arg callable invoking the original
    function; it runs only on allow / sanitize verdicts. Events are
    enqueued after ``forward()`` returns for allowed calls (so
    latency + tokens are populated), or immediately on block.
    """
    cfg = get_config()
    prev_source = get_source()
    prev_checked = get_policy_checked()

    if not prev_source:
        reset_trace()
    set_source(source)

    try:
        if prev_checked:
            return forward()

        ev = build_event(
            source=source, target=target, payload=payload, model=model, stream=stream
        )
        _attribute_event(ev, payload)
        ev["prompt_chars"] = len(prompt_text or "")
        ev["prompt_preview"] = _safe_text_preview(prompt_text)
        ev["_prompt_text_original"] = prompt_text

        set_policy_checked(True)
        reset_policy_usage()
        try:
            policy_started = time.monotonic()
            decision = evaluate(
                InputCall(
                    source=source,
                    target=target,
                    model=model,
                    prompt_text=prompt_text,
                    stream=stream,
                )
            )
            ev["policy_latency_ms"] = int((time.monotonic() - policy_started) * 1000)
            policy_in, policy_out = get_policy_usage()
            ev["policy_tokens_in"] = policy_in
            ev["policy_tokens_out"] = policy_out
            ev["verdict"] = decision.verdict
            ev["reason_code"] = decision.reason_code
            ev["reason"] = decision.message
            ev["matched_policy"] = decision.matched_policy
            ev["matched_policies"] = _serialize_matched_policies(decision)

            if decision.verdict == "block":
                ev["latency_ms"] = 0
                enqueue(ev)
                if cfg.on_block == "raise":
                    raise PermissionError(
                        f"[egisai] {decision.message or 'blocked by policy'} "
                        f"(matched={decision.matched_policy})"
                    )
                if stub_factory is None:
                    raise PermissionError(
                        f"[egisai] {decision.message or 'blocked by policy'} "
                        f"(matched={decision.matched_policy})"
                    )
                return stub_factory(decision, ensure_trace_id(), model)

            if decision.verdict == "sanitize":
                _apply_sanitization(decision=decision, payload=payload, ev=ev)

            model_started = time.monotonic()
            try:
                response = forward()
            except BaseException:
                ev["latency_ms"] = int((time.monotonic() - model_started) * 1000)
                ev["error"] = "call failed"
                enqueue(ev)
                raise
            ev["latency_ms"] = int((time.monotonic() - model_started) * 1000)
            _stamp_usage(ev, response, extract_usage)
            enqueue(ev)
            return response
        finally:
            set_policy_checked(prev_checked)
    finally:
        set_source(prev_source)


async def async_gate_call(
    *,
    source: str,
    target: str,
    model: str,
    prompt_text: str,
    stream: bool,
    payload: Any,
    stub_factory: Optional[Callable[[PolicyDecision, str, str], Any]] = None,
    extract_usage: Optional[ExtractUsage] = None,
    forward: Callable[[], Any],
) -> Any:
    """Async sibling of gate_call. Same semantics; awaits ``forward()``."""
    cfg = get_config()
    prev_source = get_source()
    prev_checked = get_policy_checked()

    if not prev_source:
        reset_trace()
    set_source(source)

    try:
        if prev_checked:
            return await forward()

        ev = build_event(
            source=source, target=target, payload=payload, model=model, stream=stream
        )
        _attribute_event(ev, payload)
        ev["prompt_chars"] = len(prompt_text or "")
        ev["prompt_preview"] = _safe_text_preview(prompt_text)
        ev["_prompt_text_original"] = prompt_text

        set_policy_checked(True)
        reset_policy_usage()
        try:
            policy_started = time.monotonic()
            decision = evaluate(
                InputCall(
                    source=source,
                    target=target,
                    model=model,
                    prompt_text=prompt_text,
                    stream=stream,
                )
            )
            ev["policy_latency_ms"] = int((time.monotonic() - policy_started) * 1000)
            policy_in, policy_out = get_policy_usage()
            ev["policy_tokens_in"] = policy_in
            ev["policy_tokens_out"] = policy_out
            ev["verdict"] = decision.verdict
            ev["reason_code"] = decision.reason_code
            ev["reason"] = decision.message
            ev["matched_policy"] = decision.matched_policy
            ev["matched_policies"] = _serialize_matched_policies(decision)

            if decision.verdict == "block":
                ev["latency_ms"] = 0
                enqueue(ev)
                if cfg.on_block == "raise":
                    raise PermissionError(
                        f"[egisai] {decision.message or 'blocked by policy'} "
                        f"(matched={decision.matched_policy})"
                    )
                if stub_factory is None:
                    raise PermissionError(
                        f"[egisai] {decision.message or 'blocked by policy'} "
                        f"(matched={decision.matched_policy})"
                    )
                return stub_factory(decision, ensure_trace_id(), model)

            if decision.verdict == "sanitize":
                _apply_sanitization(decision=decision, payload=payload, ev=ev)

            model_started = time.monotonic()
            try:
                response = await forward()
            except BaseException:
                ev["latency_ms"] = int((time.monotonic() - model_started) * 1000)
                ev["error"] = "call failed"
                enqueue(ev)
                raise
            ev["latency_ms"] = int((time.monotonic() - model_started) * 1000)
            _stamp_usage(ev, response, extract_usage)
            enqueue(ev)
            return response
        finally:
            set_policy_checked(prev_checked)
    finally:
        set_source(prev_source)
