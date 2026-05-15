"""Adapter: turn a captured framework call into PolicyContext + decision.

Each patched function calls ``evaluate()`` with a structured view of
the call. We translate it into a ``PolicyContext`` and run the
cached rules. Output-side rules run on the response via
``evaluate_output()``.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from egisai._config import get_config_optional
from egisai._context import get_context
from egisai._policy_cache import get_rules
from egisai.policy import (
    OutputPolicyContext,
    PolicyContext,
    PolicyDecision,
    _pii_loader,
    evaluate_output_policies,
    evaluate_policies,
)
from egisai.policy.semantic import SemanticBlocker

LOGGER = logging.getLogger("egisai.evaluator")

_blocker_lock = threading.Lock()
_blocker: SemanticBlocker | None = None

# ── PII analyzer first-call gate ────────────────────────────────────
#
# The Presidio + spaCy analyzer is warmed in a daemon thread from
# ``egisai.init()`` (see ``policy/_pii_loader.py``). It takes ~1–3 s
# on a warm machine and up to ~90 s the first time the SDK is ever
# installed (one-time ~750 MB model download). The hot path falls
# back to regex+checksum detection while the analyzer is cold, which
# means **names / addresses / GDPR special-category text are not
# masked on call #1 of a fresh process**.
#
# For long-running services (FastAPI backends, agent harnesses) the
# analyzer is warm by the time the first request lands and this is a
# non-issue. For test harnesses, demo scripts, and serverless cold
# starts that fire a model call within ~1 s of ``init()``, that first
# call slips through with regex-only coverage — silently.
#
# The gate below closes that window WITHOUT regressing init() back to
# blocking I/O. On the first ``evaluate()`` (or ``evaluate_output()``)
# of the process, IF the current rule set has an active ``pii_scan``
# rule scoped to this call AND the analyzer is still loading, we block
# briefly (cap 2.0 s by default) waiting for warm-up. Subsequent calls
# never wait — the one-shot flag is set regardless of outcome.
#
# Env knobs (no kwarg on init() — keeping that surface non-blocking
# by contract):
#   * ``EGISAI_PII_WARMUP_TIMEOUT_SECS`` — cap in seconds, default
#     ``2.0``. Set to ``0`` (or negative) to opt out entirely; the
#     SDK then behaves exactly as pre-0.25 (silent regex fallback on
#     call #1 if you race the loader). Recommended for AWS Lambda
#     and any environment where a 2 s cold-start blip in the first
#     request's tail latency is unacceptable.
_WARMUP_DEFAULT_SECS = 2.0
_warmup_lock = threading.Lock()
_warmup_done = False


def _warmup_timeout_secs() -> float:
    """Resolve the per-call-#1 cap from the env var, defaulting to 2 s.

    Re-read on every gate invocation so operators can flip it without
    restarting. The gate is one-shot per process so the cost of re-
    reading is paid at most once.
    """
    raw = os.getenv("EGISAI_PII_WARMUP_TIMEOUT_SECS")
    if not raw:
        return _WARMUP_DEFAULT_SECS
    try:
        return float(raw)
    except ValueError:
        return _WARMUP_DEFAULT_SECS


def _has_active_pii_rule(rules: list) -> bool:
    """Does this call's scoped rule set actually need the NER analyzer?

    If the org has only ``semantic_guard`` / ``deny_regex`` / etc.
    rules active for this agent, blocking on the PII engine is just
    wasted latency on call #1. Skip the gate in that case.
    """
    return any(getattr(r, "type", None) == "pii_scan" for r in rules)


def _maybe_wait_for_pii_analyzer(rules: list) -> None:
    """One-shot pre-evaluation gate. Idempotent across both phases.

    Conditions for the wait to actually fire:
      * we haven't waited yet in this process;
      * the org has at least one active ``pii_scan`` rule scoped to
        this call;
      * the analyzer is still loading (settled implies either warm
        or permanently failed — neither benefits from waiting).

    Logs one stderr line when it fires so operators can see "egisai
    paid X ms on the first call to warm the PII engine" — important
    observability for tuning ``EGISAI_PII_WARMUP_TIMEOUT_SECS``.
    """
    global _warmup_done
    if _warmup_done:
        return
    with _warmup_lock:
        if _warmup_done:
            return
        timeout = _warmup_timeout_secs()
        if timeout <= 0:
            # Operator opt-out (Lambda etc). Don't wait, don't log —
            # they chose this path explicitly.
            _warmup_done = True
            return
        if _pii_loader.is_settled():
            # Warm (or permanently failed). Either way, no wait helps.
            _warmup_done = True
            return
        if not _has_active_pii_rule(rules):
            # No PII rule in scope. NER warm-up wouldn't change the
            # decision on this call — skip to keep call #1 fast.
            # Don't flip the one-shot: a later call from a different
            # agent (with different scoping) might need the analyzer.
            return
        started = time.monotonic()
        warm = _pii_loader.wait_for_warm(timeout)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        _warmup_done = True
        # Both branches go through the ``egisai.evaluator`` logger
        # rather than ``sys.stderr`` directly. Default Python logging
        # config is WARNING+, so SUCCESS (.info) is silent unless the
        # operator opts in; TIMEOUT (.warning) is visible by default
        # to surface the rare honest degradation (NER coverage missed
        # call #1, regex fallback is in effect). Either way the line
        # is observable at ``logging.getLogger("egisai.evaluator")``
        # for ops who want to track cold-start cost in a structured
        # log pipeline (Datadog, CloudWatch, Loki).
        if warm:
            LOGGER.info(
                "[egisai] waited %d ms on first call to warm PII NER "
                "analyzer (Presidio + spaCy). Subsequent calls add "
                "zero overhead. Set EGISAI_PII_WARMUP_TIMEOUT_SECS=0 "
                "to disable.",
                elapsed_ms,
            )
        else:
            LOGGER.warning(
                "[egisai] PII NER analyzer not warm after %d ms — "
                "proceeding with regex+checksum fallback for THIS "
                "call. Names / addresses / GDPR special-category text "
                "will not be flagged until the daemon thread finishes "
                "loading (typical: 1–3 s warm machine, up to ~90 s on "
                "first install with model download). Raise "
                "EGISAI_PII_WARMUP_TIMEOUT_SECS if your environment "
                "needs more headroom on call #1.",
                elapsed_ms,
            )


def _reset_warmup_gate_for_tests() -> None:
    """Wipe the one-shot flag so tests can drive multiple gate scenarios.

    Production callers must not use this — the gate is intentionally
    one-shot per process. Mirrors ``_pii_loader.reset_for_tests`` in
    intent.
    """
    global _warmup_done
    with _warmup_lock:
        _warmup_done = False


def _has_semantic_rule(rules: list) -> bool:
    return any(getattr(r, "type", None) == "semantic_guard" for r in rules)


def _active_agent_id() -> str:
    """Return the agent UUID this call should be attributed to.

    ``set_context`` overrides the per-process default from ``init()``.
    """
    ctx = get_context()
    cfg = get_config_optional()
    aid = ctx.agent_id or (cfg.agent_id if cfg is not None else None)
    return (aid or "").strip().lower()


def _scope_filter(rules: list, agent_id: str) -> list:
    """Drop rules whose ``agent_ids`` excludes us.

    Empty / missing ``agent_ids`` means "applies to all". When
    ``agent_id`` is unknown, targeted rules are skipped to avoid
    enforcing a rule on the wrong agent.
    """
    out = []
    for r in rules:
        targets = tuple(getattr(r, "agent_ids", ()) or ())
        if not targets:
            out.append(r)
            continue
        if agent_id and agent_id in targets:
            out.append(r)
    return out


def _get_semantic_blocker() -> SemanticBlocker | None:
    """Lazy-construct a process-wide ``SemanticBlocker``.

    Returns ``None`` until ``egisai.init()`` has run.
    """
    global _blocker

    if _blocker is not None:
        return _blocker

    cfg = get_config_optional()
    if cfg is None:
        return None

    with _blocker_lock:
        if _blocker is None:
            _blocker = SemanticBlocker(
                platform_api_key=cfg.api_key,
                platform_base_url=cfg.base_url,
                on_outage=cfg.semantic_on_outage,
            )
    return _blocker


def _close_semantic_blocker() -> None:
    """Called from shutdown(). Idempotent."""
    global _blocker
    if _blocker is None:
        return
    try:
        _blocker.close()
    except Exception:  # noqa: BLE001
        pass
    _blocker = None


@dataclass(frozen=True)
class InputCall:
    """Captures the input side of a model call before it goes upstream."""

    source: str           # openai|anthropic|genai|httpx
    target: str           # e.g. "openai.chat.completions.create"
    model: str
    prompt_text: str
    stream: bool = False
    tenant: str | None = None


@dataclass(frozen=True)
class OutputCall:
    source: str
    target: str
    model: str
    text: str
    tool_names: list[str]
    tool_calls: list[dict]
    mcp_targets: list[str]
    stream: bool = False
    tenant: str | None = None
    # ``allow_sanitize`` flips ``pii_scan`` from "always block on the
    # output side" to "honour the operator's action='sanitize'". Only
    # set by output paths that have an atomic mutation point after
    # the model produced bytes — today that's the ``claude_agent_sdk``
    # PostToolUse hook, which can swap the tool result via
    # ``updatedToolOutput`` / ``updatedMCPToolOutput`` before Claude
    # is shown it. See ``OutputPolicyContext`` for the full
    # rationale. Default ``False`` keeps every existing caller's
    # behavior identical.
    allow_sanitize: bool = False


def evaluate(call: InputCall) -> PolicyDecision:
    """Run the cached input rules against an in-flight call."""
    rules = get_rules()
    if not rules:
        return PolicyDecision.allow()
    rules = _scope_filter(rules, _active_agent_id())
    if not rules:
        return PolicyDecision.allow()
    # First-call gate: if the analyzer is still warming and a
    # ``pii_scan`` rule is scoped to this call, briefly wait. After
    # this returns the analyzer is either warm OR we've decided to
    # proceed with the regex fallback. Idempotent across phases.
    _maybe_wait_for_pii_analyzer(rules)
    ctx = PolicyContext(
        tenant=call.tenant or "",
        model=call.model,
        prompt_text=call.prompt_text,
        prompt_chars=len(call.prompt_text),
        stream=call.stream,
    )
    blocker = _get_semantic_blocker() if _has_semantic_rule(rules) else None
    try:
        return evaluate_policies(rules, ctx, semantic_blocker=blocker)
    except Exception:  # noqa: BLE001
        LOGGER.warning("policy evaluator errored, allowing by default", exc_info=True)
        return PolicyDecision.allow()


def evaluate_output(call: OutputCall) -> PolicyDecision:
    rules = get_rules()
    if not rules:
        return PolicyDecision.allow()
    rules = _scope_filter(rules, _active_agent_id())
    if not rules:
        return PolicyDecision.allow()
    # Same first-call gate as ``evaluate`` — covers frameworks (e.g.
    # ``claude_agent_sdk``) where the first patched entry point is the
    # output / tool-result hook rather than the model-call entrypoint.
    _maybe_wait_for_pii_analyzer(rules)
    ctx = OutputPolicyContext(
        tenant=call.tenant or "",
        model=call.model,
        text=call.text,
        tool_names=list(call.tool_names),
        tool_calls=list(call.tool_calls),
        mcp_targets=list(call.mcp_targets),
        stream=call.stream,
        allow_sanitize=call.allow_sanitize,
    )
    blocker = _get_semantic_blocker() if _has_semantic_rule(rules) else None
    try:
        return evaluate_output_policies(rules, ctx, semantic_blocker=blocker)
    except Exception:  # noqa: BLE001
        LOGGER.warning("output evaluator errored, allowing by default", exc_info=True)
        return PolicyDecision.allow()


def extract_prompt_text(messages: Any) -> str:
    """Flatten OpenAI-style messages into a single searchable string.

    System messages are excluded — policies evaluate user input, not
    the developer's instructions to the model.
    """
    if messages is None:
        return ""
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        chunks: list[str] = []
        for m in messages:
            if isinstance(m, str):
                chunks.append(m)
                continue
            if isinstance(m, dict):
                if m.get("role") == "system":
                    continue
                content = m.get("content")
                if isinstance(content, str):
                    chunks.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict):
                            t = part.get("text") or part.get("content")
                            if isinstance(t, str):
                                chunks.append(t)
                        elif isinstance(part, str):
                            chunks.append(part)
        return "\n".join(c for c in chunks if c)
    return str(messages)


def extract_anthropic_prompt(messages: Any, system: Any = None) -> str:
    """Flatten Anthropic-shaped messages.

    The ``system`` kwarg is intentionally ignored (same reason as
    ``extract_prompt_text``) and kept on the signature for API
    stability.
    """
    _ = system
    return extract_prompt_text(messages)


def extract_payload_text(payload: Any) -> str:
    """Concatenate every user-visible text field from a captured payload.

    Read-side counterpart of ``mutate_prompt_text``. Handles
    OpenAI / Anthropic / OpenAI Responses / Gemini shapes. Returns
    ``""`` for unrecognised payloads.
    """
    if not isinstance(payload, dict):
        return ""

    chunks: list[str] = []

    def _walk_messages(messages: Any) -> None:
        if not isinstance(messages, list):
            return
        for m in messages:
            if not isinstance(m, dict) or m.get("role") == "system":
                continue
            content = m.get("content")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    t = part.get("text") or part.get("content")
                    if isinstance(t, str):
                        chunks.append(t)

    _walk_messages(payload.get("messages"))

    inp = payload.get("input")
    if isinstance(inp, str):
        chunks.append(inp)
    elif isinstance(inp, list):
        _walk_messages(inp)

    contents = payload.get("contents")
    if isinstance(contents, str):
        # Google Gemini's ergonomic shape — ``client.models.
        # generate_content(model=..., contents="hi gemini")`` —
        # passes the whole prompt as a top-level string. Without
        # this branch the prompt is invisible to PII scanners and
        # to label_redact, which means a raw SSN in that string
        # would never trip pii_scan and never get masked. The
        # smoke-battery test ``test_genai_async_path_sanitize``
        # pinned this regression.
        chunks.append(contents)
    elif isinstance(contents, list):
        for item in contents:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                parts = item.get("parts")
                if isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            chunks.append(part["text"])
                elif isinstance(parts, str):
                    chunks.append(parts)

    return "\n".join(c for c in chunks if c)


def mutate_prompt_text(payload: Any, transform: Callable[[str], str]) -> bool:
    """Apply ``transform`` to every user-visible text field in ``payload``.

    Mutates the same dict / list objects the framework SDK holds, so
    the upstream HTTP body picks up the change. System messages are
    skipped. Returns ``True`` if anything changed.
    """
    if not isinstance(payload, dict):
        return False
    mutated = False

    def _apply_to_text(value: Any) -> Any:
        nonlocal mutated
        if isinstance(value, str):
            new = transform(value)
            if new != value:
                mutated = True
            return new
        return value

    def _walk_messages(messages: Any) -> None:
        if not isinstance(messages, list):
            return
        for m in messages:
            if not isinstance(m, dict) or m.get("role") == "system":
                continue
            content = m.get("content")
            if isinstance(content, str):
                m["content"] = _apply_to_text(content)
            elif isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if isinstance(part.get("text"), str):
                        part["text"] = _apply_to_text(part["text"])
                    elif isinstance(part.get("content"), str):
                        part["content"] = _apply_to_text(part["content"])

    _walk_messages(payload.get("messages"))

    inp = payload.get("input")
    if isinstance(inp, str):
        payload["input"] = _apply_to_text(inp)
    elif isinstance(inp, list):
        _walk_messages(inp)

    contents = payload.get("contents")
    if isinstance(contents, str):
        # Mirror of ``extract_payload_text``: Gemini's
        # ``contents="..."`` ergonomic shape passes the whole
        # prompt as a top-level string. Mutating that key in
        # place ensures the SDK's outbound kwargs carry the
        # sanitized text rather than the original.
        payload["contents"] = _apply_to_text(contents)
    elif isinstance(contents, list):
        for i, item in enumerate(contents):
            if isinstance(item, str):
                contents[i] = _apply_to_text(item)
            elif isinstance(item, dict):
                parts = item.get("parts")
                if isinstance(parts, list):
                    for part in parts:
                        if isinstance(part, dict) and isinstance(part.get("text"), str):
                            part["text"] = _apply_to_text(part["text"])
                elif isinstance(parts, str):
                    item["parts"] = _apply_to_text(parts)

    return mutated


def extract_gemini_prompt(contents: Any) -> str:
    """Gemini accepts a string, a list of parts, or a list of {role, parts}."""
    if isinstance(contents, str):
        return contents
    if isinstance(contents, list):
        chunks: list[str] = []
        for item in contents:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                parts = item.get("parts", item.get("content"))
                if isinstance(parts, list):
                    for p in parts:
                        if isinstance(p, str):
                            chunks.append(p)
                        elif isinstance(p, dict):
                            t = p.get("text")
                            if isinstance(t, str):
                                chunks.append(t)
                elif isinstance(parts, str):
                    chunks.append(parts)
        return "\n".join(chunks)
    return str(contents)
