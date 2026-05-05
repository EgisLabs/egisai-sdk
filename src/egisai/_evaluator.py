"""Adapter: turn a captured framework call into PolicyContext + decision.

Each patched function calls ``evaluate()`` with a structured view of
the call. We translate it into a ``PolicyContext`` and run the
cached rules. Output-side rules run on the response via
``evaluate_output()``.
"""

from __future__ import annotations

import logging
import threading
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
    evaluate_output_policies,
    evaluate_policies,
)
from egisai.policy.semantic import SemanticBlocker

LOGGER = logging.getLogger("egisai.evaluator")

_blocker_lock = threading.Lock()
_blocker: SemanticBlocker | None = None


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


def evaluate(call: InputCall) -> PolicyDecision:
    """Run the cached input rules against an in-flight call."""
    rules = get_rules()
    if not rules:
        return PolicyDecision.allow()
    rules = _scope_filter(rules, _active_agent_id())
    if not rules:
        return PolicyDecision.allow()
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
    ctx = OutputPolicyContext(
        tenant=call.tenant or "",
        model=call.model,
        text=call.text,
        tool_names=list(call.tool_names),
        tool_calls=list(call.tool_calls),
        mcp_targets=list(call.mcp_targets),
        stream=call.stream,
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
    if isinstance(contents, list):
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
    if isinstance(contents, list):
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
