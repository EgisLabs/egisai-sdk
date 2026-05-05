"""Anthropic Python SDK patcher.

Targets ``anthropic.resources.messages.Messages.create`` (and async sibling).
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from egisai._evaluator import extract_anthropic_prompt
from egisai._patches import has_module
from egisai._patches._common import async_gate_call, gate_call
from egisai.policy import PolicyDecision

LOGGER = logging.getLogger("egisai.patches.anthropic")


def _read(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict):
            v = obj.get(name)
        else:
            v = getattr(obj, name, None)
        if v is not None:
            return v
    return None


def _extract_message_usage(response: Any) -> dict[str, Any]:
    """Pull token counts off ``Message.usage``."""
    usage = _read(response, "usage")
    if usage is None:
        return {}
    return {
        "tokens_in": _read(usage, "input_tokens"),
        "tokens_out": _read(usage, "output_tokens"),
    }


def _stub_message(decision: PolicyDecision, trace_id: str, model: str):
    """Build a stub Anthropic ``Message``-shaped object."""
    from types import SimpleNamespace

    blurb = (
        f"[POLICY BLOCK] {decision.message or 'Blocked by policy.'} "
        f"(matched={decision.matched_policy or 'unknown'})"
    )
    return SimpleNamespace(
        id=f"egis-blocked-{trace_id[:8]}",
        type="message",
        role="assistant",
        model=model,
        content=[SimpleNamespace(type="text", text=blurb)],
        stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=0, output_tokens=0),
        egis={"blocked": True, "reason": decision.message, "matched_policy": decision.matched_policy},
    )


def _wrap_messages_create(orig: Callable[..., Any], is_async: bool) -> Callable[..., Any]:
    target = "anthropic.messages.create"

    if is_async:
        async def aw(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            model = kwargs.get("model", "unknown")
            messages = kwargs.get("messages")
            system = kwargs.get("system")
            stream = bool(kwargs.get("stream", False))
            return await async_gate_call(
                source="anthropic",
                target=target,
                model=model,
                prompt_text=extract_anthropic_prompt(messages, system),
                stream=stream,
                payload={"messages": messages, "system": system, "tools": kwargs.get("tools")},
                stub_factory=_stub_message,
                extract_usage=_extract_message_usage,
                forward=lambda: orig(self, *args, **kwargs),
            )

        setattr(aw, "__egisai_wrapped__", True)
        return aw

    def w(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages")
        system = kwargs.get("system")
        stream = bool(kwargs.get("stream", False))
        return gate_call(
            source="anthropic",
            target=target,
            model=model,
            prompt_text=extract_anthropic_prompt(messages, system),
            stream=stream,
            payload={"messages": messages, "system": system, "tools": kwargs.get("tools")},
            stub_factory=_stub_message,
            extract_usage=_extract_message_usage,
            forward=lambda: orig(self, *args, **kwargs),
        )

    setattr(w, "__egisai_wrapped__", True)
    return w


def _patch_class(module_path: str, attr: str, *, is_async: bool) -> bool:
    try:
        mod = __import__(module_path, fromlist=[attr])
    except Exception:
        return False
    cls = getattr(mod, attr, None)
    if cls is None:
        return False
    orig = getattr(cls, "create", None)
    if not callable(orig):
        return False
    if getattr(orig, "__egisai_wrapped__", False):
        return True
    setattr(cls, "create", _wrap_messages_create(orig, is_async=is_async))
    return True


def apply() -> bool:
    if not has_module("anthropic"):
        return False
    any_patched = False
    if _patch_class("anthropic.resources.messages", "Messages", is_async=False):
        any_patched = True
    if _patch_class("anthropic.resources.messages", "AsyncMessages", is_async=True):
        any_patched = True
    return any_patched
