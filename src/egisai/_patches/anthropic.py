"""Anthropic Python SDK patcher.

Targets ``anthropic.resources.messages.Messages.create`` (and async sibling).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from egisai._evaluator import extract_anthropic_prompt
from egisai._output_signals import extract_anthropic
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
    """Build a stub Anthropic ``Message``-shaped object.

    The upstream ``anthropic.types.Usage`` exposes
    ``cache_creation_input_tokens``, ``cache_read_input_tokens``,
    ``server_tool_use`` and ``service_tier`` alongside the basic
    ``input_tokens``/``output_tokens`` pair. Frameworks that wrap
    Anthropic (LangChain, LangGraph, CrewAI, …) frequently read
    those extras directly off the response — a stub that omits
    them crashes those frameworks with ``AttributeError`` the
    moment a policy fires with ``on_block="stub"``. We populate
    every documented field at a sensible "zero/no-cache" value
    so the stub is structurally indistinguishable from a real
    "blocked, no token usage" turn.

    Similarly, the top-level ``Message`` exposes ``stop_sequence``
    and ``container`` (both Optional on the upstream model); we
    spell them as ``None`` so attribute access never raises.
    """
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
        stop_sequence=None,
        container=None,
        usage=SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            server_tool_use=None,
            service_tier=None,
        ),
        egis={"blocked": True, "reason": decision.message, "matched_policy": decision.matched_policy},
    )


# ── Smart Model Routing ──────────────────────────────────────────────


def _message_from_canonical(result: dict[str, Any]) -> Any:
    """Wrap a canonical cross-provider result into a Message shape.

    Used when the routing engine served an Anthropic-originated call
    on a different provider. Mirrors :func:`_stub_message`'s field
    surface (every documented ``Message`` / ``Usage`` attribute
    present) so frameworks reading the response never hit an
    ``AttributeError``; carries real content + token usage plus an
    additive ``egis.routing`` marker.
    """
    import uuid as _uuid
    from types import SimpleNamespace

    text = str(result.get("text") or "")
    served_model = str(result.get("model") or "")
    return SimpleNamespace(
        id=f"egis-routed-{_uuid.uuid4().hex[:12]}",
        type="message",
        role="assistant",
        model=served_model,
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason="end_turn",
        stop_sequence=None,
        container=None,
        usage=SimpleNamespace(
            input_tokens=int(result.get("tokens_in") or 0),
            output_tokens=int(result.get("tokens_out") or 0),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            server_tool_use=None,
            service_tier=None,
        ),
        egis={"routing": {"applied": True, "served_model": served_model}},
    )


def _routing_adapter_messages(kwargs: dict[str, Any]) -> Any:
    """Routing adapter for ``messages.create`` — see the openai twin.

    Same-provider swaps rewrite ``kwargs["model"]`` (the forward
    lambda reads kwargs at call time). Cross-provider swaps translate
    the Anthropic ``messages`` + ``system`` payload to the canonical
    form, execute against the target provider directly, and come back
    ``Message``-shaped. The gate restricts cross swaps to plain-text,
    non-streaming, tool-free calls; ``canonicalize_anthropic_messages``
    additionally bails on any content block it can't represent
    faithfully.
    """
    try:
        from egisai._routing import (
            RoutingAdapter,
            canonicalize_anthropic_messages,
            execute_cross_call,
        )
    except Exception:  # noqa: BLE001
        return None

    def _apply(new_model: str) -> bool:
        kwargs["model"] = new_model
        return True

    def _cross(decision: dict[str, Any]):  # noqa: ANN202
        messages = canonicalize_anthropic_messages(
            kwargs.get("messages"), kwargs.get("system")
        )
        if messages is None:
            return None
        params = {
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens"),
        }

        def _forward() -> Any:
            return _message_from_canonical(
                execute_cross_call(
                    provider=str(decision.get("provider") or ""),
                    model=str(decision.get("model") or ""),
                    messages=messages,
                    params=params,
                )
            )

        return _forward

    return RoutingAdapter(
        apply_same_provider=_apply, build_cross_forward=_cross
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
                extract_output_signals=extract_anthropic,
                # Multi-step waterfall: when the assistant turn carries
                # ``tool_use`` blocks, append one ``tool_call`` step per
                # tool so the dashboard timeline reads
                # ``model -> tool -> model -> tool -> ...`` instead of
                # collapsing the whole turn into a single ``model_call``
                # row. Same posture as the openai patch; the parent
                # model_call's output policy already had a chance to
                # gate each tool request before this step lands so the
                # per-tool steps are stamped ``enforced`` (distinct
                # from the agentic-subprocess case where execution
                # happened ahead of observation).
                emit_tool_call_steps=True,
                routing_adapter=_routing_adapter_messages(kwargs),
                forward=lambda: orig(self, *args, **kwargs),
            )

        aw.__egisai_wrapped__ = True  # type: ignore[attr-defined]
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
            extract_output_signals=extract_anthropic,
            emit_tool_call_steps=True,
            routing_adapter=_routing_adapter_messages(kwargs),
            forward=lambda: orig(self, *args, **kwargs),
        )

    w.__egisai_wrapped__ = True  # type: ignore[attr-defined]
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
    cls.create = _wrap_messages_create(orig, is_async=is_async)
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
