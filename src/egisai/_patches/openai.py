"""OpenAI Python SDK patcher.

Patches Chat Completions and Responses (sync + async) plus the
legacy v0 ``openai.ChatCompletion.create`` path.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from egisai._evaluator import extract_prompt_text
from egisai._patches import has_module
from egisai._patches._common import async_gate_call, gate_call
from egisai.policy import PolicyDecision

LOGGER = logging.getLogger("egisai.patches.openai")


def _read(obj: Any, *names: str) -> Any:
    """Best-effort attribute / dict lookup over a response-shaped value."""
    for name in names:
        if isinstance(obj, dict):
            v = obj.get(name)
        else:
            v = getattr(obj, name, None)
        if v is not None:
            return v
    return None


def _extract_chat_usage(response: Any) -> dict[str, Any]:
    """Pull token counts off ``ChatCompletion.usage``."""
    usage = _read(response, "usage")
    if usage is None:
        return {}
    return {
        "tokens_in": _read(usage, "prompt_tokens"),
        "tokens_out": _read(usage, "completion_tokens"),
    }


def _extract_responses_usage(response: Any) -> dict[str, Any]:
    """Pull token counts off ``Responses.usage``."""
    usage = _read(response, "usage")
    if usage is None:
        return {}
    return {
        "tokens_in": _read(usage, "input_tokens", "prompt_tokens"),
        "tokens_out": _read(usage, "output_tokens", "completion_tokens"),
    }


def _stub_chat_completion(decision: PolicyDecision, trace_id: str, model: str):
    """Build a stub ``ChatCompletion``-shaped object."""
    blurb = (
        f"[POLICY BLOCK] {decision.message or 'Blocked by policy.'} "
        f"(matched={decision.matched_policy or 'unknown'})"
    )
    from types import SimpleNamespace

    msg = SimpleNamespace(role="assistant", content=blurb)
    choice = SimpleNamespace(index=0, message=msg, finish_reason="stop")
    usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    return SimpleNamespace(
        id=f"egis-blocked-{trace_id[:8]}",
        object="chat.completion",
        model=model,
        choices=[choice],
        usage=usage,
        egis={"blocked": True, "reason": decision.message, "matched_policy": decision.matched_policy},
    )


def _stub_response(decision: PolicyDecision, trace_id: str, model: str):
    """OpenAI Responses-API shaped stub."""
    from types import SimpleNamespace

    blurb = (
        f"[POLICY BLOCK] {decision.message or 'Blocked by policy.'} "
        f"(matched={decision.matched_policy or 'unknown'})"
    )
    return SimpleNamespace(
        id=f"egis-blocked-{trace_id[:8]}",
        object="response",
        status="completed",
        model=model,
        output=[
            SimpleNamespace(
                type="message",
                role="assistant",
                content=[SimpleNamespace(type="output_text", text=blurb)],
            )
        ],
        usage=SimpleNamespace(input_tokens=0, output_tokens=0, total_tokens=0),
        egis={"blocked": True, "reason": decision.message, "matched_policy": decision.matched_policy},
    )


def _wrap_create_chat(orig: Callable[..., Any], is_async: bool) -> Callable[..., Any]:
    target = "openai.chat.completions.create"

    if is_async:
        async def aw(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            model = kwargs.get("model", "unknown")
            messages = kwargs.get("messages")
            stream = bool(kwargs.get("stream", False))
            return await async_gate_call(
                source="openai",
                target=target,
                model=model,
                prompt_text=extract_prompt_text(messages),
                stream=stream,
                payload={"messages": messages, "tools": kwargs.get("tools")},
                stub_factory=_stub_chat_completion,
                extract_usage=_extract_chat_usage,
                forward=lambda: orig(self, *args, **kwargs),
            )

        setattr(aw, "__egisai_wrapped__", True)
        return aw

    def w(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages")
        stream = bool(kwargs.get("stream", False))
        return gate_call(
            source="openai",
            target=target,
            model=model,
            prompt_text=extract_prompt_text(messages),
            stream=stream,
            payload={"messages": messages, "tools": kwargs.get("tools")},
            stub_factory=_stub_chat_completion,
            extract_usage=_extract_chat_usage,
            forward=lambda: orig(self, *args, **kwargs),
        )

    setattr(w, "__egisai_wrapped__", True)
    return w


def _wrap_create_responses(orig: Callable[..., Any], is_async: bool) -> Callable[..., Any]:
    target = "openai.responses.create"

    if is_async:
        async def aw(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            model = kwargs.get("model", "unknown")
            inp = kwargs.get("input")
            stream = bool(kwargs.get("stream", False))
            return await async_gate_call(
                source="openai",
                target=target,
                model=model,
                prompt_text=extract_prompt_text(inp),
                stream=stream,
                payload={"input": inp, "tools": kwargs.get("tools")},
                stub_factory=_stub_response,
                extract_usage=_extract_responses_usage,
                forward=lambda: orig(self, *args, **kwargs),
            )

        setattr(aw, "__egisai_wrapped__", True)
        return aw

    def w(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        model = kwargs.get("model", "unknown")
        inp = kwargs.get("input")
        stream = bool(kwargs.get("stream", False))
        return gate_call(
            source="openai",
            target=target,
            model=model,
            prompt_text=extract_prompt_text(inp),
            stream=stream,
            payload={"input": inp, "tools": kwargs.get("tools")},
            stub_factory=_stub_response,
            extract_usage=_extract_responses_usage,
            forward=lambda: orig(self, *args, **kwargs),
        )

    setattr(w, "__egisai_wrapped__", True)
    return w


def _patch_class(module_path: str, attr: str, *, is_async: bool, kind: str) -> bool:
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
    if kind == "chat":
        wrapped = _wrap_create_chat(orig, is_async=is_async)
    elif kind == "responses":
        wrapped = _wrap_create_responses(orig, is_async=is_async)
    else:
        return False
    setattr(cls, "create", wrapped)
    return True


def apply() -> bool:
    if not has_module("openai"):
        return False
    any_patched = False
    if _patch_class("openai.resources.chat.completions", "Completions", is_async=False, kind="chat"):
        any_patched = True
    if _patch_class("openai.resources.chat.completions", "AsyncCompletions", is_async=True, kind="chat"):
        any_patched = True
    if _patch_class("openai.resources.responses", "Responses", is_async=False, kind="responses"):
        any_patched = True
    if _patch_class("openai.resources.responses", "AsyncResponses", is_async=True, kind="responses"):
        any_patched = True

    try:
        import openai  # noqa: WPS433

        chat = getattr(openai, "ChatCompletion", None)
        if chat is not None and hasattr(chat, "create"):
            orig = chat.create
            if callable(orig) and not getattr(orig, "__egisai_wrapped__", False):
                chat.create = _wrap_create_chat(orig, is_async=False)  # type: ignore[assignment]
                any_patched = True
    except Exception:  # noqa: BLE001
        pass

    return any_patched
