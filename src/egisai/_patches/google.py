"""``google.generativeai`` patcher.

Targets ``google.generativeai.GenerativeModel.generate_content`` and
the async sibling ``generate_content_async``.

The ``google.genai`` package is patched separately by
``egisai._patches.genai``; both patchers run side-by-side when both
packages are installed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from egisai._evaluator import extract_gemini_prompt
from egisai._output_signals import extract_google
from egisai._patches import has_module
from egisai._patches._common import async_gate_call, gate_call
from egisai.policy import PolicyDecision

LOGGER = logging.getLogger("egisai.patches.google")


def _read(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict):
            v = obj.get(name)
        else:
            v = getattr(obj, name, None)
        if v is not None:
            return v
    return None


def _extract_gemini_usage(response: Any) -> dict[str, Any]:
    """Pull token counts off ``GenerateContentResponse.usage_metadata``."""
    meta = _read(response, "usage_metadata")
    if meta is None:
        return {}
    return {
        "tokens_in": _read(meta, "prompt_token_count"),
        "tokens_out": _read(meta, "candidates_token_count"),
    }


def _stub_response(decision: PolicyDecision, trace_id: str, model: str):
    """Legacy ``google.generativeai`` ``GenerateContentResponse``-shaped stub.

    Mirror of ``genai._stub_response``. The upstream
    ``google.generativeai.types.GenerateContentResponse`` carries
    a richer ``usage_metadata`` shape than the basic three-counter
    triple — frameworks that wrap the legacy SDK read those
    extras directly off the response. A stub that omits them
    crashes those frameworks with ``AttributeError`` the moment
    a policy fires with ``on_block="stub"``. We populate every
    documented field at zero/None so the stub is structurally
    indistinguishable from a real "blocked, no token usage" turn.
    """
    from types import SimpleNamespace

    blurb = (
        f"[POLICY BLOCK] {decision.message or 'Blocked by policy.'} "
        f"(matched={decision.matched_policy or 'unknown'})"
    )
    candidate = SimpleNamespace(
        content=SimpleNamespace(
            parts=[SimpleNamespace(text=blurb, function_call=None)],
            role="model",
        ),
        finish_reason="STOP",
        finish_message=None,
        safety_ratings=None,
        citation_metadata=None,
        token_count=0,
        avg_logprobs=None,
        grounding_metadata=None,
        index=0,
    )
    return SimpleNamespace(
        text=blurb,
        candidates=[candidate],
        prompt_feedback=None,
        usage_metadata=SimpleNamespace(
            prompt_token_count=0,
            candidates_token_count=0,
            total_token_count=0,
            cached_content_token_count=0,
        ),
        function_calls=[],
        egis={"blocked": True, "reason": decision.message, "matched_policy": decision.matched_policy},
        _trace_id=trace_id,
        _model=model,
    )


def _wrap_generate(orig: Callable[..., Any], is_async: bool) -> Callable[..., Any]:
    target = "google.generativeai.generate_content"

    if is_async:
        async def aw(self, contents, *args, **kwargs):  # type: ignore[no-untyped-def]
            model = getattr(self, "model_name", "unknown")
            stream = bool(kwargs.get("stream", False))
            # ``payload`` is the mutable view the gate hands to
            # ``mutate_prompt_text`` on a sanitize verdict.
            # ``contents`` is a positional arg whose binding here is
            # an immutable scalar (string) in the common case, so the
            # forward must re-read the post-sanitization value from
            # ``payload["contents"]`` and pass THAT to the SDK -
            # otherwise the raw prompt would still leave the SDK
            # boundary even though policy stamped sanitize. Same
            # rationale as ``_patches.genai._wrap_sync``.
            payload: dict[str, Any] = {
                "contents": contents,
                "tools": kwargs.get("tools"),
            }
            return await async_gate_call(
                source="genai",
                target=target,
                model=model,
                prompt_text=extract_gemini_prompt(contents),
                stream=stream,
                payload=payload,
                stub_factory=_stub_response,
                extract_usage=_extract_gemini_usage,
                extract_output_signals=extract_google,
                # Multi-step waterfall: see ``_patches.openai._wrap_create_chat``
                # for the dashboard timeline rationale. Each
                # ``function_call`` part the model emitted becomes one
                # ``tool_call`` step on the Run.
                emit_tool_call_steps=True,
                forward=lambda: orig(self, payload["contents"], *args, **kwargs),
            )

        aw.__egisai_wrapped__ = True  # type: ignore[attr-defined]
        return aw

    def w(self, contents, *args, **kwargs):  # type: ignore[no-untyped-def]
        model = getattr(self, "model_name", "unknown")
        stream = bool(kwargs.get("stream", False))
        # See the async branch above for the immutable-scalar
        # ``contents`` rationale.
        payload: dict[str, Any] = {
            "contents": contents,
            "tools": kwargs.get("tools"),
        }
        return gate_call(
            source="genai",
            target=target,
            model=model,
            prompt_text=extract_gemini_prompt(contents),
            stream=stream,
            payload=payload,
            stub_factory=_stub_response,
            extract_usage=_extract_gemini_usage,
            extract_output_signals=extract_google,
            emit_tool_call_steps=True,
            forward=lambda: orig(self, payload["contents"], *args, **kwargs),
        )

    w.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return w


def apply() -> bool:
    if not has_module("google.generativeai"):
        return False
    any_patched = False
    try:
        from google.generativeai import GenerativeModel  # type: ignore
    except Exception:  # noqa: BLE001
        return False

    sync = getattr(GenerativeModel, "generate_content", None)
    if callable(sync) and not getattr(sync, "__egisai_wrapped__", False):
        GenerativeModel.generate_content = _wrap_generate(sync, is_async=False)  # type: ignore[assignment]
        any_patched = True

    asyn = getattr(GenerativeModel, "generate_content_async", None)
    if callable(asyn) and not getattr(asyn, "__egisai_wrapped__", False):
        GenerativeModel.generate_content_async = _wrap_generate(asyn, is_async=True)  # type: ignore[assignment]
        any_patched = True

    return any_patched
