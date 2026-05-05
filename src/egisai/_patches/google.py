"""Google Generative AI (Gemini) patcher.

Targets ``google.generativeai.GenerativeModel.generate_content`` and the
async sibling ``generate_content_async``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from egisai._evaluator import extract_gemini_prompt
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
    from types import SimpleNamespace

    blurb = (
        f"[POLICY BLOCK] {decision.message or 'Blocked by policy.'} "
        f"(matched={decision.matched_policy or 'unknown'})"
    )
    return SimpleNamespace(
        text=blurb,
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text=blurb)], role="model"),
                finish_reason="STOP",
                index=0,
            )
        ],
        prompt_feedback=None,
        usage_metadata=SimpleNamespace(prompt_token_count=0, candidates_token_count=0, total_token_count=0),
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
            return await async_gate_call(
                source="genai",
                target=target,
                model=model,
                prompt_text=extract_gemini_prompt(contents),
                stream=stream,
                payload={"contents": contents, "tools": kwargs.get("tools")},
                stub_factory=_stub_response,
                extract_usage=_extract_gemini_usage,
                forward=lambda: orig(self, contents, *args, **kwargs),
            )

        setattr(aw, "__egisai_wrapped__", True)
        return aw

    def w(self, contents, *args, **kwargs):  # type: ignore[no-untyped-def]
        model = getattr(self, "model_name", "unknown")
        stream = bool(kwargs.get("stream", False))
        return gate_call(
            source="genai",
            target=target,
            model=model,
            prompt_text=extract_gemini_prompt(contents),
            stream=stream,
            payload={"contents": contents, "tools": kwargs.get("tools")},
            stub_factory=_stub_response,
            extract_usage=_extract_gemini_usage,
            forward=lambda: orig(self, contents, *args, **kwargs),
        )

    setattr(w, "__egisai_wrapped__", True)
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
