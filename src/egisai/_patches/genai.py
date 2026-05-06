"""``google.genai`` patcher.

Targets ``google.genai.models.Models.generate_content`` (sync) and
``google.genai.models.AsyncModels.generate_content`` (async), plus the
streaming siblings ``generate_content_stream``.

The ``google.generativeai`` package is patched separately by
``egisai._patches.google``; both patchers run side-by-side when both
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

LOGGER = logging.getLogger("egisai.patches.genai")


def _read(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, dict):
            v = obj.get(name)
        else:
            v = getattr(obj, name, None)
        if v is not None:
            return v
    return None


def _extract_genai_usage(response: Any) -> dict[str, Any]:
    """Pull token counts off ``GenerateContentResponse.usage_metadata``."""
    meta = _read(response, "usage_metadata")
    if meta is None:
        return {}
    return {
        "tokens_in": _read(meta, "prompt_token_count"),
        "tokens_out": _read(meta, "candidates_token_count"),
    }


def _tools_from_config(config: Any) -> list[Any] | None:
    """Best-effort: pull ``tools`` off a ``GenerateContentConfig``.

    Returned as-is so the output-signal extractor can apply its own
    shape-tolerant lookup. Returns ``None`` when ``config`` is missing
    or doesn't expose ``tools``.
    """
    if config is None:
        return None
    return _read(config, "tools")


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
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text=blurb)], role="model"
                ),
                finish_reason="STOP",
                index=0,
            )
        ],
        prompt_feedback=None,
        usage_metadata=SimpleNamespace(
            prompt_token_count=0,
            candidates_token_count=0,
            total_token_count=0,
        ),
        egis={
            "blocked": True,
            "reason": decision.message,
            "matched_policy": decision.matched_policy,
        },
        _trace_id=trace_id,
        _model=model,
    )


def _wrap_sync(orig: Callable[..., Any], stream: bool) -> Callable[..., Any]:
    target = (
        "google.genai.models.generate_content_stream"
        if stream
        else "google.genai.models.generate_content"
    )

    def w(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        model = kwargs.get("model") or "unknown"
        contents = kwargs.get("contents")
        config = kwargs.get("config")
        return gate_call(
            source="genai",
            target=target,
            model=model,
            prompt_text=extract_gemini_prompt(contents),
            stream=stream,
            payload={"contents": contents, "tools": _tools_from_config(config)},
            stub_factory=_stub_response,
            extract_usage=_extract_genai_usage,
            extract_output_signals=extract_google,
            forward=lambda: orig(self, *args, **kwargs),
        )

    w.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return w


def _wrap_async(orig: Callable[..., Any], stream: bool) -> Callable[..., Any]:
    target = (
        "google.genai.models.async.generate_content_stream"
        if stream
        else "google.genai.models.async.generate_content"
    )

    async def aw(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        model = kwargs.get("model") or "unknown"
        contents = kwargs.get("contents")
        config = kwargs.get("config")
        return await async_gate_call(
            source="genai",
            target=target,
            model=model,
            prompt_text=extract_gemini_prompt(contents),
            stream=stream,
            payload={"contents": contents, "tools": _tools_from_config(config)},
            stub_factory=_stub_response,
            extract_usage=_extract_genai_usage,
            extract_output_signals=extract_google,
            forward=lambda: orig(self, *args, **kwargs),
        )

    aw.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return aw


def _patch_class(cls: Any, *, async_class: bool) -> bool:
    """Wrap ``generate_content`` / ``generate_content_stream`` on ``cls``.

    Idempotent: a method already carrying ``__egisai_wrapped__`` is
    left alone so re-running ``init()`` doesn't double-wrap.
    """
    any_patched = False
    wrap = _wrap_async if async_class else _wrap_sync

    sync_method = getattr(cls, "generate_content", None)
    if callable(sync_method) and not getattr(
        sync_method, "__egisai_wrapped__", False
    ):
        cls.generate_content = wrap(sync_method, stream=False)
        any_patched = True

    stream_method = getattr(cls, "generate_content_stream", None)
    if callable(stream_method) and not getattr(
        stream_method, "__egisai_wrapped__", False
    ):
        cls.generate_content_stream = wrap(stream_method, stream=True)
        any_patched = True

    return any_patched


def apply() -> bool:
    """Patch ``google.genai`` if it is importable.

    Returns ``True`` if any methods were wrapped during this call.
    Safe to call multiple times.
    """
    if not has_module("google.genai"):
        return False
    try:
        from google.genai.models import (  # type: ignore[import-not-found]
            AsyncModels,
            Models,
        )
    except Exception:  # noqa: BLE001
        return False

    any_patched = False
    if _patch_class(Models, async_class=False):
        any_patched = True
    if _patch_class(AsyncModels, async_class=True):
        any_patched = True
    return any_patched
