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
    """Build a stub ``google.genai`` ``GenerateContentResponse``-shaped object.

    The upstream ``google.genai.types.GenerateContentResponseUsageMetadata``
    exposes a wide set of counters beyond the basic
    ``prompt_token_count``/``candidates_token_count``/``total_token_count``
    triple (``cached_content_token_count``, ``thoughts_token_count``,
    ``tool_use_prompt_token_count``, ``prompt_tokens_details``, …).
    Agentic wrappers built on top of ``google-genai`` (notably the
    Vertex Agent Builder shims and LangChain's ``ChatGoogleGenerativeAI``)
    read those extras directly off the response — a stub that omits
    them crashes those frameworks with ``AttributeError`` the moment
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
        model_version=model,
        response_id=f"egis-blocked-{trace_id[:8]}",
        usage_metadata=SimpleNamespace(
            prompt_token_count=0,
            candidates_token_count=0,
            total_token_count=0,
            cached_content_token_count=0,
            thoughts_token_count=0,
            tool_use_prompt_token_count=0,
            prompt_tokens_details=None,
            candidates_tokens_details=None,
            cache_tokens_details=None,
            traffic_type=None,
        ),
        # ``function_calls`` is a convenience aggregator the SDK
        # exposes that walks every candidate's parts looking for
        # ``function_call`` entries. Frameworks read it directly;
        # spell it as an empty list to mirror a normal text-only
        # response.
        function_calls=[],
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
        # ``payload`` is the canonical mutable view that ``gate_call``
        # hands to ``mutate_prompt_text`` on the sanitize verdict.
        # When the policy engine replaces ``payload["contents"]``
        # with a masked string (Gemini's ergonomic
        # ``contents="..."`` shape), the local ``contents`` variable
        # and ``kwargs["contents"]`` still reference the original
        # unmasked string (strings are immutable, so the in-place
        # write hits ``payload``, not ``kwargs``). The forward
        # lambda below reads the post-sanitization value back out of
        # ``payload`` and ships THAT — never the raw prompt — to
        # the upstream Gemini SDK. This is the privacy contract
        # ``security-and-compliance.mdc`` §1 spells out (mask
        # locally before any third party sees the bytes).
        payload: dict[str, Any] = {
            "contents": contents,
            "tools": _tools_from_config(config),
        }
        return gate_call(
            source="genai",
            target=target,
            model=model,
            prompt_text=extract_gemini_prompt(contents),
            stream=stream,
            payload=payload,
            stub_factory=_stub_response,
            extract_usage=_extract_genai_usage,
            extract_output_signals=extract_google,
            # Multi-step waterfall: per-tool ``tool_call`` step rows for
            # every Gemini ``function_call`` the model returned. Same
            # rationale as the OpenAI / Anthropic patches - keeps the
            # dashboard timeline consistent across direct LLM providers
            # so an operator reading a Run never sees "this provider
            # collapses tools, that one doesn't".
            emit_tool_call_steps=True,
            forward=lambda: orig(
                self,
                *args,
                **{**kwargs, "contents": payload["contents"]},
            ),
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
        # See ``_wrap_sync`` for the rationale - top-level scalar
        # ``contents`` is immutable, so the sanitized value lives on
        # ``payload`` and the forward lambda mirrors it back into
        # the upstream SDK kwargs at call time.
        payload: dict[str, Any] = {
            "contents": contents,
            "tools": _tools_from_config(config),
        }
        return await async_gate_call(
            source="genai",
            target=target,
            model=model,
            prompt_text=extract_gemini_prompt(contents),
            stream=stream,
            payload=payload,
            stub_factory=_stub_response,
            extract_usage=_extract_genai_usage,
            extract_output_signals=extract_google,
            emit_tool_call_steps=True,
            forward=lambda: orig(
                self,
                *args,
                **{**kwargs, "contents": payload["contents"]},
            ),
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
        from google.genai.models import (  # type: ignore[import-not-found, import-untyped]
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
