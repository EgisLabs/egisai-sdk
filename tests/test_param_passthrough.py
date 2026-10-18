"""Parameter-passthrough contract for every direct-LLM provider patch.

The egisai SDK wraps the customer's own client. Its promise is that
EVERY request parameter the customer passes to
``client.chat.completions.create(...)`` (and the Anthropic / Gemini /
Responses equivalents) reaches the provider **unchanged** — same keys,
same values — unless a documented, conditional transformation applies.

This file pins that promise. Each test sends a fully-populated request
(every knob the providers expose set to a non-default value) through
the patched ``create`` with NO policies in scope, and asserts the
kwargs the underlying provider SDK actually received are byte-for-byte
the kwargs the caller sent.

It also pins the two *intentional* deviations so they can't silently
widen:

* OpenAI Chat Completions, ``stream=True`` — the patch injects
  ``stream_options={"include_usage": True}`` so streamed audit rows
  carry real token counts. It merges (never clobbers) and honours an
  explicit ``include_usage: False``.
* A ``sparse`` request (only ``model`` + ``messages``) must gain NO
  extra keys — this is the regression guard against anyone adding a
  default (e.g. a default ``max_tokens`` for Anthropic).

If a future change drops a field, renames one, or defaults one that
the caller omitted, the relevant assertion here fails.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

# ── Shared ──────────────────────────────────────────────────────────


def _init_sdk(app: str = "param-passthrough") -> None:
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app=app,
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )


def _text_response() -> Any:
    return types.SimpleNamespace(
        id="resp",
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="stop",
            )
        ],
        usage=types.SimpleNamespace(prompt_tokens=3, completion_tokens=1),
    )


# ── OpenAI Chat Completions ─────────────────────────────────────────

#: Every documented chat-completions knob set to a non-default value.
_OPENAI_CHAT_FULL: dict[str, Any] = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hello world"}],
    "temperature": 0.3,
    "top_p": 0.9,
    "max_tokens": 256,
    "max_completion_tokens": 256,
    "stop": ["\n\n"],
    "n": 2,
    "seed": 42,
    "presence_penalty": 0.5,
    "frequency_penalty": 0.25,
    "logit_bias": {"50256": -100},
    "logprobs": True,
    "top_logprobs": 5,
    "response_format": {"type": "json_object"},
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ],
    "tool_choice": "auto",
    "parallel_tool_calls": False,
    "reasoning_effort": "high",
    "verbosity": "low",
    "user": "end-user-1",
    "metadata": {"trace": "abc"},
    "store": True,
    "service_tier": "auto",
    "extra_body": {"custom_provider_field": 1},
    "extra_headers": {"OpenAI-Project": "proj_x"},
}


def _install_fake_openai() -> tuple[type, type, type, type]:
    fake = types.ModuleType("openai")
    res = types.ModuleType("openai.resources")
    chat = types.ModuleType("openai.resources.chat")
    completions = types.ModuleType("openai.resources.chat.completions")
    responses = types.ModuleType("openai.resources.responses")

    class Completions:
        _captured: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            type(self)._captured.append(kwargs)
            if kwargs.get("stream"):
                return []  # drained by _materialize_sync_stream
            return _text_response()

    class AsyncCompletions:
        _captured: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            type(self)._captured.append(kwargs)
            if kwargs.get("stream"):
                return []
            return _text_response()

    class Responses:
        _captured: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            type(self)._captured.append(kwargs)
            return types.SimpleNamespace(
                id="r",
                output=[],
                output_text="ok",
                usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    class AsyncResponses:
        _captured: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            type(self)._captured.append(kwargs)
            return types.SimpleNamespace(
                id="r",
                output=[],
                output_text="ok",
                usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
            )

    completions.Completions = Completions
    completions.AsyncCompletions = AsyncCompletions
    responses.Responses = Responses
    responses.AsyncResponses = AsyncResponses
    sys.modules.update(
        {
            "openai": fake,
            "openai.resources": res,
            "openai.resources.chat": chat,
            "openai.resources.chat.completions": completions,
            "openai.resources.responses": responses,
        }
    )
    return Completions, AsyncCompletions, Responses, AsyncResponses


@pytest.fixture
def openai_patched(fake_backend: Any) -> Iterator[tuple[type, type, type, type]]:
    _init_sdk()
    classes = _install_fake_openai()
    for cls in classes:
        cls._captured = []
    from egisai._patches import openai as patch

    assert patch.apply() is True
    yield classes
    for mod in (
        "openai",
        "openai.resources",
        "openai.resources.chat",
        "openai.resources.chat.completions",
        "openai.resources.responses",
    ):
        sys.modules.pop(mod, None)


def test_openai_chat_full_params_forwarded_unchanged(openai_patched: Any) -> None:
    Completions, _, _, _ = openai_patched
    Completions().create(**dict(_OPENAI_CHAT_FULL))
    assert Completions._captured, "provider create() was never called"
    assert Completions._captured[-1] == _OPENAI_CHAT_FULL


def test_openai_chat_async_full_params_forwarded_unchanged(
    openai_patched: Any,
) -> None:
    _, AsyncCompletions, _, _ = openai_patched

    async def _go() -> None:
        await AsyncCompletions().create(**dict(_OPENAI_CHAT_FULL))

    asyncio.run(_go())
    assert AsyncCompletions._captured[-1] == _OPENAI_CHAT_FULL


def test_openai_chat_sparse_request_gains_no_defaults(openai_patched: Any) -> None:
    """Only ``model`` + ``messages`` sent → nothing else appears.

    This is the guard against anyone adding a default param (a
    ``max_tokens``, a ``temperature``, …) on the SDK side.
    """
    Completions, _, _, _ = openai_patched
    sent = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
    Completions().create(**dict(sent))
    assert Completions._captured[-1] == sent


def test_openai_chat_stream_injects_only_include_usage(openai_patched: Any) -> None:
    """``stream=True`` → the ONLY added key is
    ``stream_options.include_usage=True``; everything else is verbatim."""
    Completions, _, _, _ = openai_patched
    sent = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "temperature": 0.7,
    }
    Completions().create(**dict(sent))
    got = Completions._captured[-1]
    assert got == {**sent, "stream_options": {"include_usage": True}}


def test_openai_chat_stream_respects_explicit_include_usage_false(
    openai_patched: Any,
) -> None:
    """An explicit ``include_usage: False`` is never overridden."""
    Completions, _, _, _ = openai_patched
    sent = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "stream_options": {"include_usage": False},
    }
    Completions().create(**dict(sent))
    assert Completions._captured[-1] == sent


def test_openai_responses_full_params_forwarded_unchanged(
    openai_patched: Any,
) -> None:
    _, _, Responses, _ = openai_patched
    sent = {
        "model": "gpt-4o",
        "input": "hello",
        "temperature": 0.3,
        "top_p": 0.9,
        "max_output_tokens": 512,
        "reasoning": {"effort": "high"},
        "tools": [{"type": "function", "name": "f", "parameters": {}}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "metadata": {"k": "v"},
        "store": False,
        "service_tier": "auto",
        "extra_headers": {"OpenAI-Organization": "org_x"},
    }
    Responses().create(**dict(sent))
    assert Responses._captured[-1] == sent


# ── Anthropic Messages ──────────────────────────────────────────────

_ANTHROPIC_FULL: dict[str, Any] = {
    "model": "claude-3-5-sonnet-20241022",
    "messages": [{"role": "user", "content": "hello"}],
    "system": "You are helpful.",
    "max_tokens": 512,
    "temperature": 0.2,
    "top_p": 0.8,
    "top_k": 40,
    "stop_sequences": ["STOP"],
    "metadata": {"user_id": "u-1"},
    "tools": [
        {"name": "t", "input_schema": {"type": "object", "properties": {}}}
    ],
    "tool_choice": {"type": "auto"},
    "thinking": {"type": "enabled", "budget_tokens": 1024},
    "service_tier": "auto",
    "extra_headers": {"anthropic-beta": "token-counting-2024-11-01"},
}


def _install_fake_anthropic() -> tuple[type, type]:
    fake = types.ModuleType("anthropic")
    res = types.ModuleType("anthropic.resources")
    messages_mod = types.ModuleType("anthropic.resources.messages")

    def _msg() -> Any:
        return types.SimpleNamespace(
            id="m",
            content=[types.SimpleNamespace(type="text", text="ok")],
            usage=types.SimpleNamespace(input_tokens=2, output_tokens=1),
        )

    class Messages:
        _captured: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            type(self)._captured.append(kwargs)
            return _msg()

    class AsyncMessages:
        _captured: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            type(self)._captured.append(kwargs)
            return _msg()

    messages_mod.Messages = Messages
    messages_mod.AsyncMessages = AsyncMessages
    sys.modules.update(
        {
            "anthropic": fake,
            "anthropic.resources": res,
            "anthropic.resources.messages": messages_mod,
        }
    )
    return Messages, AsyncMessages


@pytest.fixture
def anthropic_patched(fake_backend: Any) -> Iterator[tuple[type, type]]:
    _init_sdk()
    classes = _install_fake_anthropic()
    for cls in classes:
        cls._captured = []
    from egisai._patches import anthropic as patch

    assert patch.apply() is True
    yield classes
    for mod in ("anthropic", "anthropic.resources", "anthropic.resources.messages"):
        sys.modules.pop(mod, None)


def test_anthropic_full_params_forwarded_unchanged(anthropic_patched: Any) -> None:
    Messages, _ = anthropic_patched
    Messages().create(**dict(_ANTHROPIC_FULL))
    assert Messages._captured[-1] == _ANTHROPIC_FULL


def test_anthropic_async_full_params_forwarded_unchanged(
    anthropic_patched: Any,
) -> None:
    _, AsyncMessages = anthropic_patched

    async def _go() -> None:
        await AsyncMessages().create(**dict(_ANTHROPIC_FULL))

    asyncio.run(_go())
    assert AsyncMessages._captured[-1] == _ANTHROPIC_FULL


def test_anthropic_sparse_request_gains_no_max_tokens_default(
    anthropic_patched: Any,
) -> None:
    """Guard: the SDK must NOT inject a default ``max_tokens`` for
    Anthropic (a tempting "helpful" default — Anthropic requires it)."""
    Messages, _ = anthropic_patched
    sent = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
    }
    Messages().create(**dict(sent))
    assert Messages._captured[-1] == sent
    # And a request that omits max_tokens entirely stays omitted.
    no_max = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "hi"}],
    }
    Messages().create(**dict(no_max))
    assert Messages._captured[-1] == no_max


# ── Google GenAI ────────────────────────────────────────────────────


def _install_fake_genai() -> tuple[type, type]:
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    models_mod = types.ModuleType("google.genai.models")

    def _resp() -> Any:
        return types.SimpleNamespace(
            text="ok",
            candidates=[],
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=1, candidates_token_count=1
            ),
            function_calls=[],
        )

    class Models:
        _captured: list[dict[str, Any]] = []

        def generate_content(self, **kwargs: Any) -> Any:
            type(self)._captured.append(kwargs)
            return _resp()

    class AsyncModels:
        _captured: list[dict[str, Any]] = []

        async def generate_content(self, **kwargs: Any) -> Any:
            type(self)._captured.append(kwargs)
            return _resp()

    models_mod.Models = Models
    models_mod.AsyncModels = AsyncModels
    genai.models = models_mod
    google.genai = genai
    sys.modules.update(
        {
            "google": google,
            "google.genai": genai,
            "google.genai.models": models_mod,
        }
    )
    return Models, AsyncModels


@pytest.fixture
def genai_patched(fake_backend: Any) -> Iterator[tuple[type, type]]:
    _init_sdk()
    classes = _install_fake_genai()
    for cls in classes:
        cls._captured = []
    from egisai._patches import genai as patch

    assert patch.apply() is True
    yield classes
    for mod in ("google.genai.models", "google.genai", "google"):
        sys.modules.pop(mod, None)


def test_genai_full_params_forwarded_unchanged(genai_patched: Any) -> None:
    Models, _ = genai_patched
    config = types.SimpleNamespace(
        temperature=0.4,
        top_p=0.9,
        top_k=20,
        max_output_tokens=800,
        stop_sequences=["X"],
        response_mime_type="application/json",
        tools=None,
    )
    sent = {"model": "gemini-2.0-flash", "contents": "hello", "config": config}
    Models().generate_content(**dict(sent))
    got = Models._captured[-1]
    assert got["model"] == "gemini-2.0-flash"
    assert got["contents"] == "hello"
    # ``config`` object is forwarded unchanged (same instance, so every
    # knob on it rides along verbatim).
    assert got["config"] is config


def test_genai_async_full_params_forwarded_unchanged(genai_patched: Any) -> None:
    _, AsyncModels = genai_patched
    config = types.SimpleNamespace(temperature=0.4, top_k=20, tools=None)
    sent = {"model": "gemini-2.0-flash", "contents": "hi", "config": config}

    async def _go() -> None:
        await AsyncModels().generate_content(**dict(sent))

    asyncio.run(_go())
    got = AsyncModels._captured[-1]
    assert got["model"] == "gemini-2.0-flash"
    assert got["contents"] == "hi"
    assert got["config"] is config
