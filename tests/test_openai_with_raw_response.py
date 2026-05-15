"""Regression contract: ``client.chat.completions.with_raw_response.create``
returns a ``LegacyAPIResponse``-shaped object on block AND allow paths.

``langchain-openai>=1.2`` (which ``langgraph``, ``langchain.agents.
create_agent``, and the classic ``AgentExecutor`` all sit on top of)
uses the raw-response code path in ``_generate``::

    raw_response = self.client.with_raw_response.create(**payload)
    response = raw_response.parse()
    base_generation_info = {"headers": dict(raw_response.headers)}

Before v0.25.9 the egisai OpenAI patch returned the synthesised
``ChatCompletion`` stub directly when an input policy fired, which
crashed the very next statement with::

    AttributeError: 'ChatCompletion' object has no attribute 'parse'

This file pins the contract that the with_raw_response path keeps
working on every verdict (allow / sanitize / block) and through
both the Chat Completions and Responses APIs, sync and async.
"""

from __future__ import annotations

import asyncio
import datetime
from typing import Any

import pytest

from egisai._patches.openai import (
    _extract_chat_signals_raw,
    _extract_chat_usage_raw,
    _extract_openai_responses_raw,
    _extract_responses_usage_raw,
    _is_raw_response_call,
    _parse_if_raw,
    _RawResponseStub,
    _stub_chat_completion_raw,
    _stub_response_raw,
)
from egisai.policy.engine import PolicyDecision


def _decision() -> PolicyDecision:
    return PolicyDecision.deny(
        reason_code="test_block",
        message="blocked by raw-response contract test",
        matched_policy="contract-test-policy",
    )


# ---------------------------------------------------------------------------
# Raw-mode detection
# ---------------------------------------------------------------------------


def test_is_raw_response_call_detects_marker_header() -> None:
    """``to_raw_response_wrapper`` injects ``X-Stainless-Raw-Response: true``
    into ``extra_headers``. Our sniff must read it back."""
    assert _is_raw_response_call(
        {"extra_headers": {"X-Stainless-Raw-Response": "true"}}
    ) is True
    # Case-insensitive value: be defensive about future SDK shifts.
    assert _is_raw_response_call(
        {"extra_headers": {"X-Stainless-Raw-Response": "TRUE"}}
    ) is True


def test_is_raw_response_call_negative_paths() -> None:
    """Absence of the marker, a different value, or a non-dict
    ``extra_headers`` must all return False — otherwise we'd
    flip every normal call into raw mode and break the standard
    extractors."""
    assert _is_raw_response_call({}) is False
    assert _is_raw_response_call({"extra_headers": None}) is False
    assert _is_raw_response_call({"extra_headers": {"X-Other": "true"}}) is False
    assert _is_raw_response_call(
        {"extra_headers": {"X-Stainless-Raw-Response": "false"}}
    ) is False


# ---------------------------------------------------------------------------
# LegacyAPIResponse-shaped stubs
# ---------------------------------------------------------------------------


def test_raw_chat_stub_satisfies_langchain_unpack() -> None:
    """Mirror of ``langchain_openai/chat_models/base.py::_generate`` lines
    1650-1665. The exact sequence the framework runs against the raw
    response — ``.parse()``, ``dict(.headers)``, ``hasattr(., 'http_response')``,
    ``.parse()`` again for caching idempotency — MUST all succeed on
    a synthesised block stub."""
    raw = _stub_chat_completion_raw(_decision(), trace_id="t" * 16, model="gpt-4o")

    # ``.parse()`` returns the ChatCompletion stub.
    response = raw.parse()
    assert response.choices[0].message.content.startswith("[POLICY BLOCK]")
    assert response.model == "gpt-4o"

    # ``dict(raw.headers)`` — langchain reads this into
    # ``base_generation_info`` for the chat result. Empty is fine,
    # the framework just stores it.
    assert isinstance(raw.headers, dict)
    assert dict(raw.headers) == {}

    # ``hasattr(raw_response, 'http_response')`` is the error-path
    # guard in langchain (line 1657). Truthy attribute presence is
    # the contract; the value itself can be ``None`` because we
    # never actually made an HTTP request on the block path.
    assert hasattr(raw, "http_response")

    # ``raw_response.parse()`` is also called from
    # ``_stream`` when ``include_response_headers=True``. Second
    # call must return the same parsed object so frameworks that
    # call ``.parse()`` multiple times don't end up with two
    # different ChatCompletion instances.
    assert raw.parse() is response


def test_raw_chat_stub_exposes_legacy_api_response_surface() -> None:
    """The ``LegacyAPIResponse`` surface langchain (and any other
    upstream framework that walks the raw object) reads goes beyond
    ``.parse()`` / ``.headers``. We model the documented surface so
    a future framework upgrade that adds a new accessor (request id,
    elapsed time) doesn't surface as an AttributeError on the next
    blocked turn at a customer."""
    raw = _stub_chat_completion_raw(_decision(), trace_id="t" * 16, model="gpt-4o")

    assert raw.status_code == 200
    assert raw.request_id is None
    assert raw.content == b""
    assert raw.text == ""
    assert raw.http_response is None
    assert isinstance(raw.elapsed, datetime.timedelta)
    assert raw.retries_taken == 0


def test_raw_responses_stub_satisfies_langchain_unpack() -> None:
    """The Responses API sibling of the Chat-Completions contract."""
    raw = _stub_response_raw(_decision(), trace_id="t" * 16, model="gpt-5")

    response = raw.parse()
    assert response.output_text.startswith("[POLICY BLOCK]")
    assert response.model == "gpt-5"

    assert hasattr(raw, "headers")
    assert hasattr(raw, "http_response")
    assert raw.parse() is response


# ---------------------------------------------------------------------------
# Extractor unwrapping
# ---------------------------------------------------------------------------


def test_parse_if_raw_passes_through_non_raw_responses() -> None:
    """Plain ``ChatCompletion`` (the non-raw path's response) must
    NOT be wrapped or re-parsed — that's the existing extractor
    contract and would double-charge the cache."""

    class _PlainChatCompletion:
        usage = type("U", (), {"prompt_tokens": 7, "completion_tokens": 11})()
        choices: list = []

    plain = _PlainChatCompletion()
    assert _parse_if_raw(plain) is plain
    assert _parse_if_raw(None) is None


def test_parse_if_raw_unwraps_raw_responses() -> None:
    """A ``LegacyAPIResponse``-shaped object is dereferenced via
    ``.parse()`` so the underlying ``ChatCompletion`` is what the
    gate's extractor sees."""

    parsed = object()

    class _FakeLegacy:
        def parse(self) -> Any:
            return parsed

        @property
        def http_response(self) -> Any:
            return None

    assert _parse_if_raw(_FakeLegacy()) is parsed


def test_parse_if_raw_fails_open_when_parse_raises() -> None:
    """If ``.parse()`` raises (rare — a malformed response or a
    custom subclass), we degrade to ``None`` instead of propagating
    so the extractor sees an empty response and stamps zero usage."""

    class _BadLegacy:
        def parse(self) -> Any:
            raise RuntimeError("parse failed")

        @property
        def http_response(self) -> Any:
            return None

    assert _parse_if_raw(_BadLegacy()) is None


def test_extract_chat_usage_raw_walks_through_legacy_wrapper() -> None:
    """The audit row's ``tokens_in`` / ``tokens_out`` must be
    populated for raw-mode allow turns (this was the second silent
    failure pre-0.25.9 — the response carried real usage but our
    extractor saw a ``LegacyAPIResponse`` instead of the parsed
    ChatCompletion and dropped the counts)."""

    parsed = type(
        "ChatCompletion",
        (),
        {
            "usage": type(
                "Usage",
                (),
                {"prompt_tokens": 137, "completion_tokens": 64},
            )()
        },
    )()

    class _FakeLegacy:
        def parse(self) -> Any:
            return parsed

        @property
        def http_response(self) -> Any:
            return None

    usage = _extract_chat_usage_raw(_FakeLegacy())
    assert usage["tokens_in"] == 137
    assert usage["tokens_out"] == 64


def test_extract_responses_usage_raw_walks_through_legacy_wrapper() -> None:
    """Responses-API sibling of the chat-usage extractor contract."""

    parsed = type(
        "Response",
        (),
        {
            "usage": type(
                "Usage",
                (),
                {"input_tokens": 211, "output_tokens": 92},
            )()
        },
    )()

    class _FakeLegacy:
        def parse(self) -> Any:
            return parsed

        @property
        def http_response(self) -> Any:
            return None

    usage = _extract_responses_usage_raw(_FakeLegacy())
    assert usage["tokens_in"] == 211
    assert usage["tokens_out"] == 92


def test_extract_chat_signals_raw_walks_through_legacy_wrapper() -> None:
    """Output-side policies (``deny_tool_call``, output ``semantic_guard``)
    walk the extracted ``tool_calls`` list. The raw-mode extractor
    must surface those even when the response is wrapped in a
    LegacyAPIResponse so blocks fire as they would on the non-raw
    path."""
    from types import SimpleNamespace

    parsed = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="hello",
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="lookup_customer",
                                arguments='{"q": "foo"}',
                            )
                        )
                    ],
                )
            )
        ]
    )

    class _FakeLegacy:
        def parse(self) -> Any:
            return parsed

        @property
        def http_response(self) -> Any:
            return None

    text, tool_names, tool_calls, mcp_targets = _extract_chat_signals_raw(
        _FakeLegacy(), payload={}
    )
    assert text == "hello"
    assert tool_calls == [{"name": "lookup_customer", "arguments": '{"q": "foo"}'}]


def test_extract_openai_responses_raw_walks_through_legacy_wrapper() -> None:
    """Responses-API sibling: ``response.output[].content[].text`` and
    ``function_call`` walks must survive the wrapper unwrap."""
    from types import SimpleNamespace

    parsed = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                role="assistant",
                content=[SimpleNamespace(type="output_text", text="hi")],
            )
        ]
    )

    class _FakeLegacy:
        def parse(self) -> Any:
            return parsed

        @property
        def http_response(self) -> Any:
            return None

    text, _, _, _ = _extract_openai_responses_raw(_FakeLegacy(), payload={})
    assert text == "hi"


# ---------------------------------------------------------------------------
# End-to-end: the actual langchain-openai unpack on a blocked turn
# ---------------------------------------------------------------------------


def test_langchain_openai_generate_unpack_does_not_crash() -> None:
    """Reproduce the exact failing 4-line sequence from
    ``langchain_openai/chat_models/base.py::_generate`` (lines
    1650-1665) against the raw block stub. The
    ``AttributeError: 'ChatCompletion' object has no attribute 'parse'``
    that crashed every blocked langgraph / langchain run before
    v0.25.9 must NOT raise here."""
    raw_response = _stub_chat_completion_raw(
        _decision(), trace_id="t" * 16, model="gpt-4o"
    )

    response = raw_response.parse()
    base_generation_info: dict[str, Any] = {}
    if hasattr(raw_response, "headers"):
        base_generation_info["headers"] = dict(raw_response.headers)

    assert response.choices[0].message.content.startswith("[POLICY BLOCK]")
    assert base_generation_info["headers"] == {}


def test_raw_response_stub_works_in_async_context() -> None:
    """``langchain-openai``'s ``_agenerate`` runs the same unpack as
    ``_generate``. Pin that the raw stub is usable in an async
    function (no await on ``.parse()``, no coroutine surprises)."""

    async def _async_consume() -> Any:
        raw = _stub_chat_completion_raw(_decision(), trace_id="t" * 16, model="gpt-4o")
        return raw.parse()

    response = asyncio.run(_async_consume())
    assert response.choices[0].message.content.startswith("[POLICY BLOCK]")


# ---------------------------------------------------------------------------
# Raw-stub wrapper is independent — wrapping a non-stub object too
# ---------------------------------------------------------------------------


def test_raw_response_stub_wraps_arbitrary_parsed_value() -> None:
    """The ``_RawResponseStub`` shouldn't care what's inside — any
    parsed value should round-trip through ``.parse()``. This makes
    it usable as a generic raw-response shim should a future framework
    need one (e.g. for the Responses API or a custom output type)."""
    sentinel = object()
    raw = _RawResponseStub(sentinel)
    assert raw.parse() is sentinel
    assert raw.parse(to=None) is sentinel


def test_raw_response_stub_passes_headers_through_constructor() -> None:
    """Future-proofing: callers should be able to inject deterministic
    headers (e.g. for tracing breadcrumbs) without subclassing."""
    raw = _RawResponseStub("body", headers={"x-egis-trace": "abc"})
    assert raw.headers == {"x-egis-trace": "abc"}
    # The internal copy must be defensive — mutating the dict
    # passed in shouldn't leak into the stub.
    src = {"x-egis-trace": "abc"}
    raw2 = _RawResponseStub("body", headers=src)
    src["x-egis-trace"] = "changed"
    assert raw2.headers["x-egis-trace"] == "abc"


# ---------------------------------------------------------------------------
# Pydantic instance contract preserved
# ---------------------------------------------------------------------------


def test_raw_chat_stub_parses_to_real_chatcompletion_when_openai_installed() -> None:
    """The unwrapped block stub must still be a real upstream
    ``ChatCompletion`` Pydantic instance — frameworks that
    isinstance-validate the response (autogen's ``model_dump()``
    plus downstream Pydantic dataclasses in the agents-SDK)
    depend on it. Pinning here keeps the raw-mode path from
    quietly degrading to a duck-typed namespace."""
    try:
        from openai.types.chat import ChatCompletion
    except ImportError:
        pytest.skip("openai not installed in this env")

    raw = _stub_chat_completion_raw(_decision(), trace_id="t" * 16, model="gpt-4o")
    assert isinstance(raw.parse(), ChatCompletion)
