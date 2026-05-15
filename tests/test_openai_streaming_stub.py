"""Streaming-shape contract for the OpenAI patch's block stub.

Frameworks like ``langchain-openai``'s ``_stream`` consume the
OpenAI ``client.chat.completions.create(...)`` return value via the
``Stream``-protocol surface:

    response = self.client.create(**payload)
    with response as response:
        for chunk in response:
            chunk = chunk.model_dump()
            ...

The classic ``langchain.agents.AgentExecutor`` (and its 1.x
back-compat sibling ``langchain_classic.agents.AgentExecutor``)
forces this code path on every invocation because the agent's
``stream_runnable=True`` default means the inner LLM is always
called via ``Runnable.stream(...)``, which sets ``stream=True`` on
the OpenAI client. Without the streaming-aware path here, a
``on_block="stub"`` decision returns a bare ``SimpleNamespace`` —
``with response:`` raises ``TypeError: 'types.SimpleNamespace'
object does not support the context manager protocol`` before the
caller ever sees the policy verdict.

Tests:

* :func:`test_stream_stub_supports_context_manager_and_iteration`
  — block-stub on the streaming path satisfies both ``with`` and
  ``for`` and the chunks expose ``.model_dump()``.
* :func:`test_stream_stub_aggregates_usage_on_response_surface`
  — the aggregated response surface carries zero token usage so
  ``_extract_chat_usage`` records ``tokens_in=0`` /
  ``tokens_out=0`` (matching the non-streaming block contract).
* :func:`test_stream_replay_captures_real_stream_usage`
  — the materialise-then-replay wrapper used on the allow path
  preserves the final-chunk ``usage`` from a real Stream so the
  audit row reports real token counts instead of zeros.
* :func:`test_stream_replay_aggregates_tool_calls_across_deltas`
  — incremental tool-call delta fragments are reassembled into a
  single ``choices[0].message.tool_calls[*]`` entry so the gate's
  per-tool waterfall fires for streaming responses too.
"""

from __future__ import annotations

import pytest

from egisai._output_signals import extract_openai_chat
from egisai._patches.openai import (
    _aggregate_chat_stream,
    _build_chat_chunk,
    _ensure_stream_usage,
    _extract_chat_usage,
    _materialize_sync_stream,
    _StreamReplay,
    _stub_chat_stream,
)
from egisai.policy import PolicyDecision

openai = pytest.importorskip("openai")


def _decision() -> PolicyDecision:
    return PolicyDecision.deny(
        reason_code="semantic_blocked",
        message="blocked by stream contract test",
        matched_policy="Block refund issuing",
    )


def test_stream_stub_supports_context_manager_and_iteration() -> None:
    """The block-stub must look like a ``Stream`` to the framework.

    ``langchain_openai.ChatOpenAI._stream`` does
    ``with response as response: for chunk in response: ...`` —
    failing the ``with`` step was the v0.25.4 regression that
    motivated this fix. Each yielded chunk must also expose
    ``.model_dump()`` because langchain's chunk-to-generation
    converter calls it unconditionally for non-dict chunks.
    """
    response = _stub_chat_stream(_decision(), trace_id="t" * 16, model="gpt-4o")
    assert isinstance(response, _StreamReplay)

    # Frameworks call ``with response as r: for chunk in r: ...``.
    # Pin both halves of the protocol.
    seen_blurb = False
    saw_stop = False
    with response as inner:
        assert inner is response
        for chunk in inner:
            dumped = chunk.model_dump()
            delta = dumped["choices"][0].get("delta") or {}
            if delta.get("content"):
                seen_blurb = "POLICY BLOCK" in delta["content"]
            if dumped["choices"][0].get("finish_reason") == "stop":
                saw_stop = True
    assert seen_blurb, "block-stub stream must surface the blurb in a delta"
    assert saw_stop, "block-stub stream must terminate with finish_reason='stop'"


def test_stream_stub_aggregates_usage_on_response_surface() -> None:
    """The replay's aggregated ``usage`` lets ``_extract_chat_usage``
    record zero token counts for the block step.

    Pre-fix the gate ran ``_extract_chat_usage`` on a stream whose
    ``.usage`` had no defined accessor → ``tokens_in=None``,
    ``cost_usd=None``. The audit row then failed every
    "block-only run carries zero tokens" assertion in the
    per-framework battery.
    """
    response = _stub_chat_stream(_decision(), trace_id="t" * 16, model="gpt-4o")

    assert response.usage is not None
    assert response.usage.prompt_tokens == 0
    assert response.usage.completion_tokens == 0
    assert response.usage.total_tokens == 0
    # Real upstream Pydantic types — see the rationale in
    # ``_stub_chat_completion``'s docstring.
    from openai.types.completion_usage import CompletionUsage

    assert isinstance(response.usage, CompletionUsage)

    extracted = _extract_chat_usage(response)
    assert extracted == {"tokens_in": 0, "tokens_out": 0}


def test_stream_replay_captures_real_stream_usage() -> None:
    """The allow path materialises the upstream Stream into a replay
    that preserves the final-chunk ``usage``.

    ``langchain-openai`` enables ``stream_options.include_usage=True``
    by default when the user hits the upstream OpenAI base URL, so
    the upstream Stream emits a usage-only chunk at the tail. The
    replay must surface that ``usage`` on its aggregated response
    surface so the gate's ``_extract_chat_usage`` returns real
    counts — anything else would leave ``runs.tokens_in/out`` as 0
    on the dashboard for every streamed turn.
    """
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import (
        Choice,
        ChoiceDelta,
    )
    from openai.types.completion_usage import CompletionUsage

    def fake_stream():
        yield ChatCompletionChunk(
            id="r1",
            object="chat.completion.chunk",
            created=0,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(role="assistant", content="Hi "),
                    finish_reason=None,
                )
            ],
        )
        yield ChatCompletionChunk(
            id="r1",
            object="chat.completion.chunk",
            created=0,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(content="there!"),
                    finish_reason="stop",
                )
            ],
        )
        yield ChatCompletionChunk(
            id="r1",
            object="chat.completion.chunk",
            created=0,
            model="gpt-4o",
            choices=[],
            usage=CompletionUsage(
                prompt_tokens=11,
                completion_tokens=3,
                total_tokens=14,
            ),
        )

    replay = _materialize_sync_stream(fake_stream())

    usage = _extract_chat_usage(replay)
    assert usage == {"tokens_in": 11, "tokens_out": 3}

    # The replay must still be iterable for the framework consumer
    # (langchain re-iterates inside ``with response as response``).
    chunks = list(replay)
    assert len(chunks) == 3
    # Per-chunk ``.model_dump()`` survives — the framework calls it
    # on every non-dict chunk.
    assert chunks[0].model_dump()["choices"][0]["delta"]["content"] == "Hi "

    text, _tools, _calls, _mcp = extract_openai_chat(replay, payload={})
    assert text == "Hi there!"


def test_stream_replay_aggregates_tool_calls_across_deltas() -> None:
    """Streaming tool-call deltas reassemble into a single
    ``choices[0].message.tool_calls`` entry.

    Upstream emits tool calls as a sequence of delta fragments —
    the first carries ``id``/``name`` plus an opening JSON brace,
    subsequent deltas carry the rest of the argument string. The
    gate's per-tool waterfall (``_dispatch_per_tool_steps``) reads
    ``response.choices[*].message.tool_calls[*].function.name``
    after the stream is consumed, so the replay's aggregated
    surface must collapse those fragments into a single coherent
    entry.
    """
    from openai.types.chat import ChatCompletionChunk
    from openai.types.chat.chat_completion_chunk import (
        Choice,
        ChoiceDelta,
        ChoiceDeltaToolCall,
        ChoiceDeltaToolCallFunction,
    )

    def fake_stream():
        yield ChatCompletionChunk(
            id="r1",
            object="chat.completion.chunk",
            created=0,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        role="assistant",
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                id="call_a1",
                                type="function",
                                function=ChoiceDeltaToolCallFunction(
                                    name="lookup_customer",
                                    arguments="",
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        )
        yield ChatCompletionChunk(
            id="r1",
            object="chat.completion.chunk",
            created=0,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                function=ChoiceDeltaToolCallFunction(
                                    arguments='{"name',
                                ),
                            )
                        ],
                    ),
                    finish_reason=None,
                )
            ],
        )
        yield ChatCompletionChunk(
            id="r1",
            object="chat.completion.chunk",
            created=0,
            model="gpt-4o",
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(
                        tool_calls=[
                            ChoiceDeltaToolCall(
                                index=0,
                                function=ChoiceDeltaToolCallFunction(
                                    arguments='":"Maria"}',
                                ),
                            )
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
        )

    replay = _materialize_sync_stream(fake_stream())

    _text, _tools, tool_calls, _mcp = extract_openai_chat(replay, payload={})
    assert tool_calls == [
        {"name": "lookup_customer", "arguments": '{"name":"Maria"}'}
    ]


def test_aggregate_handles_simple_namespace_chunks() -> None:
    """The aggregator must work on the ``SimpleNamespace`` fallback
    chunks too — they're what the stub falls back to when the
    upstream ``openai`` install is too old to provide
    ``ChatCompletionChunk``. The fallback path is hit by
    ``_build_chat_chunk`` whenever Pydantic construction raises.
    """
    chunks = [
        _build_chat_chunk(
            chunk_id="c1",
            model="gpt-4o",
            delta_kwargs={"role": "assistant", "content": "first"},
            finish_reason=None,
        ),
        _build_chat_chunk(
            chunk_id="c1",
            model="gpt-4o",
            delta_kwargs={"content": " second"},
            finish_reason="stop",
        ),
    ]
    aggregated = _aggregate_chat_stream(chunks)
    assert aggregated.choices[0].message.content == "first second"
    assert aggregated.choices[0].finish_reason == "stop"


def test_stream_stub_close_is_idempotent() -> None:
    """The replay's ``close()`` and double-``__exit__`` must be
    safe to call repeatedly — frameworks sometimes call ``close``
    explicitly after ``__exit__`` for paranoid resource cleanup."""
    response = _stub_chat_stream(_decision(), trace_id="t" * 16, model="gpt-4o")
    with response:
        list(response)
    response.close()
    response.close()
    # Re-iterable after close — frameworks that hold a reference
    # and re-read the stream don't crash.
    assert sum(1 for _ in response) >= 1


def test_ensure_stream_usage_injects_include_usage_when_absent() -> None:
    """Streaming callers that don't set ``stream_options`` get the
    flag added so the upstream emits a final usage chunk.

    LlamaIndex's ``OpenAI._astream_chat`` /
    ``OpenAI._stream_chat`` pass ``stream=True`` but never set
    ``stream_options``; without the flag, the upstream Stream
    emits no usage chunk and our materialised replay's
    ``response.usage`` is ``None``. The audit row would then carry
    ``tokens_in=None`` / ``tokens_out=None`` for every streamed
    turn — which is what the agentic harness flagged
    pre-fix. ``_ensure_stream_usage`` closes that gap.
    """
    kwargs: dict = {"model": "gpt-4o", "stream": True}
    _ensure_stream_usage(kwargs)
    assert kwargs["stream_options"] == {"include_usage": True}


def test_ensure_stream_usage_merges_with_existing_options() -> None:
    """Existing ``stream_options`` keys survive the merge."""
    kwargs: dict = {
        "model": "gpt-4o",
        "stream": True,
        "stream_options": {"other_flag": True},
    }
    _ensure_stream_usage(kwargs)
    assert kwargs["stream_options"] == {
        "other_flag": True,
        "include_usage": True,
    }


def test_ensure_stream_usage_honours_explicit_false() -> None:
    """If the caller explicitly opted OUT of usage, we don't
    override their choice — the audit row will record zeros and
    that's the caller's call to make."""
    kwargs: dict = {
        "model": "gpt-4o",
        "stream": True,
        "stream_options": {"include_usage": False},
    }
    _ensure_stream_usage(kwargs)
    assert kwargs["stream_options"] == {"include_usage": False}


def test_stream_stub_supports_async_iteration_and_async_context() -> None:
    """``llama-index-llms-openai`` consumes streaming responses via
    ``async with stream as r: async for chunk in r: ...`` (see
    ``OpenAI._astream_chat``). The replay must satisfy both halves
    of the async protocol — without it, the v0.40.x regression
    raised ``TypeError: 'async for' requires an object with
    __aiter__ method, got _StreamReplay`` and broke every
    LlamaIndex-driven OpenAI streaming call.
    """
    import asyncio

    response = _stub_chat_stream(_decision(), trace_id="t" * 16, model="gpt-4o")

    async def drive() -> tuple[bool, bool, int]:
        seen_blurb = False
        saw_stop = False
        count = 0
        async with response as inner:
            assert inner is response
            async for chunk in inner:
                count += 1
                dumped = chunk.model_dump()
                delta = dumped["choices"][0].get("delta") or {}
                if delta.get("content") and "POLICY BLOCK" in delta["content"]:
                    seen_blurb = True
                if dumped["choices"][0].get("finish_reason") == "stop":
                    saw_stop = True
        await response.aclose()
        return seen_blurb, saw_stop, count

    seen_blurb, saw_stop, count = asyncio.run(drive())
    assert count > 0
    assert seen_blurb, "async path must surface the blurb in a delta"
    assert saw_stop, "async path must terminate with finish_reason='stop'"

    async def drive_again() -> int:
        n = 0
        async for _ in response:
            n += 1
        return n

    assert asyncio.run(drive_again()) > 0, (
        "re-iteration after exhaustion must yield the same chunks again"
    )
