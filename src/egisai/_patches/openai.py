"""OpenAI Python SDK patcher.

Patches Chat Completions and Responses (sync + async) plus the
legacy v0 ``openai.ChatCompletion.create`` path.

Streaming notes
~~~~~~~~~~~~~~~

When ``stream=True`` is passed, the upstream OpenAI SDK returns a
``Stream`` (sync) / ``AsyncStream`` (async) object that is BOTH a
context manager AND an iterable of ``ChatCompletionChunk`` objects.
Frameworks like ``langchain-openai``'s ``_stream`` and the classic
``langchain.agents.AgentExecutor`` (whose ``stream_runnable=True`` is
the default) consume the response exactly that way:

    response = self.client.create(**payload)
    with response as response:
        for chunk in response:
            ...

So when we synthesize a block-stub OR forward to the real API on
the streaming path, our return value MUST also satisfy
``__enter__/__exit__`` and ``__iter__``. A bare ``SimpleNamespace``
fails the ``with response:`` step (``TypeError: 'types.SimpleNamespace'
object does not support the context manager protocol``); a real
``Stream`` works for ``with`` and ``for`` but its token usage isn't
visible to ``_stamp_usage`` / ``extract_openai_chat`` until the
caller has finished iterating — by which point the gate has long
since dispatched the audit row with zeros.

The fix is :class:`_StreamReplay`, a small adapter that:

* exposes ``__enter__`` / ``__exit__`` / ``__iter__`` so the
  framework consumer keeps working (block stub OR allowed call);
* materializes the chunks at gate time so ``extract_chat_usage``
  and ``extract_openai_chat`` see a fully-shaped response (the
  aggregated ``choices[0].message`` and the final ``usage``
  fields) and the audit row carries real token counts instead of
  zeros.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from egisai import _gateway
from egisai._evaluator import extract_prompt_text
from egisai._output_signals import extract_openai_chat, extract_openai_responses
from egisai._patches import has_module
from egisai._patches._common import async_gate_call, gate_call
from egisai.policy import PolicyDecision

LOGGER = logging.getLogger("egisai.patches.openai")


# ``openai._constants.RAW_RESPONSE_HEADER`` — the marker the upstream
# ``to_raw_response_wrapper`` injects into ``extra_headers`` when a
# caller reaches the gate via ``client.chat.completions.with_raw_response``
# (langchain-openai's ``_generate`` does exactly this on every
# non-streaming turn). Hard-coded so the module stays importable when
# ``openai`` isn't installed; the patch itself is import-gated on
# ``has_module("openai")`` so on real installs the constant is always
# in sync with the upstream value.
_RAW_RESPONSE_HEADER = "X-Stainless-Raw-Response"


class _RawResponseStub:
    """``LegacyAPIResponse``-shaped wrapper around a synthesised stub.

    When a framework calls ``client.chat.completions.with_raw_response.create(...)``
    upstream returns a ``LegacyAPIResponse`` whose ``.parse()`` method
    deserialises the JSON into a real ``ChatCompletion``. langchain-openai
    1.2+'s ``_generate`` does exactly that on every non-streaming
    turn::

        raw_response = self.client.with_raw_response.create(**payload)
        response = raw_response.parse()

    Before this wrapper existed, an input-side block fired our
    ``_stub_chat_completion`` and returned the ``ChatCompletion``
    *directly* into langchain's call site — which then crashed with
    ``AttributeError: 'ChatCompletion' object has no attribute
    'parse'`` because ``ChatCompletion`` is the *parsed* shape, not
    the raw wrapper. This class fills the gap so the
    ``raw_response.parse()`` step on the caller side keeps working
    on a blocked turn.

    We mirror the surface of ``openai._legacy_response.LegacyAPIResponse``
    that callers actually read (``parse``, ``headers``,
    ``http_response``, ``status_code``, ``request_id``,
    ``content``, ``text``, ``elapsed``, ``retries_taken``). Anything
    we don't model returns a sane default rather than raising — the
    block path is supposed to look like a "successful but synthesised"
    response to every downstream consumer.
    """

    __slots__ = ("_parsed", "_headers")

    def __init__(self, parsed: Any, headers: dict[str, str] | None = None) -> None:
        self._parsed = parsed
        # ``httpx.Headers`` would be the ideal type here, but importing
        # it conditionally is more trouble than it's worth — every
        # known consumer accepts a plain dict via ``dict(raw.headers)``.
        self._headers: dict[str, str] = dict(headers or {})

    def parse(self, *, to: Any = None) -> Any:
        # ``to`` is the upstream optional re-cast target. We don't
        # support cross-type reparsing on a stub — the parsed body is
        # already the correctly-shaped final object; cross-casting it
        # would only happen if a caller asked for a different type
        # than the bound ``cast_to``, which never happens for the
        # framework code paths this patch exists to support.
        return self._parsed

    @property
    def headers(self) -> dict[str, str]:
        return self._headers

    @property
    def http_response(self) -> Any:
        # langchain-openai's error path reads ``raw_response.http_response``
        # to attach the underlying ``httpx.Response`` to a re-raised
        # exception. On a synthesised block we have no real httpx
        # transaction to expose, so we return ``None`` and let the
        # caller's ``hasattr(..., "http_response")`` guard pass while
        # the attribute itself is benign.
        return None

    @property
    def status_code(self) -> int:
        return 200

    @property
    def request_id(self) -> str | None:
        return None

    @property
    def content(self) -> bytes:
        return b""

    @property
    def text(self) -> str:
        return ""

    @property
    def elapsed(self) -> Any:
        import datetime

        return datetime.timedelta(0)

    @property
    def retries_taken(self) -> int:
        return 0


def _stub_chat_completion_raw(
    decision: PolicyDecision, trace_id: str, model: str
) -> _RawResponseStub:
    """``with_raw_response``-shaped sibling of :func:`_stub_chat_completion`.

    Returns a ``LegacyAPIResponse``-shaped wrapper whose ``.parse()``
    yields the same ``ChatCompletion`` stub the non-raw path returns.
    Caller code:

        raw = client.chat.completions.with_raw_response.create(...)
        resp = raw.parse()  # ``resp`` is the ChatCompletion stub

    keeps working unchanged on a blocked turn.
    """
    return _RawResponseStub(_stub_chat_completion(decision, trace_id, model))


def _stub_response_raw(
    decision: PolicyDecision, trace_id: str, model: str
) -> _RawResponseStub:
    """``with_raw_response``-shaped sibling of :func:`_stub_response`."""
    return _RawResponseStub(_stub_response(decision, trace_id, model))


def _extract_chat_usage_raw(response: Any) -> dict[str, Any]:
    """``_extract_chat_usage`` that first unwraps a ``LegacyAPIResponse``."""
    return _extract_chat_usage(_parse_if_raw(response))


def _extract_responses_usage_raw(response: Any) -> dict[str, Any]:
    """``_extract_responses_usage`` that first unwraps a ``LegacyAPIResponse``."""
    return _extract_responses_usage(_parse_if_raw(response))


def _extract_chat_signals_raw(
    response: Any, payload: Any
) -> tuple[str, list[str], list[dict[str, Any]], list[str]]:
    """``extract_openai_chat`` that first unwraps a ``LegacyAPIResponse``."""
    return extract_openai_chat(_parse_if_raw(response), payload)


def _extract_openai_responses_raw(
    response: Any, payload: Any
) -> tuple[str, list[str], list[dict[str, Any]], list[str]]:
    """``extract_openai_responses`` that first unwraps a ``LegacyAPIResponse``."""
    return extract_openai_responses(_parse_if_raw(response), payload)


def _is_raw_response_call(kwargs: dict[str, Any]) -> bool:
    """True when the call came through ``with_raw_response``.

    ``to_raw_response_wrapper`` injects
    ``extra_headers["X-Stainless-Raw-Response"] = "true"`` on every
    call routed through ``client.chat.completions.with_raw_response``
    (and the async / Responses-API siblings). We sniff that marker
    BEFORE handing off to the gate so the stub factory and the
    usage / output extractors can run in raw-mode for the whole
    call instead of guessing at the response shape on the way out.
    """
    extra = kwargs.get("extra_headers")
    if not isinstance(extra, dict):
        return False
    val = extra.get(_RAW_RESPONSE_HEADER)
    return isinstance(val, str) and val.lower() == "true"


def _parse_if_raw(response: Any) -> Any:
    """Coerce a ``LegacyAPIResponse`` to its parsed body, else passthrough.

    The gate's extractors (``_extract_chat_usage``,
    ``extract_openai_chat``, …) walk the parsed shape's attributes
    (``response.usage.prompt_tokens``, ``response.choices[0].message``).
    A ``LegacyAPIResponse`` doesn't expose those — the parsed body
    is reached via ``.parse()``. We call that here so the same
    extractors keep working when the upstream call was routed
    through ``with_raw_response``. ``.parse()`` is cached on the
    upstream object (``_parsed_by_type``), so calling it multiple
    times during one gate pass is free.

    Fail-open: if ``.parse()`` blows up (rare — usually a streaming
    response or a custom client subclass), we return ``None`` so
    extractors silently degrade to zero-usage instead of crashing
    the whole gate.
    """
    if response is None:
        return None
    parse = getattr(response, "parse", None)
    if callable(parse) and hasattr(response, "http_response"):
        try:
            return parse()
        except Exception:  # noqa: BLE001
            return None
    return response


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


class _StreamReplay:
    """Re-iterable, context-managed replay of a chunk list.

    Used on the streaming path (``stream=True``) for both the
    block-stub case and the allow case. Frameworks call
    ``with response as response: for chunk in response: ...``;
    this class makes that work for synthetic stubs (zero real
    chunks) and for materialised real-stream chunks alike.

    The instance also surfaces the **aggregated** response shape
    (``id``, ``object``, ``model``, ``choices`` with a synthesised
    ``message`` per choice, and the final-chunk ``usage``) so the
    gate's existing extractors — ``_extract_chat_usage`` and
    ``extract_openai_chat`` — keep working without a streaming
    detour. They walk the same attribute paths a real
    ``ChatCompletion`` would expose, so the audit event ends up
    with real ``tokens_in`` / ``tokens_out`` / ``cost_usd`` plus
    accurate ``response_decision`` and per-tool waterfall steps.

    Async + sync iteration are both supported. Frameworks like
    ``llama-index-llms-openai`` consume the return value of
    ``await aclient.chat.completions.create(stream=True)`` via
    ``async for response in <stream>``; upstream ``AsyncStream``
    satisfies that contract, so the replay must too. The async
    iterator yields the same materialised chunks the sync side
    does, so a single block-stub or allow-path replay can be
    consumed by either flavour of framework.
    """

    __slots__ = (
        "_chunks",
        "id",
        "object",
        "created",
        "model",
        "choices",
        "usage",
        "system_fingerprint",
        "service_tier",
        "egis",
    )

    def __init__(self, chunks: list[Any], aggregated: Any) -> None:
        self._chunks = chunks
        self.id = getattr(aggregated, "id", "")
        self.object = getattr(aggregated, "object", "chat.completion")
        self.created = getattr(aggregated, "created", 0)
        self.model = getattr(aggregated, "model", "")
        self.choices = getattr(aggregated, "choices", [])
        self.usage = getattr(aggregated, "usage", None)
        self.system_fingerprint = getattr(aggregated, "system_fingerprint", None)
        self.service_tier = getattr(aggregated, "service_tier", None)
        self.egis = getattr(aggregated, "egis", None)

    def __iter__(self):  # noqa: D401 — iteration protocol
        yield from self._chunks

    def __aiter__(self) -> _StreamReplayAsyncIter:
        return _StreamReplayAsyncIter(iter(self._chunks))

    def __enter__(self) -> _StreamReplay:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None

    async def __aenter__(self) -> _StreamReplay:
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class _StreamReplayAsyncIter:
    """Async iterator over a materialized chunk list.

    Returned by ``_StreamReplay.__aiter__`` so frameworks that
    consume ``stream=True`` via ``async for chunk in stream``
    (notably ``llama-index-llms-openai``'s
    ``_astream_chat``) see the same chunk sequence the sync
    iterator yields. Each ``__aiter__`` call returns a fresh
    iterator so re-iteration after exhaustion still works the
    same way the sync ``__iter__`` does.
    """

    __slots__ = ("_it",)

    def __init__(self, it: Any) -> None:
        self._it = it

    def __aiter__(self) -> _StreamReplayAsyncIter:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration from None


def _aggregate_chat_stream(chunks: list[Any]) -> Any:
    """Collapse a list of ``ChatCompletionChunk`` into a ChatCompletion shape.

    The OpenAI streaming protocol emits incremental deltas of the
    assistant turn — content fragments, partial tool-call argument
    strings, an optional usage block on the final chunk. The gate's
    downstream extractors expect the *completed* response shape
    (``choices[*].message.content`` and ``choices[*].message.tool_calls``
    with their full argument JSON), so we reassemble it here. This
    preserves the contract that output policy evaluation runs against
    the same surface for streaming and non-streaming alike.
    """
    text_parts_by_index: dict[int, list[str]] = {}
    tool_calls_by_index: dict[int, dict[int, dict[str, Any]]] = {}
    finish_reason_by_index: dict[int, str | None] = {}
    role_by_index: dict[int, str] = {}
    chunk_id = ""
    created = 0
    model = ""
    system_fingerprint: Any = None
    service_tier: Any = None
    final_usage: Any = None

    for chunk in chunks:
        cid = _read(chunk, "id")
        if isinstance(cid, str) and cid:
            chunk_id = cid
        c_created = _read(chunk, "created")
        if isinstance(c_created, int) and c_created:
            created = c_created
        c_model = _read(chunk, "model")
        if isinstance(c_model, str) and c_model:
            model = c_model
        sf = _read(chunk, "system_fingerprint")
        if sf is not None:
            system_fingerprint = sf
        st = _read(chunk, "service_tier")
        if st is not None:
            service_tier = st

        usage = _read(chunk, "usage")
        if usage is not None:
            final_usage = usage

        choices = _read(chunk, "choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            idx = _read(choice, "index")
            if not isinstance(idx, int):
                idx = 0
            delta = _read(choice, "delta")
            if delta is None:
                continue
            role = _read(delta, "role")
            if isinstance(role, str) and role:
                role_by_index.setdefault(idx, role)
            content = _read(delta, "content")
            if isinstance(content, str) and content:
                text_parts_by_index.setdefault(idx, []).append(content)
            tcs = _read(delta, "tool_calls")
            if isinstance(tcs, list):
                slot = tool_calls_by_index.setdefault(idx, {})
                for tc in tcs:
                    tc_idx = _read(tc, "index")
                    if not isinstance(tc_idx, int):
                        tc_idx = len(slot)
                    entry = slot.setdefault(
                        tc_idx,
                        {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    tc_id = _read(tc, "id")
                    if isinstance(tc_id, str) and tc_id:
                        entry["id"] = tc_id
                    tc_type = _read(tc, "type")
                    if isinstance(tc_type, str) and tc_type:
                        entry["type"] = tc_type
                    fn = _read(tc, "function")
                    if fn is not None:
                        fn_name = _read(fn, "name")
                        if isinstance(fn_name, str) and fn_name:
                            entry["function"]["name"] = fn_name
                        fn_args = _read(fn, "arguments")
                        if isinstance(fn_args, str) and fn_args:
                            entry["function"]["arguments"] += fn_args
            fr = _read(choice, "finish_reason")
            if isinstance(fr, str) and fr:
                finish_reason_by_index[idx] = fr

    indices = sorted(
        set(text_parts_by_index) | set(tool_calls_by_index) | set(finish_reason_by_index)
    ) or [0]

    synthesised_choices: list[Any] = []
    for idx in indices:
        text = "".join(text_parts_by_index.get(idx, [])) or None
        tcs = tool_calls_by_index.get(idx, {})
        synthesised_tcs: list[Any] = []
        for tc_idx in sorted(tcs):
            entry = tcs[tc_idx]
            synthesised_tcs.append(
                SimpleNamespace(
                    id=entry["id"] or f"call_{tc_idx}",
                    type=entry["type"] or "function",
                    function=SimpleNamespace(
                        name=entry["function"]["name"],
                        arguments=entry["function"]["arguments"],
                    ),
                )
            )
        message = SimpleNamespace(
            role=role_by_index.get(idx, "assistant"),
            content=text,
            tool_calls=synthesised_tcs or None,
            refusal=None,
            function_call=None,
        )
        synthesised_choices.append(
            SimpleNamespace(
                index=idx,
                message=message,
                finish_reason=finish_reason_by_index.get(idx),
                logprobs=None,
            )
        )

    return SimpleNamespace(
        id=chunk_id,
        object="chat.completion",
        created=created,
        model=model,
        choices=synthesised_choices,
        usage=final_usage,
        system_fingerprint=system_fingerprint,
        service_tier=service_tier,
    )


def _build_chat_chunk(
    *,
    chunk_id: str,
    model: str,
    delta_kwargs: dict[str, Any],
    finish_reason: str | None,
    usage: Any = None,
) -> Any:
    """Return a single ``ChatCompletionChunk`` (Pydantic if available,
    a structurally-compatible ``SimpleNamespace`` otherwise).

    Same fallback rationale as ``_build_input_tokens_details``: real
    Pydantic types keep frameworks that validate the chunk shape
    (langchain-openai calls ``chunk.model_dump()``) happy; the
    fallback preserves attribute access for installs on an unusual
    ``openai`` pin.
    """
    try:  # pragma: no cover - exercised in real-SDK env
        from openai.types.chat import ChatCompletionChunk  # type: ignore[import-not-found]
        from openai.types.chat.chat_completion_chunk import (  # type: ignore[import-not-found]
            Choice,
            ChoiceDelta,
        )

        return ChatCompletionChunk(
            id=chunk_id,
            object="chat.completion.chunk",
            created=0,
            model=model,
            choices=[
                Choice(
                    index=0,
                    delta=ChoiceDelta(**delta_kwargs),
                    finish_reason=finish_reason,  # type: ignore[arg-type]
                )
            ],
            usage=usage,
        )
    except Exception:
        return SimpleNamespace(
            id=chunk_id,
            object="chat.completion.chunk",
            created=0,
            model=model,
            choices=[
                SimpleNamespace(
                    index=0,
                    delta=SimpleNamespace(**delta_kwargs),
                    finish_reason=finish_reason,
                    logprobs=None,
                )
            ],
            usage=usage,
        )


def _stub_chat_stream(
    decision: PolicyDecision, trace_id: str, model: str
) -> _StreamReplay:
    """Streaming-shaped sibling of :func:`_stub_chat_completion`.

    Yields a two-chunk replay (``role+content`` then ``finish_reason="stop"``)
    so frameworks that opened a ``with response:`` block followed by
    ``for chunk in response:`` keep working — see the module docstring.
    The aggregated form mirrors the non-streaming stub so the audit
    row stays consistent across modes (zero tokens, the ``egis``
    metadata block, the same blurb in the assistant message).
    """
    blurb = (
        f"[POLICY BLOCK] {decision.message or 'Blocked by policy.'} "
        f"(matched={decision.matched_policy or 'unknown'})"
    )
    chunk_id = f"egis-blocked-{trace_id[:8]}"
    # Two chunks: the content delta (so frameworks that read
    # ``choices[0].delta.content`` per chunk see the blurb), then
    # the close-out delta with ``finish_reason="stop"``. We
    # deliberately do NOT attach ``usage`` to either chunk — the
    # upstream Pydantic ``ChatCompletionChunk`` validates ``usage``
    # against the real ``CompletionUsage`` type, and the zero-usage
    # shape lives on the aggregated response surface anyway (which
    # is what ``_extract_chat_usage`` reads).
    chunks = [
        _build_chat_chunk(
            chunk_id=chunk_id,
            model=model,
            delta_kwargs={"role": "assistant", "content": blurb},
            finish_reason=None,
        ),
        _build_chat_chunk(
            chunk_id=chunk_id,
            model=model,
            delta_kwargs={},
            finish_reason="stop",
        ),
    ]
    aggregated = _aggregate_chat_stream(chunks)
    aggregated.id = chunk_id
    aggregated.usage = _build_zero_chat_usage()
    aggregated.egis = {
        "blocked": True,
        "reason": decision.message,
        "matched_policy": decision.matched_policy,
    }
    return _StreamReplay(chunks, aggregated)


def _build_zero_chat_usage() -> Any:
    """Zero-token usage with the documented sub-objects.

    Mirrors the surface of :func:`_stub_chat_completion`'s usage
    block so consumers (and our own ``_extract_chat_usage``) see
    the same shape on both modes. We try the upstream Pydantic
    ``CompletionUsage`` first so frameworks that validate the
    response (langchain-openai's ``UsageMetadata`` model, the
    agents-SDK's ``Usage`` dataclass) accept it; the
    ``SimpleNamespace`` fallback keeps attribute access working
    on unusual ``openai`` pins.
    """
    try:  # pragma: no cover - exercised in real-SDK env
        from openai.types.completion_usage import CompletionUsage  # type: ignore[import-not-found]

        return CompletionUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=_build_prompt_tokens_details(),
            completion_tokens_details=_build_completion_tokens_details(),
        )
    except Exception:
        return SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=_build_prompt_tokens_details(),
            completion_tokens_details=_build_completion_tokens_details(),
        )


def _materialize_sync_stream(stream_obj: Any) -> _StreamReplay:
    """Drain a sync openai ``Stream`` into a re-iterable replay.

    Honours the inner stream's context-manager contract so its
    underlying ``httpx.Response`` is closed even though our gate
    will return the replay to the caller (who will run its own
    ``with`` block over our replay).
    """
    chunks: list[Any] = []
    enter = getattr(stream_obj, "__enter__", None)
    exit_ = getattr(stream_obj, "__exit__", None)
    if callable(enter) and callable(exit_):
        entered = enter()
        try:
            for chunk in entered:
                chunks.append(chunk)
        finally:
            try:
                exit_(None, None, None)
            except Exception:  # noqa: BLE001
                LOGGER.debug("stream __exit__ raised", exc_info=True)
    else:
        for chunk in stream_obj:
            chunks.append(chunk)
    aggregated = _aggregate_chat_stream(chunks)
    return _StreamReplay(chunks, aggregated)


async def _materialize_async_stream(stream_obj: Any) -> _StreamReplay:
    """Async sibling of :func:`_materialize_sync_stream`.

    Supports both async-context-managed streams (the standard
    upstream ``AsyncStream``) and plain async iterables for
    forward compatibility with framework wrappers that pre-strip
    the context manager.
    """
    chunks: list[Any] = []
    aenter = getattr(stream_obj, "__aenter__", None)
    aexit = getattr(stream_obj, "__aexit__", None)
    if callable(aenter) and callable(aexit):
        entered = await aenter()
        try:
            async for chunk in entered:
                chunks.append(chunk)
        finally:
            try:
                await aexit(None, None, None)
            except Exception:  # noqa: BLE001
                LOGGER.debug("async stream __aexit__ raised", exc_info=True)
    else:
        async for chunk in stream_obj:
            chunks.append(chunk)
    aggregated = _aggregate_chat_stream(chunks)
    return _StreamReplay(chunks, aggregated)


def _stub_chat_completion(decision: PolicyDecision, trace_id: str, model: str):
    """Build a stub ``ChatCompletion``-shaped object.

    The upstream ``openai.types.CompletionUsage`` exposes
    ``prompt_tokens_details`` and ``completion_tokens_details`` as
    optional sub-objects (``audio_tokens``, ``cached_tokens``,
    ``reasoning_tokens``, …). Some agentic frameworks
    (``openai-agents``, ``langchain-openai``) access these fields
    directly off the response to populate their own usage tracking
    — a stub that omits them crashes those frameworks with
    ``AttributeError`` the moment a policy fires with
    ``on_block="stub"``. We populate every documented field at
    zero so the stub is structurally indistinguishable from a
    real "no token usage" response.

    We also build a real upstream Pydantic ``ChatCompletion``
    instance when possible, because some agentic frameworks call
    ``response.model_dump()`` directly on the result (notably
    ``autogen-ext.models.openai`` does so to populate its
    ``LLMCallEvent`` log payload — a bare ``SimpleNamespace``
    crashes with ``AttributeError: 'types.SimpleNamespace' object
    has no attribute 'model_dump'`` the moment a policy fires
    with ``on_block="stub"``). The ``SimpleNamespace`` fallback
    preserves attribute-access compatibility on installs where
    Pydantic construction unexpectedly fails (very old ``openai``
    pin, future shape change) — the same posture every other
    ``_build_*`` helper in this file takes.
    """
    blurb = (
        f"[POLICY BLOCK] {decision.message or 'Blocked by policy.'} "
        f"(matched={decision.matched_policy or 'unknown'})"
    )
    chat_id = f"egis-blocked-{trace_id[:8]}"
    egis_meta = {
        "blocked": True,
        "reason": decision.message,
        "matched_policy": decision.matched_policy,
    }

    try:  # pragma: no cover - exercised in real-SDK env
        from openai.types.chat import (  # type: ignore[import-not-found]
            ChatCompletion,
            ChatCompletionMessage,
        )
        from openai.types.chat.chat_completion import Choice  # type: ignore[import-not-found]
        from openai.types.completion_usage import CompletionUsage  # type: ignore[import-not-found]

        msg = ChatCompletionMessage(
            role="assistant",
            content=blurb,
            tool_calls=None,
            refusal=None,
            annotations=None,
            audio=None,
            function_call=None,
        )
        choice = Choice(
            index=0,
            message=msg,
            finish_reason="stop",
            logprobs=None,
        )
        usage = CompletionUsage(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            # See docstring — both ``*_tokens_details`` sub-objects
            # are populated even though their inner fields are
            # Optional on the upstream BaseModel. Frameworks read
            # them through the dotted path.
            prompt_tokens_details=_build_prompt_tokens_details(),
            completion_tokens_details=_build_completion_tokens_details(),
        )
        completion = ChatCompletion(
            id=chat_id,
            object="chat.completion",
            created=0,
            model=model,
            choices=[choice],
            usage=usage,
            system_fingerprint=None,
            service_tier=None,
        )
        # ``ChatCompletion`` is configured with ``extra='allow'``
        # (verified on openai>=1.40), so attaching the ``egis``
        # marker passes Pydantic validation and survives
        # ``.model_dump()`` round-trips for downstream loggers.
        completion.egis = egis_meta  # type: ignore[attr-defined]
        return completion
    except Exception:
        msg_ns = SimpleNamespace(role="assistant", content=blurb, tool_calls=None)
        choice_ns = SimpleNamespace(
            index=0, message=msg_ns, finish_reason="stop", logprobs=None
        )
        usage_ns = SimpleNamespace(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            prompt_tokens_details=_build_prompt_tokens_details(),
            completion_tokens_details=_build_completion_tokens_details(),
        )
        return SimpleNamespace(
            id=chat_id,
            object="chat.completion",
            created=0,
            model=model,
            choices=[choice_ns],
            usage=usage_ns,
            system_fingerprint=None,
            service_tier=None,
            egis=egis_meta,
        )


def _build_prompt_tokens_details() -> Any:
    """Return a Pydantic ``PromptTokensDetails`` instance (or a
    structurally-compatible fallback) for the Chat Completions
    usage block. See :func:`_build_input_tokens_details` for the
    full rationale on why ``SimpleNamespace`` alone is not
    enough."""
    try:  # pragma: no cover - exercised in real-SDK env
        from openai.types.completion_usage import (  # type: ignore[import-not-found]
            PromptTokensDetails,
        )

        return PromptTokensDetails(audio_tokens=0, cached_tokens=0)
    except Exception:
        from types import SimpleNamespace

        return SimpleNamespace(audio_tokens=0, cached_tokens=0)


def _build_completion_tokens_details() -> Any:
    """Counterpart of :func:`_build_prompt_tokens_details` for the
    output side of Chat Completions."""
    try:  # pragma: no cover - exercised in real-SDK env
        from openai.types.completion_usage import (  # type: ignore[import-not-found]
            CompletionTokensDetails,
        )

        return CompletionTokensDetails(
            accepted_prediction_tokens=0,
            audio_tokens=0,
            reasoning_tokens=0,
            rejected_prediction_tokens=0,
        )
    except Exception:
        from types import SimpleNamespace

        return SimpleNamespace(
            accepted_prediction_tokens=0,
            audio_tokens=0,
            reasoning_tokens=0,
            rejected_prediction_tokens=0,
        )


def _build_response_output_message(blurb: str, trace_id: str) -> Any:
    """Build a single ``ResponseOutputMessage`` for the stub.

    Same Pydantic-vs-SimpleNamespace tension as the usage block:
    ``ModelResponse(output=response.output, ...)`` in the
    ``openai-agents`` SDK is a ``pydantic.dataclasses.dataclass``
    whose ``output`` field is typed ``list[TResponseOutputItem]``.
    Pydantic v2 *will* validate each item against the union of
    real upstream output-item types — a bare ``SimpleNamespace``
    is rejected. Building the real upstream
    ``ResponseOutputMessage`` (with a nested
    ``ResponseOutputText``) keeps the stub round-trippable through
    every Pydantic boundary in the agents-SDK and the OpenAI
    Python SDK itself.

    The fallback ``SimpleNamespace`` path preserves the structure
    so customers on an unusual ``openai`` pin still get a
    response object with the right surface; downstream Pydantic
    validation may still reject it, but the surrounding gate
    contract (return *some* shaped response, never raise on
    block-with-stub) is preserved.
    """
    try:  # pragma: no cover - exercised in real-SDK env
        from openai.types.responses.response_output_message import (  # type: ignore[import-not-found]
            ResponseOutputMessage,
        )
        from openai.types.responses.response_output_text import (  # type: ignore[import-not-found]
            ResponseOutputText,
        )

        text = ResponseOutputText(annotations=[], text=blurb, type="output_text")
        return ResponseOutputMessage(
            id=f"egis-blocked-msg-{trace_id[:8]}",
            content=[text],
            role="assistant",
            status="completed",
            type="message",
        )
    except Exception:
        from types import SimpleNamespace

        return SimpleNamespace(
            id=f"egis-blocked-msg-{trace_id[:8]}",
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text=blurb, annotations=[])],
        )


def _build_input_tokens_details() -> Any:
    """Return a Pydantic ``InputTokensDetails`` instance (or a
    structurally-compatible fallback).

    Frameworks like ``openai-agents>=0.5`` treat the OpenAI
    Responses ``usage`` block as a Pydantic v2 model — its
    ``Usage`` dataclass uses ``BeforeValidator``s that only
    accept ``None`` / ``PromptTokensDetails`` /
    ``InputTokensDetails`` instances. A bare ``SimpleNamespace``
    with the right attribute names is duck-typed enough for
    naive ``getattr`` access, but blows up with a Pydantic
    ``ValidationError`` the moment the agents-SDK reconstructs
    its own ``Usage`` from it.

    We import the real upstream type at call-time (this whole
    module is only loaded when ``openai`` is importable, so the
    import is cheap and guaranteed to succeed in practice). If
    the import unexpectedly fails — e.g. a customer pinned a
    very old ``openai`` that predates these types — we fall
    back to a ``SimpleNamespace`` so the gate still returns
    *something* with the right surface (the framework that
    consumes it may still crash on Pydantic validation, but the
    surrounding fail-open contract is preserved).
    """
    try:  # pragma: no cover - exercised in real-SDK env, skipped in unit tests with no openai installed
        from openai.types.responses.response_usage import (  # type: ignore[import-not-found]
            InputTokensDetails,
        )

        return InputTokensDetails(cached_tokens=0)
    except Exception:
        from types import SimpleNamespace

        return SimpleNamespace(cached_tokens=0)


def _build_output_tokens_details() -> Any:
    """Counterpart of :func:`_build_input_tokens_details` for the
    output side."""
    try:  # pragma: no cover - exercised in real-SDK env
        from openai.types.responses.response_usage import (  # type: ignore[import-not-found]
            OutputTokensDetails,
        )

        return OutputTokensDetails(reasoning_tokens=0)
    except Exception:
        from types import SimpleNamespace

        return SimpleNamespace(reasoning_tokens=0)


def _stub_response(decision: PolicyDecision, trace_id: str, model: str):
    """OpenAI Responses-API shaped stub.

    The upstream ``openai.types.responses.ResponseUsage`` REQUIRES
    ``input_tokens_details`` and ``output_tokens_details``
    sub-objects — they're plain (non-Optional) fields on the
    Pydantic model. The ``openai-agents`` Runner reads them
    directly when unpacking the response into its internal
    ``ModelResponse``:

        usage = Usage(
            requests=1,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.total_tokens,
            input_tokens_details=response.usage.input_tokens_details,
            output_tokens_details=response.usage.output_tokens_details,
        )

    A stub that omits those sub-objects crashes ``openai-agents``
    with ``AttributeError`` the moment a policy fires with
    ``on_block="stub"``. *Worse*, the agents-SDK's ``Usage`` is
    itself a Pydantic v2 dataclass — it rejects a
    ``SimpleNamespace`` even when the attribute names match,
    because Pydantic does isinstance-based validation. So the
    sub-objects MUST be real upstream ``InputTokensDetails`` /
    ``OutputTokensDetails`` Pydantic instances — see
    :func:`_build_input_tokens_details`.
    """
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
        # ``output`` items must be real upstream Pydantic types
        # (``ResponseOutputMessage`` containing ``ResponseOutputText``)
        # because the ``agents.items.ModelResponse`` dataclass is
        # itself a ``pydantic.dataclasses.dataclass`` that
        # validates every element. See
        # :func:`_build_response_output_message` for the full
        # rationale and the fallback path.
        output=[_build_response_output_message(blurb, trace_id)],
        # ``output_text`` is the convenience aggregator the OpenAI
        # Python SDK exposes on ResponseObject. Some agentic
        # frameworks read it instead of walking ``output[].content``.
        output_text=blurb,
        usage=SimpleNamespace(
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            input_tokens_details=_build_input_tokens_details(),
            output_tokens_details=_build_output_tokens_details(),
        ),
        # ``incomplete_details`` is read by ``openai-agents`` to
        # decide whether to retry; we populate it as None to
        # mirror a normal completed response.
        incomplete_details=None,
        error=None,
        egis={"blocked": True, "reason": decision.message, "matched_policy": decision.matched_policy},
    )


# ── Smart Model Routing ──────────────────────────────────────────────


def _completion_from_canonical(result: dict[str, Any]) -> Any:
    """Wrap a canonical cross-provider result into a ChatCompletion shape.

    Used when the routing engine served an OpenAI-originated call on a
    different provider: the caller's code still receives the
    ``ChatCompletion`` surface it was written against — real content,
    real token usage, the served model id on ``model``, and an
    additive ``egis.routing`` marker for programmatic consumers. Same
    Pydantic-first / SimpleNamespace-fallback posture as the stub
    builders above.
    """
    text = str(result.get("text") or "")
    served_model = str(result.get("model") or "")
    tokens_in = int(result.get("tokens_in") or 0)
    tokens_out = int(result.get("tokens_out") or 0)
    import uuid as _uuid

    chat_id = f"egis-routed-{_uuid.uuid4().hex[:12]}"
    egis_meta = {"routing": {"applied": True, "served_model": served_model}}
    try:  # pragma: no cover - exercised in real-SDK env
        from openai.types.chat import (  # type: ignore[import-not-found]
            ChatCompletion,
            ChatCompletionMessage,
        )
        from openai.types.chat.chat_completion import Choice  # type: ignore[import-not-found]
        from openai.types.completion_usage import CompletionUsage  # type: ignore[import-not-found]

        completion = ChatCompletion(
            id=chat_id,
            object="chat.completion",
            created=0,
            model=served_model,
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant", content=text
                    ),
                    finish_reason="stop",
                    logprobs=None,
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
                total_tokens=tokens_in + tokens_out,
                prompt_tokens_details=_build_prompt_tokens_details(),
                completion_tokens_details=_build_completion_tokens_details(),
            ),
            system_fingerprint=None,
            service_tier=None,
        )
        completion.egis = egis_meta  # type: ignore[attr-defined]
        return completion
    except Exception:
        msg_ns = SimpleNamespace(role="assistant", content=text, tool_calls=None)
        return SimpleNamespace(
            id=chat_id,
            object="chat.completion",
            created=0,
            model=served_model,
            choices=[
                SimpleNamespace(
                    index=0, message=msg_ns, finish_reason="stop", logprobs=None
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
                total_tokens=tokens_in + tokens_out,
                prompt_tokens_details=_build_prompt_tokens_details(),
                completion_tokens_details=_build_completion_tokens_details(),
            ),
            system_fingerprint=None,
            service_tier=None,
            egis=egis_meta,
        )


def _routing_adapter_chat(kwargs: dict[str, Any], *, allow_cross: bool) -> Any:
    """Build the chat-completions routing adapter for one call.

    Same-provider swaps rewrite ``kwargs["model"]`` in place — the
    ``forward`` lambda reads ``kwargs`` at call time, so the original
    client library executes the routed model over its own auth and
    wire format. Cross-provider swaps (plain-text, non-streaming,
    tool-free calls only — enforced again by the gate) execute
    directly against the target provider's REST API and come back as
    a ``ChatCompletion``-shaped response. Fail-open at every step.
    """
    try:
        from egisai._routing import (
            RoutingAdapter,
            canonicalize_openai_messages,
            execute_cross_call,
        )
    except Exception:  # noqa: BLE001
        return None

    def _apply(new_model: str) -> bool:
        kwargs["model"] = new_model
        return True

    def _cross(decision: dict[str, Any]):  # noqa: ANN202
        messages = canonicalize_openai_messages(kwargs.get("messages"))
        if messages is None:
            return None
        params = {
            "temperature": kwargs.get("temperature"),
            "max_tokens": kwargs.get("max_tokens")
            or kwargs.get("max_completion_tokens"),
        }

        def _forward() -> Any:
            return _completion_from_canonical(
                execute_cross_call(
                    provider=str(decision.get("provider") or ""),
                    model=str(decision.get("model") or ""),
                    messages=messages,
                    params=params,
                )
            )

        return _forward

    return RoutingAdapter(
        apply_same_provider=_apply,
        build_cross_forward=_cross if allow_cross else None,
    )


def _ensure_stream_usage(kwargs: dict[str, Any]) -> None:
    """Inject ``stream_options={"include_usage": True}`` when streaming.

    Streamed Chat Completions only emit a final ``usage`` chunk when
    the caller passes ``stream_options={"include_usage": True}``.
    Several agentic frameworks — notably
    ``llama-index-llms-openai``'s ``_stream_chat`` /
    ``_astream_chat`` — never set this, so the materialised stream
    arrives at the gate with ``response.usage is None`` and the
    audit row ends up with no ``tokens_in`` / ``tokens_out`` /
    ``cost_usd``.

    Forcing ``include_usage`` here is safe:

    * The extra usage chunk carries an empty ``choices`` list, which
      every streaming consumer we patch (``langchain-openai``,
      ``llama-index``, ``openai-agents``, raw ``openai``) handles by
      falling through to ``delta = ChoiceDelta()`` — no content is
      emitted, no callbacks fire on it.
    * If the caller explicitly set ``include_usage: False`` we honour
      that decision (no override). The injection only fills the
      missing-or-true gap.
    * Idempotent across re-entry: the wrapper marks ``kwargs`` as a
      local dict per call, so mutations don't leak back to the
      caller's own dict.
    """
    existing = kwargs.get("stream_options")
    if isinstance(existing, dict):
        if "include_usage" in existing:
            return
        merged = dict(existing)
        merged["include_usage"] = True
        kwargs["stream_options"] = merged
    else:
        kwargs["stream_options"] = {"include_usage": True}


def _wrap_create_chat(orig: Callable[..., Any], is_async: bool) -> Callable[..., Any]:
    target = "openai.chat.completions.create"

    if is_async:
        async def aw(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            # Gateway path: either gateway mode is on
            # (``init(gateway=True)``) or the client is already
            # pointed at the Gateway (``egisai.Client`` / manual
            # ``base_url``). The Gateway evaluates + audits
            # server-side — the local gate is skipped to avoid double
            # governance; per-call context (``set_context`` →
            # ``X-Egis-Agent``) is injected on the way out. Falls
            # back to the local path when the reroute can't be
            # constructed (fail open).
            if _gateway.should_carry(self):
                try:
                    return await _gateway.forward_chat_async(
                        self, orig, args, kwargs
                    )
                except _gateway.RerouteUnavailable as exc:
                    LOGGER.warning(
                        "gateway reroute unavailable (%s) — falling back "
                        "to in-process governance for this call",
                        exc,
                    )
            model = kwargs.get("model", "unknown")
            messages = kwargs.get("messages")
            stream = bool(kwargs.get("stream", False))
            # See ``_is_raw_response_call`` — when langchain-openai
            # (and any other framework that uses the upstream
            # ``with_raw_response.create`` shape) routes the call
            # through us, the caller will do
            # ``raw_response.parse()`` to get the real
            # ``ChatCompletion``. We must therefore return a
            # ``LegacyAPIResponse``-shaped object on both the
            # block-stub and allow paths so that ``.parse()`` step
            # keeps working. The gate's extractors are wrapped to
            # peek through ``raw_response.parse()`` so the audit
            # row still carries real tokens / tool calls / verdict.
            raw_mode = _is_raw_response_call(kwargs)
            stub_factory_chat: Callable[[PolicyDecision, str, str], Any] = (
                _stub_chat_completion
            )
            extract_usage_fn: Callable[[Any], dict[str, Any]] = _extract_chat_usage
            extract_signals_fn = extract_openai_chat
            if raw_mode:
                stub_factory_chat = _stub_chat_completion_raw
                extract_usage_fn = _extract_chat_usage_raw
                extract_signals_fn = _extract_chat_signals_raw

            # When the caller asked for a streaming response (the
            # langchain-classic AgentExecutor's default code path is
            # ``stream_runnable=True`` → ``ChatOpenAI._astream`` →
            # ``self.async_client.create(stream=True)``), our return
            # value must satisfy the framework's
            # ``async with response as r: async for chunk in r: ...``
            # contract. Materialise the inner stream into a replay
            # so block-stub AND allow paths look identical to the
            # consumer and the gate still sees aggregated usage /
            # tool calls for the audit row. See the module docstring
            # for the full motivation.
            if stream:
                # Ensure the upstream emits a final usage chunk so
                # the audit row carries real tokens_in / tokens_out
                # instead of zeros. See ``_ensure_stream_usage``.
                _ensure_stream_usage(kwargs)

                async def forward_async() -> Any:
                    return await _materialize_async_stream(
                        await orig(self, *args, **kwargs)
                    )
                return await async_gate_call(
                    source="openai",
                    target=target,
                    model=model,
                    prompt_text=extract_prompt_text(messages),
                    stream=True,
                    payload={"messages": messages, "tools": kwargs.get("tools")},
                    stub_factory=_stub_chat_stream,
                    extract_usage=_extract_chat_usage,
                    extract_output_signals=extract_openai_chat,
                    emit_tool_call_steps=True,
                    # Streams route same-provider only (the swap is a
                    # kwargs rewrite; the real client keeps streaming).
                    routing_adapter=_routing_adapter_chat(
                        kwargs, allow_cross=False
                    ),
                    forward=forward_async,
                )
            return await async_gate_call(
                source="openai",
                target=target,
                model=model,
                prompt_text=extract_prompt_text(messages),
                stream=stream,
                payload={"messages": messages, "tools": kwargs.get("tools")},
                stub_factory=stub_factory_chat,
                extract_usage=extract_usage_fn,
                extract_output_signals=extract_signals_fn,
                # Raw-mode responses must stay ``LegacyAPIResponse``-
                # shaped end to end, so cross-provider execution (which
                # returns a plain ChatCompletion) is disabled there.
                routing_adapter=_routing_adapter_chat(
                    kwargs, allow_cross=not raw_mode
                ),
                # Multi-step waterfall: when the model returns
                # tool_calls, the gate appends a ``tool_call`` step
                # per tool so the dashboard's RunTimelineModal shows
                # ``model -> tool -> model -> tool -> ...`` instead
                # of collapsing the whole turn into one box. The
                # parent model_call's output policy already gated
                # each tool request before this step lands.
                emit_tool_call_steps=True,
                forward=lambda: orig(self, *args, **kwargs),
            )

        aw.__egisai_wrapped__ = True  # type: ignore[attr-defined]
        return aw

    def w(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        # Gateway path — see the async twin above.
        if _gateway.should_carry(self):
            try:
                return _gateway.forward_chat(self, orig, args, kwargs)
            except _gateway.RerouteUnavailable as exc:
                LOGGER.warning(
                    "gateway reroute unavailable (%s) — falling back "
                    "to in-process governance for this call",
                    exc,
                )
        model = kwargs.get("model", "unknown")
        messages = kwargs.get("messages")
        stream = bool(kwargs.get("stream", False))
        raw_mode = _is_raw_response_call(kwargs)
        stub_factory_chat: Callable[[PolicyDecision, str, str], Any] = (
            _stub_chat_completion
        )
        extract_usage_fn: Callable[[Any], dict[str, Any]] = _extract_chat_usage
        extract_signals_fn = extract_openai_chat
        if raw_mode:
            stub_factory_chat = _stub_chat_completion_raw
            extract_usage_fn = _extract_chat_usage_raw
            extract_signals_fn = _extract_chat_signals_raw
        if stream:
            # Sync sibling of the async streaming branch above —
            # same materialise-then-replay rationale.
            _ensure_stream_usage(kwargs)

            def forward_sync() -> Any:
                return _materialize_sync_stream(orig(self, *args, **kwargs))
            return gate_call(
                source="openai",
                target=target,
                model=model,
                prompt_text=extract_prompt_text(messages),
                stream=True,
                payload={"messages": messages, "tools": kwargs.get("tools")},
                stub_factory=_stub_chat_stream,
                extract_usage=_extract_chat_usage,
                extract_output_signals=extract_openai_chat,
                emit_tool_call_steps=True,
                routing_adapter=_routing_adapter_chat(
                    kwargs, allow_cross=False
                ),
                forward=forward_sync,
            )
        return gate_call(
            source="openai",
            target=target,
            model=model,
            prompt_text=extract_prompt_text(messages),
            stream=stream,
            payload={"messages": messages, "tools": kwargs.get("tools")},
            stub_factory=stub_factory_chat,
            extract_usage=extract_usage_fn,
            extract_output_signals=extract_signals_fn,
            emit_tool_call_steps=True,
            routing_adapter=_routing_adapter_chat(
                kwargs, allow_cross=not raw_mode
            ),
            forward=lambda: orig(self, *args, **kwargs),
        )

    w.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return w


def _wrap_create_responses(orig: Callable[..., Any], is_async: bool) -> Callable[..., Any]:
    target = "openai.responses.create"

    if is_async:
        async def aw(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            model = kwargs.get("model", "unknown")
            inp = kwargs.get("input")
            stream = bool(kwargs.get("stream", False))
            raw_mode = _is_raw_response_call(kwargs)
            stub_factory_resp: Callable[[PolicyDecision, str, str], Any] = (
                _stub_response
            )
            extract_usage_fn: Callable[[Any], dict[str, Any]] = _extract_responses_usage
            extract_signals_fn = extract_openai_responses
            if raw_mode:
                stub_factory_resp = _stub_response_raw
                extract_usage_fn = _extract_responses_usage_raw
                extract_signals_fn = _extract_openai_responses_raw
            # ``input`` on the OpenAI Responses API is either an
            # immutable string (ergonomic shape) or a list of
            # message dicts. ``mutate_prompt_text`` updates
            # ``payload["input"]`` in place; the forward lambda
            # mirrors that back into the SDK kwargs so the sanitized
            # text - not the raw prompt - is what physically leaves
            # the SDK boundary. Same rationale as the
            # ``_patches.genai`` patch. For the list-of-dicts shape
            # the same write is a no-op (the dicts are mutated in
            # place upstream), but the explicit re-bind is cheap
            # and keeps both shapes correct.
            payload: dict[str, Any] = {
                "input": inp,
                "tools": kwargs.get("tools"),
            }
            return await async_gate_call(
                source="openai",
                target=target,
                model=model,
                prompt_text=extract_prompt_text(inp),
                stream=stream,
                payload=payload,
                stub_factory=stub_factory_resp,
                extract_usage=extract_usage_fn,
                extract_output_signals=extract_signals_fn,
                # Responses-API calls route same-provider only — the
                # cross-provider translator targets the Chat
                # Completions shape, not Responses output items.
                routing_adapter=_routing_adapter_chat(
                    kwargs, allow_cross=False
                ),
                # See _wrap_create_chat for the per-tool waterfall
                # rationale. The Responses API uses
                # ``function_call`` output items rather than
                # ``tool_calls``; ``extract_openai_responses``
                # normalises both shapes to the same
                # ``{"name", "arguments"}`` list, so a single flag
                # covers the two surfaces.
                emit_tool_call_steps=True,
                forward=lambda: orig(
                    self,
                    *args,
                    **{**kwargs, "input": payload["input"]},
                ),
            )

        aw.__egisai_wrapped__ = True  # type: ignore[attr-defined]
        return aw

    def w(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        model = kwargs.get("model", "unknown")
        inp = kwargs.get("input")
        stream = bool(kwargs.get("stream", False))
        raw_mode = _is_raw_response_call(kwargs)
        stub_factory_resp: Callable[[PolicyDecision, str, str], Any] = _stub_response
        extract_usage_fn: Callable[[Any], dict[str, Any]] = _extract_responses_usage
        extract_signals_fn = extract_openai_responses
        if raw_mode:
            stub_factory_resp = _stub_response_raw
            extract_usage_fn = _extract_responses_usage_raw
            extract_signals_fn = _extract_openai_responses_raw
        # See the async branch above for the immutable-scalar
        # ``input`` rationale.
        payload: dict[str, Any] = {
            "input": inp,
            "tools": kwargs.get("tools"),
        }
        return gate_call(
            source="openai",
            target=target,
            model=model,
            prompt_text=extract_prompt_text(inp),
            stream=stream,
            payload=payload,
            stub_factory=stub_factory_resp,
            extract_usage=extract_usage_fn,
            extract_output_signals=extract_signals_fn,
            emit_tool_call_steps=True,
            # Same-provider only — see the async sibling above.
            routing_adapter=_routing_adapter_chat(kwargs, allow_cross=False),
            forward=lambda: orig(
                self,
                *args,
                **{**kwargs, "input": payload["input"]},
            ),
        )

    w.__egisai_wrapped__ = True  # type: ignore[attr-defined]
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
    cls.create = wrapped
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
        import openai  # type: ignore[import-not-found]  # noqa: WPS433

        chat = getattr(openai, "ChatCompletion", None)
        if chat is not None and hasattr(chat, "create"):
            orig = chat.create
            if callable(orig) and not getattr(orig, "__egisai_wrapped__", False):
                chat.create = _wrap_create_chat(orig, is_async=False)  # type: ignore[assignment]
                any_patched = True
    except Exception:  # noqa: BLE001
        pass

    return any_patched
