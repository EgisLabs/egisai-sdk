"""Regression contract: every block-stub mirrors the upstream provider SDK's attribute shape.

When a policy fires with ``on_block="stub"``, every provider patch
returns a ``types.SimpleNamespace`` posing as the upstream SDK's
response object. Agentic frameworks (``openai-agents``,
``langchain-openai``, ``langgraph``, CrewAI, …) consume those
stubs as if they were real responses — they walk
``response.usage.<field>``, ``response.output[0].content[0].text``,
``response.candidates[0].content.parts[0].text``, etc., directly.

If our stub omits a single field the upstream SDK reads, the
framework crashes on the *next* statement after the gate returns
with an ``AttributeError`` — exactly what
``openai-agents>=0.5`` did to the OpenAI Responses stub in
v0.25.0:

    File "/site-packages/agents/models/openai_responses.py", line 495
        input_tokens_details=response.usage.input_tokens_details,
    AttributeError: 'types.SimpleNamespace' object has no attribute
                    'input_tokens_details'

This file is the contract test that pins every documented
attribute the upstream SDK reads. Every new ``_stub_*`` factory
MUST appear here with the exact access pattern the relevant
frameworks use. If you add a field to a stub, add the
corresponding accessor here so the contract is enforceable.

This is also defensive against future upstream SDK upgrades:
when ``openai-agents`` adds another field to its
``ModelResponse`` unpack — say ``cached_tokens`` — the failing
agent harness will be the first signal, but this test gives us a
single file to patch instead of debugging through stack traces.
"""
from __future__ import annotations

from egisai._patches.anthropic import _stub_message
from egisai._patches.genai import _stub_response as _stub_genai_response
from egisai._patches.google import _stub_response as _stub_google_response
from egisai._patches.openai import _stub_chat_completion, _stub_response
from egisai.policy.engine import PolicyDecision


def _decision() -> PolicyDecision:
    return PolicyDecision.deny(
        reason_code="test_block",
        message="blocked by contract test",
        matched_policy="contract-test-policy",
    )


# ---------------------------------------------------------------------------
# OpenAI Responses API
# ---------------------------------------------------------------------------


def test_openai_responses_stub_matches_openai_agents_unpack() -> None:
    """Mirror of ``agents/models/openai_responses.py::get_response`` line 495.

    The Runner reads every one of these fields off ``response.usage``
    to populate its internal ``Usage`` object. A missing field
    crashes the entire Runner — see the v0.25.1 regression that
    motivated this test.

    Beyond attribute presence, the sub-objects must be real
    upstream ``InputTokensDetails`` / ``OutputTokensDetails``
    Pydantic instances — the ``agents.usage.Usage`` dataclass
    uses ``isinstance``-based Pydantic v2 validators that reject
    a bare ``SimpleNamespace`` with the right attribute names.
    """
    response = _stub_response(_decision(), trace_id="t" * 16, model="gpt-5")

    assert response.usage.input_tokens == 0
    assert response.usage.output_tokens == 0
    assert response.usage.total_tokens == 0
    assert response.usage.input_tokens_details.cached_tokens == 0
    assert response.usage.output_tokens_details.reasoning_tokens == 0

    # Real upstream Pydantic types must be used so downstream
    # ``isinstance``-style Pydantic validation succeeds.
    from openai.types.responses.response_usage import (
        InputTokensDetails,
        OutputTokensDetails,
    )

    assert isinstance(response.usage.input_tokens_details, InputTokensDetails)
    assert isinstance(response.usage.output_tokens_details, OutputTokensDetails)


def test_openai_responses_stub_feeds_agents_usage_unpack_end_to_end() -> None:
    """End-to-end smoke test: reproduce the exact failing line from
    ``agents/models/openai_responses.py:495`` with our patched stub.

    Before v0.25.1 this raised
    ``AttributeError: 'types.SimpleNamespace' object has no
    attribute 'input_tokens_details'`` — pinning the green path
    here means any future regression of the stub shape (or of
    the upstream agents-SDK pulling another field) lights up
    this single test instead of a customer's agent harness."""
    try:
        from agents.usage import Usage
    except ImportError:
        import pytest

        pytest.skip("openai-agents not installed in this env")

    response = _stub_response(_decision(), trace_id="t" * 16, model="gpt-5")

    usage = Usage(
        requests=1,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        total_tokens=response.usage.total_tokens,
        input_tokens_details=response.usage.input_tokens_details,
        output_tokens_details=response.usage.output_tokens_details,
    )

    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.total_tokens == 0
    assert usage.input_tokens_details.cached_tokens == 0
    assert usage.output_tokens_details.reasoning_tokens == 0


def test_openai_responses_stub_exposes_output_text_aggregator() -> None:
    """The OpenAI Python SDK's ``Response`` object exposes a flat
    ``output_text`` convenience property that walks every
    ``output[].content[].text`` for the caller. Frameworks
    (LangChain, LlamaIndex) prefer the flat shortcut; spell it on
    the stub so they don't crash."""
    response = _stub_response(_decision(), trace_id="t" * 16, model="gpt-5")
    assert isinstance(response.output_text, str)
    assert "[POLICY BLOCK]" in response.output_text


def test_openai_responses_stub_walks_output_list_shape() -> None:
    """Agentic frameworks that want fine-grained tool-call vs text
    detection walk ``response.output[].content[].text`` directly.
    Pin the shape so an SDK refactor that breaks the walk fails
    here, not in a customer's agent loop. We also assert the
    items are real upstream ``ResponseOutputMessage`` instances
    so ``ModelResponse(output=...)`` Pydantic validation
    downstream succeeds."""
    response = _stub_response(_decision(), trace_id="t" * 16, model="gpt-5")
    item = response.output[0]
    assert item.type == "message"
    assert item.role == "assistant"
    content = item.content[0]
    assert content.type == "output_text"
    assert isinstance(content.text, str) and content.text

    from openai.types.responses.response_output_message import ResponseOutputMessage
    from openai.types.responses.response_output_text import ResponseOutputText

    assert isinstance(item, ResponseOutputMessage)
    assert isinstance(content, ResponseOutputText)


def test_openai_responses_stub_feeds_full_model_response_end_to_end() -> None:
    """Deeper smoke: build the full ``ModelResponse`` Pydantic
    dataclass the agents-SDK constructs after every model call.

    This is the test that catches *both* failures the user hit on
    v0.25.0:

    1. ``AttributeError: 'types.SimpleNamespace' object has no
       attribute 'input_tokens_details'`` on the usage unpack.
    2. ``pydantic_core.ValidationError`` on the
       ``ModelResponse.output: list[TResponseOutputItem]`` field
       when each item is a ``SimpleNamespace`` instead of a
       ``ResponseOutputMessage``.

    Any regression of either fix lights up here."""
    try:
        from agents.items import ModelResponse
        from agents.usage import Usage
    except ImportError:
        import pytest

        pytest.skip("openai-agents not installed in this env")

    response = _stub_response(_decision(), trace_id="t" * 16, model="gpt-5")

    usage = Usage(
        requests=1,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        total_tokens=response.usage.total_tokens,
        input_tokens_details=response.usage.input_tokens_details,
        output_tokens_details=response.usage.output_tokens_details,
    )
    model_response = ModelResponse(
        output=response.output,
        usage=usage,
        response_id=response.id,
        request_id=None,
    )
    assert model_response.response_id.startswith("egis-blocked-")
    assert len(model_response.output) == 1
    assert model_response.output[0].type == "message"


def test_openai_responses_stub_has_optional_top_level_fields() -> None:
    """``openai-agents`` retry logic reads ``incomplete_details``
    and ``error`` to decide whether to retry the call. Both must
    be settable to ``None`` (not absent)."""
    response = _stub_response(_decision(), trace_id="t" * 16, model="gpt-5")
    assert response.incomplete_details is None
    assert response.error is None
    assert response.status == "completed"


# ---------------------------------------------------------------------------
# OpenAI Chat Completions API
# ---------------------------------------------------------------------------


def test_openai_chat_completion_stub_matches_langchain_openai_unpack() -> None:
    """LangChain's ``ChatOpenAI`` reads
    ``response.choices[0].message.tool_calls`` and the *_tokens_details
    sub-objects on the usage block. Pin the shape."""
    response = _stub_chat_completion(_decision(), trace_id="t" * 16, model="gpt-5")

    choice = response.choices[0]
    assert choice.message.role == "assistant"
    assert isinstance(choice.message.content, str)
    assert choice.message.tool_calls is None
    assert choice.finish_reason == "stop"
    assert choice.logprobs is None

    usage = response.usage
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0
    # Both sub-objects are nominally Optional on the upstream
    # ``CompletionUsage`` model — but frameworks that read e.g.
    # ``usage.completion_tokens_details.reasoning_tokens`` need
    # the *sub-object* to exist, not be ``None``. We spell every
    # documented inner field.
    assert usage.prompt_tokens_details.audio_tokens == 0
    assert usage.prompt_tokens_details.cached_tokens == 0
    assert usage.completion_tokens_details.accepted_prediction_tokens == 0
    assert usage.completion_tokens_details.audio_tokens == 0
    assert usage.completion_tokens_details.reasoning_tokens == 0
    assert usage.completion_tokens_details.rejected_prediction_tokens == 0

    # Same rationale as the Responses-API side — frameworks that
    # do ``isinstance(...)``-based Pydantic validation need real
    # upstream types, not duck-typed namespaces. We only assert
    # this when ``openai`` is importable (it always is when the
    # OpenAI patch is *active*; pyproject pins it as an optional
    # dep but the import is guarded for non-OpenAI test envs).
    try:
        from openai.types.completion_usage import (
            CompletionTokensDetails,
            PromptTokensDetails,
        )
    except ImportError:
        return
    assert isinstance(usage.prompt_tokens_details, PromptTokensDetails)
    assert isinstance(usage.completion_tokens_details, CompletionTokensDetails)


def test_openai_chat_completion_stub_is_model_dump_capable() -> None:
    """``autogen-ext.models.openai`` calls ``response.model_dump()``
    directly to populate its ``LLMCallEvent`` log payload:

        logger.info(
            LLMCallEvent(
                messages=...,
                response=result.model_dump(),
                ...
            )
        )

    Before v0.25.6 the stub was a bare ``SimpleNamespace`` with no
    ``.model_dump()`` method — every blocked autogen turn raised
    ``AttributeError: 'types.SimpleNamespace' object has no
    attribute 'model_dump'`` between the gate returning and the
    operator seeing the policy verdict. The fix is to build a
    real upstream ``ChatCompletion`` Pydantic instance. This test
    pins the contract so any future refactor of the stub keeps
    autogen working.
    """
    response = _stub_chat_completion(_decision(), trace_id="t" * 16, model="gpt-5")

    assert hasattr(response, "model_dump"), (
        "blocked stub must expose .model_dump() — autogen-ext "
        "and other frameworks call it directly on the response"
    )
    dumped = response.model_dump()
    assert isinstance(dumped, dict)
    assert dumped["choices"][0]["message"]["content"].startswith("[POLICY BLOCK]")
    assert dumped["choices"][0]["message"]["role"] == "assistant"
    assert dumped["choices"][0]["finish_reason"] == "stop"
    assert dumped["usage"]["prompt_tokens"] == 0
    assert dumped["model"] == "gpt-5"
    # Our ``egis`` marker rides along through the dump so
    # downstream loggers can spot a blocked turn.
    assert dumped.get("egis") == {
        "blocked": True,
        "reason": "blocked by contract test",
        "matched_policy": "contract-test-policy",
    }

    # Same isinstance contract as the langchain side — frameworks
    # that downstream-validate via Pydantic need the real upstream
    # type, not a duck-typed namespace. ``ChatCompletion`` is
    # configured ``extra='allow'`` so the ``egis`` marker rides
    # along.
    try:
        from openai.types.chat import ChatCompletion
    except ImportError:
        return
    assert isinstance(response, ChatCompletion)


def test_openai_chat_completion_stub_matches_autogen_unpack() -> None:
    """Mirror of ``autogen_ext.models.openai._openai_client.create``
    lines 712-736: ``getattr(result.usage, 'prompt_tokens', 0)``,
    ``result.model_dump()``, ``result.model``, ``result.choices[0]``.

    A direct end-to-end smoke test of every accessor autogen runs
    against a ``ChatCompletion`` between receiving it from the
    OpenAI client and converting it into its internal
    ``CreateResult``. Any future stub regression that breaks one
    of these accessors fails this single test instead of
    surfacing in a customer's autogen agent loop."""
    response = _stub_chat_completion(_decision(), trace_id="t" * 16, model="gpt-5")

    prompt_tokens = (
        getattr(response.usage, "prompt_tokens", 0) if response.usage is not None else 0
    )
    completion_tokens = (
        getattr(response.usage, "completion_tokens", 0)
        if response.usage is not None
        else 0
    )
    assert prompt_tokens == 0
    assert completion_tokens == 0

    dumped = response.model_dump()
    assert "choices" in dumped

    assert response.model == "gpt-5"

    choice = response.choices[0]
    assert choice.finish_reason == "stop"
    assert choice.message.role == "assistant"
    assert choice.message.tool_calls is None


def test_openai_chat_completion_stub_has_metadata_fields() -> None:
    """``response.model``, ``response.system_fingerprint``,
    ``response.service_tier`` are all read by various wrappers
    (LangChain Tracing, LangSmith, OpenAI Python SDK's
    ``model_dump``). Pin them."""
    response = _stub_chat_completion(_decision(), trace_id="t" * 16, model="gpt-5")
    assert response.model == "gpt-5"
    assert response.system_fingerprint is None
    assert response.service_tier is None
    assert response.created == 0


# ---------------------------------------------------------------------------
# Anthropic Messages API
# ---------------------------------------------------------------------------


def test_anthropic_message_stub_matches_langchain_anthropic_unpack() -> None:
    """LangChain's ``ChatAnthropic`` (and CrewAI's Anthropic
    adapter) read ``cache_creation_input_tokens`` and
    ``cache_read_input_tokens`` directly off the usage block.
    Pin the shape so they don't crash on a blocked stub."""
    response = _stub_message(_decision(), trace_id="t" * 16, model="claude-3-7-sonnet")

    assert response.type == "message"
    assert response.role == "assistant"
    assert response.model == "claude-3-7-sonnet"
    # Both stop_* are documented Optional fields the SDK exposes;
    # frameworks read them directly.
    assert response.stop_reason == "end_turn"
    assert response.stop_sequence is None
    assert response.container is None

    usage = response.usage
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.cache_creation_input_tokens == 0
    assert usage.cache_read_input_tokens == 0
    assert usage.server_tool_use is None
    assert usage.service_tier is None


def test_anthropic_message_stub_walks_content_blocks() -> None:
    """Frameworks walk ``response.content[]`` looking for
    ``type == 'text'`` vs ``type == 'tool_use'``. Pin the shape."""
    response = _stub_message(_decision(), trace_id="t" * 16, model="claude-3-7-sonnet")
    block = response.content[0]
    assert block.type == "text"
    assert isinstance(block.text, str)
    assert "[POLICY BLOCK]" in block.text


# ---------------------------------------------------------------------------
# Google GenAI (new ``google-genai`` SDK)
# ---------------------------------------------------------------------------


def test_genai_stub_matches_langchain_google_unpack() -> None:
    """LangChain's ``ChatGoogleGenerativeAI`` and Vertex Agent
    Builder shims read the extended ``usage_metadata`` shape
    (``cached_content_token_count``, ``thoughts_token_count``,
    ``tool_use_prompt_token_count``) directly off the response.
    Pin the shape."""
    response = _stub_genai_response(_decision(), trace_id="t" * 16, model="gemini-2.5-pro")

    assert isinstance(response.text, str)
    assert response.model_version == "gemini-2.5-pro"
    assert response.response_id.startswith("egis-blocked-")
    assert response.function_calls == []

    um = response.usage_metadata
    assert um.prompt_token_count == 0
    assert um.candidates_token_count == 0
    assert um.total_token_count == 0
    assert um.cached_content_token_count == 0
    assert um.thoughts_token_count == 0
    assert um.tool_use_prompt_token_count == 0
    # The ``*_tokens_details`` sub-objects are Optional on the
    # upstream model; ``None`` is fine here because no framework
    # we're aware of reads them when they're absent on a normal
    # text-only response.
    assert um.prompt_tokens_details is None
    assert um.candidates_tokens_details is None


def test_genai_stub_walks_candidates_parts() -> None:
    """Frameworks walk ``response.candidates[].content.parts[]``
    looking for ``text`` vs ``function_call`` entries. Pin the
    shape."""
    response = _stub_genai_response(_decision(), trace_id="t" * 16, model="gemini-2.5-pro")
    cand = response.candidates[0]
    assert cand.finish_reason == "STOP"
    assert cand.content.role == "model"
    part = cand.content.parts[0]
    assert isinstance(part.text, str)
    assert part.function_call is None


# ---------------------------------------------------------------------------
# Google generative-ai (legacy ``google.generativeai`` SDK)
# ---------------------------------------------------------------------------


def test_google_legacy_stub_matches_langchain_google_unpack() -> None:
    """Mirror of ``test_genai_stub_matches_langchain_google_unpack``
    for the legacy ``google.generativeai`` SDK shape."""
    response = _stub_google_response(_decision(), trace_id="t" * 16, model="gemini-1.5-pro")

    assert isinstance(response.text, str)
    assert response.function_calls == []

    um = response.usage_metadata
    assert um.prompt_token_count == 0
    assert um.candidates_token_count == 0
    assert um.total_token_count == 0
    assert um.cached_content_token_count == 0

    cand = response.candidates[0]
    assert cand.finish_reason == "STOP"
    part = cand.content.parts[0]
    assert isinstance(part.text, str)
    assert part.function_call is None


# ---------------------------------------------------------------------------
# Audit invariant: every stub MUST flag itself as blocked
# ---------------------------------------------------------------------------


def test_every_stub_carries_egis_block_marker() -> None:
    """The ``egis`` dict on every stub is the marker that the
    audit pipeline reads to know the response was synthesized,
    not a real model output. If a refactor drops the marker the
    audit log silently records a "successful" response — that's
    a compliance bug. Pin it on every factory."""
    decision = _decision()
    for stub in (
        _stub_response(decision, "trace-1234567890", "gpt-5"),
        _stub_chat_completion(decision, "trace-1234567890", "gpt-5"),
        _stub_message(decision, "trace-1234567890", "claude-3-7-sonnet"),
        _stub_genai_response(decision, "trace-1234567890", "gemini-2.5-pro"),
        _stub_google_response(decision, "trace-1234567890", "gemini-1.5-pro"),
    ):
        assert stub.egis["blocked"] is True
        assert stub.egis["reason"] == "blocked by contract test"
        assert stub.egis["matched_policy"] == "contract-test-policy"
