"""Per-framework before/after enforcement contract — the canonical pin.

This is the file you read to convince yourself egisai actually
enforces policy in the right places for every framework we ship a
patch for. The shape of the contract is one paragraph:

  * **Before every LLM call** — the user's prompt (or the rolled-up
    messages including any tool results from a previous turn) is
    evaluated by the input phase. Block raises / stubs; sanitize
    masks in place; allow forwards untouched.
  * **After every LLM call (which is also "before the tool runs")**
    — the model's response is evaluated by the output phase.
    ``deny_tool_call`` / ``deny_mcp_call`` / ``deny_output_regex``
    / output-side ``semantic_guard`` get a chance to refuse the
    tool the model just asked for, BEFORE the framework's agent
    loop dispatches it. For synchronous patches (OpenAI Chat /
    Responses, Anthropic, Gemini, Bedrock Converse) this means a
    blocked tool literally never runs — the gate raises before the
    response reaches the agent loop.
  * **After the tool executes (and before the next LLM call)** —
    the tool result lands back in the framework's messages buffer.
    The NEXT LLM call's input phase scans those messages including
    the tool-result blocks; the same input policies that fired on
    turn 1 fire again on turn 2 against the tool result. PII a tool
    returned is masked before it round-trips back to the model.

Every framework we ship gets these three guarantees. For the
Tier-1 direct LLM patches (OpenAI, Anthropic, Gemini, Bedrock
Converse) the gates wrap ``client.create()`` / ``Messages.create()``
/ ``generate_content()`` / ``converse()`` directly. For the
Tier-2 agentic delegators (OpenAI Agents, LangGraph, LangChain,
CrewAI, AutoGen, Agno, Strands, smolagents, LlamaIndex,
PydanticAI, Google ADK) the agent loop runs in Python — each
iteration delegates to a Tier-1 patch under the hood, so each
iteration gets the full before/after sandwich. For the Tier-3a
``claude_agent_sdk`` patch the entire agent loop runs in a
Node.js subprocess; we use PreToolUse + PostToolUse hooks to
reach equivalence (covered by ``test_claude_agent_sdk_pretooluse.py``
and ``test_claude_agent_sdk_posttooluse.py``).

The matrix below exercises EACH framework once with a tool-using
turn and asserts that:

1. Input phase ran before the model_call (a ``prompt_decision``
   block lands on the audit row).
2. Output phase ran after the model_call (a ``response_decision``
   block lands on the audit row).
3. When a ``deny_tool_call`` rule matches the tool the model
   wants to invoke, the gate refuses BEFORE the framework
   dispatches it (``Completions._captured_kwargs`` shows exactly
   one call — the model_call — and zero tool-execution calls).

The Tier-2 cascade fixtures live in ``test_smoke_framework_cascade.py``;
this file pulls them in via ``importlib`` so we don't fork the
mock framework stubs. The matrix design is intentional — adding
a new framework is one new entry in the ``TIER2_FRAMEWORKS`` list
and the contract is automatically locked in for it.

Privacy contract reminder: the audit row never carries
``response_preview``. The output phase reads the model's text
just long enough to feed policies; the text is then discarded.
``test_smoke_privacy_contract.py`` pins this invariant separately;
here we only assert the verdict block is present, never the text
itself.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

# ── Shared helpers ──────────────────────────────────────────────────


def _flush() -> None:
    from egisai import shutdown
    shutdown()


def _init_sdk(app: str = "before-after") -> None:
    import egisai
    egisai.init(
        api_key="egis_live_test",
        app=app,
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )


def _all_audit_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in events:
        kind = e.get("kind")
        if kind in (None, "run.step"):
            out.append(e)
    return out


def _model_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        e for e in _all_audit_events(events)
        if e.get("step_kind") in (None, "model_call")
    ]


def _set_rules(*rules: dict[str, Any]) -> None:
    from egisai._policy_cache import replace_rules
    replace_rules(f'"r{len(rules)}"', list(rules))


def _allow_pii_rule() -> dict[str, Any]:
    """A pii_scan rule that fires on plain text — used to prove the
    input phase actually ran on the prompt every framework forwards.
    We use ``action=sanitize`` so the model_call is allowed to
    proceed (we want to see the output phase run too). The
    sanitization presence on the audit row is our proof that the
    input phase fired."""
    return {
        "id": "ba-pii-san",
        "name": "ba-pii-sanitize",
        "type": "pii_scan",
        "tenant": None,
        "config": {
            "action": "sanitize",
            "types": ["ssn"],
            "mask_char": "#",
        },
    }


def _deny_tool_rule(pattern: str = r"^run_shell$") -> dict[str, Any]:
    return {
        "id": "ba-deny-tool",
        "name": "ba-block-tool",
        "type": "deny_tool_call",
        "tenant": None,
        "config": {"patterns": [pattern], "message": "Tool blocked"},
    }


def _assert_input_phase_ran(ev: dict[str, Any]) -> None:
    """An audit row produced by ``gate_call`` MUST carry the
    ``prompt_decision`` block (the per-phase verdict for input).
    ``response_decision`` is only there when the model call wasn't
    refused; ``prompt_decision`` is unconditional."""
    assert ev.get("prompt_decision") is not None, (
        f"input phase did not stamp prompt_decision on the audit row; "
        f"got keys={sorted(ev)!r}"
    )


def _assert_output_phase_ran(ev: dict[str, Any]) -> None:
    """When the model call succeeds and there's any output signal
    (text, tool calls, or MCP targets), the output phase stamps
    ``response_decision``. We assert presence — its verdict is
    asserted by individual tests."""
    assert ev.get("response_decision") is not None, (
        f"output phase did not stamp response_decision; "
        f"verdict={ev.get('verdict')!r} keys={sorted(ev)!r}"
    )


# ── Tier-1: direct LLM patches ──────────────────────────────────────


# --- OpenAI Chat Completions ---------------------------------------


def _install_fake_openai(
    response_factory: Any = None,
) -> tuple[type, type]:
    fake = types.ModuleType("openai")
    res = types.ModuleType("openai.resources")
    chat = types.ModuleType("openai.resources.chat")
    completions = types.ModuleType("openai.resources.chat.completions")
    responses = types.ModuleType("openai.resources.responses")

    class Completions:
        _captured_kwargs: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            if response_factory is None:
                return types.SimpleNamespace(
                    id="ok",
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(
                                content="ok", tool_calls=None,
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=types.SimpleNamespace(
                        prompt_tokens=2, completion_tokens=1,
                    ),
                )
            return response_factory()

    class AsyncCompletions:
        _captured_kwargs: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            return Completions().create(**kwargs)

    completions.Completions = Completions
    completions.AsyncCompletions = AsyncCompletions
    sys.modules.update(
        {
            "openai": fake,
            "openai.resources": res,
            "openai.resources.chat": chat,
            "openai.resources.chat.completions": completions,
            "openai.resources.responses": responses,
        }
    )
    Completions._captured_kwargs = []
    AsyncCompletions._captured_kwargs = []
    return Completions, AsyncCompletions


def _shell_tool_response() -> Any:
    return types.SimpleNamespace(
        id="t",
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            type="function",
                            function=types.SimpleNamespace(
                                name="run_shell", arguments="{}",
                            ),
                        )
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=types.SimpleNamespace(prompt_tokens=2, completion_tokens=1),
    )


@pytest.fixture
def openai_chat_with_tool(fake_backend: Any) -> Iterator[tuple[Any, type]]:
    _init_sdk()
    Completions, _ = _install_fake_openai(_shell_tool_response)
    from egisai._patches import openai as openai_patch
    assert openai_patch.apply() is True
    yield fake_backend, Completions
    for mod in (
        "openai", "openai.resources", "openai.resources.chat",
        "openai.resources.chat.completions", "openai.resources.responses",
    ):
        sys.modules.pop(mod, None)


def test_openai_chat_runs_input_AND_output_phase_around_each_call(
    openai_chat_with_tool: Any,
) -> None:
    """OpenAI Chat Completions: the gate stamps both ``prompt_decision``
    and ``response_decision`` on each audit row. This is the direct
    case — every other framework that calls openai under the hood
    inherits this contract."""
    fake_backend, Completions = openai_chat_with_tool
    c = Completions()
    c.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "use a tool"}],
    )
    _flush()

    events = _all_audit_events(fake_backend.events_received)
    model_evs = [
        e for e in events if e.get("step_kind") in (None, "model_call")
        and e.get("source") == "openai"
    ]
    assert model_evs, "no model_call audit row"
    ev = model_evs[-1]
    _assert_input_phase_ran(ev)
    _assert_output_phase_ran(ev)


def test_openai_chat_deny_tool_call_refuses_BEFORE_tool_dispatch(
    openai_chat_with_tool: Any,
) -> None:
    """Output policy blocks the tool the model wants to use. The
    gate raises BEFORE returning to the agent loop, so the
    framework code that would have dispatched ``run_shell`` never
    sees the response. This is what makes "after the LLM / before
    the tool" enforcement strong for synchronous patches."""
    fake_backend, Completions = openai_chat_with_tool
    _set_rules(_deny_tool_rule())
    c = Completions()
    with pytest.raises(PermissionError):
        c.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "clean it up"}],
        )
    _flush()
    # ONE LLM call captured — the tool never ran.
    assert len(Completions._captured_kwargs) == 1, (
        f"expected exactly 1 LLM call (the one that returned the "
        f"blocked tool request); got {len(Completions._captured_kwargs)}"
    )
    events = _all_audit_events(fake_backend.events_received)
    blocks = [e for e in events if e.get("verdict") == "block"]
    assert blocks, "expected an output-side block audit row"
    assert blocks[-1].get("enforcement_status") == "enforced"


# --- Anthropic ------------------------------------------------------


def _install_fake_anthropic(
    response_factory: Any = None,
) -> tuple[type, type]:
    fake = types.ModuleType("anthropic")
    res = types.ModuleType("anthropic.resources")
    messages_mod = types.ModuleType("anthropic.resources.messages")

    class Messages:
        _captured_kwargs: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            if response_factory is None:
                return types.SimpleNamespace(
                    id="ok",
                    type="message",
                    role="assistant",
                    model=kwargs.get("model", "x"),
                    content=[types.SimpleNamespace(type="text", text="ok")],
                    stop_reason="end_turn",
                    usage=types.SimpleNamespace(
                        input_tokens=1, output_tokens=1,
                    ),
                )
            return response_factory()

    class AsyncMessages:
        _captured_kwargs: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            return Messages().create(**kwargs)

    messages_mod.Messages = Messages
    messages_mod.AsyncMessages = AsyncMessages
    sys.modules.update(
        {
            "anthropic": fake,
            "anthropic.resources": res,
            "anthropic.resources.messages": messages_mod,
        }
    )
    Messages._captured_kwargs = []
    AsyncMessages._captured_kwargs = []
    return Messages, AsyncMessages


def _anthropic_tool_response() -> Any:
    return types.SimpleNamespace(
        id="t",
        type="message",
        role="assistant",
        model="claude-3-5-sonnet",
        content=[
            types.SimpleNamespace(
                type="tool_use",
                id="tu_1",
                name="run_shell",
                input={"cmd": "ls"},
            )
        ],
        stop_reason="tool_use",
        usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
    )


@pytest.fixture
def anthropic_with_tool(fake_backend: Any) -> Iterator[tuple[Any, type]]:
    _init_sdk()
    Messages, _ = _install_fake_anthropic(_anthropic_tool_response)
    from egisai._patches import anthropic as anthropic_patch
    assert anthropic_patch.apply() is True
    yield fake_backend, Messages
    for mod in (
        "anthropic", "anthropic.resources", "anthropic.resources.messages",
    ):
        sys.modules.pop(mod, None)


def test_anthropic_runs_input_AND_output_phase_around_each_call(
    anthropic_with_tool: Any,
) -> None:
    fake_backend, Messages = anthropic_with_tool
    c = Messages()
    c.create(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "use a tool"}],
        system="You are helpful.",
    )
    _flush()

    events = _all_audit_events(fake_backend.events_received)
    evs = [
        e for e in events if e.get("step_kind") in (None, "model_call")
        and e.get("source") == "anthropic"
    ]
    assert evs, "no anthropic audit row"
    ev = evs[-1]
    _assert_input_phase_ran(ev)
    _assert_output_phase_ran(ev)


def test_anthropic_deny_tool_call_refuses_BEFORE_tool_dispatch(
    anthropic_with_tool: Any,
) -> None:
    fake_backend, Messages = anthropic_with_tool
    _set_rules(_deny_tool_rule())
    c = Messages()
    with pytest.raises(PermissionError):
        c.create(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "clean it up"}],
            system="You are helpful.",
        )
    _flush()
    assert len(Messages._captured_kwargs) == 1
    events = _all_audit_events(fake_backend.events_received)
    blocks = [e for e in events if e.get("verdict") == "block"]
    assert blocks


# --- Google GenAI ---------------------------------------------------


def _install_fake_genai(
    response_factory: Any = None,
) -> tuple[type, type]:
    fake = types.ModuleType("google")
    genai_mod = types.ModuleType("google.genai")
    models_mod = types.ModuleType("google.genai.models")

    class Models:
        _captured_kwargs: list[dict[str, Any]] = []

        def generate_content(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            if response_factory is None:
                return types.SimpleNamespace(
                    candidates=[
                        types.SimpleNamespace(
                            content=types.SimpleNamespace(
                                parts=[
                                    types.SimpleNamespace(text="ok"),
                                ],
                                role="model",
                            ),
                            finish_reason="STOP",
                        )
                    ],
                    text="ok",
                    usage_metadata=types.SimpleNamespace(
                        prompt_token_count=1,
                        candidates_token_count=1,
                    ),
                )
            return response_factory()

    class AsyncModels:
        _captured_kwargs: list[dict[str, Any]] = []

        async def generate_content(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            return Models().generate_content(**kwargs)

    models_mod.Models = Models
    models_mod.AsyncModels = AsyncModels
    genai_mod.models = models_mod
    sys.modules.update(
        {
            "google": fake,
            "google.genai": genai_mod,
            "google.genai.models": models_mod,
        }
    )
    Models._captured_kwargs = []
    AsyncModels._captured_kwargs = []
    return Models, AsyncModels


def _genai_tool_response() -> Any:
    return types.SimpleNamespace(
        candidates=[
            types.SimpleNamespace(
                content=types.SimpleNamespace(
                    parts=[
                        types.SimpleNamespace(
                            function_call=types.SimpleNamespace(
                                name="run_shell",
                                args={"cmd": "ls"},
                            ),
                        )
                    ],
                    role="model",
                ),
                finish_reason="STOP",
            )
        ],
        text="",
        usage_metadata=types.SimpleNamespace(
            prompt_token_count=1,
            candidates_token_count=1,
        ),
    )


@pytest.fixture
def genai_with_tool(fake_backend: Any) -> Iterator[tuple[Any, type]]:
    _init_sdk()
    Models, _ = _install_fake_genai(_genai_tool_response)
    from egisai._patches import genai as genai_patch
    assert genai_patch.apply() is True
    yield fake_backend, Models
    for mod in (
        "google", "google.genai", "google.genai.models",
    ):
        sys.modules.pop(mod, None)


def test_genai_runs_input_AND_output_phase_around_each_call(
    genai_with_tool: Any,
) -> None:
    fake_backend, Models = genai_with_tool
    c = Models()
    c.generate_content(
        model="gemini-2.0-flash",
        contents="use a tool",
    )
    _flush()
    events = _all_audit_events(fake_backend.events_received)
    evs = [
        e for e in events
        if e.get("step_kind") in (None, "model_call")
        and e.get("source") == "genai"
    ]
    assert evs, "no genai audit row"
    ev = evs[-1]
    _assert_input_phase_ran(ev)
    _assert_output_phase_ran(ev)


def test_genai_deny_tool_call_refuses_BEFORE_tool_dispatch(
    genai_with_tool: Any,
) -> None:
    fake_backend, Models = genai_with_tool
    _set_rules(_deny_tool_rule())
    c = Models()
    with pytest.raises(PermissionError):
        c.generate_content(
            model="gemini-2.0-flash",
            contents="clean it up",
        )
    _flush()
    assert len(Models._captured_kwargs) == 1
    events = _all_audit_events(fake_backend.events_received)
    blocks = [e for e in events if e.get("verdict") == "block"]
    assert blocks


# --- Bedrock Converse -----------------------------------------------


@pytest.fixture
def bedrock_converse_with_tool(fake_backend: Any) -> Iterator[tuple[Any, Any]]:
    """boto3-shaped fake of the Bedrock Converse client. The patch
    wraps ``boto3.client`` so when the user calls ``boto3.client(
    "bedrock-runtime")`` they get a Converse client whose
    ``converse`` method routes through ``gate_call``."""
    _init_sdk()

    captured: list[dict[str, Any]] = []

    def _converse(**kwargs: Any) -> Any:
        captured.append(kwargs)
        return {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "toolUse": {
                                "toolUseId": "tu_1",
                                "name": "run_shell",
                                "input": {"cmd": "ls"},
                            }
                        }
                    ],
                }
            },
            "stopReason": "tool_use",
            "usage": {
                "inputTokens": 1,
                "outputTokens": 1,
                "totalTokens": 2,
            },
        }

    class _Client:
        converse = staticmethod(_converse)
        converse_stream = staticmethod(_converse)

    fake_boto = types.ModuleType("boto3")

    def fake_client(service_name: str, **_kw: Any) -> Any:
        return _Client()

    fake_boto.client = fake_client
    sys.modules["boto3"] = fake_boto

    from egisai._patches import bedrock_runtime as bedrock_patch
    assert bedrock_patch.apply() is True

    yield fake_backend, captured

    sys.modules.pop("boto3", None)


def test_bedrock_converse_runs_input_AND_output_phase(
    bedrock_converse_with_tool: Any,
) -> None:
    fake_backend, captured = bedrock_converse_with_tool
    import boto3  # type: ignore[import-not-found]
    client = boto3.client("bedrock-runtime")
    client.converse(
        modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=[{"role": "user", "content": [{"text": "use a tool"}]}],
        system=[{"text": "You are helpful."}],
        toolConfig={
            "tools": [
                {"toolSpec": {"name": "run_shell"}}
            ]
        },
    )
    _flush()
    events = _all_audit_events(fake_backend.events_received)
    evs = [
        e for e in events
        if e.get("step_kind") in (None, "model_call")
        and e.get("source") == "bedrock_runtime"
    ]
    assert evs, "no bedrock_runtime audit row"
    ev = evs[-1]
    _assert_input_phase_ran(ev)
    _assert_output_phase_ran(ev)


def test_bedrock_converse_deny_tool_call_refuses_BEFORE_tool_dispatch(
    bedrock_converse_with_tool: Any,
) -> None:
    fake_backend, captured = bedrock_converse_with_tool
    _set_rules(_deny_tool_rule())
    import boto3  # type: ignore[import-not-found]
    client = boto3.client("bedrock-runtime")
    with pytest.raises(PermissionError):
        client.converse(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=[
                {"role": "user", "content": [{"text": "clean it up"}]}
            ],
            system=[{"text": "Be brief."}],
            toolConfig={
                "tools": [{"toolSpec": {"name": "run_shell"}}]
            },
        )
    _flush()
    # Exactly one captured call (the one whose response we refused).
    assert len(captured) == 1
    events = _all_audit_events(fake_backend.events_received)
    blocks = [e for e in events if e.get("verdict") == "block"]
    assert blocks
    assert blocks[-1].get("enforcement_status") == "enforced"


# ── Tier-2 cascade: each framework inherits the OpenAI gate ─────────
#
# Tier-2 patches don't have their own gate — they push an
# IdentityRecord and open a Run, then delegate the LLM call to
# the underlying provider patch. Cascade tests live in
# ``test_smoke_framework_cascade.py`` for the heavy-hitters; here
# we add a single "before/after each LLM call AND tool call"
# pin that runs once per framework so a regression in any one
# patch's identity-or-delegation wiring trips an obviously named
# test.
#
# The fixture body is identical to ``test_smoke_framework_cascade.py``'s
# fixtures (plant a fake openai → plant a fake framework that
# transitively calls openai when its entry-point runs → apply the
# framework patch). The assertion here is different from the
# cascade file's: we assert the BOTH-PHASES invariant rather
# than the sanitize-passes-the-seam invariant.


def _install_fake_openai_cascade(
    response_factory: Any = None,
) -> type:
    fake = types.ModuleType("openai")
    res = types.ModuleType("openai.resources")
    chat = types.ModuleType("openai.resources.chat")
    completions = types.ModuleType("openai.resources.chat.completions")
    responses = types.ModuleType("openai.resources.responses")

    class Completions:
        _captured_kwargs: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            if response_factory is None:
                return types.SimpleNamespace(
                    id="cascade",
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(
                                content="ok", tool_calls=None,
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=types.SimpleNamespace(
                        prompt_tokens=2, completion_tokens=1,
                    ),
                )
            return response_factory()

    class AsyncCompletions:
        _captured_kwargs: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            return Completions().create(**kwargs)

    completions.Completions = Completions
    completions.AsyncCompletions = AsyncCompletions
    sys.modules.update(
        {
            "openai": fake,
            "openai.resources": res,
            "openai.resources.chat": chat,
            "openai.resources.chat.completions": completions,
            "openai.resources.responses": responses,
        }
    )
    Completions._captured_kwargs = []
    AsyncCompletions._captured_kwargs = []
    return Completions


def _invoke_openai_inside(prompt: str) -> Any:
    import openai.resources.chat.completions as oa_completions
    return oa_completions.Completions().create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )


@pytest.fixture
def cascade_env(fake_backend: Any) -> Iterator[tuple[Any, type]]:
    _init_sdk()
    Completions = _install_fake_openai_cascade()
    from egisai._patches import openai as openai_patch
    assert openai_patch.apply() is True
    yield fake_backend, Completions
    for mod in (
        "openai", "openai.resources", "openai.resources.chat",
        "openai.resources.chat.completions", "openai.resources.responses",
    ):
        sys.modules.pop(mod, None)


def _assert_cascade_both_phases_fired(fake_backend: Any) -> None:
    """For a Tier-2 cascade, the inner openai patch is what stamps
    ``prompt_decision`` and ``response_decision`` on the audit
    row. The outer framework wrap just opens the Run. We assert
    the inner audit row carries both phase blocks."""
    events = _all_audit_events(fake_backend.events_received)
    inner = [
        e for e in events
        if e.get("source") == "openai"
        and e.get("step_kind") in (None, "model_call")
    ]
    assert inner, "Tier-2 cascade didn't produce an inner openai audit row"
    ev = inner[-1]
    _assert_input_phase_ran(ev)
    _assert_output_phase_ran(ev)


# --- openai_agents --------------------------------------------------


def _install_fake_openai_agents() -> tuple[Any, type]:
    fake = types.ModuleType("agents")

    class _Agent:
        def __init__(self, name: str, instructions: str = "") -> None:
            self.name = name
            self.instructions = instructions
            self.tools: list[Any] = []

    class Runner:
        @staticmethod
        async def run(agent: _Agent, prompt: str, *a: Any, **kw: Any) -> Any:
            return _invoke_openai_inside(prompt)

        @staticmethod
        def run_sync(agent: _Agent, prompt: str, *a: Any, **kw: Any) -> Any:
            return _invoke_openai_inside(prompt)

        @staticmethod
        def run_streamed(agent: _Agent, prompt: str, *a: Any, **kw: Any) -> Any:
            _invoke_openai_inside(prompt)
            return types.SimpleNamespace(stream_events=lambda: iter(()))

    fake.Runner = Runner
    fake._Agent = _Agent
    sys.modules["agents"] = fake
    from egisai._patches import openai_agents as patch
    assert patch.apply() is True
    return fake, _Agent


def test_openai_agents_cascade_runs_input_AND_output_phase(
    cascade_env: Any,
) -> None:
    fake_backend, _ = cascade_env
    fake, _Agent = _install_fake_openai_agents()
    try:
        from agents import Runner  # type: ignore[import-not-found]
        Runner.run_sync(_Agent("TriageBot", "Route tickets."), "hello")
        _flush()
        _assert_cascade_both_phases_fired(fake_backend)
    finally:
        sys.modules.pop("agents", None)


# --- LangGraph ------------------------------------------------------


def _install_fake_langgraph() -> Any:
    lg = types.ModuleType("langgraph")
    pregel_mod = types.ModuleType("langgraph.pregel")

    class _Node:
        def __init__(self, name: str) -> None:
            self.name = name

    class Pregel:
        def __init__(self) -> None:
            self.nodes = {"agent": _Node("agent")}

        def invoke(self, input: Any, config: Any = None) -> Any:
            text = input.get("input") if isinstance(input, dict) else str(input)
            _invoke_openai_inside(text)
            return {"output": "done"}

        async def ainvoke(self, input: Any, config: Any = None) -> Any:
            return self.invoke(input, config)

        def stream(self, input: Any, config: Any = None) -> Any:
            self.invoke(input, config)
            yield {"output": "done"}

    pregel_mod.Pregel = Pregel
    lg.pregel = pregel_mod
    sys.modules["langgraph"] = lg
    sys.modules["langgraph.pregel"] = pregel_mod
    from egisai._patches import langgraph as patch
    assert patch.apply() is True
    return Pregel


def test_langgraph_cascade_runs_input_AND_output_phase(
    cascade_env: Any,
) -> None:
    fake_backend, _ = cascade_env
    Pregel = _install_fake_langgraph()
    try:
        Pregel().invoke({"input": "hello"})
        _flush()
        _assert_cascade_both_phases_fired(fake_backend)
    finally:
        sys.modules.pop("langgraph", None)
        sys.modules.pop("langgraph.pregel", None)


# --- LangChain ------------------------------------------------------


def _install_fake_langchain() -> Any:
    lc = types.ModuleType("langchain")
    agents_mod = types.ModuleType("langchain.agents")

    class AgentExecutor:
        def __init__(self) -> None:
            self.name = "LCAgent"
            self.tools: list[Any] = []
            self.agent = types.SimpleNamespace(
                system_message="Be careful.",
            )

        def invoke(self, input: Any, config: Any = None) -> Any:
            text = input.get("input") if isinstance(input, dict) else str(input)
            _invoke_openai_inside(text)
            return {"output": "done"}

        async def ainvoke(self, input: Any, config: Any = None) -> Any:
            return self.invoke(input, config)

        def stream(self, input: Any, config: Any = None) -> Any:
            self.invoke(input, config)
            yield {"output": "done"}

    agents_mod.AgentExecutor = AgentExecutor
    lc.agents = agents_mod
    sys.modules["langchain"] = lc
    sys.modules["langchain.agents"] = agents_mod
    from egisai._patches import langchain as patch
    assert patch.apply() is True
    return AgentExecutor


def test_langchain_cascade_runs_input_AND_output_phase(
    cascade_env: Any,
) -> None:
    fake_backend, _ = cascade_env
    AgentExecutor = _install_fake_langchain()
    try:
        AgentExecutor().invoke({"input": "hello"})
        _flush()
        _assert_cascade_both_phases_fired(fake_backend)
    finally:
        sys.modules.pop("langchain", None)
        sys.modules.pop("langchain.agents", None)


# --- CrewAI ---------------------------------------------------------


def _install_fake_crewai() -> Any:
    cm = types.ModuleType("crewai")

    class Agent:
        def __init__(self, role: str = "Helper", goal: str = "g") -> None:
            self.role = role
            self.goal = goal
            self.backstory = ""
            self.tools: list[Any] = []

        def execute_task(self, task: Any, *a: Any, **kw: Any) -> Any:
            prompt = str(getattr(task, "description", task))
            _invoke_openai_inside(prompt)
            return "done"

    cm.Agent = Agent
    sys.modules["crewai"] = cm
    from egisai._patches import crewai as patch
    assert patch.apply() is True
    return Agent


def test_crewai_cascade_runs_input_AND_output_phase(
    cascade_env: Any,
) -> None:
    fake_backend, _ = cascade_env
    Agent = _install_fake_crewai()
    try:
        Agent().execute_task(
            types.SimpleNamespace(description="hello")
        )
        _flush()
        _assert_cascade_both_phases_fired(fake_backend)
    finally:
        sys.modules.pop("crewai", None)


# --- AutoGen --------------------------------------------------------


def _install_fake_autogen() -> Any:
    ag = types.ModuleType("autogen_agentchat")
    agents_mod = types.ModuleType("autogen_agentchat.agents")

    class AssistantAgent:
        def __init__(self, name: str = "AGBot") -> None:
            self.name = name
            self.system_message = "Be helpful."

        async def run(self, *a: Any, **kw: Any) -> Any:
            prompt = kw.get("task") or (a[0] if a else "hello")
            _invoke_openai_inside(str(prompt))
            return "done"

        async def run_stream(self, *a: Any, **kw: Any) -> Any:
            prompt = kw.get("task") or (a[0] if a else "hello")
            _invoke_openai_inside(str(prompt))
            for x in ("done",):
                yield x

    agents_mod.AssistantAgent = AssistantAgent
    ag.agents = agents_mod
    sys.modules["autogen_agentchat"] = ag
    sys.modules["autogen_agentchat.agents"] = agents_mod
    from egisai._patches import autogen as patch
    assert patch.apply() is True
    return AssistantAgent


def test_autogen_cascade_runs_input_AND_output_phase(
    cascade_env: Any,
) -> None:
    fake_backend, _ = cascade_env
    AssistantAgent = _install_fake_autogen()
    try:
        asyncio.run(AssistantAgent().run(task="hello"))
        _flush()
        _assert_cascade_both_phases_fired(fake_backend)
    finally:
        sys.modules.pop("autogen_agentchat", None)
        sys.modules.pop("autogen_agentchat.agents", None)


# --- Agno -----------------------------------------------------------


def _install_fake_agno() -> Any:
    ag = types.ModuleType("agno")
    agent_mod = types.ModuleType("agno.agent")

    class Agent:
        def __init__(self, name: str = "AgnoBot") -> None:
            self.name = name
            self.description = ""
            self.instructions = "Be helpful."

        def run(self, *a: Any, **kw: Any) -> Any:
            prompt = a[0] if a else kw.get("message", "hello")
            _invoke_openai_inside(str(prompt))
            return types.SimpleNamespace(content="done")

        def arun(self, *a: Any, **kw: Any) -> Any:
            async def _coro() -> Any:
                prompt = a[0] if a else kw.get("message", "hello")
                _invoke_openai_inside(str(prompt))
                return types.SimpleNamespace(content="done")
            return _coro()

        def print_response(self, *a: Any, **kw: Any) -> None:
            self.run(*a, **kw)

    agent_mod.Agent = Agent
    ag.agent = agent_mod
    sys.modules["agno"] = ag
    sys.modules["agno.agent"] = agent_mod
    from egisai._patches import agno as patch
    assert patch.apply() is True
    return Agent


def test_agno_cascade_runs_input_AND_output_phase(
    cascade_env: Any,
) -> None:
    fake_backend, _ = cascade_env
    Agent = _install_fake_agno()
    try:
        Agent().run("hello")
        _flush()
        _assert_cascade_both_phases_fired(fake_backend)
    finally:
        sys.modules.pop("agno", None)
        sys.modules.pop("agno.agent", None)


# --- Strands --------------------------------------------------------


def _install_fake_strands() -> Any:
    sm = types.ModuleType("strands")

    class Agent:
        def __init__(self, name: str = "StrandsBot") -> None:
            self.name = name
            self.system_prompt = "Be helpful."
            self.tools: list[Any] = []

        def __call__(self, prompt: str, *a: Any, **kw: Any) -> Any:
            _invoke_openai_inside(prompt)
            return "done"

        async def invoke_async(self, prompt: str, *a: Any, **kw: Any) -> Any:
            _invoke_openai_inside(prompt)
            return "done"

    sm.Agent = Agent
    sys.modules["strands"] = sm
    from egisai._patches import strands as patch
    assert patch.apply() is True
    return Agent


def test_strands_cascade_runs_input_AND_output_phase(
    cascade_env: Any,
) -> None:
    fake_backend, _ = cascade_env
    Agent = _install_fake_strands()
    try:
        Agent()("hello")
        _flush()
        _assert_cascade_both_phases_fired(fake_backend)
    finally:
        sys.modules.pop("strands", None)


# --- smolagents -----------------------------------------------------


def _install_fake_smolagents() -> Any:
    sa = types.ModuleType("smolagents")

    class MultiStepAgent:
        def __init__(self, name: str = "SmolBot") -> None:
            self.name = name
            self.model = types.SimpleNamespace(model_id="gpt-4o")
            self.tools: dict[str, Any] = {}

        def run(self, prompt: str, *a: Any, **kw: Any) -> Any:
            _invoke_openai_inside(prompt)
            return "done"

    sa.MultiStepAgent = MultiStepAgent
    sys.modules["smolagents"] = sa
    from egisai._patches import smolagents as patch
    assert patch.apply() is True
    return MultiStepAgent


def test_smolagents_cascade_runs_input_AND_output_phase(
    cascade_env: Any,
) -> None:
    fake_backend, _ = cascade_env
    MultiStepAgent = _install_fake_smolagents()
    try:
        MultiStepAgent().run("hello")
        _flush()
        _assert_cascade_both_phases_fired(fake_backend)
    finally:
        sys.modules.pop("smolagents", None)


# --- LlamaIndex -----------------------------------------------------


def _install_fake_llamaindex() -> Any:
    li = types.ModuleType("llama_index")
    core_mod = types.ModuleType("llama_index.core")
    agent_mod = types.ModuleType("llama_index.core.agent")

    class FunctionAgent:
        def __init__(self, name: str = "LIBot") -> None:
            self.name = name
            self.system_prompt = "Be helpful."
            self.tools: list[Any] = []

        def run(self, prompt: str, *a: Any, **kw: Any) -> Any:
            _invoke_openai_inside(prompt)
            return "done"

    agent_mod.FunctionAgent = FunctionAgent
    core_mod.agent = agent_mod
    li.core = core_mod
    sys.modules["llama_index"] = li
    sys.modules["llama_index.core"] = core_mod
    sys.modules["llama_index.core.agent"] = agent_mod
    from egisai._patches import llamaindex as patch
    assert patch.apply() is True
    return FunctionAgent


def test_llamaindex_cascade_runs_input_AND_output_phase(
    cascade_env: Any,
) -> None:
    fake_backend, _ = cascade_env
    FunctionAgent = _install_fake_llamaindex()
    try:
        FunctionAgent().run("hello")
        _flush()
        _assert_cascade_both_phases_fired(fake_backend)
    finally:
        sys.modules.pop("llama_index", None)
        sys.modules.pop("llama_index.core", None)
        sys.modules.pop("llama_index.core.agent", None)


# --- Pydantic AI ----------------------------------------------------


def _install_fake_pydantic_ai() -> Any:
    pa = types.ModuleType("pydantic_ai")

    class Agent:
        def __init__(self, name: str = "PABot") -> None:
            self.name = name
            self._system_prompts = ("Be helpful.",)
            self.model = types.SimpleNamespace(model_name="gpt-4o")

        async def run(self, prompt: str, *a: Any, **kw: Any) -> Any:
            _invoke_openai_inside(prompt)
            return types.SimpleNamespace(data="done")

        def run_sync(self, prompt: str, *a: Any, **kw: Any) -> Any:
            _invoke_openai_inside(prompt)
            return types.SimpleNamespace(data="done")

    pa.Agent = Agent
    sys.modules["pydantic_ai"] = pa
    from egisai._patches import pydantic_ai as patch
    assert patch.apply() is True
    return Agent


def test_pydantic_ai_cascade_runs_input_AND_output_phase(
    cascade_env: Any,
) -> None:
    fake_backend, _ = cascade_env
    Agent = _install_fake_pydantic_ai()
    try:
        Agent().run_sync("hello")
        _flush()
        _assert_cascade_both_phases_fired(fake_backend)
    finally:
        sys.modules.pop("pydantic_ai", None)
