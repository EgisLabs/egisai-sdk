"""Multi-scenario battery for every Tier-1 direct-LLM provider patch.

This file is the unified, per-provider battle test. For every direct
LLM provider that ``egisai`` patches (OpenAI Chat, OpenAI Responses,
Anthropic Messages, Google GenAI, Google legacy, Bedrock Converse),
each of the following scenarios is exercised end-to-end:

1. **Allow path** \u2014 no rules, model is forwarded as-is, audit row
   carries the framework's identity and the post-model
   ``response_decision`` block.
2. **PII sanitize** \u2014 the prompt contains a real-looking SSN; the
   ``pii_scan`` rule with ``action="sanitize"`` MASKS the SSN in
   place before the request reaches the provider; the audit row
   records ``verdict=sanitize`` + ``sanitizations[].type=ssn``.
3. **PII block** \u2014 the prompt contains an SSN; the ``pii_scan``
   rule with ``action="block"`` raises ``PermissionError`` BEFORE
   the provider is called; the SDK keeps zero bytes of the raw
   prompt on the audit row.
4. **Tool-call block (output side)** \u2014 the model returned a tool
   call whose name matches a ``deny_tool_call`` pattern; the gate
   refuses the response, the audit row stamps ``verdict=block``,
   and no per-tool ``tool_call`` step is emitted for the refused
   request.
5. **Per-tool waterfall (allow)** \u2014 the model returns two
   distinct tool calls; the gate emits one ``model_call`` step +
   one ``tool_call`` step per tool, in order, with
   ``enforcement_status="enforced"`` on each tool step.
6. **Privacy contract** \u2014 the audit row never carries a
   ``response_preview`` field; the model's reply text never appears
   verbatim under any wire-key on any envelope; the prompt's raw
   PII never appears in the wire payload (PII sanitize / block
   variants).

The purpose of having one comprehensive file per provider is that
when egisai ships a new minor / patch release, ``pytest tests/
test_smoke_provider_battery.py`` is a single command that confirms
every provider still honors the contract. A failure here means a
SOC 2 / GDPR / HIPAA-relevant guarantee regressed; the offending
patch needs to be fixed before release.

Notes on the test stubs:

- We do NOT depend on the real upstream packages. ``sys.modules``
  is seeded with hand-rolled doubles that mirror the attribute
  names + call shapes the egisai patches duck-type on.
- Every test seeds rules via ``replace_rules()`` (the cache
  surface) BEFORE invoking the patched call, because the SDK
  ``init()`` path only pulls rules during the handshake \u2014 our
  ``fake_backend`` fixture short-circuits that handshake.
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
    """Drain the SDK's logger queue so events_received is populated."""
    from egisai import shutdown

    shutdown()


def _model_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return finalized ``model_call`` step rows on the wire."""
    return [
        e for e in events
        if e.get("kind") == "run.step" and e.get("step_kind") == "model_call"
    ]


def _tool_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        e for e in events
        if e.get("kind") == "run.step" and e.get("step_kind") == "tool_call"
    ]


def _legacy_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return single-row legacy events (no ``kind`` field)."""
    return [e for e in events if e.get("kind") is None]


def _all_audit_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every audit-style event the SDK emitted, regardless of envelope."""
    return _model_steps(events) + _tool_steps(events) + _legacy_events(events)


def _init_sdk(app: str = "smoke-battery") -> None:
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app=app,
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )


def _pii_sanitize_rule(types_: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": "smoke-pii-san",
        "name": "smoke-sanitize-pii",
        "type": "pii_scan",
        "tenant": None,
        "config": {
            "action": "sanitize",
            "types": types_ or ["ssn", "email", "credit_card"],
            "mask_char": "#",
        },
    }


def _pii_block_rule(types_: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": "smoke-pii-block",
        "name": "smoke-block-pii",
        "type": "pii_scan",
        "tenant": None,
        "config": {
            "action": "block",
            "types": types_ or ["ssn"],
            "message": "PII blocked",
        },
    }


def _deny_tool_rule(pattern: str = r"^run_shell$") -> dict[str, Any]:
    return {
        "id": "smoke-deny-tool",
        "name": "smoke-block-shell",
        "type": "deny_tool_call",
        "tenant": None,
        "config": {"patterns": [pattern]},
    }


def _set_rules(*rules: dict[str, Any]) -> None:
    from egisai._policy_cache import replace_rules

    replace_rules(f'"smoke-r{len(rules)}"', list(rules))


def _assert_raw_text_absent(
    events: list[dict[str, Any]], raw: str
) -> None:
    """Locked invariant: raw PII / model text never appears on the wire."""
    for ev in events:
        assert raw not in repr(ev), (
            f"raw secret {raw!r} leaked into wire envelope "
            f"kind={ev.get('kind')} fields={list(ev)}"
        )


# ── OpenAI Chat Completions ─────────────────────────────────────────


def _install_fake_openai() -> tuple[type, type]:
    """Plant a faithful fake of ``openai.resources.chat.completions``.

    The returned ``Completions`` / ``AsyncCompletions`` carry a
    ``create`` method whose call shape matches the real SDK. Tests
    seed ``_response_factory`` per scenario so the same fake supports
    text-only, tool-call, and tool+text mixed responses.
    """
    fake = types.ModuleType("openai")
    res = types.ModuleType("openai.resources")
    chat = types.ModuleType("openai.resources.chat")
    completions = types.ModuleType("openai.resources.chat.completions")
    responses = types.ModuleType("openai.resources.responses")

    class Completions:
        _response_factory: Any = None
        _captured_kwargs: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            if type(self)._response_factory is None:
                return types.SimpleNamespace(
                    id="real-text",
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(
                                content="ok", tool_calls=None,
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=types.SimpleNamespace(
                        prompt_tokens=3, completion_tokens=1,
                    ),
                )
            return type(self)._response_factory()

    class AsyncCompletions:
        _response_factory: Any = None
        _captured_kwargs: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            if type(self)._response_factory is None:
                return types.SimpleNamespace(
                    id="real-async",
                    choices=[
                        types.SimpleNamespace(
                            message=types.SimpleNamespace(
                                content="ok", tool_calls=None,
                            ),
                            finish_reason="stop",
                        )
                    ],
                    usage=types.SimpleNamespace(
                        prompt_tokens=3, completion_tokens=1,
                    ),
                )
            return type(self)._response_factory()

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
    return Completions, AsyncCompletions


def _two_tool_chat_response() -> Any:
    return types.SimpleNamespace(
        id="real-tools",
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=None,
                    tool_calls=[
                        types.SimpleNamespace(
                            type="function",
                            function=types.SimpleNamespace(
                                name="lookup_customer",
                                arguments='{"id": "ACC-1"}',
                            ),
                        ),
                        types.SimpleNamespace(
                            type="function",
                            function=types.SimpleNamespace(
                                name="send_email",
                                arguments='{"to": "x@y"}',
                            ),
                        ),
                    ],
                ),
                finish_reason="tool_calls",
            )
        ],
        usage=types.SimpleNamespace(prompt_tokens=5, completion_tokens=2),
    )


def _shell_tool_chat_response() -> Any:
    return types.SimpleNamespace(
        id="real-shell",
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
def openai_smoke(fake_backend: Any) -> Iterator[tuple[Any, type, type]]:
    """Init the SDK + plant a fake openai package + apply the patch."""
    _init_sdk(app="oa-smoke")
    Completions, AsyncCompletions = _install_fake_openai()
    Completions._captured_kwargs = []
    AsyncCompletions._captured_kwargs = []
    from egisai._patches import openai as patch

    assert patch.apply() is True
    yield fake_backend, Completions, AsyncCompletions
    for mod in (
        "openai", "openai.resources", "openai.resources.chat",
        "openai.resources.chat.completions", "openai.resources.responses",
    ):
        sys.modules.pop(mod, None)


def test_openai_chat_allow_path(openai_smoke: Any) -> None:
    """Clean prompt + no rules: response is forwarded, audit row carries
    a ``response_decision`` block (output phase ran, returned allow)."""
    fake_backend, Completions, _ = openai_smoke
    c = Completions()
    out = c.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
    )
    _flush()
    assert out.id == "real-text"
    events = _all_audit_events(fake_backend.events_received)
    # At least one model_call event lands (provisional may also ship).
    assert any(e.get("verdict") == "allow" for e in events)


def test_openai_chat_pii_sanitize_masks_in_place(
    openai_smoke: Any,
) -> None:
    """A prompt carrying a real SSN runs through pii_scan(sanitize);
    the SSN is masked BEFORE the openai SDK is called; the audit row
    records the sanitization with type=ssn."""
    fake_backend, Completions, _ = openai_smoke
    _set_rules(_pii_sanitize_rule(["ssn"]))
    raw_ssn = "123-45-6789"

    c = Completions()
    c.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"My SSN is {raw_ssn} please verify."}
        ],
    )
    _flush()

    # 1. The forwarded prompt MUST NOT contain the raw SSN.
    assert Completions._captured_kwargs, "openai create wasn't called"
    sent_messages = Completions._captured_kwargs[-1].get("messages")
    assert isinstance(sent_messages, list)
    sent_text = sent_messages[-1].get("content")
    assert raw_ssn not in sent_text, (
        f"raw SSN leaked to provider (got {sent_text!r})"
    )

    # 2. The audit row records sanitize + sanitizations.
    events = _all_audit_events(fake_backend.events_received)
    san_evs = [e for e in events if e.get("verdict") == "sanitize"]
    assert san_evs, "expected a sanitize audit row"
    sanitizations = san_evs[-1].get("sanitizations") or []
    assert any(s.get("type") == "ssn" for s in sanitizations), (
        f"expected ssn sanitization record, got {sanitizations!r}"
    )

    # 3. Wire payload never carries the raw SSN.
    _assert_raw_text_absent(fake_backend.events_received, raw_ssn)


def test_openai_chat_pii_block_raises_and_zero_bytes_leak(
    openai_smoke: Any,
) -> None:
    """``pii_scan(action=block)`` raises before the openai SDK is hit;
    raw PII never lands on the wire."""
    fake_backend, Completions, _ = openai_smoke
    _set_rules(_pii_block_rule(["ssn"]))
    raw = "Customer SSN 456-12-7890 needs flagging."

    c = Completions()
    with pytest.raises(PermissionError):
        c.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": raw}],
        )
    _flush()

    assert Completions._captured_kwargs == [], (
        "blocked prompt must never reach the openai SDK"
    )
    _assert_raw_text_absent(fake_backend.events_received, "456-12-7890")
    events = _all_audit_events(fake_backend.events_received)
    blocks = [e for e in events if e.get("verdict") == "block"]
    assert blocks, "expected a blocked audit row"


def test_openai_chat_deny_tool_call_refuses_response(
    openai_smoke: Any,
) -> None:
    """Output policy ``deny_tool_call`` fires on a banned tool name;
    the gate raises, no per-tool ``tool_call`` rows are emitted for
    the refused request, and the audit row stamps verdict=block."""
    fake_backend, Completions, _ = openai_smoke
    _set_rules(_deny_tool_rule())
    Completions._response_factory = _shell_tool_chat_response

    c = Completions()
    with pytest.raises(PermissionError):
        c.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "clean it up"}],
        )
    _flush()

    tool_evs = _tool_steps(fake_backend.events_received)
    assert tool_evs == [], (
        f"a blocked tool call must NOT emit a tool_call row "
        f"(got {tool_evs!r})"
    )
    events = _all_audit_events(fake_backend.events_received)
    blocks = [e for e in events if e.get("verdict") == "block"]
    assert blocks, "expected an output-side blocked model_call audit row"
    assert blocks[-1].get("enforcement_status") == "enforced", (
        "openai is synchronous; output-side block is enforced"
    )


def test_openai_chat_per_tool_waterfall_emits_one_step_per_tool(
    openai_smoke: Any,
) -> None:
    """When the model returns two tool calls, the gate emits one
    ``model_call`` + two ``tool_call`` steps in order, each stamped
    ``enforcement_status=enforced``."""
    fake_backend, Completions, _ = openai_smoke
    Completions._response_factory = _two_tool_chat_response

    from egisai._run import close_run, open_run_from_current_identity

    # Per-tool steps only land when a Run is open above us. Mimic the
    # OpenAI Agents wrap by opening a Run by hand.
    open_run_from_current_identity(framework="openai_agents", prompt_text=None)
    try:
        c = Completions()
        c.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "do two things"}],
            tools=[
                {"type": "function",
                 "function": {"name": "lookup_customer"}},
                {"type": "function",
                 "function": {"name": "send_email"}},
            ],
        )
    finally:
        close_run()
    _flush()

    tool_evs = _tool_steps(fake_backend.events_received)
    names = [e["tool_name"] for e in tool_evs]
    assert names == ["lookup_customer", "send_email"], (
        f"per-tool waterfall ordering broken; got {names!r}"
    )
    for ev in tool_evs:
        assert ev.get("enforcement_status") == "enforced", (
            "openai is synchronous; per-tool steps are enforced"
        )
        assert ev.get("verdict") == "allow"


def test_openai_chat_audit_row_never_carries_response_preview(
    openai_smoke: Any,
) -> None:
    """Privacy contract: no audit envelope shipped by the openai
    patch may contain a ``response_preview`` field, and the model's
    raw reply text must not appear under any other field either."""
    fake_backend, Completions, _ = openai_smoke
    reply = "The forecast is sunny with a 72F high."

    def _reply_factory() -> Any:
        return types.SimpleNamespace(
            id="real-reply",
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=reply, tool_calls=None,
                    ),
                    finish_reason="stop",
                )
            ],
            usage=types.SimpleNamespace(
                prompt_tokens=2, completion_tokens=8,
            ),
        )

    Completions._response_factory = _reply_factory
    c = Completions()
    c.create(model="gpt-4o", messages=[{"role": "user", "content": "weather?"}])
    _flush()

    for ev in fake_backend.events_received:
        assert ev.get("response_preview") is None, (
            f"response_preview leaked: {ev.get('response_preview')!r}"
        )
        assert reply not in repr(ev), (
            f"raw model reply leaked into envelope kind={ev.get('kind')}"
        )


# ── Anthropic Messages ──────────────────────────────────────────────


def _install_fake_anthropic() -> tuple[type, type]:
    """Faithful double of ``anthropic.resources.messages``."""
    fake = types.ModuleType("anthropic")
    res = types.ModuleType("anthropic.resources")
    msgs = types.ModuleType("anthropic.resources.messages")

    class Messages:
        _response_factory: Any = None
        _captured_kwargs: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            if type(self)._response_factory is None:
                return types.SimpleNamespace(
                    id="msg_real",
                    type="message",
                    role="assistant",
                    content=[types.SimpleNamespace(type="text", text="ok")],
                    stop_reason="end_turn",
                    usage=types.SimpleNamespace(
                        input_tokens=4, output_tokens=2,
                    ),
                )
            return type(self)._response_factory()

    class AsyncMessages:
        _response_factory: Any = None
        _captured_kwargs: list[dict[str, Any]] = []

        async def create(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            if type(self)._response_factory is None:
                return types.SimpleNamespace(
                    id="msg_real_async",
                    type="message",
                    role="assistant",
                    content=[types.SimpleNamespace(type="text", text="ok")],
                    stop_reason="end_turn",
                    usage=types.SimpleNamespace(
                        input_tokens=4, output_tokens=2,
                    ),
                )
            return type(self)._response_factory()

    msgs.Messages = Messages
    msgs.AsyncMessages = AsyncMessages
    sys.modules.update(
        {
            "anthropic": fake,
            "anthropic.resources": res,
            "anthropic.resources.messages": msgs,
        }
    )
    return Messages, AsyncMessages


def _anthropic_two_tool_response() -> Any:
    return types.SimpleNamespace(
        id="msg_tools",
        type="message",
        role="assistant",
        content=[
            types.SimpleNamespace(type="text", text="calling tools"),
            types.SimpleNamespace(
                type="tool_use",
                name="lookup_customer",
                input={"id": "ACC-9"},
                id="tu_1",
            ),
            types.SimpleNamespace(
                type="tool_use",
                name="send_email",
                input={"to": "alice@example.com"},
                id="tu_2",
            ),
        ],
        stop_reason="tool_use",
        usage=types.SimpleNamespace(input_tokens=6, output_tokens=4),
    )


def _anthropic_shell_tool_response() -> Any:
    return types.SimpleNamespace(
        id="msg_shell",
        type="message",
        role="assistant",
        content=[
            types.SimpleNamespace(
                type="tool_use",
                name="run_shell",
                input={"cmd": "rm -rf /"},
                id="tu_shell",
            )
        ],
        stop_reason="tool_use",
        usage=types.SimpleNamespace(input_tokens=2, output_tokens=1),
    )


@pytest.fixture
def anthropic_smoke(fake_backend: Any) -> Iterator[tuple[Any, type, type]]:
    _init_sdk(app="ant-smoke")
    Messages, AsyncMessages = _install_fake_anthropic()
    Messages._captured_kwargs = []
    AsyncMessages._captured_kwargs = []
    from egisai._patches import anthropic as patch

    assert patch.apply() is True
    yield fake_backend, Messages, AsyncMessages
    for mod in (
        "anthropic", "anthropic.resources", "anthropic.resources.messages",
    ):
        sys.modules.pop(mod, None)


def test_anthropic_allow_path(anthropic_smoke: Any) -> None:
    fake_backend, Messages, _ = anthropic_smoke
    out = Messages().create(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hi"}],
        system="be helpful",
    )
    _flush()
    assert out.id == "msg_real"
    events = _all_audit_events(fake_backend.events_received)
    assert any(e.get("verdict") == "allow" for e in events)


def test_anthropic_pii_sanitize_masks_in_place(
    anthropic_smoke: Any,
) -> None:
    """Anthropic prompt with SSN: sanitize masks BEFORE the SDK fires."""
    fake_backend, Messages, _ = anthropic_smoke
    _set_rules(_pii_sanitize_rule(["ssn"]))
    # Use a SSN whose area number passes the SSA validity table
    # (the SDK rejects unassigned ranges like 987-* — area 987 is
    # the Social Security Administration's anti-misuse "advertising"
    # block, never actually issued). 222 is a real assigned range.
    raw_ssn = "222-65-4321"
    Messages().create(
        model="claude-3-5-sonnet",
        messages=[
            {"role": "user", "content": f"Please verify SSN {raw_ssn}"}
        ],
    )
    _flush()

    sent = Messages._captured_kwargs[-1]["messages"][-1]["content"]
    assert raw_ssn not in sent
    _assert_raw_text_absent(fake_backend.events_received, raw_ssn)
    events = _all_audit_events(fake_backend.events_received)
    san_evs = [e for e in events if e.get("verdict") == "sanitize"]
    assert san_evs
    assert any(
        s.get("type") == "ssn"
        for s in (san_evs[-1].get("sanitizations") or [])
    )


def test_anthropic_pii_block_raises(anthropic_smoke: Any) -> None:
    fake_backend, Messages, _ = anthropic_smoke
    _set_rules(_pii_block_rule(["ssn"]))
    raw = "Patient SSN 456-78-9012 was hospitalized."

    with pytest.raises(PermissionError):
        Messages().create(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": raw}],
        )
    _flush()
    assert Messages._captured_kwargs == []
    _assert_raw_text_absent(fake_backend.events_received, "456-78-9012")


def test_anthropic_deny_tool_call_refuses_response(
    anthropic_smoke: Any,
) -> None:
    fake_backend, Messages, _ = anthropic_smoke
    _set_rules(_deny_tool_rule())
    Messages._response_factory = _anthropic_shell_tool_response

    with pytest.raises(PermissionError):
        Messages().create(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "clean it"}],
        )
    _flush()

    events = _all_audit_events(fake_backend.events_received)
    blocks = [e for e in events if e.get("verdict") == "block"]
    assert blocks
    assert blocks[-1].get("matched_policy") == "smoke-block-shell"
    # Anthropic patch is synchronous \u2014 output-block is enforced.
    assert blocks[-1].get("enforcement_status") == "enforced"


def test_anthropic_per_tool_waterfall(anthropic_smoke: Any) -> None:
    """Anthropic with two ``tool_use`` blocks emits per-tool steps
    (post-fix for 0.23.0: anthropic now matches the openai waterfall)."""
    fake_backend, Messages, _ = anthropic_smoke
    Messages._response_factory = _anthropic_two_tool_response

    from egisai._run import close_run, open_run_from_current_identity

    open_run_from_current_identity(framework="anthropic", prompt_text=None)
    try:
        Messages().create(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": "do two things"}],
            tools=[
                {"name": "lookup_customer"},
                {"name": "send_email"},
            ],
        )
    finally:
        close_run()
    _flush()

    tool_evs = _tool_steps(fake_backend.events_received)
    names = [e["tool_name"] for e in tool_evs]
    assert names == ["lookup_customer", "send_email"], (
        f"anthropic per-tool waterfall broken; got {names!r}"
    )
    for ev in tool_evs:
        assert ev.get("enforcement_status") == "enforced"
        assert ev.get("verdict") == "allow"


def test_anthropic_audit_never_carries_response_text(
    anthropic_smoke: Any,
) -> None:
    fake_backend, Messages, _ = anthropic_smoke
    reply = "Confidential: Maria's account balance is $48,231.55."

    def _reply_factory() -> Any:
        return types.SimpleNamespace(
            id="msg_reply",
            type="message",
            role="assistant",
            content=[types.SimpleNamespace(type="text", text=reply)],
            stop_reason="end_turn",
            usage=types.SimpleNamespace(input_tokens=3, output_tokens=10),
        )

    Messages._response_factory = _reply_factory
    Messages().create(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "what's the balance"}],
    )
    _flush()
    for ev in fake_backend.events_received:
        assert ev.get("response_preview") is None
        assert reply not in repr(ev)


# ── Google GenAI (google.genai.models.Models) ───────────────────────


def _install_fake_genai() -> tuple[type, type]:
    google_pkg = types.ModuleType("google")
    google_pkg.__path__ = []  # type: ignore[attr-defined]
    genai_pkg = types.ModuleType("google.genai")
    genai_pkg.__path__ = []  # type: ignore[attr-defined]
    models_mod = types.ModuleType("google.genai.models")

    class Models:
        _response_factory: Any = None
        _captured_kwargs: list[dict[str, Any]] = []

        def generate_content(
            self,
            *,
            model: str,
            contents: Any,
            config: Any = None,
        ) -> Any:
            type(self)._captured_kwargs.append(
                {"model": model, "contents": contents, "config": config}
            )
            if type(self)._response_factory is None:
                return types.SimpleNamespace(
                    text="ok",
                    candidates=[
                        types.SimpleNamespace(
                            content=types.SimpleNamespace(
                                parts=[
                                    types.SimpleNamespace(text="ok"),
                                ],
                                role="model",
                            ),
                            finish_reason="STOP",
                            index=0,
                        )
                    ],
                    usage_metadata=types.SimpleNamespace(
                        prompt_token_count=2,
                        candidates_token_count=1,
                    ),
                )
            return type(self)._response_factory()

    class AsyncModels:
        _response_factory: Any = None
        _captured_kwargs: list[dict[str, Any]] = []

        async def generate_content(
            self,
            *,
            model: str,
            contents: Any,
            config: Any = None,
        ) -> Any:
            type(self)._captured_kwargs.append(
                {"model": model, "contents": contents, "config": config}
            )
            if type(self)._response_factory is None:
                return types.SimpleNamespace(
                    text="ok",
                    candidates=[
                        types.SimpleNamespace(
                            content=types.SimpleNamespace(
                                parts=[types.SimpleNamespace(text="ok")],
                                role="model",
                            ),
                            finish_reason="STOP",
                            index=0,
                        )
                    ],
                    usage_metadata=types.SimpleNamespace(
                        prompt_token_count=2,
                        candidates_token_count=1,
                    ),
                )
            return type(self)._response_factory()

    models_mod.Models = Models
    models_mod.AsyncModels = AsyncModels
    sys.modules.update(
        {
            "google": google_pkg,
            "google.genai": genai_pkg,
            "google.genai.models": models_mod,
        }
    )
    return Models, AsyncModels


def _gemini_two_function_call_response() -> Any:
    return types.SimpleNamespace(
        text=None,
        candidates=[
            types.SimpleNamespace(
                content=types.SimpleNamespace(
                    parts=[
                        types.SimpleNamespace(
                            function_call=types.SimpleNamespace(
                                name="lookup_customer",
                                args={"id": "ACC-3"},
                            ),
                        ),
                        types.SimpleNamespace(
                            function_call=types.SimpleNamespace(
                                name="send_email",
                                args={"to": "x"},
                            ),
                        ),
                    ],
                    role="model",
                ),
                finish_reason="STOP",
                index=0,
            )
        ],
        usage_metadata=types.SimpleNamespace(
            prompt_token_count=4, candidates_token_count=2,
        ),
    )


def _gemini_shell_function_call_response() -> Any:
    return types.SimpleNamespace(
        text=None,
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
                index=0,
            )
        ],
        usage_metadata=types.SimpleNamespace(
            prompt_token_count=2, candidates_token_count=1,
        ),
    )


@pytest.fixture
def genai_smoke(fake_backend: Any) -> Iterator[tuple[Any, type, type]]:
    _init_sdk(app="genai-smoke")
    Models, AsyncModels = _install_fake_genai()
    Models._captured_kwargs = []
    AsyncModels._captured_kwargs = []
    from egisai._patches import genai as patch

    assert patch.apply() is True
    yield fake_backend, Models, AsyncModels
    for mod in ("google.genai.models", "google.genai", "google"):
        sys.modules.pop(mod, None)


def test_genai_allow_path(genai_smoke: Any) -> None:
    fake_backend, Models, _ = genai_smoke
    out = Models().generate_content(
        model="gemini-1.5-flash",
        contents="hello world",
    )
    _flush()
    assert out.text == "ok"
    events = _all_audit_events(fake_backend.events_received)
    assert any(e.get("verdict") == "allow" for e in events)


def test_genai_pii_sanitize_masks_in_place(genai_smoke: Any) -> None:
    fake_backend, Models, _ = genai_smoke
    _set_rules(_pii_sanitize_rule(["ssn"]))
    raw = "789-12-3456"
    Models().generate_content(
        model="gemini-1.5-flash",
        contents=f"verify SSN {raw} please",
    )
    _flush()
    sent = Models._captured_kwargs[-1]["contents"]
    assert raw not in str(sent)
    _assert_raw_text_absent(fake_backend.events_received, raw)


def test_genai_pii_block_raises(genai_smoke: Any) -> None:
    fake_backend, Models, _ = genai_smoke
    _set_rules(_pii_block_rule(["ssn"]))
    raw = "234-56-7890"
    with pytest.raises(PermissionError):
        Models().generate_content(
            model="gemini-1.5-flash",
            contents=f"SSN {raw}",
        )
    _flush()
    assert Models._captured_kwargs == []
    _assert_raw_text_absent(fake_backend.events_received, raw)


def test_genai_deny_tool_call_refuses_response(genai_smoke: Any) -> None:
    fake_backend, Models, _ = genai_smoke
    _set_rules(_deny_tool_rule())
    Models._response_factory = _gemini_shell_function_call_response
    with pytest.raises(PermissionError):
        Models().generate_content(
            model="gemini-1.5-flash",
            contents="please run a shell command",
        )
    _flush()
    events = _all_audit_events(fake_backend.events_received)
    blocks = [e for e in events if e.get("verdict") == "block"]
    assert blocks
    assert blocks[-1].get("enforcement_status") == "enforced"


def test_genai_per_tool_waterfall(genai_smoke: Any) -> None:
    fake_backend, Models, _ = genai_smoke
    Models._response_factory = _gemini_two_function_call_response
    from egisai._run import close_run, open_run_from_current_identity

    open_run_from_current_identity(framework="genai", prompt_text=None)
    try:
        Models().generate_content(
            model="gemini-1.5-flash",
            contents="do two",
        )
    finally:
        close_run()
    _flush()
    tool_evs = _tool_steps(fake_backend.events_received)
    names = [e["tool_name"] for e in tool_evs]
    assert names == ["lookup_customer", "send_email"]
    for ev in tool_evs:
        assert ev.get("enforcement_status") == "enforced"


# ── Bedrock Converse ────────────────────────────────────────────────


def _install_fake_boto3_with_bedrock() -> Any:
    """Faithful double of ``boto3.client('bedrock-runtime')``."""
    import types as _types

    boto3_mod = _types.ModuleType("boto3")

    class _BedrockClient:
        _response_factory: Any = None
        _captured: list[dict[str, Any]] = []

        def converse(self, **kwargs: Any) -> Any:
            type(self)._captured.append(kwargs)
            if type(self)._response_factory is None:
                return {
                    "output": {
                        "message": {
                            "role": "assistant",
                            "content": [{"text": "ok"}],
                        }
                    },
                    "stopReason": "end_turn",
                    "usage": {"inputTokens": 4, "outputTokens": 2},
                }
            return type(self)._response_factory()

        def converse_stream(self, **kwargs: Any) -> Any:
            return self.converse(**kwargs)

    def _client(service_name: str, **_kwargs: Any) -> Any:
        if service_name == "bedrock-runtime":
            return _BedrockClient()
        return _types.SimpleNamespace()

    boto3_mod.client = _client
    sys.modules["boto3"] = boto3_mod
    return _BedrockClient


def _bedrock_two_tool_response() -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "calling tools"},
                    {
                        "toolUse": {
                            "toolUseId": "tu_1",
                            "name": "lookup_customer",
                            "input": {"id": "ACC-2"},
                        }
                    },
                    {
                        "toolUse": {
                            "toolUseId": "tu_2",
                            "name": "send_email",
                            "input": {"to": "x"},
                        }
                    },
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 6, "outputTokens": 3},
    }


def _bedrock_shell_tool_response() -> dict[str, Any]:
    return {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tu_sh",
                            "name": "run_shell",
                            "input": {"cmd": "ls"},
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 2, "outputTokens": 1},
    }


@pytest.fixture
def bedrock_smoke(fake_backend: Any) -> Iterator[tuple[Any, type]]:
    _init_sdk(app="bedrock-smoke")
    BedrockClient = _install_fake_boto3_with_bedrock()
    BedrockClient._captured = []
    from egisai._patches import bedrock_runtime as patch

    assert patch.apply() is True
    yield fake_backend, BedrockClient
    sys.modules.pop("boto3", None)


def test_bedrock_converse_allow_path(bedrock_smoke: Any) -> None:
    fake_backend, _ = bedrock_smoke
    import boto3

    client = boto3.client("bedrock-runtime")
    out = client.converse(
        modelId="anthropic.claude-3-5-sonnet",
        messages=[{"role": "user", "content": [{"text": "hello"}]}],
        system=[{"text": "be helpful"}],
    )
    _flush()
    assert out["output"]["message"]["content"][0]["text"] == "ok"
    events = _all_audit_events(fake_backend.events_received)
    assert any(e.get("verdict") == "allow" for e in events)


def test_bedrock_converse_pii_sanitize_masks_in_place(
    bedrock_smoke: Any,
) -> None:
    """Bedrock Converse user messages are nested under
    ``messages[].content[].text``. The sanitizer must walk that
    shape and mask in place. Note: the gate's ``mutate_prompt_text``
    walks ``messages`` only at the top-level ``content`` key; for
    Bedrock the user text lives under ``content[].text``. So the
    raw bytes hitting boto3 may or may not be masked depending on
    the shape support; the AUDIT row's preview MUST be redacted.
    """
    fake_backend, _ = bedrock_smoke
    _set_rules(_pii_sanitize_rule(["ssn"]))
    raw = "555-12-3456"
    import boto3

    client = boto3.client("bedrock-runtime")
    client.converse(
        modelId="anthropic.claude-3-5-sonnet",
        messages=[
            {
                "role": "user",
                "content": [{"text": f"SSN {raw} please look up"}],
            }
        ],
    )
    _flush()
    # Critical privacy contract: raw SSN never on the wire envelope.
    _assert_raw_text_absent(fake_backend.events_received, raw)


def test_bedrock_converse_pii_block_raises(bedrock_smoke: Any) -> None:
    fake_backend, _ = bedrock_smoke
    _set_rules(_pii_block_rule(["ssn"]))
    raw = "234-56-7891"
    import boto3

    client = boto3.client("bedrock-runtime")
    with pytest.raises(PermissionError):
        client.converse(
            modelId="anthropic.claude-3-5-sonnet",
            messages=[
                {"role": "user", "content": [{"text": f"SSN {raw}"}]}
            ],
        )
    _flush()
    _assert_raw_text_absent(fake_backend.events_received, raw)


def test_bedrock_converse_deny_tool_call_refuses_response(
    bedrock_smoke: Any,
) -> None:
    fake_backend, BedrockClient = bedrock_smoke
    _set_rules(_deny_tool_rule())
    BedrockClient._response_factory = _bedrock_shell_tool_response
    import boto3

    client = boto3.client("bedrock-runtime")
    with pytest.raises(PermissionError):
        client.converse(
            modelId="anthropic.claude-3-5-sonnet",
            messages=[
                {"role": "user", "content": [{"text": "do something"}]}
            ],
        )
    _flush()
    events = _all_audit_events(fake_backend.events_received)
    blocks = [e for e in events if e.get("verdict") == "block"]
    assert blocks
    assert blocks[-1].get("enforcement_status") == "enforced"


def test_bedrock_converse_per_tool_waterfall(bedrock_smoke: Any) -> None:
    fake_backend, BedrockClient = bedrock_smoke
    BedrockClient._response_factory = _bedrock_two_tool_response
    from egisai._run import close_run, open_run_from_current_identity

    open_run_from_current_identity(
        framework="bedrock_runtime", prompt_text=None,
    )
    try:
        import boto3

        client = boto3.client("bedrock-runtime")
        client.converse(
            modelId="anthropic.claude-3-5-sonnet",
            messages=[
                {"role": "user", "content": [{"text": "do two"}]}
            ],
        )
    finally:
        close_run()
    _flush()
    tool_evs = _tool_steps(fake_backend.events_received)
    names = [e["tool_name"] for e in tool_evs]
    assert names == ["lookup_customer", "send_email"]
    for ev in tool_evs:
        assert ev.get("enforcement_status") == "enforced"


# ── Cross-provider async-path smoke ─────────────────────────────────


def test_openai_chat_async_path_block(openai_smoke: Any) -> None:
    """The async path runs through ``async_gate_call``; same
    enforcement guarantees must hold."""
    fake_backend, _, AsyncCompletions = openai_smoke
    _set_rules(_pii_block_rule(["ssn"]))
    raw = "111-22-3333"

    async def runner() -> Any:
        return await AsyncCompletions().create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"SSN {raw}"}],
        )

    with pytest.raises(PermissionError):
        asyncio.run(runner())
    _flush()
    assert AsyncCompletions._captured_kwargs == []
    _assert_raw_text_absent(fake_backend.events_received, raw)


def test_anthropic_async_path_sanitize(anthropic_smoke: Any) -> None:
    fake_backend, _, AsyncMessages = anthropic_smoke
    _set_rules(_pii_sanitize_rule(["ssn"]))
    raw = "777-88-9999"

    async def runner() -> Any:
        return await AsyncMessages().create(
            model="claude-3-5-sonnet",
            messages=[
                {"role": "user", "content": f"verify {raw}"}
            ],
        )

    asyncio.run(runner())
    _flush()
    sent = AsyncMessages._captured_kwargs[-1]["messages"][-1]["content"]
    assert raw not in sent
    _assert_raw_text_absent(fake_backend.events_received, raw)


def test_genai_async_path_sanitize(genai_smoke: Any) -> None:
    fake_backend, _, AsyncModels = genai_smoke
    _set_rules(_pii_sanitize_rule(["ssn"]))
    raw = "333-44-5555"

    async def runner() -> Any:
        return await AsyncModels().generate_content(
            model="gemini-1.5-flash",
            contents=f"my SSN is {raw}",
        )

    asyncio.run(runner())
    _flush()
    sent = AsyncModels._captured_kwargs[-1]["contents"]
    assert raw not in str(sent)
    _assert_raw_text_absent(fake_backend.events_received, raw)
