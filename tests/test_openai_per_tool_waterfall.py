"""Per-tool ``tool_call`` step emission for the OpenAI patches.

Pins the multi-step waterfall contract for synchronous OpenAI
traffic — Chat Completions and the Responses API. The dashboard's
``RunTimelineModal`` expects one ``tool_call`` step per tool the
model invoked, so the timeline reads top-to-bottom as

    Request → [Pre-policy] → Model → [Post-policy] →
              Tool: lookup_customer → [Pre-policy] →
              Model → [Post-policy] →
              Tool: send_email → … → Returned

Before this PR the OpenAI patch only emitted one ``model_call``
step per turn; the tools the model invoked were buried inside the
parent step's ``response_decision`` and invisible on the
timeline. The fix is two lines per wrap site
(``emit_tool_call_steps=True``); the shared helper in
``_patches/_common.py`` walks the response with the existing
``extract_output_signals`` function and appends one step per
tool. ``enforcement_status`` is ``"enforced"`` (not ``"advisory"``)
because the parent model_call's output policy had a chance to
refuse the request before it left the gate — distinct from
``claude_agent_sdk`` where the Node subprocess has already
executed the tool by the time we see it.

The per-tool steps attach to the currently-open Run. In
production this Run is opened by the framework wrap above the
gate (``Runner.run`` for OpenAI Agents). These tests simulate that
wrap by opening a Run via ``open_run_from_current_identity``
before exercising the gate, then closing it with ``close_run``.
Raw ``client.chat.completions.create()`` calls without an
agentic framework above them deliberately do NOT emit per-tool
steps — the legacy single-event ingest path keeps producing a
one-step Run on the backend for that case, preserving wire
compatibility with pre-0.20 SDKs.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import pytest

from egisai._output_signals import extract_openai_chat, extract_openai_responses


def _tool_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        e for e in events
        if e.get("kind") == "run.step" and e.get("step_kind") == "tool_call"
    ]


def _model_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        e for e in events
        if e.get("kind") == "run.step" and e.get("step_kind") == "model_call"
    ]


def _flush() -> None:
    from egisai import shutdown
    shutdown()


@contextmanager
def _framework_run(framework: str = "openai_agents") -> Iterator[None]:
    """Simulate the framework wrap above the gate.

    OpenAI Agents' ``Runner.run`` patch opens a Run for the duration
    of an agent invocation; every inner ``responses.create`` call
    attaches its model_call + tool_call steps to that Run. These
    tests mimic the wrap by opening + closing a Run by hand so the
    per-tool emission has somewhere to land.
    """
    from egisai._run import close_run, open_run_from_current_identity

    open_run_from_current_identity(framework=framework, prompt_text=None)
    try:
        yield
    finally:
        close_run()


# ── Chat Completions ────────────────────────────────────────────────


def test_chat_completions_emits_one_tool_call_step_per_tool(
    fake_backend: Any,
) -> None:
    """A single Chat Completions call that returns two tool_calls
    must produce ONE ``model_call`` step + TWO ``tool_call`` steps
    in order, so the timeline shows the agent's invocation pattern
    clearly."""
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="oa-waterfall",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    from egisai._patches._common import gate_call

    fake_response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup_customer",
                                "arguments": '{"account_id": "ACC-1"}',
                            },
                        },
                        {
                            "type": "function",
                            "function": {
                                "name": "send_email",
                                "arguments": '{"to": "x@example.com"}',
                            },
                        },
                    ],
                }
            }
        ]
    }

    with _framework_run():
        result = gate_call(
            source="openai",
            target="openai.chat.completions.create",
            model="gpt-4o",
            prompt_text="please help",
            stream=False,
            payload={
                "messages": [{"role": "user", "content": "please help"}],
                "tools": [
                    {"type": "function", "function": {"name": "lookup_customer"}},
                    {"type": "function", "function": {"name": "send_email"}},
                ],
            },
            extract_output_signals=extract_openai_chat,
            emit_tool_call_steps=True,
            forward=lambda: fake_response,
        )
        assert result is fake_response
    _flush()

    model_evs = _model_steps(fake_backend.events_received)
    tool_evs = _tool_steps(fake_backend.events_received)
    assert len(model_evs) == 1, (
        f"expected exactly 1 model_call step, got {len(model_evs)}"
    )
    assert len(tool_evs) == 2, (
        f"expected 2 tool_call steps (one per tool the model "
        f"invoked), got {len(tool_evs)}; "
        f"names={[e.get('tool_name') for e in tool_evs]}"
    )
    names = [e.get("tool_name") for e in tool_evs]
    assert names == ["lookup_customer", "send_email"]
    # Per-tool step ordering: each must follow the parent
    # model_call on the run timeline so the dashboard renders
    # model → tool → tool top-to-bottom.
    assert tool_evs[0]["seq"] > model_evs[0]["seq"]
    assert tool_evs[1]["seq"] > tool_evs[0]["seq"]
    # OpenAI is a synchronous patch — the model_call's output
    # policy had the chance to block the tool requests before we
    # got here. Each per-tool step is therefore an *enforced*
    # record, distinct from the agentic-subprocess case
    # (claude_agent_sdk → advisory).
    for ev in tool_evs:
        assert ev.get("enforcement_status") == "enforced"
        assert ev.get("verdict") == "allow"

    # Regression for 0.27.1 — every tool_call event MUST ship the
    # tool input under the wire key ``prompt_preview``. The backend
    # reads the audit row's preview text from ``ev.get(
    # "prompt_preview")`` (see ``app.routers.sdk
    # ._build_request_log_row``); shipping it under ``request_text``
    # (the DB column name) silently dropped the value on the floor,
    # which left every tool_call row's ``request_text`` NULL and
    # collapsed the intent-summary LLM onto generic "Open ended
    # assistant chat" / "General chat follow up question" labels.
    #
    # The preview is *post-redaction* by design
    # (``security-and-compliance.mdc`` §1) — ``_label_redact_tool_input``
    # routes the JSON-serialized input through ``label_redact``
    # before stamping the event. So we check for the field name
    # (which is never redacted) plus an unredacted sentinel from
    # the input (``ACC-1`` survives because it doesn't match any
    # PII pattern).
    for ev in tool_evs:
        preview = ev.get("prompt_preview")
        assert isinstance(preview, str) and preview, (
            "tool_call events must ship the tool input under "
            f"``prompt_preview`` (got {preview!r}; full keys="
            f"{sorted(ev.keys())})"
        )
        assert ev["tool_name"] in {"lookup_customer", "send_email"}
        if ev["tool_name"] == "lookup_customer":
            # Field names aren't redacted; ``ACC-1`` doesn't match
            # any PII pattern so it survives end-to-end.
            assert "account_id" in preview
            assert "ACC-1" in preview
        else:
            # ``send_email`` carried ``"x@example.com"`` which the
            # URL detector correctly redacts to ``x@<URL>``. The
            # important invariant is that the field key reaches
            # the backend so the intent-summary LLM can describe
            # the call.
            assert "to" in preview
            assert "x@" in preview
        # Defense in depth: the legacy key MUST NOT be set, or the
        # backend's compatibility fallback would still read it and
        # the test would mask a future regression.
        assert "request_text" not in ev


def test_chat_completions_no_tool_calls_emits_no_tool_steps(
    fake_backend: Any,
) -> None:
    """When the model returns just text (no tool calls), the gate
    must NOT append phantom tool_call steps. The timeline collapses
    to a clean single model_call row."""
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="oa-waterfall",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    from egisai._patches._common import gate_call

    fake_response = {
        "choices": [
            {"message": {"content": "All clear, no tools needed."}}
        ]
    }

    with _framework_run():
        gate_call(
            source="openai",
            target="openai.chat.completions.create",
            model="gpt-4o",
            prompt_text="status?",
            stream=False,
            payload={"messages": [{"role": "user", "content": "status?"}]},
            extract_output_signals=extract_openai_chat,
            emit_tool_call_steps=True,
            forward=lambda: fake_response,
        )
    _flush()

    model_evs = _model_steps(fake_backend.events_received)
    tool_evs = _tool_steps(fake_backend.events_received)
    assert len(model_evs) == 1
    assert tool_evs == []


def test_chat_completions_blocked_call_does_not_emit_tool_steps(
    fake_backend: Any,
) -> None:
    """When the output-side ``deny_tool_call`` policy refuses the
    response, the gate raises (or stubs) BEFORE the per-tool
    emission runs — the audit row carries a single blocked
    model_call step and NO tool_call rows. The tool never got a
    green light to invoke, so it would be misleading to record
    individual ``allow``-flavoured tool steps for it."""
    fake_backend.set_rules(
        [
            {
                "id": "1",
                "name": "block-shell-tool",
                "type": "deny_tool_call",
                "tenant": None,
                "config": {
                    "patterns": [r"^run_shell$"],
                    "message": "Tool call blocked by policy.",
                },
            }
        ],
        etag='"deny-shell"',
    )
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="oa-waterfall",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )

    from egisai._patches._common import gate_call

    fake_response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "run_shell", "arguments": "{}"},
                        }
                    ],
                }
            }
        ]
    }

    with _framework_run():
        with pytest.raises(PermissionError):
            gate_call(
                source="openai",
                target="openai.chat.completions.create",
                model="gpt-4o",
                prompt_text="please help",
                stream=False,
                payload={
                    "messages": [{"role": "user", "content": "please help"}],
                    "tools": [
                        {"type": "function", "function": {"name": "run_shell"}}
                    ],
                },
                extract_output_signals=extract_openai_chat,
                emit_tool_call_steps=True,
                forward=lambda: fake_response,
            )
    _flush()

    model_evs = _model_steps(fake_backend.events_received)
    tool_evs = _tool_steps(fake_backend.events_received)
    assert len(model_evs) == 1
    assert model_evs[0].get("verdict") == "block"
    assert tool_evs == [], (
        "tool_call steps must NOT be emitted when the parent "
        "model_call was refused — the tool was never invoked"
    )


def test_tool_call_step_input_is_label_redacted(
    fake_backend: Any,
) -> None:
    """If the model passes PII as a tool argument, the tool_call
    step's ``request_text`` must be label-redacted by the SDK before
    the audit row reaches the backend. The original PII value must
    not appear anywhere on the wire — same contract the
    claude_agent_sdk per-tool emission already pins."""
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="oa-waterfall",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    from egisai._patches._common import gate_call

    leaked_ssn = "123-45-6789"
    fake_response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "lookup_customer",
                                "arguments": (
                                    '{"ssn": "' + leaked_ssn + '"}'
                                ),
                            },
                        }
                    ],
                }
            }
        ]
    }

    with _framework_run():
        gate_call(
            source="openai",
            target="openai.chat.completions.create",
            model="gpt-4o",
            prompt_text="lookup",
            stream=False,
            payload={
                "messages": [{"role": "user", "content": "lookup"}],
                "tools": [
                    {"type": "function", "function": {"name": "lookup_customer"}}
                ],
            },
            extract_output_signals=extract_openai_chat,
            emit_tool_call_steps=True,
            forward=lambda: fake_response,
        )
    _flush()

    tool_evs = _tool_steps(fake_backend.events_received)
    assert len(tool_evs) == 1
    request_text = tool_evs[0].get("request_text") or ""
    assert leaked_ssn not in request_text, (
        f"raw SSN must not appear in tool_call request_text "
        f"(got {request_text!r})"
    )
    # Belt-and-braces: it must not appear ANYWHERE on this step's
    # wire payload either (a future regression that stamps the
    # value under a different field name still fails this).
    assert leaked_ssn not in repr(tool_evs[0])


# ── Responses API ───────────────────────────────────────────────────


def test_responses_api_emits_one_tool_call_step_per_function_call(
    fake_backend: Any,
) -> None:
    """The Responses API wraps tool invocations as ``function_call``
    output items (vs. ``tool_calls`` on the Chat side). The same
    waterfall contract must hold: one ``tool_call`` step per
    function the model invoked. This is the surface OpenAI Agents
    SDK uses end-to-end."""
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="oa-waterfall",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    from egisai._patches._common import gate_call

    fake_response = {
        "output": [
            {
                "type": "function_call",
                "name": "lookup_customer",
                "arguments": '{"account_id": "ACC-1"}',
            },
            {
                "type": "function_call",
                "name": "issue_refund",
                "arguments": '{"amount": 100}',
            },
        ]
    }

    with _framework_run():
        gate_call(
            source="openai",
            target="openai.responses.create",
            model="gpt-4o",
            prompt_text="process refund",
            stream=False,
            payload={
                "input": "process refund",
                "tools": [
                    {"type": "function", "name": "lookup_customer"},
                    {"type": "function", "name": "issue_refund"},
                ],
            },
            extract_output_signals=extract_openai_responses,
            emit_tool_call_steps=True,
            forward=lambda: fake_response,
        )
    _flush()

    tool_evs = _tool_steps(fake_backend.events_received)
    assert len(tool_evs) == 2, (
        f"expected 2 tool_call steps for the Responses API "
        f"function_calls, got {len(tool_evs)}"
    )
    names = [e.get("tool_name") for e in tool_evs]
    assert names == ["lookup_customer", "issue_refund"]
    for ev in tool_evs:
        assert ev.get("enforcement_status") == "enforced"
