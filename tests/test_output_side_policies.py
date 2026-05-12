"""Output-side policies (``deny_tool_call``, ``deny_mcp_call``,
``deny_output_regex``) actually fire when the model response carries
a banned signal.

Before 0.11.0, ``evaluate_output()`` existed but was never called
by the framework patchers — these tests lock in the wiring.
"""

from __future__ import annotations

from typing import Any

import pytest

from egisai._evaluator import OutputCall, evaluate_output
from egisai._output_signals import (
    extract_anthropic,
    extract_bedrock_converse,
    extract_google,
    extract_openai_chat,
    extract_openai_responses,
)


def _deny_tool_rule(name: str = "block-shell-tool") -> dict[str, Any]:
    return {
        "id": "1",
        "name": name,
        "type": "deny_tool_call",
        "tenant": None,
        "config": {
            "patterns": [r"^delete_user$", r"^run_shell$"],
            "message": "Tool call blocked by policy.",
        },
    }


def _deny_mcp_rule(name: str = "block-prod-mcp") -> dict[str, Any]:
    return {
        "id": "2",
        "name": name,
        "type": "deny_mcp_call",
        "tenant": None,
        "config": {
            "patterns": [r"prod\.acmecorp\.com"],
            "message": "MCP call to prod blocked.",
        },
    }


def _deny_output_regex_rule(name: str = "block-secret-output") -> dict[str, Any]:
    return {
        "id": "3",
        "name": name,
        "type": "deny_output_regex",
        "tenant": None,
        "config": {
            "pattern": r"sk-[A-Za-z0-9]{16,}",
            "message": "Model output contained a secret-looking string.",
        },
    }


# ── End-to-end: evaluate_output blocks on the right signals ──────────


def test_evaluate_output_blocks_on_tool_call_match(fake_backend) -> None:
    fake_backend.set_rules([_deny_tool_rule()], etag='"out-tool"')

    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    decision = evaluate_output(
        OutputCall(
            source="openai",
            target="chat.completions",
            model="gpt-4o",
            text="",
            tool_names=[],
            tool_calls=[{"name": "run_shell", "arguments": '{"cmd": "rm -rf /"}'}],
            mcp_targets=[],
        )
    )
    assert decision.verdict == "block"
    assert decision.matched_policy == "block-shell-tool"


def test_evaluate_output_blocks_on_mcp_target_match(fake_backend) -> None:
    fake_backend.set_rules([_deny_mcp_rule()], etag='"out-mcp"')

    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    decision = evaluate_output(
        OutputCall(
            source="openai",
            target="chat.completions",
            model="gpt-4o",
            text="",
            tool_names=[],
            tool_calls=[],
            mcp_targets=["mcp://prod.acmecorp.com/db"],
        )
    )
    assert decision.verdict == "block"
    assert decision.matched_policy == "block-prod-mcp"


def test_evaluate_output_blocks_on_assistant_text_match(fake_backend) -> None:
    fake_backend.set_rules([_deny_output_regex_rule()], etag='"out-text"')

    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    decision = evaluate_output(
        OutputCall(
            source="openai",
            target="chat.completions",
            model="gpt-4o",
            text="Here's the key: sk-abcdefghijklmnopqrstuvwxyz0123",
            tool_names=[],
            tool_calls=[],
            mcp_targets=[],
        )
    )
    assert decision.verdict == "block"
    assert decision.matched_policy == "block-secret-output"


def test_evaluate_output_allows_when_signals_dont_match(fake_backend) -> None:
    fake_backend.set_rules([_deny_tool_rule()], etag='"out-tool-no-match"')

    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    decision = evaluate_output(
        OutputCall(
            source="openai",
            target="chat.completions",
            model="gpt-4o",
            text="Hello, how can I help?",
            tool_names=[],
            tool_calls=[{"name": "search_kb", "arguments": '{"q": "x"}'}],
            mcp_targets=[],
        )
    )
    assert decision.verdict == "allow"


# ── Output signal extractors handle each provider's response shape ──


def test_openai_chat_signal_extractor_pulls_tool_calls() -> None:
    response = {
        "choices": [
            {
                "message": {
                    "content": "calling shell",
                    "tool_calls": [
                        {
                            "id": "x1",
                            "type": "function",
                            "function": {
                                "name": "run_shell",
                                "arguments": '{"cmd": "ls"}',
                            },
                        }
                    ],
                }
            }
        ]
    }
    payload = {
        "messages": [],
        "tools": [{"type": "function", "function": {"name": "run_shell"}}],
    }
    text, names, calls, mcp = extract_openai_chat(response, payload)
    assert "calling shell" in text
    assert names == ["run_shell"]
    assert calls == [{"name": "run_shell", "arguments": '{"cmd": "ls"}'}]
    assert mcp == []


def test_openai_responses_signal_extractor_handles_tool_call_items() -> None:
    response = {
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "hello"}]},
            {"type": "tool_call", "name": "delete_user", "arguments": '{"id": 7}'},
        ],
    }
    text, _names, calls, _mcp = extract_openai_responses(response, {})
    assert "hello" in text
    assert calls == [{"name": "delete_user", "arguments": '{"id": 7}'}]


def test_anthropic_signal_extractor_pulls_tool_use_blocks() -> None:
    response = {
        "content": [
            {"type": "text", "text": "Let me look that up."},
            {"type": "tool_use", "name": "search_kb", "input": {"q": "policies"}},
        ],
    }
    payload = {"tools": [{"name": "search_kb"}]}
    text, names, calls, _mcp = extract_anthropic(response, payload)
    assert "look that up" in text
    assert names == ["search_kb"]
    assert calls and calls[0]["name"] == "search_kb"
    # Arguments are coerced to a stable JSON string.
    assert "policies" in calls[0]["arguments"]


def test_bedrock_converse_signal_extractor_pulls_tool_uses() -> None:
    """Bedrock Converse normalises providers (Anthropic/Mistral/Cohere/
    Meta/Amazon) onto ``output.message.content`` with text + toolUse
    blocks. The extractor must pull both so deny_tool_call /
    deny_output_regex fire post-model — without this, Bedrock-routed
    traffic skipped the output phase entirely (gap fixed in 0.18.x).
    """
    response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {"text": "I'll handle that now."},
                    {
                        "toolUse": {
                            "toolUseId": "tu_1",
                            "name": "run_shell",
                            "input": {"cmd": "ls"},
                        }
                    },
                ],
            }
        },
        "stopReason": "tool_use",
    }
    payload = {
        "toolConfig": {
            "tools": [
                {"toolSpec": {"name": "run_shell"}},
                {"toolSpec": {"name": "search_kb"}},
            ]
        }
    }
    text, names, calls, mcp = extract_bedrock_converse(response, payload)
    assert "handle that now" in text
    assert set(names) == {"run_shell", "search_kb"}
    assert calls and calls[0]["name"] == "run_shell"
    assert "ls" in calls[0]["arguments"]
    assert mcp == []


def test_google_signal_extractor_pulls_function_calls() -> None:
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "Result: 42"},
                        {
                            "function_call": {
                                "name": "compute",
                                "args": {"a": 1, "b": 2},
                            }
                        },
                    ]
                }
            }
        ]
    }
    text, _names, calls, _mcp = extract_google(response, {})
    assert "Result: 42" in text
    assert calls and calls[0]["name"] == "compute"


# ── End-to-end: gate_call invokes the output phase ──────────────────


def test_gate_call_blocks_response_with_banned_tool_call(fake_backend) -> None:
    """Going through ``gate_call`` end-to-end: a ``deny_tool_call``
    rule must turn an otherwise-allowed call into a blocked one
    when the model responds with a tool-call to a banned tool.
    """
    fake_backend.set_rules([_deny_tool_rule()], etag='"e2e"')

    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )

    from egisai._patches._common import gate_call

    # The request only declares safe tools; the model's response
    # invokes a banned tool. The output-side gate must catch that.
    fake_response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {"name": "run_shell", "arguments": '{}'},
                        }
                    ],
                }
            }
        ]
    }

    with pytest.raises(PermissionError, match="block-shell-tool"):
        gate_call(
            source="openai",
            target="openai.chat.completions.create",
            model="gpt-4o",
            prompt_text="please help",
            stream=False,
            payload={
                "messages": [{"role": "user", "content": "please help"}],
                "tools": [{"type": "function", "function": {"name": "search_kb"}}],
            },
            extract_output_signals=extract_openai_chat,
            forward=lambda: fake_response,
        )


def test_gate_call_allows_response_when_no_banned_signal(fake_backend) -> None:
    fake_backend.set_rules([_deny_tool_rule()], etag='"e2e-allow"')

    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    from egisai._patches._common import gate_call

    fake_response = {
        "choices": [
            {"message": {"content": "Sure, here's the info you asked for."}}
        ]
    }

    result = gate_call(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4o",
        prompt_text="please help",
        stream=False,
        payload={
            "messages": [{"role": "user", "content": "please help"}],
            "tools": [{"type": "function", "function": {"name": "search_kb"}}],
        },
        extract_output_signals=extract_openai_chat,
        forward=lambda: fake_response,
    )
    assert result is fake_response


def test_gate_call_blocks_bedrock_response_with_banned_tool_call(
    fake_backend,
) -> None:
    """End-to-end Bedrock parity with the OpenAI test above. Confirms
    the output gate fires for Bedrock Converse traffic — closes the
    0.18.x gap where bedrock_runtime forwarded through ``gate_call``
    without an ``extract_output_signals`` so output policies silently
    skipped.
    """
    fake_backend.set_rules([_deny_tool_rule()], etag='"bedrock-e2e"')

    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )

    from egisai._patches._common import gate_call

    fake_response = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tu_1",
                            "name": "run_shell",
                            "input": {},
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
    }

    with pytest.raises(PermissionError, match="block-shell-tool"):
        gate_call(
            source="bedrock_runtime",
            target="bedrock.converse",
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            prompt_text="please help",
            stream=False,
            payload={
                "messages": [{"role": "user", "content": [{"text": "help"}]}],
                "toolConfig": {
                    "tools": [{"toolSpec": {"name": "search_kb"}}]
                },
            },
            extract_output_signals=extract_bedrock_converse,
            forward=lambda: fake_response,
        )
