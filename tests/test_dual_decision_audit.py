"""Audit events carry ``prompt_decision`` + ``response_decision`` blocks.

Before 0.12.4 the gate stamped a single top-level ``verdict`` that
got overwritten when the output phase blocked. The dashboard
couldn't tell whether a block happened pre- or post-model.

0.12.4 adds two structured per-phase blocks:

- ``prompt_decision``  — always present (input phase always runs).
- ``response_decision`` — present iff the post-model phase actually
  ran (i.e. the model returned and an output extractor produced
  signals to evaluate).

The legacy top-level fields stay for back-compat.
"""

from __future__ import annotations

from typing import Any

import pytest

from egisai._output_signals import extract_openai_chat


def _deny_tool_rule() -> dict[str, Any]:
    return {
        "id": "1",
        "name": "block-shell",
        "type": "deny_tool_call",
        "tenant": None,
        "config": {"patterns": [r"^run_shell$"]},
    }


def _pii_block_rule() -> dict[str, Any]:
    return {
        "id": "2",
        "name": "block-pii",
        "type": "pii_scan",
        "tenant": None,
        "config": {"action": "block"},
    }


def _init(fake_backend, rules: list[dict[str, Any]], etag: str = '"x"') -> None:
    fake_backend.set_rules(rules, etag=etag)
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )


def _gate(payload: dict[str, Any], response: Any) -> Any:
    from egisai._patches._common import gate_call

    return gate_call(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4o",
        prompt_text=payload.get("messages", [{}])[-1].get("content", ""),
        stream=False,
        payload=payload,
        extract_output_signals=extract_openai_chat,
        forward=lambda: response,
    )


# ── Allow path: both phases ran cleanly ─────────────────────────────


def test_allow_path_emits_two_decision_blocks(fake_backend) -> None:
    _init(fake_backend, [_deny_tool_rule()], etag='"a"')
    safe_response = {
        "choices": [{"message": {"content": "Sure, here's the info."}}]
    }
    _gate(
        {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function", "function": {"name": "search_kb"}}],
        },
        safe_response,
    )

    from egisai import shutdown

    shutdown()
    assert len(fake_backend.events_received) == 1
    ev = fake_backend.events_received[0]

    assert ev["verdict"] == "allow"
    assert ev["prompt_decision"]["verdict"] == "allow"
    assert ev["response_decision"]["verdict"] == "allow"


# ── Pre-model block: response phase never ran ───────────────────────


def test_pre_model_block_omits_response_decision(fake_backend) -> None:
    """When the prompt is blocked, the model is never called, so
    there is no post-model phase to record. ``response_decision``
    must be absent — the dashboard reads its absence as
    "post-model not evaluated"."""
    _init(fake_backend, [_pii_block_rule()], etag='"p"')
    safe_response = {"choices": [{"message": {"content": "..."}}]}

    with pytest.raises(PermissionError):
        _gate(
            {
                "messages": [
                    {"role": "user", "content": "My SSN is 123-45-6789."}
                ]
            },
            safe_response,
        )

    from egisai import shutdown

    shutdown()
    assert len(fake_backend.events_received) == 1
    ev = fake_backend.events_received[0]

    assert ev["verdict"] == "block"
    assert ev["prompt_decision"]["verdict"] == "block"
    assert ev["prompt_decision"]["matched_policy"] == "block-pii"
    assert "response_decision" not in ev


# ── Post-model block: prompt cleared, response was refused ──────────


def test_post_model_block_records_both_phases(fake_backend) -> None:
    """The pre-model phase saw a clean prompt (allow); the
    post-model phase caught a banned tool call. Each phase's
    decision is recorded independently so the dashboard can show
    "passed pre-model, blocked post-model"."""
    _init(fake_backend, [_deny_tool_rule()], etag='"q"')
    bad_response = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "run_shell",
                                "arguments": "{}",
                            },
                        }
                    ],
                }
            }
        ]
    }
    with pytest.raises(PermissionError):
        _gate(
            {
                "messages": [{"role": "user", "content": "please help"}],
                "tools": [
                    {"type": "function", "function": {"name": "search_kb"}}
                ],
            },
            bad_response,
        )

    from egisai import shutdown

    shutdown()
    assert len(fake_backend.events_received) == 1
    ev = fake_backend.events_received[0]

    assert ev["verdict"] == "block"
    assert ev["prompt_decision"]["verdict"] == "allow"
    assert ev["response_decision"]["verdict"] == "block"
    assert ev["response_decision"]["matched_policy"] == "block-shell"
