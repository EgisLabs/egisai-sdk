"""Per-call declared tool / MCP inventory stamped on audit events.

Pins the contract behind the request-detail / run-timeline "Tools &
MCP" section: every ``model_call`` audit event carries a compact,
metadata-only ``tools`` list derived from the same access-inventory
extraction that feeds the per-agent Access tab.

Three properties matter:

* **Shape** — ``compact_tool_inventory`` keeps name / kind /
  server_name / description / param_names and drops the internal
  ``schema_hash`` join key.
* **Privacy** — only metadata leaves the SDK: a raw argument value
  passed to a tool, and the tool's input-schema JSON, never appear in
  the stamped ``tools`` list (compliance rule §1).
* **Survival** — the field is a top-level, non-underscore key, so it
  survives the logger's ``payload``-stripping and actually reaches the
  backend (which is the whole bug this feature fixes).
"""

from __future__ import annotations

from typing import Any

from egisai._access import compact_tool_inventory, extract_access_items

_OPENAI_V1 = {
    "model": "gpt-4o",
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "delete_customer",
                "description": "Remove a customer record permanently.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                },
            },
        },
    ],
}

_MCP_TAGGED = {
    "model": "gpt-4o",
    "tools": [
        {
            "name": "search_issues",
            "description": "Search the tracker.",
            "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
            "mcp": {"server": "github"},
        },
    ],
}


# ── Pure extraction + compaction ─────────────────────────────────────


def test_compact_shape_from_openai_v1() -> None:
    items = extract_access_items(_OPENAI_V1)
    compact = compact_tool_inventory(items)
    assert len(compact) == 1
    entry = compact[0]
    assert entry["kind"] == "tool"
    assert entry["name"] == "delete_customer"
    # Parameter *names* only — never values or the schema JSON.
    assert entry["param_names"] == ["customer_id", "reason"]
    assert "Remove a customer record" in entry["description"]
    # The internal join key is dropped from the per-call display shape.
    assert "schema_hash" not in entry


def test_compact_mcp_tagged_yields_server_and_tool() -> None:
    compact = compact_tool_inventory(extract_access_items(_MCP_TAGGED))
    kinds = {e["kind"] for e in compact}
    assert "mcp_server" in kinds
    tool = next(e for e in compact if e["kind"] == "tool")
    assert tool["server_name"] == "github"


def test_compact_is_metadata_only() -> None:
    """A raw argument value and the schema JSON never leak."""
    payload = {
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "wire_transfer",
                    "description": "Move money.",
                    "parameters": {
                        "type": "object",
                        "properties": {"iban": {"type": "string"}},
                    },
                },
            }
        ],
    }
    compact = compact_tool_inventory(extract_access_items(payload))
    blob = repr(compact)
    # The declared parameter name is fine; a concrete value is not.
    assert "iban" in blob
    assert "type" not in compact[0]  # no raw JSON-Schema structure
    assert "properties" not in blob


def test_compact_caps_at_fifty() -> None:
    payload = {
        "tools": [
            {"type": "function", "function": {"name": f"tool_{i}"}}
            for i in range(120)
        ]
    }
    compact = compact_tool_inventory(extract_access_items(payload))
    assert len(compact) == 50


def test_empty_when_no_tools() -> None:
    assert compact_tool_inventory(extract_access_items({"model": "gpt-4o"})) == []


# ── End-to-end: the field reaches the backend ───────────────────────


def _model_steps(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        e
        for e in events
        if e.get("kind") in (None, "run.step")
        and e.get("step_kind") in (None, "model_call")
    ]


def test_tools_stamped_on_model_event_and_survive_logger(
    fake_backend: Any,
) -> None:
    """A gated call with declared tools ships a ``tools`` list to the
    backend; a tool_call step records the invoked tool via
    ``tool_name`` and carries no declared-tools list."""
    import egisai
    from egisai._output_signals import extract_openai_chat
    from egisai._run import close_run, open_run_from_current_identity

    egisai.init(
        api_key="egis_live_test",
        app="tool-inventory",
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
                                "arguments": '{"account_id": "ACC-SECRET-123"}',
                            },
                        }
                    ],
                }
            }
        ]
    }

    open_run_from_current_identity(framework="openai_agents", prompt_text=None)
    try:
        gate_call(
            source="openai",
            target="openai.chat.completions.create",
            model="gpt-4o",
            prompt_text="please help",
            stream=False,
            payload={
                "messages": [{"role": "user", "content": "please help"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup_customer",
                            "description": "Look a customer up by id.",
                            "parameters": {
                                "type": "object",
                                "properties": {"account_id": {"type": "string"}},
                            },
                        },
                    }
                ],
            },
            extract_output_signals=extract_openai_chat,
            emit_tool_call_steps=True,
            forward=lambda: fake_response,
        )
    finally:
        close_run()

    from egisai import shutdown

    shutdown()

    events = fake_backend.events_received
    model_evs = [
        e
        for e in events
        if e.get("kind") == "run.step" and e.get("step_kind") == "model_call"
    ]
    assert model_evs, "expected a model_call step event"
    tools = model_evs[0].get("tools")
    assert tools, "declared tools must ride the model_call event"
    assert tools[0]["name"] == "lookup_customer"
    assert tools[0]["param_names"] == ["account_id"]

    # Privacy: the declared tool inventory is metadata only — a
    # concrete argument value a tool was called with must never appear
    # in the ``tools`` list. (The separate tool_call step preview goes
    # through ``label_redact`` and is governed by its own contract.)
    assert "ACC-SECRET-123" not in repr(tools)

    # The tool the model actually invoked is its own tool_call step,
    # identified by ``tool_name`` — and that step carries no declared
    # inventory (declared tools belong on the model_call).
    tool_evs = [
        e
        for e in events
        if e.get("kind") == "run.step" and e.get("step_kind") == "tool_call"
    ]
    assert tool_evs, "expected a tool_call step"
    assert tool_evs[0].get("tool_name") == "lookup_customer"
    assert not tool_evs[0].get("tools")
