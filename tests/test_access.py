"""Tests for the access-inventory extractor + reporter (``egisai._access``).

Pins three contracts:

* **Shape coverage** — every declared-tool shape the provider patches
  produce (OpenAI v1, Anthropic / OpenAI legacy, Bedrock Converse
  ``toolConfig``, MCP-tagged tools) yields the same metadata-only
  item shape.
* **Privacy** — descriptions are PII-sanitized before they leave the
  extractor (fail closed to empty on sanitizer error), schemas are
  reduced to a hash + parameter names, and the raw PII value never
  appears anywhere in the extracted payload
  (``assert raw not in repr(items)`` per the compliance rule).
* **Report dedup** — ``maybe_report_access`` fires the background
  report exactly once per ``(agent, bundle_hash)`` and again only
  when the bundle actually changes; a failed report clears the cache
  entry so the next sighting retries.
"""

from __future__ import annotations

import threading
from typing import Any

import egisai._access as access
from egisai._access import (
    compute_bundle_hash,
    extract_access_items,
    maybe_report_access,
    reset_access_cache,
)

# ── Fixture payloads ─────────────────────────────────────────────────

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

_ANTHROPIC = {
    "model": "claude-3",
    "tools": [
        {
            "name": "query_db",
            "description": "Run a read-only SQL query.",
            "input_schema": {
                "type": "object",
                "properties": {"sql": {"type": "string"}},
            },
        },
    ],
}

_BEDROCK = {
    "modelId": "anthropic.claude-3",
    "toolConfig": {
        "tools": [
            {
                "toolSpec": {
                    "name": "send_email",
                    "description": "Send an email to a recipient.",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "to": {"type": "string"},
                                "body": {"type": "string"},
                            },
                        }
                    },
                }
            }
        ]
    },
}


# ── Extraction shapes ────────────────────────────────────────────────


def test_openai_v1_shape_extracts_metadata_only() -> None:
    items = extract_access_items(_OPENAI_V1)
    assert len(items) == 1
    it = items[0]
    assert it["kind"] == "tool"
    assert it["name"] == "delete_customer"
    assert it["param_names"] == ["customer_id", "reason"]
    assert isinstance(it["schema_hash"], str) and len(it["schema_hash"]) == 64
    # The schema itself must never ship — only the hash + param names.
    assert "type" not in repr(it["param_names"])


def test_anthropic_shape_uses_input_schema() -> None:
    items = extract_access_items(_ANTHROPIC)
    assert len(items) == 1
    assert items[0]["name"] == "query_db"
    assert items[0]["param_names"] == ["sql"]
    assert items[0]["schema_hash"] is not None


def test_bedrock_toolconfig_shape() -> None:
    items = extract_access_items(_BEDROCK)
    assert len(items) == 1
    assert items[0]["name"] == "send_email"
    assert items[0]["param_names"] == ["body", "to"]


def test_mcp_tagged_tool_emits_mcp_server_item() -> None:
    payload = {
        "tools": [
            {
                "name": "search_docs",
                "input_schema": {"properties": {"q": {"type": "string"}}},
                "mcp": {"server": "docs-mcp.internal"},
            }
        ]
    }
    items = extract_access_items(payload)
    kinds = {(it["kind"], it["name"]) for it in items}
    assert ("tool", "search_docs") in kinds
    assert ("mcp_server", "docs-mcp.internal") in kinds


def test_non_dict_and_toolless_payloads_yield_empty() -> None:
    assert extract_access_items(None) == []
    assert extract_access_items("nope") == []
    assert extract_access_items({"messages": []}) == []
    assert extract_access_items({"tools": "not-a-list"}) == []


def test_duplicate_tools_dedupe() -> None:
    payload = {"tools": [_ANTHROPIC["tools"][0], _ANTHROPIC["tools"][0]]}
    assert len(extract_access_items(payload)) == 1


def test_item_cap_bounds_pathological_bundles() -> None:
    payload = {
        "tools": [
            {"name": f"tool_{i}", "parameters": {}} for i in range(1000)
        ]
    }
    assert len(extract_access_items(payload)) == access._MAX_ITEMS


# ── Privacy ──────────────────────────────────────────────────────────


def test_description_pii_is_masked_before_leaving_extractor() -> None:
    """Compliance rule 1: the raw value must not survive into the
    reported inventory in any form."""
    raw_ssn = "123-45-6789"
    payload = {
        "tools": [
            {
                "name": "lookup",
                "description": f"Look up the account for SSN {raw_ssn}.",
                "parameters": {"properties": {"q": {}}},
            }
        ]
    }
    items = extract_access_items(payload)
    assert len(items) == 1
    assert raw_ssn not in repr(items)


def test_description_fails_closed_on_sanitizer_error(monkeypatch: Any) -> None:
    """A crashed sanitizer must drop the description, not leak it."""

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("sanitizer down")

    import egisai.policy.pii as pii_mod

    monkeypatch.setattr(pii_mod, "sanitize", _boom)
    items = extract_access_items(
        {
            "tools": [
                {
                    "name": "lookup",
                    "description": "contains SSN 123-45-6789",
                    "parameters": {},
                }
            ]
        }
    )
    assert len(items) == 1
    assert items[0]["description"] == ""
    assert "123-45-6789" not in repr(items)


# ── Bundle hash ──────────────────────────────────────────────────────


def test_bundle_hash_stable_across_order_and_descriptions() -> None:
    a = [
        {"kind": "tool", "name": "a", "schema_hash": "x", "description": "one"},
        {"kind": "tool", "name": "b", "schema_hash": "y", "description": "two"},
    ]
    b = [
        {"kind": "tool", "name": "b", "schema_hash": "y", "description": "CHANGED"},
        {"kind": "tool", "name": "a", "schema_hash": "x", "description": ""},
    ]
    assert compute_bundle_hash(a) == compute_bundle_hash(b)


def test_bundle_hash_changes_on_schema_change() -> None:
    a = [{"kind": "tool", "name": "a", "schema_hash": "x"}]
    b = [{"kind": "tool", "name": "a", "schema_hash": "z"}]
    assert compute_bundle_hash(a) != compute_bundle_hash(b)


# ── Report dedup + retry ─────────────────────────────────────────────


def _drain_threads() -> None:
    for t in threading.enumerate():
        if t.name == "egisai-access-report":
            t.join(timeout=5)


def test_report_fires_once_per_bundle(monkeypatch: Any) -> None:
    reset_access_cache()
    calls: list[dict[str, Any]] = []

    def _capture(**kwargs: Any) -> None:
        calls.append(kwargs)

    import egisai._backend as backend

    monkeypatch.setattr(backend, "report_agent_access", _capture)

    maybe_report_access("agent-1", _OPENAI_V1)
    maybe_report_access("agent-1", _OPENAI_V1)
    _drain_threads()
    assert len(calls) == 1
    assert calls[0]["agent_id"] == "agent-1"
    assert calls[0]["items"][0]["name"] == "delete_customer"

    # A different bundle for the same agent reports again.
    maybe_report_access("agent-1", _ANTHROPIC)
    _drain_threads()
    assert len(calls) == 2


def test_failed_report_clears_cache_for_retry(monkeypatch: Any) -> None:
    reset_access_cache()
    attempts: list[int] = []

    def _flaky(**kwargs: Any) -> None:
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("backend down")

    import egisai._backend as backend

    monkeypatch.setattr(backend, "report_agent_access", _flaky)

    maybe_report_access("agent-2", _OPENAI_V1)
    _drain_threads()
    # First attempt failed → cache evicted → same bundle retries.
    maybe_report_access("agent-2", _OPENAI_V1)
    _drain_threads()
    assert len(attempts) == 2


def test_no_report_without_agent_or_tools() -> None:
    reset_access_cache()
    # Would raise on network if it ever tried to report — reaching
    # the return-early paths is the assertion.
    maybe_report_access(None, _OPENAI_V1)
    maybe_report_access("agent-3", {"messages": []})
    _drain_threads()


# ── Agent-framework options (Claude Agent SDK shape) ─────────────────


class _FakeServer:
    """Stands in for ``mcp.server.Server`` — a plain (weakref-able)
    class, like the real thing. ``SimpleNamespace`` would NOT work
    here (it has no ``__weakref__`` slot), which is exactly why the
    registry treats non-weakref-able instances as unregistrable."""


def _sdk_tool(name: str, description: str, schema: Any) -> Any:
    """Duck-typed ``@tool``-decorated object (SdkMcpTool shape)."""
    from types import SimpleNamespace

    return SimpleNamespace(
        name=name, description=description, input_schema=schema
    )


def _agent_options() -> Any:
    """Fake ``ClaudeAgentOptions`` mirroring the investor-demo shape."""
    from types import SimpleNamespace

    instance = _FakeServer()
    access.register_sdk_mcp_server_tools(
        instance,
        [
            _sdk_tool(
                "reconcile_spend_ledger",
                "Reconcile the spend ledger against platform invoices.",
                {"campaign_ref": str, "period": str},
            ),
            _sdk_tool(
                "identify_pacing_breaks",
                "Identify budget pacing breaks for a campaign.",
                {
                    "type": "object",
                    "properties": {"campaign_ref": {"type": "string"}},
                },
            ),
        ],
    )
    return (
        SimpleNamespace(
            system_prompt="You are a reconciliation specialist.",
            mcp_servers={
                "finance-tools": {
                    "type": "sdk",
                    "name": "finance-tools",
                    "instance": instance,
                }
            },
            allowed_tools=[
                "mcp__finance-tools__reconcile_spend_ledger",
                "Bash",
                "WebSearch",
            ],
        ),
        instance,  # keep the weak-registry key alive for the test body
    )


def test_options_extractor_yields_server_and_runtime_named_tools() -> None:
    options, _keepalive = _agent_options()
    items = access.extract_access_items_from_agent_options(options)
    by_key = {(it["kind"], it["name"]): it for it in items}

    # One mcp_server item, transport-labelled.
    server = by_key[("mcp_server", "finance-tools")]
    assert "sdk" in server["description"]

    # Tools carry the RUNTIME invocation name (the tool_call join
    # key) plus the human-readable server association.
    tool = by_key[("tool", "mcp__finance-tools__reconcile_spend_ledger")]
    assert tool["server_name"] == "finance-tools"
    assert tool["description"].startswith("Reconcile")
    # Simplified ``{param: type}`` schema → keys are the param names.
    assert tool["param_names"] == ["campaign_ref", "period"]
    assert tool["schema_hash"] is not None

    # JSON-Schema tool params resolve through ``properties``.
    tool2 = by_key[("tool", "mcp__finance-tools__identify_pacing_breaks")]
    assert tool2["param_names"] == ["campaign_ref"]

    # Built-ins from allowed_tools ship as name-only tool items.
    assert ("tool", "Bash") in by_key
    assert ("tool", "WebSearch") in by_key
    # The fully-qualified allowed_tools entry deduped against the
    # richer registry-derived entry (no duplicate).
    names = [it["name"] for it in items]
    assert len(names) == len(set(names))


def test_options_extractor_handles_external_transports_and_none() -> None:
    from types import SimpleNamespace

    assert access.extract_access_items_from_agent_options(None) == []

    options = SimpleNamespace(
        mcp_servers={
            "gh": {"command": "npx", "args": ["gh-mcp"]},
            "search": {"type": "http", "url": "https://mcp.example.com"},
        },
        allowed_tools=["mcp__gh__create_issue", "mcp__search"],
    )
    items = access.extract_access_items_from_agent_options(options)
    by_key = {(it["kind"], it["name"]): it for it in items}
    assert "stdio" in by_key[("mcp_server", "gh")]["description"]
    assert "http" in by_key[("mcp_server", "search")]["description"]
    tool = by_key[("tool", "mcp__gh__create_issue")]
    assert tool["server_name"] == "gh"
    # Server-level allow (``mcp__search``) yields no phantom tool.
    assert not any(
        it["kind"] == "tool" and it["name"].startswith("mcp__search")
        for it in items
    )


def test_options_tool_description_pii_is_masked() -> None:
    from types import SimpleNamespace

    instance = _FakeServer()
    # NOT a reserved doc-domain (example.com) — the sanitizer
    # deliberately skips those; this must read as real PII.
    raw = "Escalate to the AE. Contact ops@dentsu-ops.io for pager access."
    access.register_sdk_mcp_server_tools(
        instance, [_sdk_tool("escalate", raw, {"reason": str})]
    )
    options = SimpleNamespace(
        mcp_servers={"support": {"type": "sdk", "instance": instance}},
        allowed_tools=[],
    )
    items = access.extract_access_items_from_agent_options(options)
    tool = next(it for it in items if it["kind"] == "tool")
    assert "ops@dentsu-ops.io" not in repr(items)
    assert "ops@dentsu-ops.io" not in tool["description"]


def test_tool_names_extractor_builds_servers_and_tools() -> None:
    items = access.extract_access_items_from_tool_names(
        ["ToolSearch", "Bash", "mcp__finance-tools__reconcile_spend_ledger"]
    )
    by_key = {(it["kind"], it["name"]): it for it in items}
    assert ("tool", "ToolSearch") in by_key
    assert ("mcp_server", "finance-tools") in by_key
    tool = by_key[("tool", "mcp__finance-tools__reconcile_spend_ledger")]
    assert tool["server_name"] == "finance-tools"
    # Name-only sightings never fabricate schemas or descriptions.
    assert tool["schema_hash"] is None
    assert tool["description"] == ""


def test_split_mcp_tool_name_edge_cases() -> None:
    assert access.split_mcp_tool_name("mcp__srv__tool") == ("srv", "tool")
    # The tool part may itself contain ``__``.
    assert access.split_mcp_tool_name("mcp__srv__a__b") == ("srv", "a__b")
    assert access.split_mcp_tool_name("Bash") is None
    assert access.split_mcp_tool_name("mcp__srv") is None
    assert access.split_mcp_tool_name("mcp____tool") is None


# ── Merge-mode reporting (multi-source declarations) ────────────────


def test_merge_reports_converge_instead_of_flip_flopping(
    monkeypatch: Any,
) -> None:
    """Options + init-message reports union monotonically: after both
    sources land, re-seeing either one re-reports nothing."""
    reset_access_cache()
    calls: list[dict[str, Any]] = []

    def _capture(**kwargs: Any) -> None:
        calls.append(kwargs)

    import egisai._backend as backend

    monkeypatch.setattr(backend, "report_agent_access", _capture)

    options_items = [
        {
            "kind": "tool",
            "name": "mcp__fin__reconcile",
            "description": "Reconcile ledgers.",
            "schema_hash": "a" * 64,
            "param_names": ["ref"],
            "server_name": "fin",
        }
    ]
    init_items = access.extract_access_items_from_tool_names(
        ["ToolSearch", "mcp__fin__reconcile"]
    )

    access.maybe_report_access_items("agent-m", options_items, merge=True)
    _drain_threads()
    access.maybe_report_access_items("agent-m", init_items, merge=True)
    _drain_threads()
    assert len(calls) == 2  # union grew → second report

    # Steady state: replaying either source is a no-op.
    access.maybe_report_access_items("agent-m", options_items, merge=True)
    access.maybe_report_access_items("agent-m", init_items, merge=True)
    _drain_threads()
    assert len(calls) == 2

    # The final union kept the RICH entry (schema hash from options)
    # and added the name-only built-in from the init message.
    final = {it["name"]: it for it in calls[-1]["items"]}
    assert final["mcp__fin__reconcile"]["schema_hash"] == "a" * 64
    assert "ToolSearch" in final


def test_merge_name_only_never_downgrades_rich_entry(
    monkeypatch: Any,
) -> None:
    reset_access_cache()
    calls: list[dict[str, Any]] = []

    def _capture(**kwargs: Any) -> None:
        calls.append(kwargs)

    import egisai._backend as backend

    monkeypatch.setattr(backend, "report_agent_access", _capture)

    # Name-only first (init message beat the options report).
    name_only = access.extract_access_items_from_tool_names(
        ["mcp__fin__reconcile"]
    )
    access.maybe_report_access_items("agent-d", name_only, merge=True)
    _drain_threads()
    rich = [
        {
            "kind": "tool",
            "name": "mcp__fin__reconcile",
            "description": "Reconcile ledgers.",
            "schema_hash": "b" * 64,
            "param_names": ["ref"],
            "server_name": "fin",
        }
    ]
    access.maybe_report_access_items("agent-d", rich, merge=True)
    _drain_threads()
    final = {it["name"]: it for it in calls[-1]["items"]}
    # Rich replaced name-only…
    assert final["mcp__fin__reconcile"]["schema_hash"] == "b" * 64
    # …and replaying name-only afterwards changes nothing.
    n_before = len(calls)
    access.maybe_report_access_items("agent-d", name_only, merge=True)
    _drain_threads()
    assert len(calls) == n_before


def test_registry_skips_non_weakrefable_and_evicts_on_gc() -> None:
    """A non-weakref-able instance registers nothing (graceful — the
    observed layer still covers its tools); a GC'd server's registry
    entry is evicted so stale metadata can never map to a reused id."""
    import gc
    from types import SimpleNamespace

    # SimpleNamespace has no __weakref__ slot → skipped, no crash.
    ns = SimpleNamespace()
    access.register_sdk_mcp_server_tools(
        ns, [_sdk_tool("t", "d", {"a": str})]
    )
    assert access._sdk_mcp_tools_for({"instance": ns}) == []

    inst = _FakeServer()
    access.register_sdk_mcp_server_tools(
        inst, [_sdk_tool("t", "d", {"a": str})]
    )
    key = id(inst)
    assert key in access._SDK_MCP_TOOLS
    del inst
    gc.collect()
    assert key not in access._SDK_MCP_TOOLS


# ── claude_agent_sdk patch seams ─────────────────────────────────────


def test_create_sdk_mcp_server_wrap_registers_tool_metadata() -> None:
    """The patch's wrap records tool metadata against the returned
    config's instance so the options extractor can enumerate it."""

    from egisai._patches.claude_agent_sdk import _wrap_create_sdk_mcp_server

    def fake_create(
        name: str, version: str = "1.0.0", tools: Any = None
    ) -> dict[str, Any]:
        return {"type": "sdk", "name": name, "instance": _FakeServer()}

    wrapped = _wrap_create_sdk_mcp_server(fake_create)
    cfg = wrapped(
        name="support-tools",
        version="2.0.0",
        tools=[_sdk_tool("apply_credit", "Apply a service credit.", {"amount": str})],
    )
    # Return value passes through untouched.
    assert cfg["name"] == "support-tools"

    from types import SimpleNamespace as NS

    options = NS(mcp_servers={"support-tools": cfg}, allowed_tools=[])
    items = access.extract_access_items_from_agent_options(options)
    tool = next(it for it in items if it["kind"] == "tool")
    assert tool["name"] == "mcp__support-tools__apply_credit"
    assert tool["param_names"] == ["amount"]


def test_init_system_message_reports_runtime_toolset(
    monkeypatch: Any,
) -> None:
    """The CLI's ``init`` system message merges the runtime toolset
    (built-ins included) into the agent's declared inventory."""
    reset_access_cache()
    calls: list[dict[str, Any]] = []

    def _capture(**kwargs: Any) -> None:
        calls.append(kwargs)

    import egisai._backend as backend

    monkeypatch.setattr(backend, "report_agent_access", _capture)

    from egisai._patches.claude_agent_sdk import _report_init_message_tools

    class SystemMessage:  # duck-typed: matched on class NAME
        subtype = "init"
        data = {"tools": ["ToolSearch", "mcp__finance-tools__reconcile"]}

    _report_init_message_tools("agent-i", SystemMessage())
    _drain_threads()
    assert len(calls) == 1
    names = {it["name"] for it in calls[0]["items"]}
    assert "ToolSearch" in names
    assert "finance-tools" in names  # derived mcp_server item

    class OtherMessage:
        subtype = "init"
        data = {"tools": ["Bash"]}

    # Wrong class name → no-op (dispatch is duck-typed on the name).
    _report_init_message_tools("agent-i", OtherMessage())

    class SystemMessage:  # noqa: F811 — deliberate: right name, wrong subtype
        subtype = "compact"
        data = {"tools": ["Bash"]}

    _report_init_message_tools("agent-i", SystemMessage())
    _drain_threads()
    assert len(calls) == 1


def test_init_tools_filtered_by_allowed_tools_grant(
    monkeypatch: Any,
) -> None:
    """When the operator set ``allowed_tools``, the init message's
    ungated built-ins (CronDelete, TaskStop, …) are NOT declared —
    only the granted set is. This is the dentsu-demo shape: 4 MCP
    tools granted, ~25 CLI built-ins loaded but permission-gated."""
    reset_access_cache()
    calls: list[dict[str, Any]] = []

    def _capture(**kwargs: Any) -> None:
        calls.append(kwargs)

    import egisai._backend as backend

    monkeypatch.setattr(backend, "report_agent_access", _capture)

    from types import SimpleNamespace

    from egisai._patches.claude_agent_sdk import _report_init_message_tools

    options = SimpleNamespace(
        allowed_tools=[
            "mcp__audience__build_audience_segment",
            "mcp__audience__estimate_media_reach",
        ],
    )

    class SystemMessage:
        subtype = "init"
        data = {
            "tools": [
                "Bash",
                "CronDelete",
                "TaskStop",
                "ToolSearch",
                "mcp__audience__build_audience_segment",
                "mcp__audience__estimate_media_reach",
            ]
        }

    _report_init_message_tools("agent-g", SystemMessage(), options)
    _drain_threads()
    assert len(calls) == 1
    names = {it["name"] for it in calls[0]["items"]}
    # Only the granted MCP tools + their derived server.
    assert names == {
        "mcp__audience__build_audience_segment",
        "mcp__audience__estimate_media_reach",
        "audience",
    }

    # All-gated init list → no report at all.
    reset_access_cache()
    calls.clear()

    class SystemMessage2:
        subtype = "init"
        data = {"tools": ["Bash", "TaskStop"]}

    SystemMessage2.__name__ = "SystemMessage"
    _report_init_message_tools("agent-g2", SystemMessage2(), options)
    _drain_threads()
    assert calls == []


def test_allowed_tools_grant_semantics() -> None:
    from types import SimpleNamespace

    from egisai._patches.claude_agent_sdk import (
        _allowed_tool_names,
        _init_tool_is_granted,
    )

    # No options / empty list → no restriction (None).
    assert _allowed_tool_names(None) is None
    assert _allowed_tool_names(SimpleNamespace(allowed_tools=[])) is None

    # Permission specifiers keep only the tool-name part.
    allowed = _allowed_tool_names(
        SimpleNamespace(allowed_tools=["Bash(git:*)", "mcp__audience"])
    )
    assert allowed == {"Bash", "mcp__audience"}
    assert _init_tool_is_granted("Bash", allowed)
    # Server-level allow grants every tool on that server…
    assert _init_tool_is_granted("mcp__audience__build_audience_segment", allowed)
    # …but not other servers or ungranted built-ins.
    assert not _init_tool_is_granted("mcp__other__tool", allowed)
    assert not _init_tool_is_granted("TaskStop", allowed)
