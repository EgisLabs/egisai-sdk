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
