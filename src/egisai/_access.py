"""Agent access-inventory extraction + reporting.

Backs the dashboard's per-agent "Access" tab: what tools, MCP
servers, and database-shaped capabilities can this agent reach?

The SDK already *sees* every tool definition — provider patches pass
``tools`` (or Bedrock's ``toolConfig``) inside the in-process
``payload`` dict. This module turns that bundle into a metadata-only
inventory and ships it to ``POST /v1/sdk/agents/access`` the first
time each ``(agent_id, bundle_hash)`` pair is seen.

Privacy contract (per the platform's compliance rules):

* Item metadata only — tool *name*, a PII-sanitized + truncated
  description, a SHA-256 hash of the declared input schema, and the
  parameter *names*. Never the schema JSON itself, never call
  arguments, never argument values.
* Descriptions fail closed: if the PII sanitizer errors, the
  description is dropped entirely (name + hash still ship).

Hot-path contract (per the SDK design philosophy):

* Steady state is one dict lookup per call — the inventory only
  ships when the bundle hash changes for an agent.
* The report itself runs on a daemon thread and fails open; a
  backend outage never delays or breaks the user's model call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from typing import Any

LOGGER = logging.getLogger("egisai.access")

# Caps keep the payload bounded even against a pathological tool
# bundle (an agent declaring thousands of tools).
_MAX_ITEMS = 200
_MAX_PARAM_NAMES = 40
_MAX_DESCRIPTION_CHARS = 500

# ``(agent_id) -> bundle_hash`` of the last successfully *initiated*
# report. Guarded by ``_lock`` so concurrent calls from multiple
# threads don't double-report the same bundle.
_reported: dict[str, str] = {}
_lock = threading.Lock()


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _schema_hash(schema: Any) -> str | None:
    """SHA-256 (hex) of the declared input schema — never the schema."""
    if not isinstance(schema, dict) or not schema:
        return None
    try:
        canonical = json.dumps(schema, sort_keys=True, default=str)
    except Exception:  # noqa: BLE001
        return None
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _param_names(schema: Any) -> list[str]:
    """Declared parameter *names* only — never values or examples."""
    if not isinstance(schema, dict):
        return []
    props = schema.get("properties")
    if not isinstance(props, dict):
        return []
    out = [str(k)[:128] for k in props if isinstance(k, str)]
    return sorted(out)[:_MAX_PARAM_NAMES]


def _safe_description(raw: Any) -> str:
    """PII-sanitize + truncate a tool description.

    Fail closed: any sanitizer error drops the description entirely
    — better an empty field than a leaked value on the inventory.
    """
    if not isinstance(raw, str) or not raw:
        return ""
    try:
        from egisai.policy.pii import sanitize as pii_sanitize

        masked, _records = pii_sanitize(raw)
        return masked[:_MAX_DESCRIPTION_CHARS]
    except Exception:  # noqa: BLE001
        return ""


def _tool_entry(
    *, name: Any, description: Any, schema: Any
) -> dict[str, Any] | None:
    if not isinstance(name, str) or not name:
        return None
    return {
        "kind": "tool",
        "name": name[:255],
        "description": _safe_description(description),
        "schema_hash": _schema_hash(schema),
        "param_names": _param_names(schema),
    }


def extract_access_items(payload: Any) -> list[dict[str, Any]]:
    """Build the metadata-only inventory from a request payload.

    Handles every declared-tool shape the provider patches produce:

    * OpenAI v1 — ``tools[*].function.{name,description,parameters}``
    * Anthropic — ``tools[*].{name,description,input_schema}``
    * OpenAI legacy / generic — ``tools[*].{name,description,parameters}``
    * Bedrock Converse — ``toolConfig.tools[*].toolSpec``
    * MCP-tagged tools — ``tools[*].mcp.server`` → one ``mcp_server``
      item per distinct server.

    Returns ``[]`` for payloads with no declared tools.
    """
    if not isinstance(payload, dict):
        return []

    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _add(entry: dict[str, Any] | None) -> None:
        if entry is None or len(items) >= _MAX_ITEMS:
            return
        key = (entry["kind"], entry["name"])
        if key in seen:
            return
        seen.add(key)
        items.append(entry)

    for tool in _as_list(payload.get("tools")):
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict):
            # OpenAI v1 function tool.
            _add(
                _tool_entry(
                    name=fn.get("name"),
                    description=fn.get("description"),
                    schema=fn.get("parameters"),
                )
            )
        else:
            # Anthropic (``input_schema``) / OpenAI legacy
            # (``parameters``) / generic dict with a ``name``.
            _add(
                _tool_entry(
                    name=tool.get("name"),
                    description=tool.get("description"),
                    schema=tool.get("input_schema") or tool.get("parameters"),
                )
            )
        mcp = tool.get("mcp")
        if isinstance(mcp, dict):
            server = mcp.get("server") or mcp.get("server_url")
            if isinstance(server, str) and server:
                _add(
                    {
                        "kind": "mcp_server",
                        "name": server[:255],
                        "description": "",
                        "schema_hash": None,
                        "param_names": [],
                    }
                )

    tool_config = payload.get("toolConfig")
    if isinstance(tool_config, dict):
        for tool in _as_list(tool_config.get("tools")):
            if not isinstance(tool, dict):
                continue
            spec = tool.get("toolSpec")
            if not isinstance(spec, dict):
                continue
            input_schema = spec.get("inputSchema")
            schema = (
                input_schema.get("json")
                if isinstance(input_schema, dict)
                else None
            )
            _add(
                _tool_entry(
                    name=spec.get("name"),
                    description=spec.get("description"),
                    schema=schema,
                )
            )

    return items


def compute_bundle_hash(items: list[dict[str, Any]]) -> str:
    """Stable SHA-256 over the inventory's identifying fields.

    Only ``(kind, name, schema_hash)`` participate — a description
    wording tweak doesn't count as an access change, but a schema
    change does.
    """
    canonical = sorted(
        (
            str(it.get("kind") or ""),
            str(it.get("name") or ""),
            str(it.get("schema_hash") or ""),
        )
        for it in items
    )
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _report(agent_id: str, items: list[dict[str, Any]], bundle_hash: str) -> None:
    """Daemon-thread body — POST the inventory; fail open on any error."""
    try:
        from egisai._backend import report_agent_access

        report_agent_access(
            agent_id=agent_id, items=items, bundle_hash=bundle_hash
        )
    except Exception as exc:  # noqa: BLE001
        # Fail open + allow a retry on the next bundle sighting.
        with _lock:
            if _reported.get(agent_id) == bundle_hash:
                del _reported[agent_id]
        LOGGER.debug(
            "access report failed for agent %s: %s",
            agent_id,
            exc.__class__.__name__,
        )


def maybe_report_access(agent_id: str | None, payload: Any) -> None:
    """Report the agent's declared access bundle if it changed.

    Called on the gate hot path — MUST stay cheap. Steady state is
    one dict lookup (the extraction only runs when the payload
    actually declares tools, and the report thread only spawns when
    the bundle hash is new for this agent).
    """
    if not agent_id or not isinstance(payload, dict):
        return
    raw_tools = payload.get("tools") or payload.get("toolConfig")
    if not raw_tools:
        return
    try:
        items = extract_access_items(payload)
        if not items:
            return
        digest = compute_bundle_hash(items)
        with _lock:
            if _reported.get(agent_id) == digest:
                return
            _reported[agent_id] = digest
        threading.Thread(
            target=_report,
            args=(agent_id, items, digest),
            name="egisai-access-report",
            daemon=True,
        ).start()
    except Exception:  # noqa: BLE001
        LOGGER.debug("access extraction failed", exc_info=True)


def reset_access_cache() -> None:
    """Test helper — forget every reported bundle."""
    with _lock:
        _reported.clear()
