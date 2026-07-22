"""Agent access-inventory extraction + reporting.

Backs the dashboard's per-agent "Access" tab: what tools, MCP
servers, and database-shaped capabilities can this agent reach?

The SDK already *sees* every tool definition. Three declaration
shapes feed this module:

* **Request payloads** — provider patches pass ``tools`` (or
  Bedrock's ``toolConfig``) inside the in-process ``payload`` dict
  (:func:`extract_access_items`).
* **Agent-framework options** — the Claude Agent SDK declares tools
  via ``ClaudeAgentOptions`` (``mcp_servers`` + ``allowed_tools``),
  never in the payload (:func:`extract_access_items_from_agent_options`).
  In-process SDK MCP servers register their full tool metadata via
  :func:`register_sdk_mcp_server_tools` (called by the patch's
  ``create_sdk_mcp_server`` wrap).
* **Runtime tool lists** — bare tool-name lists observed at runtime
  (e.g. the Claude CLI's ``init`` system message)
  (:func:`extract_access_items_from_tool_names`).

Each shape becomes a metadata-only inventory shipped to
``POST /v1/sdk/agents/access`` the first time each
``(agent_id, bundle_hash)`` pair is seen.

MCP-served tools carry their **runtime invocation name**
(``mcp__<server>__<tool>``) — the exact string the model emits on a
``tool_call`` — so the backend can join declared inventory against
observed usage. The human-readable server association ships
separately as ``server_name``.

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
import weakref
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
# ``(agent_id) -> {(kind, name): item}`` union of everything merge-mode
# reports have seen for the agent. Merge mode is used by agentic
# frameworks whose declarations arrive from multiple sources (options
# at query time, the CLI's init message mid-stream) — a monotonic
# union keeps the bundle hash stable instead of flip-flopping between
# per-source hashes (which would re-report every turn).
_merged_items: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
_lock = threading.Lock()

# In-process SDK MCP server instance -> metadata of the tools it
# registered. Populated by the ``claude_agent_sdk`` patch's
# ``create_sdk_mcp_server`` wrap; read by
# :func:`extract_access_items_from_agent_options`.
#
# Keyed on ``id(instance)`` (not a WeakKeyDictionary) because the
# upstream server class is not ours — it may be unhashable or define
# ``__eq__``. Each entry stores a weakref to the instance: the lookup
# verifies identity (guards against id reuse after GC) and a
# ``weakref.finalize`` evicts the entry when the server is collected,
# so a dropped server never leaks through this registry.
_SDK_MCP_TOOLS: dict[int, tuple[Any, list[dict[str, Any]]]] = {}


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
    """Declared parameter *names* only — never values or examples.

    Handles both JSON Schema (``{"properties": {...}}``) and the
    Claude Agent SDK's simplified ``@tool`` schema (a flat
    ``{param_name: type}`` mapping). The two are disambiguated the
    same way the upstream SDK does: a dict carrying JSON-Schema
    structural keys (``properties`` / ``type`` / ``$schema``) is a
    JSON Schema; any other flat dict is a param→type mapping.
    """
    if not isinstance(schema, dict) or not schema:
        return []
    props = schema.get("properties")
    if isinstance(props, dict):
        out = [str(k)[:128] for k in props if isinstance(k, str)]
        return sorted(out)[:_MAX_PARAM_NAMES]
    if "type" in schema or "$schema" in schema or "properties" in schema:
        # JSON Schema without an inspectable ``properties`` block.
        return []
    # Simplified schema — keys ARE the parameter names.
    out = [str(k)[:128] for k in schema if isinstance(k, str)]
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
    *,
    name: Any,
    description: Any,
    schema: Any,
    server_name: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(name, str) or not name:
        return None
    return {
        "kind": "tool",
        "name": name[:255],
        "description": _safe_description(description),
        "schema_hash": _schema_hash(schema),
        "param_names": _param_names(schema),
        "server_name": server_name[:255] if server_name else None,
    }


def _server_entry(name: Any, description: str = "") -> dict[str, Any] | None:
    if not isinstance(name, str) or not name:
        return None
    return {
        "kind": "mcp_server",
        "name": name[:255],
        "description": description[:_MAX_DESCRIPTION_CHARS],
        "schema_hash": None,
        "param_names": [],
        "server_name": None,
    }


def split_mcp_tool_name(name: str) -> tuple[str, str] | None:
    """Split a runtime ``mcp__<server>__<tool>`` name.

    Returns ``(server, tool)`` or ``None`` when ``name`` isn't an
    MCP-namespaced tool name. The tool part may itself contain
    ``__`` (only the first two separators are structural).
    """
    if not name.startswith("mcp__"):
        return None
    parts = name.split("__", 2)
    if len(parts) < 3 or not parts[1] or not parts[2]:
        return None
    return parts[1], parts[2]


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
        # MCP-tagged tools carry their server association; resolve it
        # first so the tool entry itself records ``server_name``.
        server: str | None = None
        mcp = tool.get("mcp")
        if isinstance(mcp, dict):
            raw_server = mcp.get("server") or mcp.get("server_url")
            if isinstance(raw_server, str) and raw_server:
                server = raw_server
                _add(_server_entry(server))
        fn = tool.get("function")
        if isinstance(fn, dict):
            # OpenAI v1 function tool.
            _add(
                _tool_entry(
                    name=fn.get("name"),
                    description=fn.get("description"),
                    schema=fn.get("parameters"),
                    server_name=server,
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
                    server_name=server,
                )
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


# ── Agent-framework declarations (Claude Agent SDK) ─────────────────


def register_sdk_mcp_server_tools(instance: Any, tools: Any) -> None:
    """Record the tool metadata an in-process SDK MCP server registered.

    Called by the ``claude_agent_sdk`` patch's ``create_sdk_mcp_server``
    wrap. ``tools`` is the raw list of ``@tool``-decorated objects
    (duck-typed: ``.name`` / ``.description`` / ``.input_schema``).
    Only metadata is retained — the handler callables never enter the
    registry.
    """
    if instance is None:
        return
    meta: list[dict[str, Any]] = []
    for t in _as_list(tools):
        name = getattr(t, "name", None)
        if not isinstance(name, str) or not name:
            continue
        meta.append(
            {
                "name": name,
                "description": getattr(t, "description", None),
                "schema": getattr(t, "input_schema", None),
            }
        )
    try:
        ref = weakref.ref(instance)
    except TypeError:
        # Non-weakref-able instance — skip; Layer-2 observed capture
        # still surfaces the tools when they're invoked.
        LOGGER.debug("SDK MCP server instance not weakref-able; skipping")
        return
    key = id(instance)
    _SDK_MCP_TOOLS[key] = (ref, meta)
    weakref.finalize(instance, _SDK_MCP_TOOLS.pop, key, None)


def _sdk_mcp_tools_for(config: Any) -> list[dict[str, Any]]:
    """Registered tool metadata for one ``mcp_servers`` config entry."""
    instance = (
        config.get("instance")
        if isinstance(config, dict)
        else getattr(config, "instance", None)
    )
    if instance is None:
        return []
    entry = _SDK_MCP_TOOLS.get(id(instance))
    if entry is None or entry[0]() is not instance:
        # Unknown instance, or the id was reused after a GC'd server —
        # the weakref identity check keeps stale metadata out.
        return []
    return entry[1]


def _server_transport(config: Any) -> str:
    """Human-readable transport label for an MCP server config."""
    if isinstance(config, dict):
        ctype = config.get("type")
        if isinstance(ctype, str) and ctype:
            return ctype
        if config.get("command"):
            return "stdio"
        if config.get("url"):
            return "http"
    elif config is not None and getattr(config, "instance", None) is not None:
        return "sdk"
    return ""


def extract_access_items_from_agent_options(options: Any) -> list[dict[str, Any]]:
    """Build the inventory from agent-framework options (Claude Agent SDK).

    Reads two declaration surfaces off ``ClaudeAgentOptions`` (attr or
    dict access — the patch never imports upstream types):

    * ``mcp_servers`` — one ``mcp_server`` item per server. In-process
      SDK servers additionally yield one ``tool`` item per registered
      tool (full metadata via :func:`register_sdk_mcp_server_tools`),
      named with the runtime form ``mcp__<server>__<tool>`` so the
      backend can join declared inventory against observed usage.
    * ``allowed_tools`` — built-in tool names (``Bash``, ``Read``, …)
      and any fully-qualified ``mcp__…`` names not already covered.

    Returns ``[]`` when ``options`` declares nothing.
    """
    if options is None:
        return []

    def _get(attr: str) -> Any:
        value = getattr(options, attr, None)
        if value is None and isinstance(options, dict):
            value = options.get(attr)
        return value

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

    mcp_servers = _get("mcp_servers")
    server_configs: dict[str, Any] = {}
    if isinstance(mcp_servers, dict):
        server_configs = {str(k): v for k, v in mcp_servers.items() if k}
    elif isinstance(mcp_servers, (list, tuple)):
        for cfg in mcp_servers:
            sname = getattr(cfg, "name", None) or (
                cfg.get("name") if isinstance(cfg, dict) else None
            )
            if isinstance(sname, str) and sname:
                server_configs[sname] = cfg

    for sname in sorted(server_configs):
        config = server_configs[sname]
        transport = _server_transport(config)
        _add(
            _server_entry(
                sname,
                description=(
                    f"MCP server ({transport} transport)" if transport else ""
                ),
            )
        )
        for tool_meta in _sdk_mcp_tools_for(config):
            _add(
                _tool_entry(
                    name=f"mcp__{sname}__{tool_meta['name']}",
                    description=tool_meta.get("description"),
                    schema=tool_meta.get("schema"),
                    server_name=sname,
                )
            )

    for raw in _as_list(_get("allowed_tools")):
        if not isinstance(raw, str) or not raw:
            continue
        split = split_mcp_tool_name(raw)
        if split is not None:
            server, _tool = split
            _add(_server_entry(server))
            _add(
                _tool_entry(
                    name=raw, description=None, schema=None, server_name=server
                )
            )
            continue
        if raw.startswith("mcp__"):
            # Server-level allow (``mcp__<server>``) — server only.
            parts = raw.split("__", 2)
            if len(parts) >= 2 and parts[1]:
                _add(_server_entry(parts[1]))
            continue
        _add(_tool_entry(name=raw, description=None, schema=None))

    return items


def extract_access_items_from_tool_names(names: Any) -> list[dict[str, Any]]:
    """Build a name-only inventory from a runtime tool-name list.

    Used for tool lists observed at runtime (e.g. the Claude CLI's
    ``init`` system message, which is the authoritative record of
    what actually loaded into the session — including built-ins like
    ``ToolSearch`` that never appear in the user's options).
    """
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for raw in _as_list(names):
        name = raw if isinstance(raw, str) else (
            raw.get("name") if isinstance(raw, dict) else None
        )
        if not isinstance(name, str) or not name:
            continue
        split = split_mcp_tool_name(name)
        server = split[0] if split is not None else None
        for entry in (
            _server_entry(server) if server else None,
            _tool_entry(
                name=name, description=None, schema=None, server_name=server
            ),
        ):
            if entry is None or len(items) >= _MAX_ITEMS:
                continue
            key = (entry["kind"], entry["name"])
            if key in seen:
                continue
            seen.add(key)
            items.append(entry)
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


def _richer(new: dict[str, Any], old: dict[str, Any]) -> bool:
    """Would replacing ``old`` with ``new`` add information?

    Merge-mode reports arrive from sources of different fidelity
    (options carry schemas + descriptions; init messages carry names
    only). A name-only sighting must never downgrade a rich entry.
    """
    if new.get("schema_hash") and not old.get("schema_hash"):
        return True
    return bool(new.get("description")) and not old.get("description")


def maybe_report_access_items(
    agent_id: str | None,
    items: list[dict[str, Any]],
    *,
    merge: bool = False,
) -> None:
    """Report a prebuilt inventory if the agent's bundle changed.

    ``merge=False`` — full-bundle semantics: ``items`` IS the agent's
    complete declared bundle (payload-shaped declarations); the
    backend tombstones anything absent from it.

    ``merge=True`` — monotonic-union semantics for frameworks whose
    declarations arrive from multiple sources across a turn (options
    at query time, the CLI init message mid-stream). Each report
    ships the union of everything seen for the agent so the bundle
    hash converges instead of flip-flopping per source (which would
    re-report every turn). Name-only sightings never downgrade rich
    entries (see :func:`_richer`).

    Hot-path contract: steady state is one dict lookup; the report
    thread only spawns when the bundle hash is new for this agent.
    """
    if not agent_id or not items:
        return
    try:
        with _lock:
            if merge:
                union = _merged_items.setdefault(agent_id, {})
                for entry in items:
                    key = (entry["kind"], entry["name"])
                    prev = union.get(key)
                    if prev is None or _richer(entry, prev):
                        union[key] = entry
                report_items = list(union.values())[:_MAX_ITEMS]
            else:
                report_items = items
            digest = compute_bundle_hash(report_items)
            if _reported.get(agent_id) == digest:
                return
            _reported[agent_id] = digest
        threading.Thread(
            target=_report,
            args=(agent_id, report_items, digest),
            name="egisai-access-report",
            daemon=True,
        ).start()
    except Exception:  # noqa: BLE001
        LOGGER.debug("access report scheduling failed", exc_info=True)


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
    except Exception:  # noqa: BLE001
        LOGGER.debug("access extraction failed", exc_info=True)
        return
    maybe_report_access_items(agent_id, items)


def reset_access_cache() -> None:
    """Test helper — forget every reported bundle."""
    with _lock:
        _reported.clear()
        _merged_items.clear()
