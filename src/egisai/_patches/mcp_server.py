"""MCP server adapter — govern inbound ``tools/call`` on hosted MCP servers.

Part of the MCP Servers add-on. Where every other patch in this
package governs an agent's *outbound* model/tool calls, this patch
governs the *inbound* tool calls that external agents make against an
MCP server the customer hosts (FastMCP v2 or the official ``mcp``
SDK's ``FastMCP``).

Design contract (``sdk-design-philosophy.mdc`` + the add-on spec):

* **Dormant unless entitled.** ``apply()`` is a no-op unless
  ``init()``'s handshake reported the org has the ``mcp_servers``
  add-on enabled (``config.mcp_servers_enabled``). A customer without
  the add-on — i.e. everyone today — gets byte-for-byte the same
  behaviour as before this module existed.
* **Auto-detect + auto-register.** On first sight of a server we
  fingerprint it (name + sorted tool schema + transport), register it
  via ``POST /v1/sdk/mcp-servers/ensure`` (cached forever after), and
  report its tool inventory.
* **Govern every inbound tool call.** Each ``tools/call`` is evaluated
  with the already-public ``evaluate_output_policies`` against the
  org's rules scoped to this server: ``block`` raises a tool error (the
  tool never runs), ``sanitize`` masks the arguments in place, ``allow``
  proceeds. Every outcome is reported as a ``source_kind="mcp_server"``
  audit event.
* **Fail open, never break the customer's server.** Any unexpected
  error in the gate falls through to the original handler so the
  customer's MCP server keeps serving even when egisai is unhappy.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from egisai._auto_agent import _hash_bundle
from egisai._backend import ensure_mcp_server
from egisai._config import get_config_optional
from egisai._events import build_event, safe_preview
from egisai._logger import enqueue
from egisai._policy_cache import get_rules
from egisai.policy import OutputPolicyContext, PolicyDecision, evaluate_output_policies
from egisai.policy.pii import sanitize as pii_sanitize

LOGGER = logging.getLogger("egisai.patches.mcp_server")

_SOURCE = "mcp_server"


# ── Per-server identity cache ───────────────────────────────────────


@dataclass
class _ServerIdentity:
    """Cached registration state for one in-process MCP server instance."""

    name: str
    identity_hash: str
    transport: str | None = None
    server_id: str | None = None
    # ``True`` once ``ensure`` has succeeded. While ``False`` we retry
    # registration on the next call (fail-open: the server keeps
    # working, we just haven't attributed it to a dashboard row yet).
    registered: bool = False


# Keyed by ``id(server_instance)`` — one entry per live FastMCP object.
_registry: dict[int, _ServerIdentity] = {}


@dataclass
class _Gate:
    """Result of preparing the gate for one inbound tool call."""

    identity: _ServerIdentity
    tool_name: str
    arguments: Any
    decision: PolicyDecision | None = None
    blocked: bool = False
    message: str = ""
    sanitizations: list[dict[str, Any]] = field(default_factory=list)


# ── Tool-error type (block enforcement) ─────────────────────────────


def _raise_block(message: str) -> None:
    """Raise the most framework-appropriate error to refuse a tool call.

    Both FastMCP flavours convert a ``ToolError`` (or any exception
    raised inside the tool dispatch path) into an MCP error result the
    calling agent sees — the tool body never runs. We prefer the
    framework's ``ToolError`` when importable so the error is rendered
    cleanly; otherwise a ``PermissionError`` propagates the same way.
    """
    for mod_name, attr in (
        ("fastmcp.exceptions", "ToolError"),
        ("mcp.server.fastmcp.exceptions", "ToolError"),
    ):
        try:
            module = __import__(mod_name, fromlist=[attr])
            err_cls = getattr(module, attr)
            raise err_cls(message)
        except ImportError:
            continue
    raise PermissionError(message)


# ── Tool inventory discovery (best-effort, fail-open) ───────────────


def _discover_tools(instance: Any) -> list[dict[str, Any]]:
    """Return ``[{name, description, schema_hash}]`` for a server.

    Best-effort across FastMCP variants: each access path is wrapped so
    an internal-API change in any framework version degrades to "no
    inventory" rather than breaking registration.
    """
    raw_tools: Any = None
    for getter in (
        lambda: instance._tool_manager.list_tools(),  # noqa: SLF001
        lambda: instance._tool_manager._tools,  # noqa: SLF001
        lambda: instance._tools,  # noqa: SLF001
    ):
        try:
            candidate = getter()
        except Exception:  # noqa: BLE001
            continue
        if candidate:
            raw_tools = candidate
            break
    if not raw_tools:
        return []

    # Normalise dict-of-tools or list-of-tools into an iterable of tool
    # objects.
    if isinstance(raw_tools, dict):
        items = list(raw_tools.values())
    else:
        items = list(raw_tools)

    out: list[dict[str, Any]] = []
    for tool in items:
        try:
            name = getattr(tool, "name", None) or (
                tool.get("name") if isinstance(tool, dict) else None
            )
            if not name:
                continue
            description = getattr(tool, "description", None) or (
                tool.get("description") if isinstance(tool, dict) else None
            )
            schema = (
                getattr(tool, "parameters", None)
                or getattr(tool, "inputSchema", None)
                or getattr(tool, "input_schema", None)
            )
            schema_hash = _hash_bundle([name, schema]) if schema else None
            out.append(
                {
                    "name": str(name)[:128],
                    "description": (str(description)[:2000] if description else ""),
                    "schema_hash": schema_hash,
                }
            )
        except Exception:  # noqa: BLE001
            continue
    return out


def _detect_transport(instance: Any) -> str | None:
    """Best-effort transport label for the server, or ``None``."""
    for attr in ("transport", "_transport"):
        try:
            value = getattr(instance, attr, None)
            if isinstance(value, str) and value:
                return value[:32]
        except Exception:  # noqa: BLE001
            continue
    return None


def _server_name(instance: Any) -> str:
    for attr in ("name", "_name"):
        try:
            value = getattr(instance, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()[:255]
        except Exception:  # noqa: BLE001
            continue
    return "MCP Server"


def _resolve_identity(instance: Any) -> _ServerIdentity:
    """Resolve (and lazily register) the identity for a server instance.

    Cached by ``id(instance)``. The first call performs the ``ensure``
    round-trip; subsequent calls are a dict lookup. A failed ensure
    leaves ``registered=False`` so we retry next time without ever
    blocking the customer's server.
    """
    key = id(instance)
    cached = _registry.get(key)
    if cached is not None and cached.registered:
        return cached

    name = _server_name(instance)
    transport = _detect_transport(instance)
    tools = _discover_tools(instance)
    identity_hash = _hash_bundle(
        [name, [t["name"] for t in tools], transport or ""]
    )

    identity = cached or _ServerIdentity(
        name=name, identity_hash=identity_hash, transport=transport
    )
    identity.name = name
    identity.transport = transport
    identity.identity_hash = identity_hash

    runtime: dict[str, Any] | None = None
    try:
        from egisai import __version__
        from egisai._runtime import collect_runtime_fingerprint

        runtime = collect_runtime_fingerprint(sdk_version=__version__)
    except Exception:  # noqa: BLE001
        runtime = None

    try:
        resp = ensure_mcp_server(
            name=name,
            transport=transport,
            identity_hash=identity_hash,
            identity_source="sdk",
            runtime=runtime,
            tools=tools or None,
        )
        server_id = resp.get("id")
        if server_id:
            identity.server_id = str(server_id)
            identity.registered = True
    except Exception as exc:  # noqa: BLE001
        # Fail open — keep serving, retry registration next call.
        LOGGER.debug("mcp server ensure failed: %s", exc.__class__.__name__)

    _registry[key] = identity
    return identity


# ── Policy scope + governance ───────────────────────────────────────


def _scope_rules(rules: list, server_id: str | None) -> list:
    """Keep only rules that apply to this MCP server.

    * A rule with non-empty ``mcp_server_ids`` applies iff this
      server's id is listed.
    * A rule with empty ``mcp_server_ids`` AND empty ``agent_ids`` is
      org-wide — it applies to both agents and MCP servers.
    * A rule scoped to agents only (``agent_ids`` set, no
      ``mcp_server_ids``) does NOT apply on the MCP side.
    """
    sid = (server_id or "").strip().lower()
    out = []
    for r in rules:
        mcp_ids = tuple(getattr(r, "mcp_server_ids", ()) or ())
        agent_ids = tuple(getattr(r, "agent_ids", ()) or ())
        if mcp_ids:
            if sid and sid in mcp_ids:
                out.append(r)
        elif not agent_ids:
            out.append(r)
    return out


def _arguments_text(arguments: Any) -> str:
    """Render tool-call arguments as text for policy evaluation."""
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    try:
        import json

        return json.dumps(arguments, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        return repr(arguments)


def _sanitize_arguments(
    arguments: Any, decision: PolicyDecision
) -> tuple[Any, list[dict[str, Any]]]:
    """Mask PII in tool-call arguments. Returns ``(new_args, records)``.

    Walks string leaves of the arguments (dict / list / str) and masks
    each via the shared PII engine using the decision's requested
    types. Records carry count + mask shape only — never the raw value
    (security-and-compliance.mdc §1).
    """
    aggregated: dict[str, dict[str, Any]] = {}

    def _mask(text: str) -> str:
        new_text, recs = pii_sanitize(
            text,
            types=decision.sanitize_types or None,
            mask_char=decision.sanitize_mask_char,
        )
        for rec in recs:
            existing = aggregated.get(rec.type)
            if existing is None:
                aggregated[rec.type] = {
                    "type": rec.type,
                    "count": rec.count,
                    "pattern": rec.pattern,
                }
            else:
                existing["count"] += rec.count
        return new_text

    def _walk(value: Any) -> Any:
        if isinstance(value, str):
            return _mask(value)
        if isinstance(value, dict):
            return {k: _walk(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_walk(v) for v in value]
        return value

    try:
        new_args = _walk(arguments)
    except Exception:  # noqa: BLE001
        return arguments, []
    return new_args, list(aggregated.values())


def _prepare_gate(instance: Any, tool_name: str, arguments: Any) -> _Gate:
    """Resolve identity, evaluate policies, and (maybe) sanitize args.

    Runs entirely on a worker thread (see the async wrapper) because it
    performs the registration round-trip and may invoke the
    ``semantic_guard`` judge — both blocking httpx calls we must keep
    off the server's event loop.
    """
    identity = _resolve_identity(instance)
    gate = _Gate(identity=identity, tool_name=tool_name, arguments=arguments)

    rules = _scope_rules(get_rules(), identity.server_id)
    if not rules:
        return gate

    ctx = OutputPolicyContext(
        tenant=identity.server_id or "",
        model="",
        text=_arguments_text(arguments),
        tool_names=[tool_name],
        tool_calls=[{"name": tool_name, "arguments": _arguments_text(arguments)}],
        mcp_targets=[identity.server_id] if identity.server_id else [],
        stream=False,
        allow_sanitize=True,
    )
    try:
        # Single-surface gate: an inbound MCP ``tools/call``. Rules
        # scoped via ``applies_to`` only fire when they cover "mcp".
        decision = evaluate_output_policies(rules, ctx, surfaces=("mcp",))
    except Exception as exc:  # noqa: BLE001
        # Fail open on availability (NOT on PII — the deterministic PII
        # checks already ran and any block among them would have been
        # returned; an exception here means the engine itself failed).
        LOGGER.debug("mcp policy eval failed: %s", exc.__class__.__name__)
        return gate

    gate.decision = decision
    if decision.verdict == "block":
        gate.blocked = True
        gate.message = (
            f"[egisai] {decision.message or 'blocked by policy'} "
            f"(matched={decision.matched_policy})"
        )
    elif decision.verdict == "sanitize":
        new_args, records = _sanitize_arguments(arguments, decision)
        gate.arguments = new_args
        gate.sanitizations = records
    return gate


# ── Audit event ─────────────────────────────────────────────────────


def _serialize_matched_policies(decision: PolicyDecision) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in decision.matched_policies:
        out.append(
            {
                "name": r.name,
                "type": r.type,
                "verdict": r.verdict,
                "reason_code": r.reason_code,
                "message": r.message,
                "sanitize_types": list(r.sanitize_types),
                "sanitize_mask_char": r.sanitize_mask_char,
            }
        )
    return out


def _emit_event(
    gate: _Gate,
    *,
    latency_ms: int,
    error: bool,
) -> None:
    """Build + enqueue a ``source_kind='mcp_server'`` audit event."""
    try:
        ev = build_event(
            source=_SOURCE,
            target=f"mcp.{gate.identity.name}.tools/call",
            payload=gate.arguments,
            model=None,
            stream=False,
        )
    except Exception:  # noqa: BLE001
        return

    decision = gate.decision
    verdict = decision.verdict if decision is not None else "allow"

    # MCP servers are not agents — null out agent attribution and
    # stamp the server linkage the backend ingest routes on.
    ev["agent_id"] = None
    ev["source_kind"] = _SOURCE
    ev["mcp_server_id"] = gate.identity.server_id
    ev["app"] = gate.identity.name
    ev["step_kind"] = "tool_call"
    ev["tool_name"] = gate.tool_name
    ev["verdict"] = verdict
    ev["latency_ms"] = latency_ms
    ev["enforcement_status"] = "enforced"
    ev["payload_preview"] = safe_preview(gate.arguments)
    if error:
        ev["error"] = "tool call failed"
    if decision is not None:
        ev["reason_code"] = decision.reason_code
        ev["reason"] = decision.message
        ev["matched_policy"] = decision.matched_policy
        ev["matched_policies"] = _serialize_matched_policies(decision)
        block = {
            "verdict": decision.verdict,
            "reason_code": decision.reason_code,
            "reason": decision.message,
            "matched_policy": decision.matched_policy,
            "matched_policies": _serialize_matched_policies(decision),
        }
        ev["response_decision"] = block
        ev["response_verdict"] = decision.verdict
    if gate.sanitizations:
        ev["sanitizations"] = gate.sanitizations

    enqueue(ev)


# ── The wrapped call_tool ───────────────────────────────────────────


def _make_wrapper(orig: Any) -> Any:
    """Wrap a FastMCP ``call_tool``-style coroutine with the gate."""

    @functools.wraps(orig)
    async def wrapper(self: Any, name: Any, arguments: Any = None, *args: Any, **kwargs: Any) -> Any:  # noqa: E501
        cfg = get_config_optional()
        if cfg is None or not cfg.mcp_servers_enabled:
            return await orig(self, name, arguments, *args, **kwargs)

        tool_name = str(name) if name is not None else ""
        eff_arguments = arguments
        if eff_arguments is None and "arguments" in kwargs:
            eff_arguments = kwargs.get("arguments")

        try:
            gate = await asyncio.to_thread(
                _prepare_gate, self, tool_name, eff_arguments
            )
        except Exception:  # noqa: BLE001
            # Fail open — never let governance break the server.
            return await orig(self, name, arguments, *args, **kwargs)

        if gate.blocked:
            try:
                await asyncio.to_thread(_emit_event, gate, latency_ms=0, error=False)
            except Exception:  # noqa: BLE001
                pass
            _raise_block(gate.message)

        # Forward the (possibly sanitized) arguments. We pass the
        # sanitized value positionally so it overrides both the
        # positional and kwarg forms.
        forward_args = gate.arguments
        if "arguments" in kwargs:
            kwargs = dict(kwargs)
            kwargs["arguments"] = forward_args
            started = time.monotonic()
            try:
                result = await orig(self, name, *args, **kwargs)
            except BaseException:
                latency = int((time.monotonic() - started) * 1000)
                try:
                    await asyncio.to_thread(
                        _emit_event, gate, latency_ms=latency, error=True
                    )
                except Exception:  # noqa: BLE001
                    pass
                raise
        else:
            started = time.monotonic()
            try:
                result = await orig(self, name, forward_args, *args, **kwargs)
            except BaseException:
                latency = int((time.monotonic() - started) * 1000)
                try:
                    await asyncio.to_thread(
                        _emit_event, gate, latency_ms=latency, error=True
                    )
                except Exception:  # noqa: BLE001
                    pass
                raise

        latency = int((time.monotonic() - started) * 1000)
        try:
            await asyncio.to_thread(
                _emit_event, gate, latency_ms=latency, error=False
            )
        except Exception:  # noqa: BLE001
            pass
        return result

    wrapper.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return wrapper


def _patch_class_method(cls: Any, method_name: str) -> bool:
    """Wrap ``cls.method_name`` with the gate. Idempotent."""
    orig = getattr(cls, method_name, None)
    if orig is None or not asyncio.iscoroutinefunction(orig):
        return False
    if getattr(orig, "__egisai_wrapped__", False):
        return False
    setattr(cls, method_name, _make_wrapper(orig))
    return True


def apply() -> bool:
    """Patch installed MCP server frameworks. No-op unless entitled.

    Returns ``True`` if at least one framework's tool-dispatch method
    was wrapped. Import-guarded for both ``fastmcp`` (v2) and the
    official ``mcp`` SDK's ``FastMCP``; either or both may be absent.
    """
    cfg = get_config_optional()
    if cfg is None or not cfg.mcp_servers_enabled:
        return False

    patched = False

    # FastMCP v2 (the standalone ``fastmcp`` package).
    try:
        from fastmcp import FastMCP as FastMCPv2  # type: ignore

        for method in ("_mcp_call_tool", "call_tool"):
            if _patch_class_method(FastMCPv2, method):
                patched = True
                break
    except Exception:  # noqa: BLE001
        pass

    # Official MCP SDK's FastMCP (``mcp.server.fastmcp.FastMCP``).
    try:
        from mcp.server.fastmcp import FastMCP as FastMCPofficial  # type: ignore

        for method in ("call_tool", "_mcp_call_tool"):
            if _patch_class_method(FastMCPofficial, method):
                patched = True
                break
    except Exception:  # noqa: BLE001
        pass

    return patched
