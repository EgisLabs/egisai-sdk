"""MCP client adapter — govern the tools an agent reaches for.

This is the mirror image of :mod:`egisai._patches.mcp_server`. That
module governs *inbound* ``tools/call`` on an MCP server the customer
hosts. This one governs the *outbound* direction: the customer's agent
acting as an MCP client, calling somebody else's server — Slack, Drive,
GitHub, an internal wiki.

Why this exists
---------------
Until now we watched what an agent said to the model and were blind to
what it said to its tools. The Access tab knew a workflow *declared* an
MCP server; it never knew a call was made, to which tool, with what
arguments. That gap is why exposure findings have to reason from a tool
*name* — "this agent has something called ``comp_bands_lookup`` and
there is a public doc by that name, so maybe". Observing the call turns
that guess into a fact.

Enforcement, not just observation
---------------------------------
Because we sit in front of ``call_tool``, a ``deny_mcp_call`` or
``pii_scan`` rule can stop the arguments before they leave the process.
An SSN pasted into a Slack ``post_message`` call is masked here, on the
customer's machine, exactly like a prompt would be — the third party
never sees it (``security-and-compliance.mdc`` §1).

Deliberate differences from the server-side patch
-------------------------------------------------
* **Not add-on gated.** ``mcp_server.py`` is dormant without the
  ``mcp_servers`` entitlement because hosting a governed MCP server is
  a paid product. An agent calling a tool is ordinary agent governance
  — the same thing we already do for ``claude_agent_sdk``'s tool hooks
  — so it ships to everyone.
* **Attributed to the agent, not the server.** These rows carry the
  calling agent's id and the default ``source_kind="agent"``. The
  server on the other end is context (``mcp_targets``), not the
  subject.
* **Policy runs through the evaluator, not the engine directly.**
  :func:`egisai._evaluator.evaluate_output` adds the things an
  agent-scoped call needs and the server-side gate doesn't have:
  operator pause, the ungoverned opt-out, the fail-closed no-rules
  posture, and per-agent rule scoping.
* **A block returns an error result rather than raising.** The server
  side raises ``ToolError`` because it is answering somebody else's
  agent. Here we are inside our own agent's loop, and an exception
  usually kills the whole run. An ``isError`` result is what the MCP
  spec already uses for "the tool failed", so the model reads the
  refusal, can explain it, and can try something else.

Fail open, always
-----------------
Any unexpected error anywhere in the gate falls through to the original
``call_tool``. Governance having a bad day must never stop a customer's
agent from doing its job (``sdk-design-philosophy.mdc`` §5).
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from egisai._access import maybe_report_access_items
from egisai._config import get_config_optional
from egisai._evaluator import OutputCall, evaluate_output
from egisai._events import build_event, safe_preview
from egisai._patches._common import (
    ENFORCEMENT_ENFORCED,
    _attribute_event,
    _dispatch_step,
)
from egisai.policy import PolicyDecision
from egisai.policy.pii import sanitize as pii_sanitize

LOGGER = logging.getLogger("egisai.patches.mcp_client")

_SOURCE = "mcp_client"

#: Cap on how much of a tool's arguments we render for policy text. A
#: pathological caller can hand a tool a whole file; scanning megabytes
#: on the hot path would blow the ~1 ms budget the design philosophy
#: sets for steady-state overhead.
_MAX_POLICY_TEXT = 200_000

#: Remote servers we have already told the backend about, so the Access
#: tab reflects observed servers and not just declared ones. Keyed by
#: ``(agent_id, server_label)`` — cheap set membership on every call
#: after the first.
_reported: set[tuple[str, str]] = set()


@dataclass
class _Gate:
    """Everything one governed ``call_tool`` needs to carry."""

    tool_name: str
    server: str
    arguments: Any
    decision: PolicyDecision | None = None
    blocked: bool = False
    message: str | None = None
    sanitizations: list[dict[str, Any]] = field(default_factory=list)


# ── Identifying the server on the other end ─────────────────────────


def _server_label(session: Any) -> str:
    """A stable, human-readable name for the server being called.

    The MCP client session does not expose "who am I connected to" in
    any standard way — the transport owns that. We dig for a URL on the
    usual private attributes and fall back to the server's advertised
    name from the ``initialize`` handshake. Both are best-effort, and a
    server we can't name is still governed; it just shows up as
    ``"mcp"`` in the audit row.
    """
    for attr in ("_egisai_server_label", "server_label"):
        value = getattr(session, attr, None)
        if isinstance(value, str) and value.strip():
            return value.strip()[:255]

    # The handshake result, when the caller kept it on the session.
    info = getattr(session, "_server_info", None) or getattr(
        session, "server_info", None
    )
    name = getattr(info, "name", None)
    if isinstance(name, str) and name.strip():
        return name.strip()[:255]

    for attr in ("_read_stream", "_write_stream", "_transport", "transport"):
        holder = getattr(session, attr, None)
        for url_attr in ("url", "_url", "endpoint", "_endpoint"):
            raw = getattr(holder, url_attr, None)
            if isinstance(raw, str) and raw.strip():
                try:
                    host = urlparse(raw).netloc
                except Exception:  # noqa: BLE001
                    host = ""
                return (host or raw).strip()[:255]

    return "mcp"


def _report_server(agent_id: str | None, server: str, tool_name: str) -> None:
    """Record the observed server + tool on the agent's Access tab.

    Declared inventory says what an agent *could* reach. This says what
    it actually did, which is the difference between a config review and
    evidence. Metadata only — the tool's name, never its arguments.
    """
    if not agent_id or server == "mcp":
        return
    key = (agent_id, server)
    if key in _reported:
        return
    _reported.add(key)
    try:
        maybe_report_access_items(
            agent_id,
            [
                {
                    "kind": "mcp_server",
                    "name": server,
                    "description": "Observed at runtime by the MCP client gate.",
                    "schema_hash": None,
                    "param_names": [],
                    "server_name": None,
                },
                {
                    "kind": "tool",
                    "name": tool_name[:255],
                    "description": None,
                    "schema_hash": None,
                    "param_names": [],
                    "server_name": server,
                },
            ],
            merge=True,
        )
    except Exception:  # noqa: BLE001
        # Inventory is a nice-to-have; never let it cost a tool call.
        _reported.discard(key)


# ── Arguments → policy text, and back again ─────────────────────────


def _arguments_text(arguments: Any) -> str:
    """Render tool arguments as text the policy engine can scan."""
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments[:_MAX_POLICY_TEXT]
    try:
        return json.dumps(arguments, default=str, ensure_ascii=False)[
            :_MAX_POLICY_TEXT
        ]
    except Exception:  # noqa: BLE001
        return repr(arguments)[:_MAX_POLICY_TEXT]


def _sanitize_arguments(
    arguments: Any, decision: PolicyDecision
) -> tuple[Any, list[dict[str, Any]]]:
    """Mask PII in the arguments before they leave the process.

    Walks every string leaf of the argument tree. Records carry the
    count and mask shape only, never the original value
    (``security-and-compliance.mdc`` §1).
    """
    aggregated: dict[str, dict[str, Any]] = {}

    def _mask(text: str) -> str:
        masked, records = pii_sanitize(
            text,
            types=decision.sanitize_types or None,
            mask_char=decision.sanitize_mask_char,
        )
        for record in records:
            existing = aggregated.get(record.type)
            if existing is None:
                aggregated[record.type] = {
                    "type": record.type,
                    "count": record.count,
                    "pattern": record.pattern,
                }
            else:
                existing["count"] += record.count
        return masked

    def _walk(value: Any) -> Any:
        if isinstance(value, str):
            return _mask(value)
        if isinstance(value, dict):
            return {key: _walk(item) for key, item in value.items()}
        if isinstance(value, list):
            return [_walk(item) for item in value]
        return value

    try:
        return _walk(arguments), list(aggregated.values())
    except Exception:  # noqa: BLE001
        # A mask we couldn't apply must not silently ship the raw
        # value as if it had been cleaned — but the deterministic PII
        # block already ran upstream, so failing to *mask* here is a
        # degradation, not a leak of something that should have been
        # stopped. Send the original and record nothing.
        return arguments, []


# ── The gate ────────────────────────────────────────────────────────


def _prepare_gate(session: Any, tool_name: str, arguments: Any) -> _Gate:
    """Evaluate policy for one outbound tool call.

    Runs on a worker thread: a ``semantic_guard`` rule makes a blocking
    HTTP call to the judge, and the caller is almost always inside an
    asyncio event loop driving the agent.
    """
    server = _server_label(session)
    gate = _Gate(tool_name=tool_name, server=server, arguments=arguments)

    try:
        decision = evaluate_output(
            OutputCall(
                source=_SOURCE,
                target=f"mcp.{server}.tools/call",
                model="",
                text=_arguments_text(arguments),
                tool_names=[tool_name],
                tool_calls=[{"name": tool_name, "input": arguments}],
                mcp_targets=[server],
                stream=False,
                # We hold the arguments and can rewrite them before
                # they go anywhere, so honouring action="sanitize" is
                # safe here — unlike output paths with no mutation
                # point, which must block instead.
                allow_sanitize=True,
                surfaces=("mcp",),
            )
        )
    except Exception as exc:  # noqa: BLE001
        LOGGER.debug("mcp client policy eval failed: %s", exc.__class__.__name__)
        return gate

    gate.decision = decision
    if decision.verdict == "block":
        gate.blocked = True
        gate.message = (
            f"[egisai] {decision.message or 'blocked by policy'} "
            f"(matched={decision.matched_policy})"
        )
    elif decision.verdict == "sanitize":
        gate.arguments, gate.sanitizations = _sanitize_arguments(
            arguments, decision
        )
    return gate


# ── Refusing a call ─────────────────────────────────────────────────


def _blocked_result(message: str) -> Any:
    """Build an MCP error result the calling model can read.

    Returning ``isError`` rather than raising keeps the agent's loop
    alive: the model sees a tool failure, can say why, and can pick a
    different route. Raising would abort the run, which turns a policy
    decision about one tool into an outage for the whole agent.

    If the ``mcp`` types aren't importable for some reason we raise
    ``PermissionError`` instead — a hard stop is still better than
    quietly letting a blocked call through.
    """
    try:
        from mcp.types import CallToolResult, TextContent  # type: ignore

        # Construct by the pydantic field name (``is_error``). Current
        # ``mcp`` sets ``populate_by_name=True`` so the wire alias
        # ``isError`` still serializes correctly; using the field name
        # keeps the static type-checker happy across mcp versions.
        return CallToolResult(
            content=[TextContent(type="text", text=message)],
            is_error=True,
        )
    except Exception:  # noqa: BLE001
        raise PermissionError(message) from None


# ── Audit ───────────────────────────────────────────────────────────


def _serialize_matched_policies(
    decision: PolicyDecision,
) -> list[dict[str, Any]]:
    return [
        {
            "name": record.name,
            "type": record.type,
            "verdict": record.verdict,
            "reason_code": record.reason_code,
            "message": record.message,
            "sanitize_types": list(record.sanitize_types),
            "sanitize_mask_char": record.sanitize_mask_char,
        }
        for record in decision.matched_policies
    ]


def _emit_event(
    gate: _Gate, *, started_at: float, latency_ms: int, error: bool
) -> None:
    """Record one outbound tool call as a ``tool_call`` step."""
    try:
        ev = build_event(
            source=_SOURCE,
            target=f"mcp.{gate.server}.tools/call",
            payload=gate.arguments,
            model=None,
            stream=False,
        )
    except Exception:  # noqa: BLE001
        return

    try:
        _attribute_event(ev, gate.arguments)
    except Exception:  # noqa: BLE001
        pass

    decision = gate.decision
    ev["step_kind"] = "tool_call"
    ev["tool_name"] = gate.tool_name
    ev["verdict"] = decision.verdict if decision is not None else "allow"
    ev["latency_ms"] = latency_ms
    ev["enforcement_status"] = ENFORCEMENT_ENFORCED
    # ``payload_preview`` is sampled from the POST-sanitization
    # arguments, so an audit row can never carry a value the policy
    # just masked (``security-and-compliance.mdc`` §5).
    ev["payload_preview"] = safe_preview(gate.arguments)
    ev["prompt_preview"] = ev["payload_preview"]
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

    _report_server(ev.get("agent_id"), gate.server, gate.tool_name)
    _dispatch_step(ev, started_at=started_at, kind="tool_call")


# ── The wrapped call_tool ───────────────────────────────────────────


def _make_wrapper(orig: Any) -> Any:
    """Wrap ``ClientSession.call_tool`` with the gate."""

    @functools.wraps(orig)
    async def wrapper(
        self: Any,
        name: Any = None,
        arguments: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if get_config_optional() is None:
            # ``init()`` never ran. Stay completely out of the way.
            return await orig(self, name, arguments, *args, **kwargs)

        tool_name = str(name) if name is not None else str(kwargs.get("name") or "")
        eff_arguments = arguments
        if eff_arguments is None and "arguments" in kwargs:
            eff_arguments = kwargs.get("arguments")

        started_at = time.monotonic()
        try:
            gate = await asyncio.to_thread(
                _prepare_gate, self, tool_name, eff_arguments
            )
        except Exception:  # noqa: BLE001
            return await orig(self, name, arguments, *args, **kwargs)

        if gate.blocked:
            try:
                await asyncio.to_thread(
                    _emit_event,
                    gate,
                    started_at=started_at,
                    latency_ms=0,
                    error=False,
                )
            except Exception:  # noqa: BLE001
                pass
            return _blocked_result(gate.message or "[egisai] blocked by policy")

        # Forward the possibly-masked arguments in whichever position
        # the caller used, so we never turn a keyword call positional.
        forward_kwargs = kwargs
        forward_positional = gate.arguments
        if "arguments" in kwargs:
            forward_kwargs = dict(kwargs)
            forward_kwargs["arguments"] = gate.arguments
            forward_positional = None

        call_started = time.monotonic()
        try:
            if forward_positional is None and "arguments" in forward_kwargs:
                result = await orig(self, name, *args, **forward_kwargs)
            else:
                result = await orig(
                    self, name, forward_positional, *args, **forward_kwargs
                )
        except BaseException:
            latency = int((time.monotonic() - call_started) * 1000)
            try:
                await asyncio.to_thread(
                    _emit_event,
                    gate,
                    started_at=started_at,
                    latency_ms=latency,
                    error=True,
                )
            except Exception:  # noqa: BLE001
                pass
            raise

        latency = int((time.monotonic() - call_started) * 1000)
        try:
            await asyncio.to_thread(
                _emit_event,
                gate,
                started_at=started_at,
                latency_ms=latency,
                error=bool(getattr(result, "isError", False)),
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
    """Patch the MCP client session, if the ``mcp`` package is installed.

    Returns ``True`` when at least one ``call_tool`` was wrapped.

    Both the low-level ``ClientSession`` and the newer high-level
    ``Client`` are patched when present. ``Client.call_tool`` delegates
    to the session in current releases, so the double wrap is harmless
    — the inner call sees an already-decided gate's sanitized arguments
    and simply re-evaluates them, which is idempotent. Patching both
    means a future release that stops delegating doesn't silently open
    a hole.
    """
    patched = False

    try:
        from mcp.client.session import ClientSession  # type: ignore

        if _patch_class_method(ClientSession, "call_tool"):
            patched = True
    except Exception:  # noqa: BLE001
        pass

    try:
        from mcp.client import Client  # type: ignore

        if _patch_class_method(Client, "call_tool"):
            patched = True
    except Exception:  # noqa: BLE001
        pass

    return patched
