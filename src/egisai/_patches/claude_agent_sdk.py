"""Governance patch for the Claude Agent SDK (Anthropic's agentic stack).

Why this patch is *not* a plain identity wrap
----------------------------------------------

Every other framework in our matrix ultimately bottoms out at an
HTTP client our SDK already governs (the patched ``openai``,
``anthropic``, ``google.genai``, and the ``httpx``/``requests``
fallback). The Claude Agent SDK is the exception: the Python
package is a thin client that pipes JSON over stdio into a
Node.js CLI (``claude``) which then calls Anthropic and drives
MCP tools. The LLM call never re-enters Python, so wrapping the
inner ``anthropic`` client (or installing an ``httpx`` interceptor)
captures nothing.

That left the 0.17.x patch as an identity-only wrap: agents were
registered on the dashboard, but ``request_logs`` stayed empty —
no input policies ran on the prompt, no output policies ran on
the streamed ``AssistantMessage`` / ``ToolUseBlock`` /
``ResultMessage`` chain, and the user's frustration in
``v0.17.5`` ("agents register but no requests come through") was
exactly that gap.

This module closes the gap by governing the *Python-visible*
boundary instead:

- ``ClaudeSDKClient.query(prompt)`` is where the prompt is
  observable (one round-trip into the subprocess). We run
  Phase 1 → Phase 2 input policies on the prompt here,
  build the audit event, and stash it on the client as an
  inflight handle.
- ``ClaudeSDKClient.receive_messages()`` is where every
  yielded message — ``AssistantMessage`` (with ``TextBlock`` /
  ``ToolUseBlock`` content), ``SystemMessage``, ``ResultMessage``
  — flows back. We accumulate ``text``, ``tool_names``,
  ``tool_calls``, and ``mcp_targets`` per turn. On
  ``ResultMessage`` (turn boundary) we run output policies,
  stamp tokens/cost/latency, and enqueue the audit event.
- ``ClaudeSDKClient.__aexit__`` flushes any unfinished
  inflight event so a user who never iterates the response
  still produces a request row.
- Module-level ``claude_agent_sdk.query(...)`` is the
  single-call async-generator API. We wrap it inline with
  the same Phase 1 → forward → Phase 2 pipeline.

A few subtleties that are *not* obvious:

1. ``ToolUseBlock`` arrives *after* the CLI has already invoked
   the MCP tool — the tool ran inside the Node.js subprocess
   before we saw the block. Output policies (``deny_tool_call``,
   ``deny_mcp_call``) therefore stamp a violation and stop the
   stream rather than preventing the call. This is the strongest
   guarantee Python can give from outside the subprocess; pre-
   execution gating would require us to fork the MCP transport.
2. Identity resolution is Tier-2B (hash bundle of system_prompt
   + allowed_tools + permission_mode + model + mcp_server_names).
   This is unchanged from 0.17.5.
3. The inflight event is held on the *client instance*; concurrent
   ``ClaudeSDKClient``s in the same task each carry their own
   handle. Multi-turn (``await client.query(t1); async for …;
   await client.query(t2); async for …``) is handled by clearing
   the handle on ``ResultMessage`` and re-opening on the next
   ``query()``. If a second ``query()`` lands while a previous
   inflight is still un-finalized (user forgot to iterate), we
   flush the stale one as ``error="never_consumed"`` before
   opening the new turn.

Import-guarded; fail-open.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import AsyncIterator
from contextlib import nullcontext
from contextvars import copy_context
from typing import Any

from egisai._auto_agent import (
    IdentityRecord,
    _derive_identity_from_system,
    identity_scope,
)
from egisai._config import get_config
from egisai._context import (
    get_init_latency,
    get_policy_checked,
    get_policy_usage,
    get_source,
    reset_init_latency,
    reset_policy_usage,
    reset_trace,
    set_policy_checked,
    set_source,
)
from egisai._evaluator import OutputCall, evaluate_output
from egisai._logger import enqueue
from egisai._patches import has_module
from egisai._patches._common import (
    ENFORCEMENT_ADVISORY,
    ENFORCEMENT_ENFORCED,
    _apply_sanitization,
    _build_input_event,
    _decision_block,
    _run_input_phase,
    _safe_text_preview,
    _serialize_matched_policies,
    _stamp_output_block,
)
from egisai._patches._framework import make_identity, patch_method
from egisai._run import (
    RunContext,
    _current_run,
    append_initial_model_call_step,
    append_step,
    close_run,
    current_run,
    finalize_or_append_model_call_step,
    open_run,
)
from egisai.policy import PolicyDecision
from egisai.policy.pii import sanitize as pii_sanitize

LOGGER = logging.getLogger("egisai.patches.claude_agent_sdk")

FRAMEWORK_SOURCE = "framework:claude_agent_sdk"
SOURCE_NAME = "claude_agent_sdk"
TARGET_DEFAULT = "claude_agent_sdk.client"
INFLIGHT_ATTR = "__egisai_inflight_event__"
INFLIGHT_SIGNALS_ATTR = "__egisai_inflight_signals__"
INFLIGHT_STARTED_ATTR = "__egisai_inflight_started__"
INFLIGHT_IDENTITY_ATTR = "__egisai_inflight_identity__"
# Per-turn dict mapping ``tool_use_id`` → ``"allow" | "block"``.
# Populated by our PreToolUse hook callback the moment a tool dispatch
# is gated (pre-execution); read by ``receive_messages`` so it does
# NOT re-emit a tool_call step the hook has already shipped.
INFLIGHT_HOOK_DECISIONS_ATTR = "__egisai_hook_decisions__"
# Truthy iff our PreToolUse hook was successfully injected into this
# turn's ``options.hooks`` (sets the audit row's enforcement_status
# semantics). False on older SDKs that don't expose ``hooks`` — those
# fall back to today's post-hoc advisory mode.
INFLIGHT_HOOKS_ACTIVE_ATTR = "__egisai_hooks_active__"
# Truthy iff the SDK supports PostToolUse hooks AND we successfully
# injected one this turn. Independent of ``INFLIGHT_HOOKS_ACTIVE_ATTR``
# because some SDK versions could conceivably expose PreToolUse but
# not PostToolUse (or vice versa); we feature-detect each event
# separately so the rollout is graceful.
INFLIGHT_POST_HOOKS_ACTIVE_ATTR = "__egisai_post_hooks_active__"
# Per-turn callback handles (the closures built in _wrap_client_query
# with all eager state). Placeholder dispatchers injected at
# ``connect()`` time look these up at hook-fire time and delegate to
# them. We can't inject the real callbacks directly into
# ``options.hooks`` because ``ClaudeSDKClient.connect()`` reads
# ``options.hooks`` ONCE at subprocess-init time — anything we mutate
# afterwards is ignored. By the time ``client.query()`` runs (the
# only point where we know the per-turn identity / Run / ev template),
# the CLI's hook table is already frozen. The two-step pattern
# (placeholder at connect time, real callback bound at query time)
# bridges that gap: the placeholder ID is registered with the CLI
# while the callable it actually invokes still gets refreshed every
# turn. See the docstring of ``_wrap_client_connect`` for the full
# wire-protocol explanation.
INFLIGHT_PRE_CALLBACK_ATTR = "__egisai_pre_cb__"
INFLIGHT_POST_CALLBACK_ATTR = "__egisai_post_cb__"


# ── Identity helpers (unchanged shape from 0.17.5) ──────────────────


def _bundle_from_options(options: Any) -> tuple[str, str, tuple[Any, ...]]:
    """Extract ``(display_name, system_prompt, bundle_tuple)`` from options."""
    if options is None:
        return ("Claude Agent", "", ("claude_agent_sdk",))
    sp_raw = getattr(options, "system_prompt", None) or (
        options.get("system_prompt") if isinstance(options, dict) else None
    )
    system_prompt = str(sp_raw or "").strip()
    allowed_tools = getattr(options, "allowed_tools", None) or (
        options.get("allowed_tools") if isinstance(options, dict) else None
    ) or []
    permission_mode = str(
        getattr(options, "permission_mode", "") or (
            options.get("permission_mode") if isinstance(options, dict) else ""
        ) or ""
    )
    model = str(
        getattr(options, "model", "") or (
            options.get("model") if isinstance(options, dict) else ""
        ) or ""
    )
    mcp_servers = getattr(options, "mcp_servers", None) or (
        options.get("mcp_servers") if isinstance(options, dict) else None
    ) or []
    mcp_names: list[str] = []
    if isinstance(mcp_servers, dict):
        mcp_names = sorted(str(k) for k in mcp_servers.keys())
    elif isinstance(mcp_servers, (list, tuple)):
        for s in mcp_servers:
            sn = getattr(s, "name", None) or (
                s.get("name") if isinstance(s, dict) else None
            )
            if isinstance(sn, str):
                mcp_names.append(sn)
        mcp_names.sort()

    tool_names = sorted(
        str(t) for t in (allowed_tools or []) if isinstance(t, (str, bytes))
    )

    display_name: str
    if system_prompt:
        _, display_name = _derive_identity_from_system(system_prompt)
    else:
        display_name = "Claude Agent"
    bundle = (
        "claude_agent_sdk",
        system_prompt,
        tuple(tool_names),
        permission_mode,
        model,
        tuple(mcp_names),
    )
    return display_name, system_prompt, bundle


def _options_for(self_or_first: Any, kwargs: dict[str, Any]) -> Any:
    """Resolve ``options`` from kwargs (module-level) or instance (client)."""
    opts = kwargs.get("options") if isinstance(kwargs, dict) else None
    if opts is None:
        opts = getattr(self_or_first, "options", None) or getattr(
            self_or_first, "_options", None
        )
    return opts


def _derive(self_or_first: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    """Build a Tier-2B identity from ``ClaudeAgentOptions``."""
    opts = _options_for(self_or_first, kwargs)
    if opts is None:
        return None
    display_name, _, bundle = _bundle_from_options(opts)
    return make_identity(
        source=FRAMEWORK_SOURCE,
        display_name=display_name,
        bundle=bundle,
    )


def _model_for(self_or_first: Any, kwargs: dict[str, Any]) -> str:
    """Pluck ``options.model`` (or ``""``) — used to populate audit row."""
    opts = _options_for(self_or_first, kwargs)
    if opts is None:
        return ""
    try:
        return str(getattr(opts, "model", "") or "")
    except Exception:  # noqa: BLE001
        return ""


# ── PreToolUse hook — pre-execution enforcement ─────────────────────
#
# The default ``claude_agent_sdk`` flow runs the entire agent loop
# (model + tool dispatch + MCP) inside a Node.js subprocess. Without
# any hook, the Python wrapper only sees tool-use blocks AFTER the
# CLI has already executed the tool. That's why pre-0.21 stamped
# ``enforcement_status="advisory"`` on tool-call audit rows — honest
# about the post-hoc nature of the gate.
#
# The Claude Agent SDK exposes a first-class ``PreToolUse`` hook via
# ``ClaudeAgentOptions.hooks`` that fires in our Python process
# BEFORE the subprocess dispatches the tool. The CLI sends a
# ``hook_callback`` control message over stdio; the SDK invokes our
# Python coroutine and forwards the response back. If we return
# ``permissionDecision: "deny"`` the CLI synthesizes a permission-
# denied tool-result block and never runs the tool.
#
# This module wires our policy evaluator into that hook so
# ``deny_tool_call`` / ``deny_mcp_call`` / ``semantic_guard`` on
# tool calls becomes REAL pre-execution enforcement. The audit row's
# ``enforcement_status`` flips from ``"advisory"`` to ``"enforced"``.
#
# Feature-detected. If the installed SDK version doesn't expose a
# ``hooks`` field on ``ClaudeAgentOptions`` (or fails to import
# ``HookMatcher``), we silently fall back to the legacy
# post-hoc advisory path. No regression on older SDKs.
#
# Composition. If the user already passed ``hooks={"PreToolUse":
# [HookMatcher(matcher="Bash", hooks=[their_cb])]}``, we APPEND our
# own ``HookMatcher`` with ``matcher=None`` (catch-all) to the list.
# The SDK invokes every matching hook for a given event; any one
# returning ``deny`` denies the call. We never clobber a user-set
# hook.


def _hooks_supported() -> bool:
    """True if the installed ``claude_agent_sdk`` exposes the hook API.

    We require BOTH the ``hooks`` field on ``ClaudeAgentOptions``
    AND a public ``HookMatcher`` class — both landed in the same
    upstream release (the public hook system). Feature-detecting
    both protects us from a partial-rollout SDK version that only
    has one or the other.
    """
    try:
        import claude_agent_sdk as _sdk  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return False
    if not hasattr(_sdk, "HookMatcher"):
        return False
    options_cls = getattr(_sdk, "ClaudeAgentOptions", None)
    if options_cls is None:
        return False
    # ``hasattr`` on a dataclass-style class doesn't tell us if the
    # field exists at the instance level (it returns False because
    # the default lives on instances), so check the dataclass fields
    # listing.
    try:
        from dataclasses import fields as _dc_fields

        return any(f.name == "hooks" for f in _dc_fields(options_cls))
    except Exception:  # noqa: BLE001
        # Not a dataclass? Fall back to instance-level probe.
        try:
            probe = options_cls()
            return hasattr(probe, "hooks")
        except Exception:  # noqa: BLE001
            return False


def _hook_response_allow() -> dict[str, Any]:
    """Build the SyncHookJSONOutput for an allow decision."""
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
        }
    }


def _hook_response_deny(reason: str) -> dict[str, Any]:
    """Build the SyncHookJSONOutput for a deny decision.

    The ``reason`` field is delivered to the CLI as
    ``permissionDecisionReason`` and surfaces in the model's
    next-turn context (the CLI feeds the denial reason back to
    the model so it can react / apologize / try a different
    approach). We prefix with ``[egisai]`` so a maintainer
    grepping CLI logs can attribute the denial to this SDK.
    """
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _build_pretooluse_callback(
    *,
    record: IdentityRecord | None,
    run_ctx: RunContext | None,
    ev_template: dict[str, Any],
    model: str,
    decisions: dict[str, str],
) -> Any:
    """Build a PreToolUse hook callback bound to this turn.

    The returned coroutine has the signature the SDK expects::

        async def cb(input: HookInput, tool_use_id: str | None,
                     context: HookContext) -> HookJSONOutput

    What it does (in order, every invocation):

    1. Pulls ``tool_name`` + ``tool_input`` + ``tool_use_id`` off
       ``input``. The SDK guarantees these for ``PreToolUse``.
    2. Enters ``identity_scope(record)`` so the policy evaluator's
       per-agent filtering picks the right ``agent_id`` (the SDK's
       hook callback runs on its own asyncio task — contextvars
       set on the outer ``query()`` task do NOT propagate).
    3. Enters the captured ``RunContext`` so ``append_step`` can
       attach the tool_call step to the correct run. Same reason
       as (2) — contextvars don't cross the SDK's task boundary.
    4. Runs ``evaluate_output(OutputCall(tool_names=[…],
       tool_calls=[{...}], mcp_targets=[…]))``. This is the same
       call the legacy advisory path made — but here, the verdict
       actually prevents execution.
    5. Builds a tool_call step row with the verdict and dispatches
       it via ``append_step``. Stamps
       ``enforcement_status="enforced"`` because this is a true
       pre-execution gate.
    6. Records the verdict in ``decisions[tool_use_id]`` so
       ``receive_messages`` knows the hook handled this tool —
       skip the receive-side fallback emission.
    7. Returns the appropriate ``HookJSONOutput`` dict.

    Failures are caught and treated as ``allow`` per the
    fail-open contract (a buggy policy MUST NOT brick the user's
    agent). Anything we couldn't gate cleanly falls through to
    the legacy post-hoc path.
    """
    # NB: closures capture ``ev_template``, ``record``, ``run_ctx``,
    # ``decisions``, ``model`` at hook-build time so the callback is
    # entirely self-contained. Even if the outer ``query()`` task
    # has long since returned, the hook still has everything it
    # needs.

    async def _hook_cb(
        hook_input: Any,
        tool_use_id: str | None,
        _context: Any,
    ) -> dict[str, Any]:
        # The SDK delivers ``hook_input`` as a TypedDict-shaped
        # dict for PreToolUse: keys ``hook_event_name``,
        # ``tool_name``, ``tool_input``, ``tool_use_id``,
        # ``session_id``, ``cwd``, ``permission_mode``.
        if not isinstance(hook_input, dict):
            return _hook_response_allow()
        if hook_input.get("hook_event_name") != "PreToolUse":
            return _hook_response_allow()

        tool_name = str(hook_input.get("tool_name") or "")
        tool_input = hook_input.get("tool_input")
        if not isinstance(tool_input, (dict, list, str, int, float, bool)):
            tool_input = None
        tuid = (
            tool_use_id
            or str(hook_input.get("tool_use_id") or "")
            or f"anon_{int(time.monotonic() * 1000)}"
        )

        # Derive MCP server-name target from the namespaced tool name
        # (``mcp__<server>__<tool>``) so deny_mcp_call rules can fire.
        mcp_target: list[str] = []
        if tool_name.startswith("mcp__"):
            parts = tool_name.split("__")
            if len(parts) >= 2 and parts[1]:
                mcp_target.append(parts[1])

        # ── Enter the captured scopes ─────────────────────────────
        # ``identity_scope`` is a context manager; ``_current_run``
        # is a ContextVar we set/reset by hand. Both are restored
        # on the way out via ``finally``.
        scope_cm = identity_scope(record) if record is not None else nullcontext()
        run_token = None
        if run_ctx is not None and not run_ctx.closed:
            run_token = _current_run.set(run_ctx)

        started = time.monotonic()
        try:
            with scope_cm:
                prev_pol_in, prev_pol_out = get_policy_usage()
                # Init-latency split: a PreToolUse hook can be the
                # very first egisai entry-point in a fresh process
                # (no input phase ran yet on this client), so the
                # one-shot PII NER warm-up may land here. Book it
                # under ``init_latency_ms`` instead of letting it
                # inflate the per-tool ``policy_latency_ms``.
                reset_init_latency()
                policy_started = time.monotonic()
                try:
                    # PreToolUse fires inside an async hook callback
                    # — running ``evaluate_output`` synchronously
                    # here would block the asyncio event loop on the
                    # judge HTTP round-trip. Park it on a worker so
                    # other coroutines (especially other PreToolUse
                    # hooks for sibling tools in the same turn) keep
                    # making progress.
                    decision = await asyncio.to_thread(
                        evaluate_output,
                        OutputCall(
                            source=SOURCE_NAME,
                            target=f"{TARGET_DEFAULT}.tool_call",
                            model=model,
                            text="",
                            tool_names=[tool_name],
                            tool_calls=[
                                {"name": tool_name, "input": tool_input}
                            ],
                            mcp_targets=mcp_target,
                            stream=True,
                        ),
                    )
                except Exception:  # noqa: BLE001
                    LOGGER.debug(
                        "PreToolUse hook policy eval failed; "
                        "failing open for tool=%s",
                        tool_name,
                        exc_info=True,
                    )
                    decisions[tuid] = "allow"
                    return _hook_response_allow()

                elapsed_policy_ms_raw = int(
                    (time.monotonic() - policy_started) * 1000
                )
                hook_init_ms = get_init_latency()
                elapsed_policy_ms = max(
                    0, elapsed_policy_ms_raw - hook_init_ms
                )
                cur_pol_in, cur_pol_out = get_policy_usage()

                # Build + dispatch the tool_call step row. We use
                # the same shape as the legacy ``_dispatch_tool_call_step``
                # path but stamp ``enforcement_status="enforced"``
                # because this is a true pre-execution gate.
                ev: dict[str, Any] = {
                    "event_id": __import__("uuid").uuid4().hex,
                    "trace_id": ev_template.get("trace_id"),
                    "timestamp": ev_template.get("timestamp"),
                    "app": ev_template.get("app"),
                    "env": ev_template.get("env"),
                    "org_id": ev_template.get("org_id"),
                    "agent_id": ev_template.get("agent_id"),
                    "user_id": ev_template.get("user_id"),
                    "user_role": ev_template.get("user_role"),
                    "session_id": ev_template.get("session_id"),
                    "workflow_id": ev_template.get("workflow_id"),
                    "end_user_id": ev_template.get("end_user_id"),
                    "source": SOURCE_NAME,
                    "target": f"{TARGET_DEFAULT}.tool_call",
                    "model": model,
                    "stream": True,
                    "tool_name": tool_name,
                    # Wire key MUST be ``prompt_preview`` — the backend
                    # reads the audit row's preview text from this key
                    # (``app.routers.sdk._build_request_log_row`` →
                    # ``ev.get("prompt_preview")``). Shipping under
                    # ``request_text`` silently drops the value on the
                    # floor (column name on the DB ≠ wire key). Bug
                    # fix in 0.27.1 — see CHANGELOG.
                    "prompt_preview": _safe_preview_tool_input(tool_input),
                    "verdict": "allow",
                    "enforcement_status": ENFORCEMENT_ENFORCED,
                    "policy_latency_ms": elapsed_policy_ms,
                    "policy_tokens_in": max(0, cur_pol_in - prev_pol_in),
                    "policy_tokens_out": max(0, cur_pol_out - prev_pol_out),
                    "latency_ms": int(
                        max(0, (time.monotonic() - started) * 1000)
                    ),
                }
                if hook_init_ms > 0:
                    ev["init_latency_ms"] = hook_init_ms

                ev["response_decision"] = _decision_block(decision)
                if decision.verdict == "block":
                    ev["verdict"] = "block"
                    ev["reason_code"] = decision.reason_code
                    ev["reason"] = decision.message
                    ev["matched_policy"] = decision.matched_policy
                    ev["matched_policies"] = _serialize_matched_policies(
                        decision
                    )

                try:
                    append_step(event=ev, kind="tool_call", started_at=started)
                except Exception:  # noqa: BLE001
                    LOGGER.debug(
                        "PreToolUse hook append_step failed", exc_info=True,
                    )

                decisions[tuid] = decision.verdict

                if decision.verdict == "block":
                    reason = (
                        f"[egisai] {decision.message or 'blocked by policy'} "
                        f"(matched={decision.matched_policy or 'unknown'})"
                    )
                    return _hook_response_deny(reason)

                return _hook_response_allow()
        finally:
            if run_token is not None:
                try:
                    _current_run.reset(run_token)
                except Exception:  # noqa: BLE001
                    # ContextVar may have been mutated in a way that
                    # makes reset() raise (e.g. token from another
                    # context). Best-effort cleanup.
                    pass

    return _hook_cb


def _inject_pretooluse_hook(options: Any, callback: Any) -> bool:
    """Append our PreToolUse hook into ``options.hooks``.

    Returns ``True`` on success, ``False`` if anything went wrong
    (which falls back to advisory mode for this turn). We never
    clobber existing hooks — user-supplied PreToolUse hooks remain
    in place; the SDK runs them all and any one returning ``deny``
    denies the call.

    Mutates ``options`` in place (or builds a fresh hooks dict if
    the existing value is None / missing). The mutation is local to
    this call's options instance; we never poke at the class-level
    default.
    """
    try:
        import claude_agent_sdk as _sdk  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return False

    hook_matcher_cls = getattr(_sdk, "HookMatcher", None)
    if hook_matcher_cls is None:
        return False

    try:
        new_matcher = hook_matcher_cls(matcher=None, hooks=[callback])
    except Exception:  # noqa: BLE001
        LOGGER.debug("Failed to instantiate HookMatcher", exc_info=True)
        return False

    try:
        existing = getattr(options, "hooks", None)
    except Exception:  # noqa: BLE001
        existing = None

    new_hooks: dict[str, list[Any]]
    if isinstance(existing, dict):
        # Shallow-copy so we don't mutate a dict the caller may be
        # holding elsewhere; deep-copy of HookMatcher lists is
        # unsafe (they hold live callables).
        new_hooks = {k: list(v) for k, v in existing.items()}
    else:
        new_hooks = {}

    matchers = new_hooks.get("PreToolUse")
    if not isinstance(matchers, list):
        matchers = []
    matchers.append(new_matcher)
    new_hooks["PreToolUse"] = matchers

    try:
        options.hooks = new_hooks
    except Exception:  # noqa: BLE001
        LOGGER.debug(
            "Failed to set options.hooks (frozen dataclass?)", exc_info=True,
        )
        return False
    return True


# ── PostToolUse hook — tool-result enforcement ──────────────────────
#
# The PreToolUse hook above governs the tool *input* (the call
# Claude wants to make). That catches ``deny_tool_call``,
# ``deny_mcp_call``, ``deny_bash_command``, and any policy whose
# regex matches the arguments the model wrote — destructive shell,
# wrong server, etc. It does NOT catch PII inside the tool's
# *response* (a CRM record with the customer's email, a database
# row with an SSN, a file the agent just read). Without
# PostToolUse, those bytes round-trip Claude unmasked, the model
# writes them into its next turn, and the only thing that catches
# the leak is end-of-turn output-text scanning — which is too late
# (the PII already reached the provider) and attributes the
# violation to ``model_call`` rather than the tool that produced
# it. Both make SOC 2 / GDPR / HIPAA audit narratives wrong.
#
# The SDK exposes a first-class ``PostToolUse`` hook event whose
# ``hookSpecificOutput`` carries ``updatedToolOutput`` (built-in
# tools) and ``updatedMCPToolOutput`` (MCP tools) — the canonical
# substitution surface. Returning either field replaces the tool
# result Claude is shown. This module wires our output policy
# evaluator into that hook so a ``pii_scan`` rule with
# ``action="block"`` refuses the result (Claude sees a denial
# string and recovers gracefully), and the same rule with
# ``action="sanitize"`` masks PII in-place (Claude sees the same
# record with ``"email": "########"`` instead of the real value).
#
# Why a separate callback (vs. extending PreToolUse): PreToolUse
# evaluates with ``text=""`` because the response doesn't exist
# yet. PostToolUse evaluates with ``text=<tool_response_text>``
# and intentionally omits ``tool_names`` / ``tool_calls`` /
# ``mcp_targets`` so name-based deny rules don't double-fire
# (they already gated at Pre). The two callbacks see the same
# tool from opposite sides — one for "should this call happen at
# all", one for "should this call's result reach the model" —
# and the audit trail carries one row per fired phase.


def _post_hooks_supported() -> bool:
    """True if ``claude_agent_sdk`` exposes the PostToolUse hook event.

    PostToolUse landed in the same upstream release as PreToolUse
    on the public hook system. We still feature-detect it
    independently in case a future SDK changes the matrix.
    Fail-quiet: if anything goes wrong probing, we report
    "unsupported" and the patch falls back to PreToolUse-only
    enforcement (today's pre-fix behavior on tool inputs is
    preserved; tool results stay un-gated — same as before).
    """
    if not _hooks_supported():
        return False
    # No dataclass-field probe needed: the ``hooks`` field accepts
    # any event-name → matchers mapping the CLI honors. If the SDK
    # supports PreToolUse it overwhelmingly also supports PostToolUse
    # (same control-protocol path). We keep this predicate separate
    # so a future asymmetry has a clean override point.
    return True


def _hook_response_post_replace(
    *,
    shape: str,
    new_value: Any,
) -> dict[str, Any]:
    """Build the SyncHookJSONOutput for a PostToolUse substitution.

    ``shape`` is the extractor's classification of the original
    response (``"mcp"`` for MCP tools, anything else for built-in
    tools). The SDK uses ``updatedMCPToolOutput`` for the former
    and ``updatedToolOutput`` for the latter; setting the wrong
    one is a silent no-op on Claude's side, which is why this
    helper centralizes the dispatch.
    """
    out: dict[str, Any] = {"hookEventName": "PostToolUse"}
    if shape == "mcp":
        out["updatedMCPToolOutput"] = new_value
    else:
        out["updatedToolOutput"] = new_value
    return {"hookSpecificOutput": out}


def _build_posttooluse_callback(
    *,
    record: IdentityRecord | None,
    run_ctx: RunContext | None,
    ev_template: dict[str, Any],
    model: str,
) -> Any:
    """Build a PostToolUse hook callback bound to this turn.

    Coroutine signature matches the SDK contract::

        async def cb(input: HookInput, tool_use_id: str | None,
                     context: HookContext) -> HookJSONOutput

    For each tool result:

    1. Extract the response text + shape (``mcp`` / ``string`` /
       ``json``).
    2. Re-enter the captured identity + Run context (the SDK
       runs hook callbacks on a separate asyncio task; our
       contextvars don't propagate automatically).
    3. Run ``evaluate_output(OutputCall(text=tool_result_text,
       allow_sanitize=True, ...))``. ``allow_sanitize=True`` flips
       ``pii_scan`` rules from "always block on output" to "honor
       operator's action setting" — the policy engine's
       ``OutputPolicyContext`` reads it via the same field. We
       deliberately omit ``tool_names`` / ``tool_calls`` /
       ``mcp_targets`` so name-based deny rules don't double-fire
       (PreToolUse already gated those).
    4. Verdict dispatch:

       * **allow** — return ``{}``. No replacement. Most tool
         results land here, so the cheap path is the common path.
       * **sanitize** — ``pii.sanitize()`` the text with the
         decision's ``sanitize_types`` + ``mask_char``, rebuild
         the response via ``_rewrite_tool_response``, return
         ``updatedToolOutput`` / ``updatedMCPToolOutput``. Emit
         a ``tool_call`` step row with ``verdict="sanitize"``
         and the per-type sanitization counts.
       * **block** — substitute the response with a denial
         payload, return the substitution. Emit a step row
         with ``verdict="block"``. The audit narrative carries
         ``matched_policy`` so SOC 2 queries find this row
         under "tool results refused by policy".

    Failures are caught and treated as allow (fail-open per
    ``sdk-design-philosophy.mdc`` §5). A buggy policy MUST NOT
    brick the user's agent.

    Privacy contract (``security-and-compliance.mdc`` §1, §5):
    the audit row's ``request_text`` preview is sampled from the
    POST-sanitize / POST-denial text — never the raw tool
    response. Raw PII goes out of scope as soon as the policy
    decision is computed.
    """

    async def _hook_cb(
        hook_input: Any,
        tool_use_id: str | None,
        _context: Any,
    ) -> dict[str, Any]:
        if not isinstance(hook_input, dict):
            return {}
        if hook_input.get("hook_event_name") != "PostToolUse":
            return {}

        tool_name = str(hook_input.get("tool_name") or "")
        tool_response = hook_input.get("tool_response")
        # ``tool_input`` and ``tool_use_id`` are intentionally not
        # unpacked here. PreToolUse already attributed the call by
        # name and id; PostToolUse only governs the RESPONSE text.
        # Keeping the closure narrow avoids accidentally including
        # tool args in audit fields where only the result belongs.

        extracted_text, shape = _extract_tool_response_text(tool_response)
        if not extracted_text:
            # Nothing to scan — empty result, image-only payload, etc.
            return {}

        scope_cm = identity_scope(record) if record is not None else nullcontext()
        run_token = None
        if run_ctx is not None and not run_ctx.closed:
            run_token = _current_run.set(run_ctx)

        started = time.monotonic()
        try:
            with scope_cm:
                prev_pol_in, prev_pol_out = get_policy_usage()
                # Init-latency split (see PreToolUse hook for the
                # full rationale): one-shot warm-up wait does NOT
                # count toward governance time on this row.
                reset_init_latency()
                policy_started = time.monotonic()
                try:
                    # PostToolUse hook is async — keep the event
                    # loop free during the (potentially blocking)
                    # judge round-trip. See PreToolUse for the
                    # full rationale.
                    decision = await asyncio.to_thread(
                        evaluate_output,
                        OutputCall(
                            source=SOURCE_NAME,
                            target=f"{TARGET_DEFAULT}.tool_result",
                            model=model,
                            text=extracted_text,
                            # Name-based deny rules already fired at
                            # PreToolUse; passing empty lists here
                            # prevents double-counting in the audit
                            # row's matched_policies.
                            tool_names=[],
                            tool_calls=[],
                            mcp_targets=[],
                            stream=True,
                            allow_sanitize=True,
                        ),
                    )
                except Exception:  # noqa: BLE001
                    LOGGER.debug(
                        "PostToolUse hook policy eval failed; "
                        "failing open for tool=%s",
                        tool_name,
                        exc_info=True,
                    )
                    return {}

                if decision.verdict == "allow":
                    # No step row on allow — the PreToolUse hook
                    # already emitted one for this tool, and the
                    # common path stays cheap (no extra audit churn
                    # for the 99% case where the tool result is
                    # PII-free).
                    return {}

                # Verdict is sanitize or block — emit an audit step
                # row AND substitute the response Claude sees.

                elapsed_policy_ms_raw = int(
                    (time.monotonic() - policy_started) * 1000
                )
                hook_init_ms = get_init_latency()
                elapsed_policy_ms = max(
                    0, elapsed_policy_ms_raw - hook_init_ms
                )
                cur_pol_in, cur_pol_out = get_policy_usage()

                # Build the replacement payload first so the step
                # row's request_text preview is post-sanitize (rule
                # #5: audit before persist).
                replacement_text: str
                sanitizations_audit: list[dict[str, Any]] = []
                if decision.verdict == "sanitize":
                    masked, records = pii_sanitize(
                        extracted_text,
                        types=decision.sanitize_types or None,
                        mask_char=decision.sanitize_mask_char,
                    )
                    replacement_text = masked
                    sanitizations_audit = [
                        {
                            "type": r.type,
                            "count": r.count,
                            "pattern": r.pattern,
                        }
                        for r in records
                    ]
                else:
                    # block — denial payload Claude can recover from.
                    replacement_text = _build_denial_payload(
                        decision=decision, tool_name=tool_name,
                    )

                replacement_response = _rewrite_tool_response(
                    tool_response, shape=shape, new_text=replacement_text,
                )

                ev: dict[str, Any] = {
                    "event_id": __import__("uuid").uuid4().hex,
                    "trace_id": ev_template.get("trace_id"),
                    "timestamp": ev_template.get("timestamp"),
                    "app": ev_template.get("app"),
                    "env": ev_template.get("env"),
                    "org_id": ev_template.get("org_id"),
                    "agent_id": ev_template.get("agent_id"),
                    "user_id": ev_template.get("user_id"),
                    "user_role": ev_template.get("user_role"),
                    "session_id": ev_template.get("session_id"),
                    "workflow_id": ev_template.get("workflow_id"),
                    "end_user_id": ev_template.get("end_user_id"),
                    "source": SOURCE_NAME,
                    "target": f"{TARGET_DEFAULT}.tool_result",
                    "model": model,
                    "stream": True,
                    "tool_name": tool_name,
                    # ``prompt_preview`` is the wire key the backend
                    # reads (see note on the PreToolUse path above).
                    "prompt_preview": _safe_text_preview(replacement_text),
                    "verdict": decision.verdict,
                    "enforcement_status": ENFORCEMENT_ENFORCED,
                    "policy_latency_ms": elapsed_policy_ms,
                    "policy_tokens_in": max(0, cur_pol_in - prev_pol_in),
                    "policy_tokens_out": max(0, cur_pol_out - prev_pol_out),
                    "latency_ms": int(
                        max(0, (time.monotonic() - started) * 1000)
                    ),
                    "response_decision": _decision_block(decision),
                }
                if hook_init_ms > 0:
                    ev["init_latency_ms"] = hook_init_ms
                if decision.verdict == "block":
                    ev["reason_code"] = decision.reason_code
                    ev["reason"] = decision.message
                    ev["matched_policy"] = decision.matched_policy
                    ev["matched_policies"] = _serialize_matched_policies(
                        decision
                    )
                else:  # sanitize
                    ev["matched_policy"] = decision.matched_policy
                    ev["matched_policies"] = _serialize_matched_policies(
                        decision
                    )
                    if sanitizations_audit:
                        ev["sanitizations"] = sanitizations_audit

                try:
                    append_step(event=ev, kind="tool_call", started_at=started)
                except Exception:  # noqa: BLE001
                    LOGGER.debug(
                        "PostToolUse hook append_step failed", exc_info=True,
                    )

                return _hook_response_post_replace(
                    shape=shape, new_value=replacement_response,
                )
        finally:
            if run_token is not None:
                try:
                    _current_run.reset(run_token)
                except Exception:  # noqa: BLE001
                    pass

    return _hook_cb


def _inject_posttooluse_hook(options: Any, callback: Any) -> bool:
    """Append our PostToolUse hook into ``options.hooks``.

    Symmetric to ``_inject_pretooluse_hook``: shallow-copy any
    existing dict, append our matcher to the ``PostToolUse`` slot,
    write back. User-supplied PostToolUse hooks remain in place;
    the CLI runs all of them and merges their decisions. Returns
    ``True`` on success.
    """
    try:
        import claude_agent_sdk as _sdk  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return False

    hook_matcher_cls = getattr(_sdk, "HookMatcher", None)
    if hook_matcher_cls is None:
        return False

    try:
        new_matcher = hook_matcher_cls(matcher=None, hooks=[callback])
    except Exception:  # noqa: BLE001
        LOGGER.debug("Failed to instantiate HookMatcher (post)", exc_info=True)
        return False

    try:
        existing = getattr(options, "hooks", None)
    except Exception:  # noqa: BLE001
        existing = None

    new_hooks: dict[str, list[Any]]
    if isinstance(existing, dict):
        new_hooks = {k: list(v) for k, v in existing.items()}
    else:
        new_hooks = {}

    matchers = new_hooks.get("PostToolUse")
    if not isinstance(matchers, list):
        matchers = []
    matchers.append(new_matcher)
    new_hooks["PostToolUse"] = matchers

    try:
        options.hooks = new_hooks
    except Exception:  # noqa: BLE001
        LOGGER.debug(
            "Failed to set options.hooks for PostToolUse "
            "(frozen dataclass?)", exc_info=True,
        )
        return False
    return True


# ── Deferred-resolution hook dispatchers ────────────────────────────
#
# The wire protocol issue this section solves:
#
# ``ClaudeSDKClient.connect()`` calls
# ``_convert_hooks_to_internal_format(self.options.hooks)`` exactly
# ONCE per session and ships the resulting matcher table to the
# Node.js CLI subprocess. Each callable in that table is registered
# with a stable callback ID; from then on the CLI invokes hooks by
# ID, never re-reading ``options.hooks``. Mutating
# ``options.hooks`` after ``connect()`` has run is a silent no-op.
#
# Our hook callbacks need per-TURN state (identity record, Run
# context, ev template, decisions dict). That state isn't known
# until ``client.query()`` is called — i.e. AFTER the CLI is
# initialized. If we wait until ``query()`` to inject hooks, the
# CLI has already frozen its matcher table and ignores them; the
# user observes "Allowed" rows on every tool because the post-hoc
# advisory fallback in ``receive_messages`` is the only thing
# emitting step rows.
#
# Solution: inject PLACEHOLDER dispatchers into ``options.hooks``
# at ``connect()`` time. Each placeholder closes over the client
# instance ``self``. At hook-fire time it reads the REAL per-turn
# callback from ``self.INFLIGHT_PRE_CALLBACK_ATTR`` /
# ``INFLIGHT_POST_CALLBACK_ATTR`` and delegates. The placeholder
# never changes — only the callable it points to does, every
# ``query()`` call. The CLI never knows the difference; from its
# point of view there's one stable callback ID per matcher, and it
# always returns a fresh decision.
#
# Fail-open: if no callback is stashed (e.g. CLI fires a hook
# before ``query()`` has set up the turn, or after the turn closed),
# the dispatcher returns ``{}`` — the SDK contract for "no
# decision; let the tool through". A no-op fail-open is safer than
# a deny here because the alternative (the CLI dispatching tools
# with no governance) is the bug we're fixing in the first place.


def _make_pretooluse_dispatcher(client_self: Any) -> Any:
    """Build the placeholder PreToolUse callback bound to a client.

    Returns an ``async`` callable matching the SDK hook contract:
    ``(hook_input, tool_use_id, context) -> dict``. At fire time it
    pulls the real per-turn callback off the client instance and
    delegates. If no callback is stashed (no turn in flight), it
    returns ``{}`` — no decision — so the tool runs normally.
    """

    async def _dispatch(
        hook_input: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        cb = getattr(client_self, INFLIGHT_PRE_CALLBACK_ATTR, None)
        if cb is None:
            return {}
        try:
            return await cb(hook_input, tool_use_id, context)
        except Exception:  # noqa: BLE001
            # Fail open — a buggy real-callback must NEVER brick
            # the customer's agent (sdk-design-philosophy.mdc §5).
            LOGGER.debug(
                "PreToolUse dispatcher: real callback raised; "
                "failing open",
                exc_info=True,
            )
            return {}

    return _dispatch


def _make_posttooluse_dispatcher(client_self: Any) -> Any:
    """Build the placeholder PostToolUse callback bound to a client.

    Symmetric to ``_make_pretooluse_dispatcher`` — reads the real
    per-turn callback at fire time, delegates if present, fails
    open otherwise.
    """

    async def _dispatch(
        hook_input: Any, tool_use_id: str | None, context: Any
    ) -> dict[str, Any]:
        cb = getattr(client_self, INFLIGHT_POST_CALLBACK_ATTR, None)
        if cb is None:
            return {}
        try:
            return await cb(hook_input, tool_use_id, context)
        except Exception:  # noqa: BLE001
            LOGGER.debug(
                "PostToolUse dispatcher: real callback raised; "
                "failing open",
                exc_info=True,
            )
            return {}

    return _dispatch


def _ensure_options_for(client_self: Any) -> Any:
    """Best-effort fetch of the ClaudeAgentOptions on a client.

    The SDK exposes the options as ``self.options``. We tolerate a
    legacy ``self._options`` private-name fallback in case an old
    version uses it. Returns ``None`` if neither is set — the
    caller should treat that as "give up and let the unwrapped
    SDK run".
    """
    return getattr(client_self, "options", None) or getattr(
        client_self, "_options", None
    )


# ── Response signal accumulation ────────────────────────────────────


def _new_signals() -> dict[str, list[Any]]:
    return {
        "text": [],
        "tool_names": [],
        "tool_calls": [],
        "mcp_targets": [],
    }


def _accumulate_response_signals(message: Any, signals: dict[str, list[Any]]) -> None:
    """Extract text + tool-use info from a yielded ``Message``.

    We duck-type on the class name and attribute shape because the
    Claude Agent SDK's internal types are not part of any public
    contract — testing against ``isinstance`` would require importing
    the package at module load and would couple our test stubs to
    real upstream classes.
    """
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return
    for block in content:
        block_type = type(block).__name__
        if block_type == "TextBlock":
            text = getattr(block, "text", None)
            if isinstance(text, str):
                signals["text"].append(text)
        elif block_type == "ToolUseBlock":
            name = getattr(block, "name", None)
            if isinstance(name, str) and name:
                signals["tool_names"].append(name)
                # Claude's MCP tool names are namespaced as
                # ``mcp__<server>__<tool>``.
                if name.startswith("mcp__"):
                    parts = name.split("__")
                    if len(parts) >= 2 and parts[1]:
                        signals["mcp_targets"].append(parts[1])
                inp = getattr(block, "input", None)
                signals["tool_calls"].append(
                    {
                        "name": name,
                        "input": inp
                        if isinstance(inp, (dict, list, str, int, float, bool))
                        else None,
                    }
                )


def _iter_tool_uses(message: Any) -> list[tuple[str, Any, str | None]]:
    """Return ``[(tool_name, input_obj, tool_use_id), …]`` for one ``AssistantMessage``.

    Used by the multi-step dispatcher in ``receive_messages`` to
    emit one ``tool_call`` step per ``ToolUseBlock``. We re-walk the
    message rather than caching from ``_accumulate_response_signals``
    because step emission needs to know WHICH AssistantMessage each
    tool came from (so the dashboard's waterfall stays ordered
    correctly: assistant turn → its tools → next assistant turn → its
    tools → …).

    ``tool_use_id`` is the SDK-assigned per-call identifier
    (``ToolUseBlock.id``). It's used to correlate this ToolUseBlock
    with a PreToolUse hook decision (same id flows on both
    channels) so the dispatcher knows the hook already shipped a
    step and skips the redundant fallback emit.
    """
    out: list[tuple[str, Any, str | None]] = []
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return out
    for block in content:
        if type(block).__name__ != "ToolUseBlock":
            continue
        name = getattr(block, "name", None)
        if not isinstance(name, str) or not name:
            continue
        inp = getattr(block, "input", None)
        if not isinstance(inp, (dict, list, str, int, float, bool)):
            inp = None
        tuid = getattr(block, "id", None)
        if not isinstance(tuid, str):
            tuid = None
        out.append((name, inp, tuid))
    return out


def _safe_preview_tool_input(inp: Any) -> str | None:
    """Render a tool's input as a compliance-safe preview.

    The input is operator-authored tool schema data (e.g.
    ``{"account_id": "ACC-2847193"}``); we still pass it through
    ``_safe_text_preview`` so any free-text values inside (a
    ``"reason"`` field a model wrote, etc.) are label-redacted.
    """
    if inp is None:
        return None
    try:
        import json

        rendered = json.dumps(inp, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        rendered = repr(inp)
    return _safe_text_preview(rendered)


# ── Tool-response extraction & substitution (PostToolUse) ──────────
#
# A ``PostToolUse`` hook fires AFTER the tool has executed in the
# Node subprocess but BEFORE the result is shown to Claude. The
# ``tool_response`` field carries the raw result; ``updatedToolOutput``
# (for built-in tools) and ``updatedMCPToolOutput`` (for MCP tools)
# let us substitute the result in place. This is the only point in
# the Python-visible boundary where we can both *observe* the tool's
# actual output and *rewrite* it before Claude sees it — which is
# the only honest enforcement story for SOC 2 / GDPR on
# subprocess-loop agents.
#
# The two shapes we need to handle:
#
# 1. **MCP-style**: ``{"content": [{"type": "text", "text": "..."},
#    {"type": "image", "data": "..."}, ...]}``. Used by every tool
#    registered via ``create_sdk_mcp_server`` (the in-process Python
#    MCP server pattern from the docs) and by stdio MCP servers.
# 2. **Built-in tools** (``Read``, ``Bash``, ``Grep``, …): the
#    response is whatever the tool's Python implementation returned.
#    Usually a string or dict. Treated as opaque JSON for scanning,
#    serialized back into the same shape after sanitization.
#
# The extractor is deliberately permissive: anything that decodes
# to text (or can be ``json.dumps``'d to text) gets scanned. PII
# hiding in a base64 image payload would *not* be detected here
# (out of scope; a future detector that decodes inline images can
# plug in alongside ``presidio`` without re-wiring this seam).


def _extract_tool_response_text(response: Any) -> tuple[str, str]:
    """Return ``(extracted_text, shape)`` for a tool's response.

    ``shape`` is one of:

    - ``"mcp"`` — the response is an MCP-style dict with a
      ``content`` list of ``{type, text|data}`` parts. The
      extracted text concatenates every ``type == "text"`` block;
      non-text parts (images, audio) are left untouched in the
      replacement payload.
    - ``"string"`` — the response is a raw string (e.g. a Bash
      tool's stdout). Extracted text is the string itself.
    - ``"json"`` — the response is some other Python object
      (dict / list / scalar). Extracted text is its JSON
      serialization. The replacement path re-serializes through
      ``updatedToolOutput`` so the model sees a structurally
      identical wrapper.
    - ``"none"`` — ``response`` is ``None`` / missing / not
      stringifiable. No text to scan; the caller should treat
      this as a fail-open allow.

    The extractor is intentionally tolerant: anything that
    *could* contain PII gets surfaced as a single text blob the
    policy engine can scan. Non-text MCP parts (image / audio)
    are preserved through sanitization in the replacement builder
    — we don't currently decode them, but stripping them would
    silently corrupt the tool result, which is a worse failure
    mode than missing PII inside binary payloads.
    """
    if response is None:
        return ("", "none")
    if isinstance(response, str):
        return (response, "string")
    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, list):
            chunks: list[str] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    t = part.get("text")
                    if isinstance(t, str):
                        chunks.append(t)
            return ("\n".join(chunks), "mcp")
    # Fallback: serialize whatever shape this is to JSON for scanning.
    try:
        import json

        return (json.dumps(response, default=str, ensure_ascii=False), "json")
    except Exception:  # noqa: BLE001
        return (repr(response), "json")


def _rewrite_tool_response(
    original: Any,
    *,
    shape: str,
    new_text: str,
) -> Any:
    """Return a tool_response with the text portion replaced.

    Mirrors the input shape ``original`` came in as so the model
    sees a structurally identical wrapper (just with masked or
    denied text). Non-text MCP parts (image / audio) survive
    untouched.

    Shape-specific behavior:

    - ``"mcp"`` — clone the ``content`` list. Replace the FIRST
      ``type == "text"`` part with ``new_text`` and drop every
      subsequent ``type == "text"`` part. The single replacement
      carries the redacted-or-denied payload; the model never
      sees more text than the policy approved. Image / audio
      blocks pass through.
    - ``"string"`` — return ``new_text`` directly.
    - ``"json"`` — return ``new_text`` as a plain string (the
      best we can do without re-deserializing into the original
      shape, which could re-introduce the masked PII if the
      shape was lossy).
    - ``"none"`` — return ``new_text`` directly (caller decides
      what to do with an empty original).
    """
    if shape == "mcp" and isinstance(original, dict):
        new_doc: dict[str, Any] = dict(original)
        content = original.get("content") or []
        new_content: list[dict[str, Any]] = []
        text_replaced = False
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                if not text_replaced:
                    new_content.append({"type": "text", "text": new_text})
                    text_replaced = True
                # subsequent text parts are dropped — single block
                # carries the post-sanitize payload.
                continue
            new_content.append(part)
        if not text_replaced:
            new_content.insert(0, {"type": "text", "text": new_text})
        new_doc["content"] = new_content
        return new_doc
    return new_text


def _build_denial_payload(
    *,
    decision: PolicyDecision,
    tool_name: str,
) -> str:
    """Build the text shown to Claude when a tool result is blocked.

    The model receives this string in place of the real tool
    response. Two design goals:

    1. **Tell the model the call failed** so it can recover
       (apologize, try a different approach, fall back). A blocked
       tool result that looks identical to a successful empty
       response confuses the model into a loop.
    2. **Tell auditors which policy fired** via the
       ``matched_policy`` name. The same value lands on the
       audit row, so the model's recovery context and the
       compliance trail use a consistent identifier.

    No ``[egisai]`` prefix here — that prefix is a Python-side
    log marker; what the model receives must read like a
    domain-level refusal so the recovery prompt makes sense.
    """
    name = decision.matched_policy or "policy"
    msg = decision.message or "Tool result withheld by governance policy."
    return (
        f"[Tool result for {tool_name!r} was withheld by governance "
        f"policy {name!r}: {msg}]"
    )


def _dispatch_tool_call_step(
    *,
    ev_template: dict[str, Any],
    tool_name: str,
    tool_input: Any,
    tool_use_id: str | None,
    model: str,
    started_at: float,
    hook_decisions: dict[str, str] | None,
) -> None:
    """Build + dispatch one ``tool_call`` step for a single ``ToolUseBlock``.

    Two paths through this function:

    1. **Hook-gated** — ``hook_decisions`` is non-None and contains
       ``tool_use_id``. The PreToolUse hook callback already emitted
       this step (with the post-evaluation verdict and
       ``enforcement_status="enforced"``) — return without re-emitting
       to avoid a duplicate row.
    2. **Fallback (post-hoc)** — ``hook_decisions`` is None (old SDK
       without the ``hooks`` field) OR the tool_use_id wasn't in the
       dict (our hook didn't fire — e.g. another hook in the chain
       denied first). Evaluate policies ourselves and emit the step
       with ``enforcement_status="advisory"`` because the Node
       subprocess has already executed the tool by the time we see
       the ToolUseBlock. SOC 2 auditors querying for tools that ran
       despite a policy block find these rows via
       ``(step_kind='tool_call', verdict='block',
       enforcement_status='advisory')``.

    Per-tool output policy evaluation in path (2) so per-tool
    ``deny_tool_call`` / ``deny_mcp_call`` rules can fire with a
    matching ``matched_policy`` on the step row (the operator can
    then see in the dashboard exactly which tool tripped the rule).
    """
    # Path 1 — the hook already shipped this step row. Skip.
    if (
        hook_decisions is not None
        and tool_use_id is not None
        and tool_use_id in hook_decisions
    ):
        return

    # Path 2 — post-hoc advisory (old SDK or hook didn't fire).
    mcp_target: list[str] = []
    if tool_name.startswith("mcp__"):
        parts = tool_name.split("__")
        if len(parts) >= 2 and parts[1]:
            mcp_target.append(parts[1])

    # Copy identity + run linkage off the template; build a fresh
    # event so per-step fields (verdict, matched_policy, …) don't
    # leak across siblings.
    ev: dict[str, Any] = {
        "event_id": __import__("uuid").uuid4().hex,
        "trace_id": ev_template.get("trace_id"),
        "timestamp": ev_template.get("timestamp"),
        "app": ev_template.get("app"),
        "env": ev_template.get("env"),
        "org_id": ev_template.get("org_id"),
        "agent_id": ev_template.get("agent_id"),
        "user_id": ev_template.get("user_id"),
        "user_role": ev_template.get("user_role"),
        "session_id": ev_template.get("session_id"),
        "workflow_id": ev_template.get("workflow_id"),
        "end_user_id": ev_template.get("end_user_id"),
        "source": SOURCE_NAME,
        "target": f"{TARGET_DEFAULT}.tool_call",
        "model": model,
        "stream": True,
        "tool_name": tool_name,
        # ``prompt_preview`` is the wire key the backend reads
        # (see note on the hook-gated path above).
        "prompt_preview": _safe_preview_tool_input(tool_input),
        "verdict": "allow",
        "enforcement_status": ENFORCEMENT_ADVISORY,
    }

    prev_pol_in, prev_pol_out = get_policy_usage()
    # Init-latency split: same accounting shape as the hook-active
    # paths so legacy advisory rows also show governance time
    # without the spaCy/Presidio cold-start cost folded in.
    reset_init_latency()
    policy_started = time.monotonic()
    try:
        decision = evaluate_output(
            OutputCall(
                source=SOURCE_NAME,
                target=f"{TARGET_DEFAULT}.tool_call",
                model=model,
                text="",
                tool_names=[tool_name],
                tool_calls=[{"name": tool_name, "input": tool_input}],
                mcp_targets=mcp_target,
                stream=True,
            )
        )
    except Exception:  # noqa: BLE001
        LOGGER.debug(
            "claude_agent_sdk per-tool output evaluator failed",
            exc_info=True,
        )
        decision = None

    elapsed_ms_raw = int((time.monotonic() - policy_started) * 1000)
    init_ms = get_init_latency()
    cur_pol_in, cur_pol_out = get_policy_usage()
    ev["policy_latency_ms"] = max(0, elapsed_ms_raw - init_ms)
    if init_ms > 0:
        ev["init_latency_ms"] = init_ms
    ev["policy_tokens_in"] = max(0, cur_pol_in - prev_pol_in)
    ev["policy_tokens_out"] = max(0, cur_pol_out - prev_pol_out)
    ev["latency_ms"] = int(max(0, (time.monotonic() - started_at) * 1000))

    if decision is not None:
        ev["response_decision"] = _decision_block(decision)
        if decision.verdict == "block":
            ev["verdict"] = "block"
            ev["reason_code"] = decision.reason_code
            ev["reason"] = decision.message
            ev["matched_policy"] = decision.matched_policy
            ev["matched_policies"] = _serialize_matched_policies(decision)

    try:
        append_step(event=ev, kind="tool_call", started_at=started_at)
    except Exception:  # noqa: BLE001
        LOGGER.debug("tool_call step emit failed", exc_info=True)


def _is_result_message(message: Any) -> bool:
    return type(message).__name__ == "ResultMessage"


def _stamp_usage_from_result(ev: dict[str, Any], result: Any) -> None:
    """Pull tokens + cost off a ``ResultMessage`` onto the audit row."""
    cost = getattr(result, "total_cost_usd", None)
    if cost is not None:
        try:
            ev["cost_usd"] = float(cost)
        except (TypeError, ValueError):
            pass
    inner = getattr(result, "usage", None)
    if isinstance(inner, dict):
        for k_in, k_out in (
            ("input_tokens", "tokens_in"),
            ("output_tokens", "tokens_out"),
        ):
            val = inner.get(k_in)
            if val is not None:
                try:
                    ev[k_out] = int(val)
                except (TypeError, ValueError):
                    pass


def _run_output_phase(
    *,
    ev: dict[str, Any],
    signals: dict[str, list[Any]],
    model: str,
    stream: bool,
    hooks_active: bool = False,
) -> PolicyDecision | None:
    """Evaluate Phase 1+2 output policies on the accumulated stream.

    Privacy contract — the model's text output is **never persisted**.
    We accumulate it here only long enough to feed
    ``deny_output_regex`` / ``semantic_guard`` / etc., then it goes
    out of scope. The audit event never carries ``response_preview``.
    See ``_common._run_output_phase``'s docstring for the full
    rationale; the same contract holds for every framework patch.

    Stamps:

    * ``policy_latency_ms`` — ADDITIVE on top of whatever the input
      phase already booked, so the row reflects total policy
      wall-clock time across both phases. Same shape for
      ``policy_tokens_in`` / ``policy_tokens_out``.
    * ``response_decision`` — per-phase block summary (mirrors the
      synchronous patches via ``_run_output_phase`` in ``_common``).
    * On block: ``verdict``, ``reason``, ``matched_policy``,
      ``matched_policies`` (re-stamped via ``_stamp_output_block``).
      The ``enforcement_status`` distinguishes **truthful MCP timing**
      from **effective withhold at the SDK boundary**:

      - ``hooks_active=False`` → always ``advisory`` (observe-only /
        subprocess already ran).
      - ``hooks_active=True`` **without** accumulated ``tool_calls``
        (pure assistant ``TextBlock`` path) → ``enforced`` on block:
        withhold matches synchronously patch semantics.
      - ``hooks_active=True`` **with** accumulated ``tool_calls`` →
        ``advisory`` on output block: payloads here reflect the MCP
        turn that already flowed through CLI; auditors correlate tool
        rows + optional PreToolUse ``enforced`` blocks.
    """
    text = "".join(t for t in signals.get("text", []) if isinstance(t, str))
    tool_names = list(signals.get("tool_names", []))
    tool_calls = list(signals.get("tool_calls", []))
    mcp_targets = list(signals.get("mcp_targets", []))

    # When PreToolUse hooks are wired we already evaluated every
    # ``tool_call`` / ``tool_name`` / ``mcp_target`` individually as
    # the CLI requested them — emitting a per-tool ``tool_call`` step
    # row apiece. Re-running those signals through ``evaluate_output``
    # here would issue a second ``semantic_guard`` judge round-trip
    # for every tool the model invoked (the engine's tool_calls loop
    # in ``_semantic_guard_match`` is sequential, so an N-tool turn
    # paid 2N round-trips before this fix). The doubling shows up on
    # the dashboard as inflated ``policy_latency_ms`` AND
    # ``policy_tokens_*`` aggregated across the Run's steps.
    #
    # The fix: when hooks were active, only feed the *text* signal to
    # this phase — the assistant's accumulated ``TextBlock`` content
    # was NOT gated by PreToolUse, so text-only rules
    # (``deny_output_regex`` / ``semantic_guard.targets=["text"]`` /
    # ``pii_scan`` on output) still need to fire here. Tool/MCP
    # signals are dropped because the PreToolUse hook already
    # emitted authoritative per-tool decisions.
    #
    # Hooks-off path is unchanged: this phase is the only place
    # those signals get evaluated, so we MUST keep them in the
    # OutputCall for advisory-mode framework users.
    saw_tool_signals_pre_filter = bool(tool_names or tool_calls or mcp_targets)
    if hooks_active:
        tool_names = []
        tool_calls = []
        mcp_targets = []

    if not (text or tool_names or tool_calls or mcp_targets):
        return None

    # ``text`` is intentionally NOT stamped onto ``ev`` — see the
    # privacy contract in the docstring.

    prev_pol_in, prev_pol_out = get_policy_usage()
    # Init-latency split: same accounting as ``_common._run_output_phase`` —
    # one-shot library cold-start (PII NER load) goes on
    # ``init_latency_ms`` instead of ``policy_latency_ms``.
    reset_init_latency()
    policy_started = time.monotonic()

    try:
        decision = evaluate_output(
            OutputCall(
                source=SOURCE_NAME,
                target=TARGET_DEFAULT,
                model=model,
                text=text,
                tool_names=tool_names,
                tool_calls=tool_calls,
                mcp_targets=mcp_targets,
                stream=stream,
            )
        )
    except Exception:  # noqa: BLE001
        LOGGER.debug(
            "claude_agent_sdk output evaluator failed", exc_info=True,
        )
        return None

    elapsed_ms = int((time.monotonic() - policy_started) * 1000)
    init_ms = get_init_latency()
    cur_pol_in, cur_pol_out = get_policy_usage()
    ev["policy_latency_ms"] = int(ev.get("policy_latency_ms") or 0) + max(
        0, elapsed_ms - init_ms
    )
    if init_ms > 0:
        ev["init_latency_ms"] = int(ev.get("init_latency_ms") or 0) + init_ms
    ev["policy_tokens_in"] = int(ev.get("policy_tokens_in") or 0) + max(
        0, cur_pol_in - prev_pol_in
    )
    ev["policy_tokens_out"] = int(ev.get("policy_tokens_out") or 0) + max(
        0, cur_pol_out - prev_pol_out
    )

    # Enforcement honesty for auditors (SOC 2 / ISO):
    # Output evaluation runs here at ``ResultMessage`` — after every
    # ``ToolUseBlock`` visible to Python means the MCP invocation was
    # already discharged inside the Claude CLI subprocess unless a
    # PreToolUse hook intercepted it beforehand. Rows that aggregated
    # structured tool payloads must therefore stamp ``advisory`` on an
    # output-phase *block* even when hooks are active: callers still
    # see ``PermissionError`` (effective withhold at the SDK boundary),
    # but claiming full "enforced-before-execution" for the MCP leg
    # would contradict the ingest timeline. Pure text-only violations
    # (regex / semantic_guard on concatenated assistant text alone)
    # keep ``enforced`` when hooks are wired.
    #
    # Per-tool Hook blocks continue to stamp ``block`` /
    # ``enforced`` on the individual ``tool_call`` rows —
    # independent signal.
    if decision.verdict == "block":
        # ``saw_tool_signals_pre_filter`` reads the unfiltered signal
        # set so the enforced-vs-advisory decision still reflects
        # **original CLI activity**. After the dedupe fix above we
        # zero out tool_calls/tool_names/mcp_targets locally when
        # hooks are active — so a naive ``len(tool_calls)`` post-
        # filter check would always be False and incorrectly stamp
        # ``enforced`` on a turn whose tools already ran in the CLI
        # subprocess. The pre-filter snapshot preserves the original
        # auditor-facing semantics:
        #   hooks_active + no tools          → ``enforced``  (text-only block)
        #   hooks_active + tools             → ``advisory``  (tools already ran)
        #   hooks_active=False (any signals) → ``advisory``  (subprocess ran)
        if (
            hooks_active and not saw_tool_signals_pre_filter
        ):
            block_status = ENFORCEMENT_ENFORCED
        else:
            block_status = ENFORCEMENT_ADVISORY
        _stamp_output_block(
            ev, decision, enforcement_status=block_status,
        )
    else:
        ev["response_decision"] = _decision_block(decision)
        ev.setdefault("enforcement_status", ENFORCEMENT_ENFORCED)
    return decision


def _safe_enqueue(ev: dict[str, Any] | None) -> None:
    """Dispatch the audit event — as a step under the current Run when
    one is open, or as a legacy single-row event otherwise.

    The run-based path is normal for 0.18+: query() opens a Run,
    receive_messages finalizes the event, and that final event lands
    here as a model_call step on the open Run. The legacy path is
    only hit when an event flush happens *after* the Run has already
    been closed (race against ``__aexit__`` finishing first) — rare
    but possible during teardown.
    """
    if ev is None:
        return
    try:
        if finalize_or_append_model_call_step(event=ev) is None:
            enqueue(ev)
    except Exception:  # noqa: BLE001
        LOGGER.debug("enqueue failed for claude_agent_sdk event", exc_info=True)


def _flush_stale_inflight(self_obj: Any) -> None:
    """Enqueue a previous inflight event that was never finalized.

    A user pattern like::

        await client.query(t1)
        await client.query(t2)   # never iterated receive_response()
        async for msg in client.receive_response():
            ...

    leaves the first turn's event open forever. We flush it as
    ``error='never_consumed'`` so the dashboard still reflects
    that ``t1`` was sent, just with no response side. Better an
    incomplete row than a silent drop.

    Also closes any Run that was opened by the previous ``query()``
    so a fresh Run can be opened cleanly.
    """
    ev = getattr(self_obj, INFLIGHT_ATTR, None)
    if ev is None:
        # Even when there's no inflight, a Run may still be open if
        # the previous turn's receive_messages never reached
        # ResultMessage. Close it so the next query() opens a fresh
        # one. ``close_run`` is idempotent.
        if current_run() is not None:
            close_run(error="never_consumed")
        return
    started = getattr(self_obj, INFLIGHT_STARTED_ATTR, None) or time.monotonic()
    ev["latency_ms"] = int(max(0, (time.monotonic() - started) * 1000))
    ev["error"] = "never_consumed"
    _safe_enqueue(ev)
    _clear_inflight(self_obj)
    if current_run() is not None:
        close_run(error="never_consumed")


def _clear_inflight(self_obj: Any) -> None:
    for attr in (
        INFLIGHT_ATTR,
        INFLIGHT_SIGNALS_ATTR,
        INFLIGHT_STARTED_ATTR,
        INFLIGHT_IDENTITY_ATTR,
        INFLIGHT_HOOK_DECISIONS_ATTR,
        INFLIGHT_HOOKS_ACTIVE_ATTR,
        INFLIGHT_POST_HOOKS_ACTIVE_ATTR,
        INFLIGHT_PRE_CALLBACK_ATTR,
        INFLIGHT_POST_CALLBACK_ATTR,
    ):
        try:
            if hasattr(self_obj, attr):
                delattr(self_obj, attr)
        except (AttributeError, TypeError):
            pass


# ── Method wrappers ─────────────────────────────────────────────────


def _wrap_client_connect(orig: Any) -> Any:
    """Inject placeholder hooks BEFORE the CLI sees ``options.hooks``.

    Why this exists — and why it's the SOC 2 / ISO 27001 fix:

    The upstream SDK builds its internal hook matcher table once
    inside ``ClaudeSDKClient.connect()`` (and ``__aenter__`` calls
    ``connect()`` for you). That table is shipped to the Node CLI
    as part of the ``initialize`` control message; each callback
    is given a stable ID and from then on the CLI invokes hooks
    by ID. Mutating ``options.hooks`` AFTER ``connect()`` returns
    is a silent no-op — the CLI never re-reads it.

    Before 0.22.1 the patch injected its PreToolUse / PostToolUse
    hooks inside ``client.query()``. That code path runs AFTER
    the user's ``async with ClaudeSDKClient(...) as client:`` has
    already triggered ``__aenter__`` → ``connect()`` → empty
    hook table sent to CLI. The CLI therefore dispatched every
    tool with no governance round-trip. The "Allowed" tool rows
    customers saw were the legacy receive-side post-hoc fallback
    in ``_dispatch_tool_call_step``, not real hook decisions.
    Worse, tool RESULTS were never evaluated because the
    PostToolUse hook never fired — meaning PII in CRM lookups,
    file reads, etc. round-tripped Claude unmasked. That is the
    bug this wrapper fixes.

    Wrap ``connect`` (not ``__aenter__``) because the user might
    call ``await client.connect()`` directly without using the
    context manager — wrapping ``connect`` covers both paths.

    The injection is idempotent at the level we care about: the
    SDK CLI sees one matcher per event (``PreToolUse``,
    ``PostToolUse``), each pointing at our placeholder dispatcher.
    Subsequent ``connect()`` calls on the same client (rare; used
    only for reconnection scenarios) would re-inject — that's
    harmless because the placeholder dispatchers are reentrant
    and re-running them per-turn is the design.
    """

    @functools.wraps(orig)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        if _hooks_supported():
            opts = _ensure_options_for(self)
            if opts is not None:
                try:
                    _inject_pretooluse_hook(
                        opts, _make_pretooluse_dispatcher(self)
                    )
                except Exception:  # noqa: BLE001
                    LOGGER.debug(
                        "PreToolUse placeholder injection failed at "
                        "connect()", exc_info=True,
                    )
                if _post_hooks_supported():
                    try:
                        _inject_posttooluse_hook(
                            opts, _make_posttooluse_dispatcher(self)
                        )
                    except Exception:  # noqa: BLE001
                        LOGGER.debug(
                            "PostToolUse placeholder injection failed "
                            "at connect()", exc_info=True,
                        )
        return await orig(self, *args, **kwargs)

    wrapped.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


def _wrap_client_query(orig: Any) -> Any:
    """``ClaudeSDKClient.query`` — Phase 1+2 input gate + stash inflight."""

    @functools.wraps(orig)
    async def wrapped(self: Any, prompt: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            record = _derive(self, *args, **kwargs)
        except Exception:  # noqa: BLE001
            record = None

        model = _model_for(self, kwargs) or "claude"
        prompt_text = prompt if isinstance(prompt, str) else ""
        # Use ``input`` (not ``prompt``) as the payload key so the
        # shared ``mutate_prompt_text`` helper picks it up when a
        # sanitize verdict fires (see ``_evaluator.mutate_prompt_text``
        # — it walks ``messages`` / ``input`` / ``contents`` only).
        payload: dict[str, Any] = {
            "input": prompt_text,
            "session_id": kwargs.get("session_id", "default"),
        }

        prev_source = get_source()
        prev_checked = get_policy_checked()
        if not prev_source:
            reset_trace()
        set_source(SOURCE_NAME)

        try:
            if prev_checked:
                # Nested inside another gate; let the outer one own
                # event emission and policy enforcement.
                return await orig(self, prompt, *args, **kwargs)

            # Flush any previous un-finalized inflight on this client
            # before opening a fresh turn.
            _flush_stale_inflight(self)

            scope_cm = identity_scope(record) if record is not None else nullcontext()
            with scope_cm:
                # Open the Run upfront — even input-side blocks ship
                # as a complete (failed) Run so the dashboard never
                # leaves a turn invisible. ``close_run`` is called on
                # every exit path below.
                opened_run_here = current_run() is None
                if opened_run_here:
                    # Compliance rule #5 (audit before persist): the
                    # raw prompt has NOT been sanitized yet — it could
                    # contain PII the input policy will redact in the
                    # next phase. We open the Run with prompt_text=None
                    # so the streaming ``run.start`` event ships no
                    # preview. The backend pulls prompt_text from the
                    # FIRST step's post-sanitize ``prompt_preview``,
                    # which is the canonical post-redaction snapshot.
                    open_run(
                        framework="claude_agent_sdk",
                        identity=record,
                        prompt_text=None,
                    )

                ev = _build_input_event(
                    source=SOURCE_NAME,
                    target=TARGET_DEFAULT,
                    model=model,
                    prompt_text=prompt_text,
                    stream=True,
                    payload=payload,
                )

                set_policy_checked(True)
                reset_policy_usage()
                try:
                    decision = _run_input_phase(
                        source=SOURCE_NAME,
                        target=TARGET_DEFAULT,
                        model=model,
                        prompt_text=prompt_text,
                        stream=True,
                        ev=ev,
                    )

                    if decision.verdict == "block":
                        ev["latency_ms"] = 0
                        # Input-side block: we raise BEFORE forwarding
                        # the prompt to the Node subprocess, so the
                        # call genuinely doesn't happen. This is real
                        # enforcement — record it as such.
                        ev["enforcement_status"] = ENFORCEMENT_ENFORCED
                        _safe_enqueue(ev)
                        if opened_run_here and current_run() is not None:
                            close_run(error="input policy block")
                        msg = (
                            f"[egisai] {decision.message or 'blocked by policy'} "
                            f"(matched={decision.matched_policy})"
                        )
                        # claude_agent_sdk doesn't ship a stub
                        # response shape — input-side block always
                        # raises regardless of on_block setting (the
                        # subprocess would otherwise still receive
                        # the original prompt).
                        raise PermissionError(msg)

                    if decision.verdict == "sanitize":
                        _apply_sanitization(
                            decision=decision, payload=payload, ev=ev
                        )
                        # The shared sanitizer mutated ``payload["input"]``;
                        # update the local ``prompt`` we'll forward to
                        # the subprocess so the masked copy goes over
                        # stdio instead of the raw original.
                        prompt = payload["input"]

                    # Stash inflight handle for receive_messages to
                    # extend. Identity record is stashed so the
                    # response side re-enters the same scope (output
                    # policy filtering reads the active agent_id).
                    setattr(self, INFLIGHT_ATTR, ev)
                    setattr(self, INFLIGHT_SIGNALS_ATTR, _new_signals())
                    setattr(self, INFLIGHT_STARTED_ATTR, time.monotonic())
                    setattr(self, INFLIGHT_IDENTITY_ATTR, record)

                    # ── Per-turn hook callback binding ───────────
                    # The actual ``HookMatcher`` instances landed on
                    # ``options.hooks`` at ``connect()`` time (see
                    # ``_wrap_client_connect``). What we do HERE is
                    # build the eagerly-bound per-turn callbacks
                    # (with this turn's identity record, Run, ev
                    # template, decisions dict) and stash them on
                    # ``self`` so the placeholder dispatchers pick
                    # them up at hook-fire time.
                    #
                    # Why not inject hooks here directly? The CLI's
                    # matcher table is frozen at ``connect()`` and
                    # never re-read; any ``options.hooks`` mutation
                    # past that point is a silent no-op. That was
                    # the 0.22.0 bug — hooks were injected here but
                    # never reached the CLI. See
                    # ``_wrap_client_connect`` for the wire-protocol
                    # explanation.
                    hook_decisions: dict[str, str] = {}
                    pre_present = (
                        _hooks_supported()
                        and getattr(self, "options", None) is not None
                    )
                    post_present = (
                        _post_hooks_supported()
                        and getattr(self, "options", None) is not None
                    )
                    if pre_present:
                        pre_callback = _build_pretooluse_callback(
                            record=record,
                            run_ctx=current_run(),
                            ev_template=ev,
                            model=model,
                            decisions=hook_decisions,
                        )
                        setattr(
                            self, INFLIGHT_PRE_CALLBACK_ATTR, pre_callback
                        )
                    if post_present:
                        post_callback = _build_posttooluse_callback(
                            record=record,
                            run_ctx=current_run(),
                            ev_template=ev,
                            model=model,
                        )
                        setattr(
                            self, INFLIGHT_POST_CALLBACK_ATTR, post_callback
                        )
                    setattr(self, INFLIGHT_HOOK_DECISIONS_ATTR, hook_decisions)
                    setattr(self, INFLIGHT_HOOKS_ACTIVE_ATTR, pre_present)
                    setattr(
                        self,
                        INFLIGHT_POST_HOOKS_ACTIVE_ATTR,
                        post_present,
                    )

                    # Provisional seq-0 row: include the same defaults the
                    # terminal event will carry so early wire payloads are
                    # not missing ``enforcement_status``.
                    ev.setdefault("enforcement_status", ENFORCEMENT_ENFORCED)

                    try:
                        append_initial_model_call_step(
                            event=ev,
                            started_at=float(
                                getattr(
                                    self,
                                    INFLIGHT_STARTED_ATTR,
                                    time.monotonic(),
                                )
                            ),
                        )
                    except Exception:  # noqa: BLE001
                        LOGGER.debug(
                            "claude_agent_sdk provisional model step failed",
                            exc_info=True,
                        )

                    return await orig(self, prompt, *args, **kwargs)
                finally:
                    set_policy_checked(prev_checked)
        finally:
            set_source(prev_source)

    wrapped.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


def _wrap_client_receive_messages(orig: Any) -> Any:
    """``ClaudeSDKClient.receive_messages`` — accumulate + finalize."""

    @functools.wraps(orig)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        record = getattr(self, INFLIGHT_IDENTITY_ATTR, None)
        if record is None:
            try:
                record = _derive(self, *args, **kwargs)
            except Exception:  # noqa: BLE001
                record = None

        model = _model_for(self, kwargs) or "claude"
        scope_cm = identity_scope(record) if record is not None else nullcontext()

        ctx = copy_context()
        async_gen = orig(self, *args, **kwargs)

        try:
            with scope_cm:
                while True:
                    try:
                        message = await ctx.run(async_gen.__anext__)
                    except StopAsyncIteration:
                        break

                    ev = getattr(self, INFLIGHT_ATTR, None)
                    signals = getattr(self, INFLIGHT_SIGNALS_ATTR, None)
                    if signals is None:
                        signals = _new_signals()
                        setattr(self, INFLIGHT_SIGNALS_ATTR, signals)

                    _accumulate_response_signals(message, signals)

                    # Emit one ``tool_call`` step per ``ToolUseBlock`` only
                    # when PreToolUse hooks are *off* (older SDK or no
                    # ``options``): in that case Python sees the block
                    # before any hook row exists and must evaluate the
                    # legacy post-hoc advisory path.
                    #
                    # When hooks are on, ``PreToolUse`` runs before the
                    # tool executes and already emitted the step — calling
                    # ``_dispatch_tool_call_step`` here as well duplicated
                    # “Allowed” rows because ``AssistantMessage`` is
                    # processed before ``hook_decisions`` is populated.
                    hooks_active = bool(
                        getattr(self, INFLIGHT_HOOKS_ACTIVE_ATTR, False),
                    )
                    if (
                        ev is not None
                        and type(message).__name__ == "AssistantMessage"
                        and current_run() is not None
                        and not hooks_active
                    ):
                        hook_decisions = getattr(
                            self, INFLIGHT_HOOK_DECISIONS_ATTR, None
                        )
                        tool_uses = _iter_tool_uses(message)
                        for tname, tinp, tuid in tool_uses:
                            _dispatch_tool_call_step(
                                ev_template=ev,
                                tool_name=tname,
                                tool_input=tinp,
                                tool_use_id=tuid,
                                model=model,
                                started_at=time.monotonic(),
                                hook_decisions=hook_decisions,
                            )

                    if _is_result_message(message):
                        if ev is not None:
                            started = getattr(
                                self, INFLIGHT_STARTED_ATTR, time.monotonic()
                            )
                            ev["latency_ms"] = int(
                                max(0, (time.monotonic() - started) * 1000)
                            )
                            _stamp_usage_from_result(ev, message)

                            # ``_run_output_phase`` calls
                            # ``evaluate_output`` which can issue a
                            # *blocking* judge HTTP round-trip for
                            # ``semantic_guard`` rules. Inside an
                            # async receive loop that would freeze
                            # every other coroutine on this event
                            # loop until the judge responds. Park
                            # the whole synchronous phase on a
                            # worker thread instead so concurrent
                            # client work (other inflight queries,
                            # streaming consumers) stays responsive.
                            decision = await asyncio.to_thread(
                                _run_output_phase,
                                ev=ev,
                                signals=signals,
                                model=model,
                                stream=True,
                                hooks_active=bool(
                                    getattr(
                                        self,
                                        INFLIGHT_HOOKS_ACTIVE_ATTR,
                                        False,
                                    )
                                ),
                            )
                            # If no output policy fired (or the
                            # extractor found nothing to evaluate),
                            # the ``enforcement_status`` field will
                            # still be unset. Default to ``enforced``
                            # — there was nothing to enforce against,
                            # so the SDK trivially didn't fail to
                            # enforce. The output-block path inside
                            # ``_run_output_phase`` overrides this to
                            # ``advisory`` when (1) a policy decided
                            # block AND (2) hooks weren't active.
                            ev.setdefault(
                                "enforcement_status", ENFORCEMENT_ENFORCED,
                            )
                            # Append as a step on the Run that
                            # query() opened. After the step lands,
                            # close the Run so the dashboard sees
                            # ONE complete run row for this turn.
                            _safe_enqueue(ev)
                            _clear_inflight(self)
                            if current_run() is not None:
                                close_run()
                            # Reset for the next turn — same
                            # ``receive_messages`` call may span
                            # multiple ``query()`` turns.
                            setattr(self, INFLIGHT_SIGNALS_ATTR, _new_signals())

                            # Raise BEFORE yielding the ResultMessage
                            # so a user loop that breaks on
                            # ``ResultMessage`` (the canonical
                            # ``receive_response`` body) still sees
                            # the block. Yielding first would let
                            # ``return`` close our generator before
                            # we got to raise.
                            if (
                                decision is not None
                                and decision.verdict == "block"
                            ):
                                cfg = get_config()
                                msg = (
                                    "[egisai] "
                                    f"{decision.message or 'blocked by policy'} "
                                    f"(matched={decision.matched_policy})"
                                )
                                if cfg.on_block == "raise":
                                    raise PermissionError(msg)

                            yield message
                            continue
                        # No inflight (user iterated without query)
                        # — pass the message through unchanged.

                    yield message
        except BaseException:
            ev = getattr(self, INFLIGHT_ATTR, None)
            if ev is not None:
                started = getattr(self, INFLIGHT_STARTED_ATTR, time.monotonic())
                ev["latency_ms"] = int(
                    max(0, (time.monotonic() - started) * 1000)
                )
                if "error" not in ev:
                    ev["error"] = "stream failed"
                _safe_enqueue(ev)
                _clear_inflight(self)
            if current_run() is not None:
                close_run(error="stream failed")
            raise

    wrapped.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


def _wrap_client_aexit(orig: Any) -> Any:
    """Flush any still-open inflight when the client context closes."""

    @functools.wraps(orig)
    async def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await orig(self, *args, **kwargs)
        finally:
            _flush_stale_inflight(self)

    wrapped.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


def _wrap_module_query(orig: Any) -> Any:
    """Module-level ``claude_agent_sdk.query`` — single-call streaming gate."""

    @functools.wraps(orig)
    async def wrapped(prompt: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        try:
            record = _derive(None, prompt, *args, **kwargs)
        except Exception:  # noqa: BLE001
            record = None

        options = kwargs.get("options")
        model = ""
        if options is not None:
            try:
                model = str(getattr(options, "model", "") or "")
            except Exception:  # noqa: BLE001
                model = ""
        model = model or "claude"

        prompt_text = prompt if isinstance(prompt, str) else ""
        payload: dict[str, Any] = {"input": prompt_text, "options": options}

        prev_source = get_source()
        prev_checked = get_policy_checked()
        if not prev_source:
            reset_trace()
        set_source(SOURCE_NAME)

        try:
            if prev_checked:
                async for item in orig(prompt, *args, **kwargs):
                    yield item
                return

            scope_cm = identity_scope(record) if record is not None else nullcontext()
            with scope_cm:
                ev = _build_input_event(
                    source=SOURCE_NAME,
                    target="claude_agent_sdk.query",
                    model=model,
                    prompt_text=prompt_text,
                    stream=True,
                    payload=payload,
                )

                set_policy_checked(True)
                reset_policy_usage()
                try:
                    decision = _run_input_phase(
                        source=SOURCE_NAME,
                        target="claude_agent_sdk.query",
                        model=model,
                        prompt_text=prompt_text,
                        stream=True,
                        ev=ev,
                    )

                    if decision.verdict == "block":
                        ev["latency_ms"] = 0
                        # Input-side block on the module-level
                        # one-shot ``query()``: the subprocess never
                        # got the prompt — real enforcement.
                        ev["enforcement_status"] = ENFORCEMENT_ENFORCED
                        _safe_enqueue(ev)
                        cfg = get_config()
                        msg = (
                            f"[egisai] {decision.message or 'blocked by policy'} "
                            f"(matched={decision.matched_policy})"
                        )
                        if cfg.on_block == "raise":
                            raise PermissionError(msg)
                        return

                    if decision.verdict == "sanitize":
                        _apply_sanitization(
                            decision=decision, payload=payload, ev=ev
                        )
                        prompt = payload["input"]

                    # Open Run for the module-level query() — the
                    # Run spans the iterator's lifetime and closes
                    # when the result arrives (or the generator is
                    # aclosed early).
                    run_opened = False
                    if current_run() is None:
                        open_run(
                            framework="claude_agent_sdk",
                            identity=record,
                            prompt_text=ev.get("prompt_preview"),
                        )
                        run_opened = True
                        ev.setdefault("enforcement_status", ENFORCEMENT_ENFORCED)
                        try:
                            append_initial_model_call_step(
                                event=ev,
                                started_at=time.monotonic(),
                            )
                        except Exception:  # noqa: BLE001
                            LOGGER.debug(
                                "claude_agent_sdk provisional model step failed",
                                exc_info=True,
                            )

                    # ── PreToolUse + PostToolUse hook injection ───
                    # See _wrap_client_query for the full rationale.
                    # We mutate ``options`` in place; the user's
                    # original instance lives for the duration of
                    # this query() call so the local mutation is
                    # scoped to this turn.
                    module_hook_decisions: dict[str, str] = {}
                    if _hooks_supported() and options is not None:
                        callback = _build_pretooluse_callback(
                            record=record,
                            run_ctx=current_run(),
                            ev_template=ev,
                            model=model,
                            decisions=module_hook_decisions,
                        )
                        _inject_pretooluse_hook(options, callback)
                    if _post_hooks_supported() and options is not None:
                        post_callback = _build_posttooluse_callback(
                            record=record,
                            run_ctx=current_run(),
                            ev_template=ev,
                            model=model,
                        )
                        _inject_posttooluse_hook(options, post_callback)

                    started = time.monotonic()
                    signals = _new_signals()
                    enqueued = False
                    ctx = copy_context()
                    async_gen = orig(prompt, *args, **kwargs)
                    try:
                        while True:
                            try:
                                message = await ctx.run(async_gen.__anext__)
                            except StopAsyncIteration:
                                break
                            _accumulate_response_signals(message, signals)
                            module_hooks_injected = (
                                _hooks_supported() and options is not None
                            )
                            if (
                                type(message).__name__ == "AssistantMessage"
                                and current_run() is not None
                                and not module_hooks_injected
                            ):
                                tool_uses = _iter_tool_uses(message)
                                for tname, tinp, tuid in tool_uses:
                                    _dispatch_tool_call_step(
                                        ev_template=ev,
                                        tool_name=tname,
                                        tool_input=tinp,
                                        tool_use_id=tuid,
                                        model=model,
                                        started_at=time.monotonic(),
                                        hook_decisions=module_hook_decisions,
                                    )
                            if _is_result_message(message):
                                ev["latency_ms"] = int(
                                    max(0, (time.monotonic() - started) * 1000)
                                )
                                _stamp_usage_from_result(ev, message)
                                module_hooks_active = _hooks_supported() and (
                                    options is not None
                                )
                                decision_out = _run_output_phase(
                                    ev=ev,
                                    signals=signals,
                                    model=model,
                                    stream=True,
                                    hooks_active=module_hooks_active,
                                )
                                # Allow path trivially enforced; the
                                # block path inside _run_output_phase
                                # already chose enforced or advisory
                                # based on hooks_active.
                                ev.setdefault(
                                    "enforcement_status",
                                    ENFORCEMENT_ENFORCED,
                                )
                                _safe_enqueue(ev)
                                enqueued = True
                                if run_opened and current_run() is not None:
                                    close_run()
                                    run_opened = False
                                if (
                                    decision_out is not None
                                    and decision_out.verdict == "block"
                                ):
                                    cfg = get_config()
                                    msg = (
                                        "[egisai] "
                                        f"{decision_out.message or 'blocked by policy'} "
                                        f"(matched={decision_out.matched_policy})"
                                    )
                                    if cfg.on_block == "raise":
                                        raise PermissionError(msg)
                                yield message
                                continue
                            yield message
                        if not enqueued:
                            ev["latency_ms"] = int(
                                max(0, (time.monotonic() - started) * 1000)
                            )
                            _safe_enqueue(ev)
                    except BaseException:
                        if not enqueued:
                            ev["latency_ms"] = int(
                                max(0, (time.monotonic() - started) * 1000)
                            )
                            if "error" not in ev:
                                ev["error"] = "stream failed"
                            _safe_enqueue(ev)
                        if run_opened and current_run() is not None:
                            close_run(error="stream failed")
                            run_opened = False
                        raise
                    finally:
                        if run_opened and current_run() is not None:
                            # Generator exhausted without a
                            # ResultMessage — close the run anyway.
                            close_run()
                            run_opened = False
                finally:
                    set_policy_checked(prev_checked)
        finally:
            set_source(prev_source)

    wrapped.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


# ── Public entry point ──────────────────────────────────────────────


def apply() -> bool:
    if not has_module("claude_agent_sdk"):
        return False

    any_patched = False

    try:
        import claude_agent_sdk as _sdk  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return False

    # Module-level ``query`` — single-shot async generator.
    if hasattr(_sdk, "query") and callable(_sdk.query):
        orig_q = _sdk.query
        if not getattr(orig_q, "__egisai_wrapped__", False):
            _sdk.query = _wrap_module_query(orig_q)
            any_patched = True

    # ``ClaudeSDKClient`` — persistent client across multi-turn convos.
    client_cls = getattr(_sdk, "ClaudeSDKClient", None)
    if client_cls is not None:
        # 1. ``connect`` — inject placeholder PreToolUse +
        #    PostToolUse hooks BEFORE the CLI initializes its
        #    matcher table. This is the SOC 2 / ISO 27001 fix:
        #    upstream reads ``options.hooks`` exactly once inside
        #    ``connect()`` and ships the result to the Node CLI;
        #    any later mutation is a silent no-op. Wrap
        #    ``connect`` (not ``__aenter__``) because the user
        #    may call it directly — and ``__aenter__`` calls
        #    ``self.connect()`` so wrapping ``connect`` covers
        #    both paths in one patch. See
        #    ``_wrap_client_connect`` for the protocol notes.
        orig_connect = getattr(client_cls, "connect", None)
        if (
            orig_connect is not None
            and callable(orig_connect)
            and not getattr(orig_connect, "__egisai_wrapped__", False)
        ):
            client_cls.connect = _wrap_client_connect(orig_connect)
            any_patched = True

        # 2. ``query`` (coroutine) — Phase 1+2 input gate.
        orig_query = getattr(client_cls, "query", None)
        if (
            orig_query is not None
            and callable(orig_query)
            and not getattr(orig_query, "__egisai_wrapped__", False)
        ):
            client_cls.query = _wrap_client_query(orig_query)
            any_patched = True

        # 2. ``receive_messages`` (async generator) — output gate +
        #    finalize. ``receive_response`` delegates to this so we
        #    cover both call paths in one patch.
        orig_recv = getattr(client_cls, "receive_messages", None)
        if (
            orig_recv is not None
            and callable(orig_recv)
            and not getattr(orig_recv, "__egisai_wrapped__", False)
        ):
            client_cls.receive_messages = _wrap_client_receive_messages(orig_recv)
            any_patched = True

        # 3. ``__aexit__`` — flush any leftover inflight on close.
        orig_aexit = getattr(client_cls, "__aexit__", None)
        if (
            orig_aexit is not None
            and callable(orig_aexit)
            and not getattr(orig_aexit, "__egisai_wrapped__", False)
        ):
            client_cls.__aexit__ = _wrap_client_aexit(orig_aexit)
            any_patched = True

    # Identity-only patch for ``query_stream`` (deprecated alias on some
    # versions). Falls back to the legacy framework wrapper if present.
    if hasattr(_sdk, "ClaudeSDKClient") and not any_patched:
        # ``patch_method`` is idempotent; safe to call as a final
        # safety net so we never end up with zero patches when the
        # upstream class shape changes.
        if patch_method(
            "claude_agent_sdk", "ClaudeSDKClient", "query",
            derive=_derive, kind="async",
        ):
            any_patched = True

    return any_patched
