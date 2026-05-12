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
    get_policy_checked,
    get_source,
    reset_policy_usage,
    reset_trace,
    set_policy_checked,
    set_source,
)
from egisai._evaluator import OutputCall, evaluate_output
from egisai._logger import enqueue
from egisai._patches import has_module
from egisai._patches._common import (
    _apply_sanitization,
    _build_input_event,
    _decision_block,
    _run_input_phase,
    _stamp_output_block,
)
from egisai._patches._framework import make_identity, patch_method
from egisai._run import append_step, close_run, current_run, open_run
from egisai.policy import PolicyDecision

LOGGER = logging.getLogger("egisai.patches.claude_agent_sdk")

FRAMEWORK_SOURCE = "framework:claude_agent_sdk"
SOURCE_NAME = "claude_agent_sdk"
TARGET_DEFAULT = "claude_agent_sdk.client"
INFLIGHT_ATTR = "__egisai_inflight_event__"
INFLIGHT_SIGNALS_ATTR = "__egisai_inflight_signals__"
INFLIGHT_STARTED_ATTR = "__egisai_inflight_started__"
INFLIGHT_IDENTITY_ATTR = "__egisai_inflight_identity__"


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
) -> PolicyDecision | None:
    """Evaluate Phase 1+2 output policies on the accumulated stream.

    Stamps ``response_decision`` on the event and, on block,
    re-stamps the top-level ``verdict``. The caller decides whether
    to surface ``PermissionError`` to the user (input-side block is
    raised at ``query`` time; output-side block is raised inline at
    ``receive_messages`` time so the iterator stops before draining
    further turns).
    """
    text = "".join(t for t in signals.get("text", []) if isinstance(t, str))
    tool_names = list(signals.get("tool_names", []))
    tool_calls = list(signals.get("tool_calls", []))
    mcp_targets = list(signals.get("mcp_targets", []))

    if not (text or tool_names or tool_calls or mcp_targets):
        return None

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

    if decision.verdict == "block":
        _stamp_output_block(ev, decision)
    else:
        ev["response_decision"] = _decision_block(decision)
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
        if append_step(event=ev, kind="model_call") is None:
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
    ):
        try:
            if hasattr(self_obj, attr):
                delattr(self_obj, attr)
        except (AttributeError, TypeError):
            pass


# ── Method wrappers ─────────────────────────────────────────────────


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

                    if _is_result_message(message):
                        if ev is not None:
                            started = getattr(
                                self, INFLIGHT_STARTED_ATTR, time.monotonic()
                            )
                            ev["latency_ms"] = int(
                                max(0, (time.monotonic() - started) * 1000)
                            )
                            _stamp_usage_from_result(ev, message)

                            decision = _run_output_phase(
                                ev=ev,
                                signals=signals,
                                model=model,
                                stream=True,
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
                            if _is_result_message(message):
                                ev["latency_ms"] = int(
                                    max(0, (time.monotonic() - started) * 1000)
                                )
                                _stamp_usage_from_result(ev, message)
                                decision_out = _run_output_phase(
                                    ev=ev,
                                    signals=signals,
                                    model=model,
                                    stream=True,
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
        # 1. ``query`` (coroutine) — Phase 1+2 input gate.
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
