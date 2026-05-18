"""Shared gate logic used by every framework patch.

For each governed call:

1. Build an audit event from the captured kwargs.
2. Auto-detect the agent identity from the system prompt
   (``set_context(agent=…)`` wins).
3. Evaluate the cached policies.
4. Allow → forward to the original function, time it, capture token
   usage. Sanitize → mask in place, then forward. Block → raise
   ``PermissionError`` or return the framework-specific stub.
5. Enqueue the event for async flushing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from egisai._auto_agent import (
    current_identity,
    resolve_identity,
)
from egisai._config import get_config
from egisai._context import (
    ensure_trace_id,
    get_context,
    get_policy_checked,
    get_policy_usage,
    get_source,
    reset_policy_usage,
    reset_trace,
    set_policy_checked,
    set_source,
)
from egisai._evaluator import (
    InputCall,
    OutputCall,
    evaluate,
    evaluate_output,
    extract_payload_text,
    mutate_prompt_text,
)
from egisai._events import build_event, safe_preview
from egisai._logger import enqueue
from egisai._run import StepKind, append_step, current_run
from egisai.policy import PolicyDecision, label_redact
from egisai.policy.pii import Sanitization
from egisai.policy.pii import sanitize as pii_sanitize


def _serialize_matched_policies(
    decision: PolicyDecision,
) -> list[dict[str, Any]]:
    """Convert ``decision.matched_policies`` to JSON-friendly dicts."""
    out: list[dict[str, Any]] = []
    for r in decision.matched_policies:
        out.append(
            {
                "name": r.name,
                "type": r.type,
                "verdict": r.verdict,
                "reason_code": r.reason_code,
                "message": r.message,
                # Wire field is now ``sanitize_types`` (matches the
                # operator-facing terminology). The backend accepts
                # both ``sanitize_types`` and the legacy
                # ``sanitize_kinds`` for one release while older SDKs
                # are still in the field.
                "sanitize_types": list(r.sanitize_types),
                "sanitize_mask_char": r.sanitize_mask_char,
            }
        )
    if not out and decision.matched_policy and decision.verdict in (
        "block",
        "sanitize",
    ):
        out.append(
            {
                "name": decision.matched_policy,
                "type": "",
                "verdict": decision.verdict,
                "reason_code": decision.reason_code or "",
                "message": decision.message or "",
                "sanitize_types": list(decision.sanitize_types),
                "sanitize_mask_char": decision.sanitize_mask_char,
            }
        )
    return out


def _decision_block(decision: PolicyDecision) -> dict[str, Any]:
    """Per-phase decision summary persisted alongside the audit row.

    Two of these are produced per call (``prompt_decision`` and
    ``response_decision``) so the dashboard can render the pre-model
    and post-model verdicts independently. The shape mirrors the
    top-level audit fields one-for-one, plus ``matched_policies``
    for the per-phase rule list.
    """
    return {
        "verdict": decision.verdict,
        "reason_code": decision.reason_code,
        "reason": decision.message,
        "matched_policy": decision.matched_policy,
        "matched_policies": _serialize_matched_policies(decision),
    }


_MAX_PREVIEW_LEN = 2048


def _safe_text_preview(text: str | None) -> str | None:
    """Label-redact + truncate.

    Any string returned by this function is safe to persist in the
    audit log even if it originally contained raw PII.
    """
    if not text:
        return text
    redacted = label_redact(text)
    if len(redacted) <= _MAX_PREVIEW_LEN:
        return redacted
    return redacted[: _MAX_PREVIEW_LEN - 3] + "..."

LOGGER = logging.getLogger("egisai.patches")

# A function that, given the framework-specific response object, returns
# a small dict with ``tokens_in``, ``tokens_out``, and (optionally)
# ``cost_usd``. Returning an empty dict is fine — the audit row just
# stays with zeros for that call.
ExtractUsage = Callable[[Any], dict[str, Any]]


def _stamp_usage(ev: dict, response: Any, extract_usage: ExtractUsage | None) -> None:
    """Copy ``tokens_in`` / ``tokens_out`` / ``cost_usd`` onto the event."""
    if extract_usage is None or response is None:
        return
    try:
        usage = extract_usage(response) or {}
    except Exception:  # noqa: BLE001
        LOGGER.debug("usage extractor failed for %s", ev.get("source"), exc_info=True)
        return
    if "tokens_in" in usage and usage["tokens_in"] is not None:
        ev["tokens_in"] = int(usage["tokens_in"])
    if "tokens_out" in usage and usage["tokens_out"] is not None:
        ev["tokens_out"] = int(usage["tokens_out"])
    if usage.get("cost_usd") is not None:
        ev["cost_usd"] = float(usage["cost_usd"])


def _apply_sanitization(
    *, decision: PolicyDecision, payload: Any, ev: dict
) -> None:
    """Mask PII in ``payload`` and stamp the audit event with what we did.

    Mutates the payload in place so the upstream framework
    serializes the masked text. The audit event records the count
    and mask shape per kind, never the original value.
    """
    aggregated: dict[str, Sanitization] = {}

    def _transform(text: str) -> str:
        new_text, records = pii_sanitize(
            text,
            types=decision.sanitize_types or None,
            mask_char=decision.sanitize_mask_char,
        )
        for rec in records:
            existing = aggregated.get(rec.type)
            if existing is None:
                aggregated[rec.type] = rec
            else:
                aggregated[rec.type] = Sanitization(
                    type=rec.type,
                    count=existing.count + rec.count,
                    pattern=existing.pattern,
                )
        return new_text

    before_text = ev.get("_prompt_text_original") or extract_payload_text(payload)
    ev["prompt_preview_before"] = _safe_text_preview(before_text)

    mutate_prompt_text(payload, _transform)

    ev["payload_preview"] = safe_preview(payload)

    after_text = extract_payload_text(payload)
    ev["prompt_preview"] = _safe_text_preview(after_text)

    # Wire field is now ``type`` (matches the operator-facing
    # terminology used everywhere else). Backend accepts both
    # ``type`` and the legacy ``kind`` on inbound audit events for
    # one release.
    ev["sanitizations"] = [
        {"type": s.type, "count": s.count, "pattern": s.pattern}
        for s in aggregated.values()
    ]


def _label_redact_tool_input(tool_input: Any) -> str | None:
    """Render a tool's input as a compliance-safe preview.

    Same shape as ``claude_agent_sdk._safe_preview_tool_input``:
    JSON-serialize the input (or ``repr`` it if that fails), then
    push through ``label_redact`` + the 2 KB truncation cap so a
    free-text argument the model wrote (an end-user email, a
    free-form reason string) never lands raw on the audit row.
    """
    if tool_input is None or tool_input == "":
        return None
    if isinstance(tool_input, str):
        rendered = tool_input
    else:
        try:
            import json

            rendered = json.dumps(tool_input, default=str, ensure_ascii=False)
        except Exception:  # noqa: BLE001
            rendered = repr(tool_input)
    return _safe_text_preview(rendered)


def _dispatch_per_tool_steps(
    *,
    response: Any,
    payload: Any,
    source: str,
    target: str,
    model: str,
    stream: bool,
    extract_output_signals: ExtractOutputSignals,
    ev_template: dict[str, Any],
    enforcement_status: str,
) -> None:
    """Emit one ``tool_call`` step per tool the model asked to invoke.

    Mirrors the multi-step waterfall that ``claude_agent_sdk``
    already paints: when a model_call returns tool calls, the
    timeline should show each tool as its own dot on the rail so
    the operator reads the agentic loop top-to-bottom as
    ``model -> tool -> model -> tool -> ...`` instead of collapsing
    every turn's tools into a single, opaque ``model_call`` row.

    For synchronous patches (OpenAI Chat / Responses) the tool
    hasn't *executed* yet by the time we see this; the next
    framework iteration is what runs it. The step is therefore
    a faithful record of what the model invoked plus the
    label-redacted argument shape. ``enforcement_status`` is
    passed in by the caller so each framework can describe its
    own actual enforcement guarantee — ``"enforced"`` for
    synchronous patches (the model_call's output policy already
    had a chance to block the tool request), ``"advisory"`` for
    streaming agentic frameworks that execute tools in a
    subprocess we can't stop.

    No per-tool output-policy re-evaluation here: the parent
    model_call step's ``response_decision`` already evaluated all
    tool calls together. We deliberately keep the per-tool step
    informational so the timeline reads ``model -> tool`` rather
    than ``model -> [redundant policy] -> tool``.
    """
    try:
        _text, _tool_names, tool_calls, _mcp_targets = extract_output_signals(
            response, payload
        )
    except Exception:  # noqa: BLE001
        LOGGER.debug(
            "per-tool extractor failed for %s", source, exc_info=True
        )
        return
    if not tool_calls:
        return

    import uuid as _uuid

    now = time.monotonic()
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if not isinstance(name, str) or not name:
            continue
        # Tools may carry their args under either ``arguments``
        # (Chat Completions) or ``input`` (Responses API / Claude
        # tool_use) depending on the extractor's normalisation.
        tool_input = call.get("arguments")
        if tool_input is None:
            tool_input = call.get("input")

        ev: dict[str, Any] = {
            "event_id": _uuid.uuid4().hex,
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
            "source": source,
            "target": f"{target}.tool_call",
            "model": model,
            "stream": stream,
            "tool_name": name,
            # Wire key MUST be ``prompt_preview`` — the backend reads
            # the audit row's preview text from this field
            # (``app.routers.sdk._build_request_log_row`` → ``ev.get(
            # "prompt_preview")``). Shipping under ``request_text``
            # (the column name on the DB side) is silently dropped
            # because the ingest reader doesn't look at that key.
            # Bug fix in 0.27.1 — see CHANGELOG.
            "prompt_preview": _label_redact_tool_input(tool_input),
            # The parent model_call's output policy already gated
            # this tool request. Each per-tool step records the
            # invocation as ``allow`` so the timeline reads green
            # for the tool box; the operator can drill into the
            # parent model_call step to see the aggregate
            # response_decision for the turn.
            "verdict": "allow",
            "enforcement_status": enforcement_status,
            "latency_ms": 0,
            "policy_latency_ms": 0,
            "policy_tokens_in": 0,
            "policy_tokens_out": 0,
        }

        try:
            append_step(event=ev, kind="tool_call", started_at=now)
        except Exception:  # noqa: BLE001
            LOGGER.debug("tool_call step emit failed", exc_info=True)


def _stamp_identity_provenance(ev: dict, record: Any) -> None:
    """Copy ``identity_source`` + ``identity_hash`` from an IdentityRecord.

    The two fields are read by ingest in two places:

    * ``run.start`` events emitted by framework patches that open a
      Run carry them through :func:`egisai._run._build_run_start_event`
      so the backend's ``_upsert_run_from_start`` writes them onto the
      ``runs`` row.
    * Legacy single-row audit events (raw LLM calls — no framework
      wrap above the gate) need them stamped here so the backend's
      synth-Run path (``ingest_events`` ``else:`` branch when
      ``ev.get("run_id") is None``) can copy them onto the Run row it
      synthesises from this event.

    Without this helper the synth Run lands with NULL
    ``identity_source`` / ``identity_hash`` and the Agent Identity
    card on the dashboard renders blank for every Bedrock Converse /
    raw OpenAI Chat / raw Anthropic call. The fields themselves are
    derived from controlled-vocabulary tokens (``identity_source``)
    plus a SHA-256 digest (``identity_hash``); neither carries raw
    prompt content, so this stamp is compliance-safe.
    """
    if record is None:
        return
    source = getattr(record, "source", None)
    digest = getattr(record, "identity_hash", None)
    if source:
        ev.setdefault("identity_source", source)
    if digest:
        ev.setdefault("identity_hash", digest)


def _attribute_event(ev: dict, payload: Any) -> None:
    """Attribute the event to the right agent identity.

    Reads from the LOCKED Run identity first (a framework wrap above
    us already opened a Run with a resolved identity — we must not
    re-derive mid-run, otherwise Tier 5 prompt-hashing drift across
    turns would register 4 agents for what is logically 1). If no Run
    is open we fall through to the per-call stack identity, and then
    to the full 7-tier resolver as a last resort.

    See ``egisai._auto_agent.resolve_identity`` for the full tier
    table.

    Identity provenance (``identity_source`` + ``identity_hash``) is
    stamped onto ``ev`` alongside ``agent_id`` whenever the resolver
    surfaces an :class:`IdentityRecord`. The backend's legacy
    synth-Run path reads these off the event when no ``run.start``
    envelope preceded the event (raw LLM calls — patches that don't
    open their own Run), so the synth ``runs`` row carries the same
    Agent Identity provenance a framework-wrapped Run would.
    """
    # Run-scoped lock — the framework patch resolved identity ONCE at
    # entry; every inner LLM call shares that identity even if their
    # per-call payload would have otherwise resolved differently.
    run = current_run()
    if run is not None and run.agent_id:
        ev["agent_id"] = run.agent_id
        ev["app"] = run.agent_name or ev.get("app")
        _stamp_identity_provenance(ev, run.identity)
        return

    ctx = get_context()
    record = current_identity()
    if record is not None and record.agent_id:
        ev["agent_id"] = record.agent_id
        ev["app"] = record.display_name
        _stamp_identity_provenance(ev, record)
        return

    # No pre-pushed identity — fall through to the resolver. This
    # path is hit by direct callers (legacy tests) and any patch
    # that doesn't wrap its forward via ``gate_call``.
    from egisai._config import get_config_optional

    cfg_opt = get_config_optional()
    hints = (
        getattr(cfg_opt, "auto_stack_hints", "loose")
        if cfg_opt is not None
        else "loose"
    )
    resolved = resolve_identity(payload, auto_stack_hints=hints)
    if resolved is not None and resolved.agent_id:
        ev["agent_id"] = resolved.agent_id
        ev["app"] = resolved.display_name
        _stamp_identity_provenance(ev, resolved)
        return

    if ctx.agent_id:
        # set_context(agent_id=…) escape hatch — explicit UUID on the
        # context with no display name. Keep the legacy behaviour.
        return

    cfg = get_config()
    if cfg.agent_id:
        return


# ── Run / step dispatch ─────────────────────────────────────────────


def _dispatch_step(
    ev: dict[str, Any],
    *,
    started_at: float,
    ended_at: float | None = None,
    kind: StepKind = "model_call",
) -> None:
    """Send a finished audit event to the right destination.

    When a Run is open (framework wrap above us, or auto-opened by
    the gate for raw LLM use), the event is recorded as a step under
    that Run and the streaming ``run.step`` envelope is enqueued.
    When no Run is open, the legacy single-row event is enqueued —
    this preserves the older wire format for tests / callers that
    intentionally bypass the run framework.
    """
    if append_step(  # type: ignore[func-returns-value]
        event=ev, kind=kind, started_at=started_at, ended_at=ended_at,
    ) is not None:
        return
    enqueue(ev)


# Output-side signal extractor: ``(response, request_payload) → (text,
# tool_names, tool_calls, mcp_targets)``. Patchers that don't care about
# output-side policies pass ``None``; the gate then skips Phase 3.
ExtractOutputSignals = Callable[
    [Any, Any],
    tuple[str, list[str], list[dict[str, Any]], list[str]],
]


def _build_input_event(
    *,
    source: str,
    target: str,
    model: str,
    prompt_text: str,
    stream: bool,
    payload: Any,
) -> dict[str, Any]:
    """Construct the audit event and attribute it to an agent identity."""
    ev = build_event(
        source=source, target=target, payload=payload, model=model, stream=stream
    )
    _attribute_event(ev, payload)
    ev["prompt_chars"] = len(prompt_text or "")
    ev["prompt_preview"] = _safe_text_preview(prompt_text)
    ev["_prompt_text_original"] = prompt_text
    return ev


def _run_input_phase(
    *,
    source: str,
    target: str,
    model: str,
    prompt_text: str,
    stream: bool,
    ev: dict[str, Any],
) -> PolicyDecision:
    """Evaluate input-side policies and stamp the verdict onto ``ev``.

    Stamps both the legacy top-level fields (``verdict``,
    ``matched_policies``, …) for older backends and the new
    structured ``prompt_decision`` block consumed by 0.12.4+.
    """
    policy_started = time.monotonic()
    decision = evaluate(
        InputCall(
            source=source,
            target=target,
            model=model,
            prompt_text=prompt_text,
            stream=stream,
        )
    )
    ev["policy_latency_ms"] = int((time.monotonic() - policy_started) * 1000)
    policy_in, policy_out = get_policy_usage()
    ev["policy_tokens_in"] = policy_in
    ev["policy_tokens_out"] = policy_out
    ev["verdict"] = decision.verdict
    ev["reason_code"] = decision.reason_code
    ev["reason"] = decision.message
    ev["matched_policy"] = decision.matched_policy
    ev["matched_policies"] = _serialize_matched_policies(decision)
    ev["prompt_decision"] = _decision_block(decision)
    return decision


def _block_response(
    *,
    decision: PolicyDecision,
    ev: dict[str, Any],
    model: str,
    stub_factory: Callable[[PolicyDecision, str, str], Any] | None,
    step_started: float | None = None,
) -> Any:
    """Dispatch the block step + return-or-raise the framework-shaped response.

    ``ev`` is dispatched as-is; the caller is responsible for setting
    ``ev["latency_ms"]`` first (zero for input-side blocks, real
    elapsed time for output-side blocks).

    Reaching this function means the SDK is about to actually refuse
    the call (synthetic stub OR raise) — so ``enforcement_status``
    is stamped ``"enforced"`` if the caller hasn't already set it.
    Patches that can't enforce at this point (agentic streaming SDKs)
    set ``enforcement_status="advisory"`` BEFORE calling this and we
    preserve their choice.
    """
    ev.setdefault("enforcement_status", ENFORCEMENT_ENFORCED)
    _dispatch_step(
        ev,
        started_at=step_started if step_started is not None else time.monotonic(),
        kind="model_call",
    )
    cfg = get_config()
    msg = (
        f"[egisai] {decision.message or 'blocked by policy'} "
        f"(matched={decision.matched_policy})"
    )
    if cfg.on_block == "raise" or stub_factory is None:
        raise PermissionError(msg)
    return stub_factory(decision, ensure_trace_id(), model)


def _run_output_phase(
    *,
    response: Any,
    payload: Any,
    source: str,
    target: str,
    model: str,
    stream: bool,
    extract_output_signals: ExtractOutputSignals | None,
    ev: dict[str, Any] | None = None,
) -> PolicyDecision | None:
    """Run output-side policies; return the full decision or ``None``.

    Returns the ``PolicyDecision`` whenever the post-model phase
    actually executed (allow OR block). Returns ``None`` when the
    phase was skipped — extractor missing, response empty, or
    nothing was extractable to evaluate. The caller uses this to
    decide whether to stamp ``response_decision`` on the audit
    event.

    Privacy contract: the model's response text is **evaluated but
    never persisted**. We extract it here only long enough to feed
    output-side policies (``deny_output_regex``, ``deny_tool_call``,
    ``semantic_guard``, etc.) and to expose match decisions on the
    audit row; the text itself is then discarded. The audit event
    never carries ``response_preview``. The dashboard surfaces what
    the model *did* (verdict, matched policy, tool names) — not what
    it *said*. This is the deliberate compliance posture: storing
    model outputs creates a perpetual leak surface (model output is
    less constrained than its input), and SOC 2 / GDPR auditors
    consistently prefer "we never had it" over "we redacted it
    well".

    When ``ev`` is provided, this function also stamps:

    * ``policy_latency_ms`` and ``policy_tokens_in`` /
      ``policy_tokens_out`` — **additive** to whatever the input
      phase already booked. The output-side ``semantic_guard``
      judge is a real LLM round-trip; its wall-clock time and
      token spend must be reflected on the row alongside the
      input-side judge's spend, not overwritten and not silently
      dropped.
    """
    if extract_output_signals is None or response is None:
        return None
    try:
        text, tool_names, tool_calls, mcp_targets = extract_output_signals(
            response, payload
        )
    except Exception:  # noqa: BLE001
        LOGGER.debug(
            "output signal extractor failed for %s", source, exc_info=True
        )
        return None

    if not (text or tool_names or tool_calls or mcp_targets):
        return None

    # NOTE: ``text`` is intentionally NOT stamped onto ``ev``. See
    # the docstring's privacy contract; the text only fuels policy
    # evaluation below and then goes out of scope with this function.

    prev_pol_in, prev_pol_out = get_policy_usage()
    policy_started = time.monotonic()

    decision = evaluate_output(
        OutputCall(
            source=source,
            target=target,
            model=model,
            text=text or "",
            tool_names=list(tool_names or []),
            tool_calls=list(tool_calls or []),
            mcp_targets=list(mcp_targets or []),
            stream=stream,
        )
    )

    if ev is not None:
        elapsed_ms = int((time.monotonic() - policy_started) * 1000)
        cur_pol_in, cur_pol_out = get_policy_usage()
        ev["policy_latency_ms"] = int(ev.get("policy_latency_ms") or 0) + elapsed_ms
        ev["policy_tokens_in"] = int(ev.get("policy_tokens_in") or 0) + max(
            0, cur_pol_in - prev_pol_in
        )
        ev["policy_tokens_out"] = int(ev.get("policy_tokens_out") or 0) + max(
            0, cur_pol_out - prev_pol_out
        )

    return decision


# ── enforcement_status ─────────────────────────────────────────────
#
# Values:
#
# * ``"enforced"`` (default) — the SDK actually prevented the call
#   from reaching its full destination. This is the case for every
#   input-side block (we never forwarded the prompt), and for every
#   output-side block on patches that synthesize a stub response or
#   raise (the user's code never sees the model's real output).
#
# * ``"advisory"`` — the policy decided block, BUT by the time the
#   SDK could act the call had already completed and (in agentic
#   frameworks) tools had already executed inside a subprocess we
#   don't own. ``claude_agent_sdk`` with ``on_block="stub"`` is the
#   canonical case: the Node.js CLI ran the entire agent loop end
#   to end before Python ever saw ``ResultMessage``, so an output
#   policy firing here is a post-hoc finding, not an enforcement.
#
# The two are intentionally separate from ``verdict`` so the audit
# row honestly reflects BOTH "what the policy decided" and "what
# actually happened to the user's call". SOC 2 / GDPR / HIPAA all
# expect this distinction: hiding it inside the verdict (e.g. by
# downgrading block -> "advisory verdict") would obscure the count
# of policies that fired, which auditors need to be able to query.

ENFORCEMENT_ENFORCED = "enforced"
ENFORCEMENT_ADVISORY = "advisory"


def _stamp_output_block(
    ev: dict[str, Any],
    decision: PolicyDecision,
    *,
    enforcement_status: str = ENFORCEMENT_ENFORCED,
) -> None:
    """Re-stamp the audit event so it reflects the output-side block.

    Input-side matches that fired (allow / sanitize) are preserved
    on ``ev["matched_policies"]``; the output match is appended so
    the audit row carries the full chain. The structured
    ``response_decision`` block carries the post-model verdict alone
    so the dashboard can render the two phases side-by-side.

    ``enforcement_status`` defaults to ``"enforced"`` because every
    caller in this module (``_gate_call_inner`` / ``_async_gate_call_inner``)
    follows the stamp with ``_block_response``, which DOES actually
    refuse the call (stub or raise). Patches whose framework doesn't
    permit enforcement at output time (notably ``claude_agent_sdk``)
    must pass ``enforcement_status=ENFORCEMENT_ADVISORY`` explicitly.
    """
    ev["verdict"] = "block"
    ev["reason_code"] = decision.reason_code
    ev["reason"] = decision.message
    ev["matched_policy"] = decision.matched_policy
    existing = ev.get("matched_policies") or []
    ev["matched_policies"] = list(existing) + _serialize_matched_policies(decision)
    ev["response_decision"] = _decision_block(decision)
    ev["enforcement_status"] = enforcement_status


def _resolve_and_scope_identity(payload: Any) -> Any:
    """Resolve the agent identity and (if non-stack) push it.

    Returns a context manager that pops the identity on exit. When
    the resolver already found a pre-pushed identity (parent framework
    patch is in scope), or the resolved tier doesn't push (stack /
    class / hash / app — those are per-call only), we return a
    no-op context.

    Called BEFORE the policy phase so:
      * ``_active_agent_id`` reads the right identity inside scoped
        rules (closes the policy-attribution gap that existed pre-0.17).
      * Inner nested ``gate_call``s in the same task see the same
        identity instead of re-deriving from their own (often empty)
        payload.
    """
    from contextlib import nullcontext

    from egisai._auto_agent import identity_scope as _scope
    from egisai._config import get_config_optional

    cfg = get_config_optional()
    hints = getattr(cfg, "auto_stack_hints", "loose") if cfg else "loose"
    record = resolve_identity(payload, auto_stack_hints=hints)
    if record is None or not record.push_to_stack:
        return nullcontext(record)
    return _scope(record)


def gate_call(
    *,
    source: str,
    target: str,
    model: str,
    prompt_text: str,
    stream: bool,
    payload: Any,
    stub_factory: Callable[[PolicyDecision, str, str], Any] | None = None,
    extract_usage: ExtractUsage | None = None,
    extract_output_signals: ExtractOutputSignals | None = None,
    emit_tool_call_steps: bool = False,
    forward: Callable[[], Any],
) -> Any:
    """Run the gate around a single synchronous model call.

    ``forward`` is a zero-arg callable invoking the original
    function; it runs only on allow / sanitize verdicts. Events are
    recorded as a step under the current Run (auto-opened when no
    framework wrap above us is hosting one).

    Output-side policies (``deny_tool_call``, ``deny_mcp_call``,
    ``deny_output_regex``, output-side ``semantic_guard``) run
    against the response when ``extract_output_signals`` is
    provided.

    ``emit_tool_call_steps`` (default ``False``) opts a framework
    into the multi-step waterfall: when the model's response carries
    tool calls and the call wasn't refused, one extra ``tool_call``
    step is appended per tool the model invoked. This is the visual
    that turns the dashboard timeline into a clear
    ``request -> [policy] -> model -> [policy] -> tool -> [policy]
    -> model -> ...`` sequence instead of a single opaque model
    row. Synchronous patches (OpenAI Chat / Responses) pass
    ``True`` here. Agentic-subprocess patches (claude_agent_sdk)
    already emit their own per-tool steps inline as ToolUseBlocks
    stream in.

    Identity resolution runs FIRST (before policy) so scoped policy
    rules can match on the auto-detected agent — pre-0.17 only
    ``set_context(agent=…)`` callers benefited from scoped rules.
    """
    prev_source = get_source()
    prev_checked = get_policy_checked()

    if not prev_source:
        reset_trace()
    set_source(source)

    try:
        if prev_checked:
            return forward()

        with _resolve_and_scope_identity(payload):
            return _gate_call_inner(
                source=source,
                target=target,
                model=model,
                prompt_text=prompt_text,
                stream=stream,
                payload=payload,
                stub_factory=stub_factory,
                extract_usage=extract_usage,
                extract_output_signals=extract_output_signals,
                emit_tool_call_steps=emit_tool_call_steps,
                forward=forward,
            )
    finally:
        set_source(prev_source)


def _gate_call_inner(
    *,
    source: str,
    target: str,
    model: str,
    prompt_text: str,
    stream: bool,
    payload: Any,
    stub_factory: Callable[[PolicyDecision, str, str], Any] | None,
    extract_usage: ExtractUsage | None,
    extract_output_signals: ExtractOutputSignals | None,
    emit_tool_call_steps: bool,
    forward: Callable[[], Any],
) -> Any:
    """Body of ``gate_call`` after identity has been resolved + pushed.

    We only reach this helper when ``prev_checked`` was False on the
    outer ``gate_call``; the outer's `prev_source` is reset by *its*
    try/finally so we don't manage it here.
    """
    ev = _build_input_event(
        source=source,
        target=target,
        model=model,
        prompt_text=prompt_text,
        stream=stream,
        payload=payload,
    )

    set_policy_checked(True)
    reset_policy_usage()
    step_started = time.monotonic()
    try:
        decision = _run_input_phase(
            source=source,
            target=target,
            model=model,
            prompt_text=prompt_text,
            stream=stream,
            ev=ev,
        )

        if decision.verdict == "block":
            ev["latency_ms"] = 0
            return _block_response(
                decision=decision,
                ev=ev,
                model=model,
                stub_factory=stub_factory,
                step_started=step_started,
            )

        if decision.verdict == "sanitize":
            _apply_sanitization(decision=decision, payload=payload, ev=ev)

        model_started = time.monotonic()
        try:
            response = forward()
        except BaseException:
            ev["latency_ms"] = int((time.monotonic() - model_started) * 1000)
            ev["error"] = "call failed"
            _dispatch_step(ev, started_at=step_started, kind="model_call")
            raise
        ev["latency_ms"] = int((time.monotonic() - model_started) * 1000)
        _stamp_usage(ev, response, extract_usage)

        output_decision = _run_output_phase(
            response=response,
            payload=payload,
            source=source,
            target=target,
            model=model,
            stream=stream,
            extract_output_signals=extract_output_signals,
            ev=ev,
        )
        if output_decision is not None and output_decision.verdict == "block":
            _stamp_output_block(ev, output_decision)
            return _block_response(
                decision=output_decision,
                ev=ev,
                model=model,
                stub_factory=stub_factory,
                step_started=step_started,
            )
        if output_decision is not None:
            ev["response_decision"] = _decision_block(output_decision)

        ev.setdefault("enforcement_status", ENFORCEMENT_ENFORCED)
        _dispatch_step(ev, started_at=step_started, kind="model_call")

        # Multi-step waterfall: if the framework opted in and the
        # model returned tool calls, append one ``tool_call`` step
        # per tool so the timeline reads ``model -> tool -> ...``
        # instead of collapsing the whole turn into a single box.
        # Synchronous patches (OpenAI) describe these as
        # ``enforced`` — the parent model_call's output policy
        # already had a chance to refuse the request before this
        # response left the gate. The per-tool emission only fires
        # when a Run is open above us (framework wrap like
        # ``Runner.run``); raw ``client.chat.completions.create()``
        # without an agent wrap falls through to the legacy
        # one-event path which the backend synthesises into a
        # one-step Run on ingest.
        if emit_tool_call_steps and extract_output_signals is not None:
            _dispatch_per_tool_steps(
                response=response,
                payload=payload,
                source=source,
                target=target,
                model=model,
                stream=stream,
                extract_output_signals=extract_output_signals,
                ev_template=ev,
                enforcement_status=ENFORCEMENT_ENFORCED,
            )
        return response
    finally:
        # Outer `gate_call` guaranteed prev_checked was False
        # (we returned early otherwise), so reset to False here.
        set_policy_checked(False)


async def async_gate_call(
    *,
    source: str,
    target: str,
    model: str,
    prompt_text: str,
    stream: bool,
    payload: Any,
    stub_factory: Callable[[PolicyDecision, str, str], Any] | None = None,
    extract_usage: ExtractUsage | None = None,
    extract_output_signals: ExtractOutputSignals | None = None,
    emit_tool_call_steps: bool = False,
    forward: Callable[[], Any],
) -> Any:
    """Async sibling of ``gate_call`` — same semantics, awaits ``forward()``."""
    prev_source = get_source()
    prev_checked = get_policy_checked()

    if not prev_source:
        reset_trace()
    set_source(source)

    try:
        if prev_checked:
            return await forward()

        with _resolve_and_scope_identity(payload):
            return await _async_gate_call_inner(
                source=source,
                target=target,
                model=model,
                prompt_text=prompt_text,
                stream=stream,
                payload=payload,
                stub_factory=stub_factory,
                extract_usage=extract_usage,
                extract_output_signals=extract_output_signals,
                emit_tool_call_steps=emit_tool_call_steps,
                forward=forward,
            )
    finally:
        set_source(prev_source)


async def _async_gate_call_inner(
    *,
    source: str,
    target: str,
    model: str,
    prompt_text: str,
    stream: bool,
    payload: Any,
    stub_factory: Callable[[PolicyDecision, str, str], Any] | None,
    extract_usage: ExtractUsage | None,
    extract_output_signals: ExtractOutputSignals | None,
    emit_tool_call_steps: bool,
    forward: Callable[[], Any],
) -> Any:
    """Body of ``async_gate_call`` after identity has been resolved + pushed."""
    ev = _build_input_event(
        source=source,
        target=target,
        model=model,
        prompt_text=prompt_text,
        stream=stream,
        payload=payload,
    )

    set_policy_checked(True)
    reset_policy_usage()
    step_started = time.monotonic()
    try:
        decision = _run_input_phase(
            source=source,
            target=target,
            model=model,
            prompt_text=prompt_text,
            stream=stream,
            ev=ev,
        )

        if decision.verdict == "block":
            ev["latency_ms"] = 0
            return _block_response(
                decision=decision,
                ev=ev,
                model=model,
                stub_factory=stub_factory,
                step_started=step_started,
            )

        if decision.verdict == "sanitize":
            _apply_sanitization(decision=decision, payload=payload, ev=ev)

        model_started = time.monotonic()
        try:
            response = await forward()
        except BaseException:
            ev["latency_ms"] = int((time.monotonic() - model_started) * 1000)
            ev["error"] = "call failed"
            _dispatch_step(ev, started_at=step_started, kind="model_call")
            raise
        ev["latency_ms"] = int((time.monotonic() - model_started) * 1000)
        _stamp_usage(ev, response, extract_usage)

        output_decision = _run_output_phase(
            response=response,
            payload=payload,
            source=source,
            target=target,
            model=model,
            stream=stream,
            extract_output_signals=extract_output_signals,
            ev=ev,
        )
        if output_decision is not None and output_decision.verdict == "block":
            _stamp_output_block(ev, output_decision)
            return _block_response(
                decision=output_decision,
                ev=ev,
                model=model,
                stub_factory=stub_factory,
                step_started=step_started,
            )
        if output_decision is not None:
            ev["response_decision"] = _decision_block(output_decision)

        ev.setdefault("enforcement_status", ENFORCEMENT_ENFORCED)
        _dispatch_step(ev, started_at=step_started, kind="model_call")

        # Multi-step waterfall — async sibling of the same per-tool
        # emission the sync path runs. See ``_gate_call_inner``.
        if emit_tool_call_steps and extract_output_signals is not None:
            _dispatch_per_tool_steps(
                response=response,
                payload=payload,
                source=source,
                target=target,
                model=model,
                stream=stream,
                extract_output_signals=extract_output_signals,
                ev_template=ev,
                enforcement_status=ENFORCEMENT_ENFORCED,
            )
        return response
    finally:
        # Outer `async_gate_call` guaranteed prev_checked was
        # False (it returned early otherwise), so reset to False.
        set_policy_checked(False)
