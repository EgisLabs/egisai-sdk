"""Run / Step model — turn N model calls into ONE auditable Run.

A *Run* is the unit of work between a framework entry point (e.g.
``Runner.run``, ``Pregel.invoke``, ``ClaudeSDKClient.query``) and its
terminal output. Inside the run we accumulate *Steps* — one per
underlying model call (today) and, in v2, one per tool call too. The
dashboard renders the run as a single timeline so the operator sees
the entire flow:

  prompt -> policy -> model -> policy -> tool -> policy -> model -> ... -> final

Wire format (streaming, option B):

  * ``run.start``  emitted when the framework entry opens. Carries
                   ``run_id``, ``agent_id``, ``framework``, identity,
                   and the initial prompt preview so the dashboard can
                   render a row immediately.
  * ``run.step``   emitted as each step completes. Carries the full
                   audit-row fields the legacy single-row event used
                   to ship (verdict, prompt/response decisions, tokens,
                   latency, etc.) plus ``run_id``, ``seq``, ``kind``.
  * ``run.end``    emitted when the framework entry exits (or the
                   streaming handler is exhausted / aexits). Carries
                   the run totals (sum tokens / latency / cost, worst
                   verdict) and the final response preview.

Identity is **locked** at run open. Inner gate calls under an open
Run use the run's owner; they cannot re-derive identity mid-run.
This is what fixes "4 agents from 1 task" — once the framework
patch resolved the identity at the entry point, subsequent LLM calls
inherit that identity even if their per-call payload would have
otherwise resolved differently (Tier 5 prompt-hash drifting across
turns, for example).

For raw LLM use (no framework wrap), the gate auto-opens an
ephemeral 1-step Run on the fly so the wire format is consistent
and the backend always sees a Run hierarchy.

ContextVar — the run pointer is stored in a ``ContextVar`` so it
inherits across asyncio tasks (with ``copy_context`` where the
patch already does that). Nested entry points open child Runs and
link via ``parent_run_id``.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from egisai._auto_agent import IdentityRecord, current_identity

LOGGER = logging.getLogger("egisai.run")

StepKind = Literal["model_call", "tool_call", "sub_agent_spawn", "policy_check"]


@dataclass
class RunStep:
    """One step inside a run.

    The ``event`` payload mirrors today's audit event shape one-for-one
    so backend ingest can persist it as a ``run_steps`` row without a
    second schema. We also keep the absolute step seq and timing here
    so the run can compute its aggregates without parsing the event.
    """

    seq: int
    kind: StepKind
    started_at: float            # ``time.monotonic()`` at step entry
    ended_at: float | None = None
    # Model/tool execution wall-clock for this step, in ms. Sourced
    # from the event's own ``latency_ms`` when the patch stamped one
    # (patches measure the forward() call only, and stamp an explicit
    # ``0`` for input-side blocks where the model was never called).
    # Falls back to ``ended_at - started_at`` for events that never
    # stamped a value. NEVER recomputed from the step timestamps when
    # the event carries a value — the timestamps span the whole gate
    # (input policy + model + output policy), so recomputing would
    # double-book governance time that ``policy_latency_ms`` already
    # accounts for, and would show phantom "model latency" on blocked
    # calls that never reached the provider.
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    policy_tokens_in: int = 0
    policy_tokens_out: int = 0
    cost_usd: float = 0.0
    verdict: str = "allow"       # worst-of for this step (allow|sanitize|block)
    model: str | None = None
    target: str | None = None
    tool_name: str | None = None
    event: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """Live state for an open run.

    Held on a ``ContextVar`` so asyncio tasks inherit it cleanly via
    ``contextvars.copy_context()`` (which our async-iter framework
    wrap already does for identity propagation).

    Concurrency: the SDK's gate path is single-threaded per call,
    but streamed generators advance on whatever asyncio task picks
    them up. The ``_lock`` here is cheap insurance for the few
    ``append_step`` / aggregate-update touchpoints.
    """

    run_id: str
    trace_id: str                # mirrored onto every step for back-compat
    framework: str               # 'openai_agents' | 'claude_agent_sdk' | 'raw' | ...
    identity: IdentityRecord | None
    agent_id: str | None         # LOCKED — set once at open, never re-derived
    agent_name: str | None
    parent_run_id: str | None
    started_at: float
    ended_at: float | None = None
    steps: list[RunStep] = field(default_factory=list)
    closed: bool = False
    error: str | None = None
    # ``prompt_text``: the FIRST step's prompt preview, displayed as
    # "what kicked this run off" on the dashboard. Post-redaction.
    #
    # ``final_text`` is intentionally always ``None``: the privacy
    # contract is that the SDK NEVER persists model responses (see
    # ``_patches/_common._run_output_phase``). The field is retained
    # on the dataclass so the wire shape of ``run.end`` stays
    # backwards-compatible — the value just goes out as ``None`` and
    # the backend / dashboard expect that.
    prompt_text: str | None = None
    final_text: str | None = None
    # Worst verdict across all steps — block > sanitize > allow.
    worst_verdict: str = "allow"
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    # Operator-visible run-summary event_id — stable across the
    # ``run.start`` / ``run.end`` boundary so the backend can update
    # in place rather than insert + update.
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # When this Run was opened nested inside another, we keep a
    # direct pointer to the parent so ``close_run`` can restore it
    # via the ContextVar without needing a separate stack. The
    # ContextVar always points to the innermost open run; the
    # ``parent`` chain reconstructs the stack on demand.
    parent: RunContext | None = None


_current_run: contextvars.ContextVar[RunContext | None] = contextvars.ContextVar(
    "egisai_current_run", default=None
)


# ── Public surface ──────────────────────────────────────────────────


def current_run() -> RunContext | None:
    """Return the run open in this context, or ``None``."""
    return _current_run.get()


def open_run(
    *,
    framework: str,
    identity: IdentityRecord | None,
    prompt_text: str | None = None,
) -> RunContext:
    """Open a new run scope.

    Pushes a fresh :class:`RunContext` onto the ContextVar. If a run is
    already open, the new run is opened as a CHILD (sub-agent / handoff)
    and ``parent_run_id`` is wired up so the dashboard can render the
    parent → child link.

    The returned context can be used to read ``run_id`` for logging or
    to attach additional metadata before the first step.
    """
    parent = _current_run.get()
    agent_id = identity.agent_id if identity is not None else None
    agent_name = identity.display_name if identity is not None else None

    ctx = RunContext(
        run_id=uuid.uuid4().hex,
        trace_id=uuid.uuid4().hex,
        framework=framework,
        identity=identity,
        agent_id=agent_id,
        agent_name=agent_name,
        parent_run_id=parent.run_id if parent is not None else None,
        started_at=time.monotonic(),
        prompt_text=prompt_text,
        parent=parent,
    )
    _current_run.set(ctx)
    # Emit run.start so the dashboard can show the row the moment the
    # framework entry fires (long-running agents otherwise stay
    # invisible until completion).
    _safe_emit_run_start(ctx)
    LOGGER.debug(
        "open_run framework=%s run_id=%s parent=%s agent=%s",
        framework, ctx.run_id[:8], (ctx.parent_run_id or "-")[:8] if ctx.parent_run_id else "-",
        agent_name,
    )
    return ctx


def close_run(*, error: str | None = None) -> RunContext | None:
    """Close the current run scope; emit ``run.end``.

    Idempotent — if the current run has already been closed (e.g. by
    a streaming handler finalising on iterator exhaustion before the
    outer wrap returns), this is a no-op and we just clear the
    ContextVar.
    """
    ctx = _current_run.get()
    if ctx is None:
        return None
    if not ctx.closed:
        with ctx._lock:
            if not ctx.closed:
                ctx.closed = True
                ctx.ended_at = time.monotonic()
                if error is not None:
                    ctx.error = error
        _safe_emit_run_end(ctx)
        LOGGER.debug(
            "close_run run_id=%s steps=%d verdict=%s tokens_in=%d tokens_out=%d",
            ctx.run_id[:8],
            len(ctx.steps),
            ctx.worst_verdict,
            sum(s.tokens_in for s in ctx.steps),
            sum(s.tokens_out for s in ctx.steps),
        )
    # Restore parent so nested wraps unwind correctly:
    # parent.start -> child.start -> child.end (this call) -> parent.end (next close)
    _current_run.set(ctx.parent)
    return ctx


def _stamp_step_event(
    ctx: RunContext,
    event: dict[str, Any],
    *,
    seq: int,
    kind: StepKind,
) -> dict[str, Any]:
    ev_copy = dict(event)
    ev_copy["run_id"] = ctx.run_id
    ev_copy["seq"] = seq
    ev_copy["kind"] = kind
    ev_copy["trace_id"] = ctx.trace_id
    if ctx.agent_id:
        ev_copy.setdefault("agent_id", ctx.agent_id)
    if ctx.agent_name:
        ev_copy.setdefault("app", ctx.agent_name)
    return ev_copy


def _make_run_step(
    ctx: RunContext,
    *,
    seq: int,
    kind: StepKind,
    event: dict[str, Any],
    started_at: float,
    ended_at: float,
) -> RunStep:
    step = RunStep(
        seq=seq,
        kind=kind,
        started_at=started_at,
        ended_at=ended_at,
        latency_ms=_step_latency_ms(
            event, started_at=started_at, ended_at=ended_at,
        ),
        tokens_in=int(event.get("tokens_in") or 0),
        tokens_out=int(event.get("tokens_out") or 0),
        policy_tokens_in=int(event.get("policy_tokens_in") or 0),
        policy_tokens_out=int(event.get("policy_tokens_out") or 0),
        cost_usd=float(event.get("cost_usd") or 0.0),
        verdict=str(event.get("verdict") or "allow"),
        model=event.get("model"),
        target=event.get("target"),
        tool_name=event.get("tool_name"),
        event=_stamp_step_event(ctx, event, seq=seq, kind=kind),
    )
    return step


def _step_latency_ms(
    event: dict[str, Any],
    *,
    started_at: float,
    ended_at: float | None,
) -> int:
    """Resolve a step's model/tool latency from its audit event.

    The event's own ``latency_ms`` is authoritative when present:
    patches stamp it from the ``forward()`` call alone (and stamp
    an explicit ``0`` when an input-side block means the provider
    was never contacted). Only when the event carries no value do
    we fall back to the step's own timestamps — that fallback spans
    the whole gate (policy + model), which is the best available
    signal for legacy callers that never stamped a latency.
    """
    raw = event.get("latency_ms")
    if raw is not None:
        try:
            return max(0, int(raw))
        except (TypeError, ValueError):
            pass
    end = ended_at if ended_at is not None else time.monotonic()
    return max(0, int((end - started_at) * 1000))


def _run_worst_verdict_from_steps(ctx: RunContext) -> str:
    worst = "allow"
    for s in ctx.steps:
        worst = _worse(worst, s.verdict)
    return worst


def _append_step_unlocked(
    ctx: RunContext,
    *,
    event: dict[str, Any],
    kind: StepKind,
    started_at: float | None,
    ended_at: float | None,
) -> RunStep:
    """Append a step while ``ctx._lock`` is already held."""
    seq = len(ctx.steps)
    sta = started_at if started_at is not None else time.monotonic()
    end = ended_at if ended_at is not None else time.monotonic()
    step = _make_run_step(ctx, seq=seq, kind=kind, event=event, started_at=sta, ended_at=end)
    ctx.steps.append(step)
    ctx.worst_verdict = _worse(ctx.worst_verdict, step.verdict)
    if seq == 0 and ctx.prompt_text is None:
        ctx.prompt_text = event.get("prompt_preview")
    return step


def append_step(
    *,
    event: dict[str, Any],
    kind: StepKind = "model_call",
    started_at: float | None = None,
    ended_at: float | None = None,
) -> RunStep | None:
    """Append a finished step to the current run; emit ``run.step``.

    Returns the step that was added, or ``None`` when no run is open
    (caller should fall back to legacy enqueue). The event MUST already
    carry the full audit-row fields (verdict, prompt_decision, tokens,
    etc.) that the legacy single-row event used to ship — we just stamp
    ``run_id`` / ``seq`` on top.
    """
    ctx = _current_run.get()
    if ctx is None or ctx.closed:
        return None
    with ctx._lock:
        step = _append_step_unlocked(
            ctx, event=event, kind=kind, started_at=started_at, ended_at=ended_at,
        )
    _safe_emit_run_step(ctx, step)
    return step


def append_initial_model_call_step(
    *,
    event: dict[str, Any],
    started_at: float | None = None,
) -> RunStep | None:
    """Append seq 0 ``model_call`` when the run is still empty.

    Used by ``claude_agent_sdk`` (and similar) so the dashboard timeline
    lists input policy + model *before* tool rows while tool steps are
    still being streamed. The row is later finalized in-place via
    :func:`finalize_or_append_model_call_step`.
    """
    ctx = _current_run.get()
    if ctx is None or ctx.closed:
        return None
    with ctx._lock:
        if ctx.steps:
            return None
        sta = started_at if started_at is not None else time.monotonic()
        step = _append_step_unlocked(
            ctx,
            event=event,
            kind="model_call",
            started_at=sta,
            ended_at=sta,
        )
    _safe_emit_run_step(ctx, step)
    return step


def finalize_or_append_model_call_step(
    *,
    event: dict[str, Any],
    started_at: float | None = None,
    ended_at: float | None = None,
) -> RunStep | None:
    """Finalize placeholder seq 0 ``model_call`` or append a new one.

    When :func:`append_initial_model_call_step` created the first step,
    this merges the terminal audit fields (tokens, output policy,
    latency, …) into that row and re-emits ``run.step`` with the same
    ``seq``. Otherwise falls back to :func:`append_step`.
    """
    ctx = _current_run.get()
    if ctx is None or ctx.closed:
        return None
    with ctx._lock:
        if (
            ctx.steps
            and ctx.steps[0].kind == "model_call"
            and ctx.steps[0].seq == 0
        ):
            step = ctx.steps[0]
            sta = started_at if started_at is not None else step.started_at
            end = ended_at if ended_at is not None else time.monotonic()
            step.started_at = sta
            step.ended_at = end
            step.latency_ms = _step_latency_ms(
                event, started_at=sta, ended_at=end,
            )
            step.tokens_in = int(event.get("tokens_in") or 0)
            step.tokens_out = int(event.get("tokens_out") or 0)
            step.policy_tokens_in = int(event.get("policy_tokens_in") or 0)
            step.policy_tokens_out = int(event.get("policy_tokens_out") or 0)
            step.cost_usd = float(event.get("cost_usd") or 0.0)
            step.verdict = str(event.get("verdict") or "allow")
            step.model = event.get("model")
            step.target = event.get("target")
            step.tool_name = event.get("tool_name")
            step.event = _stamp_step_event(ctx, event, seq=0, kind="model_call")
            ctx.worst_verdict = _run_worst_verdict_from_steps(ctx)
        else:
            step = _append_step_unlocked(
                ctx,
                event=event,
                kind="model_call",
                started_at=started_at,
                ended_at=ended_at,
            )
    _safe_emit_run_step(ctx, step)
    return step


def reset_for_tests() -> None:
    """Clear the run pointer — only used by the test reset fixture."""
    _current_run.set(None)


def finalize_run_in_place(
    ctx: RunContext, *, error: str | None = None
) -> None:
    """Mark ``ctx`` closed + emit ``run.end`` without touching the ContextVar.

    Used by callbacks scheduled on framework-handle completion
    (e.g. LlamaIndex ``WorkflowHandler._result_task.add_done_callback``)
    where the handle's work runs on a different asyncio task than
    the one that opened the run. The contextvar dance ``close_run``
    performs is the wrong tool there — the parent task has long
    since restored its own pointer, so we just need to flip
    ``closed=True`` and enqueue the ``run.end`` event for ``ctx``
    specifically.

    Idempotent — repeated calls with the same ``ctx`` are no-ops.
    """
    if ctx.closed:
        return
    with ctx._lock:
        if ctx.closed:
            return
        ctx.closed = True
        ctx.ended_at = time.monotonic()
        if error is not None:
            ctx.error = error
    _safe_emit_run_end(ctx)


# ── Wire-format helpers ─────────────────────────────────────────────


def _build_run_start_event(ctx: RunContext) -> dict[str, Any]:
    """Construct the ``run.start`` wire event."""
    from egisai._config import get_config
    from egisai._context import get_context

    cfg = get_config()
    user_ctx = get_context()
    return {
        "kind": "run.start",
        "event_id": ctx.event_id,
        "run_id": ctx.run_id,
        "trace_id": ctx.trace_id,
        "parent_run_id": ctx.parent_run_id,
        "framework": ctx.framework,
        "timestamp": _now_iso(),
        "org_id": cfg.org_id,
        "agent_id": ctx.agent_id,
        "app": ctx.agent_name or cfg.app,
        "env": cfg.env,
        "user_id": user_ctx.user_id,
        "user_role": user_ctx.user_role,
        "session_id": user_ctx.session_id,
        "workflow_id": user_ctx.workflow_id,
        "end_user_id": user_ctx.end_user_id,
        "prompt_preview": ctx.prompt_text,
        "identity_source": ctx.identity.source if ctx.identity else None,
        "identity_hash": ctx.identity.identity_hash if ctx.identity else None,
    }


def _build_run_end_event(ctx: RunContext) -> dict[str, Any]:
    """Construct the ``run.end`` wire event with aggregates."""
    from egisai._config import get_config
    from egisai._context import get_context

    cfg = get_config()
    user_ctx = get_context()
    tokens_in = sum(s.tokens_in for s in ctx.steps)
    tokens_out = sum(s.tokens_out for s in ctx.steps)
    policy_tokens_in = sum(s.policy_tokens_in for s in ctx.steps)
    policy_tokens_out = sum(s.policy_tokens_out for s in ctx.steps)
    cost_usd = sum(s.cost_usd for s in ctx.steps)
    # Run latency = SUM of per-step model/tool latencies — the same
    # contract the backend's step-reconciliation path and the
    # dashboard's "Model" latency row assume. Pre-0.41.1 this shipped
    # the run's whole wall clock (open→close), which the backend then
    # trusted over its own step sums; for a run blocked at the input
    # policy that wall clock is pure governance time, so the dashboard
    # showed nonzero "Model" latency for a model that was never
    # called. Wall clock is still derivable server-side from the
    # run's ``started_at`` / ``ended_at`` timestamps.
    latency_ms = sum(s.latency_ms for s in ctx.steps)
    # Primary model = most frequent model across model_call steps.
    model_counts: dict[str, int] = {}
    for s in ctx.steps:
        if s.kind == "model_call" and s.model:
            model_counts[s.model] = model_counts.get(s.model, 0) + 1
    primary_model = (
        max(model_counts, key=lambda k: model_counts[k]) if model_counts else None
    )
    return {
        "kind": "run.end",
        "event_id": ctx.event_id,
        "run_id": ctx.run_id,
        "trace_id": ctx.trace_id,
        "parent_run_id": ctx.parent_run_id,
        "framework": ctx.framework,
        "timestamp": _now_iso(),
        "org_id": cfg.org_id,
        "agent_id": ctx.agent_id,
        "app": ctx.agent_name or cfg.app,
        "env": cfg.env,
        "user_id": user_ctx.user_id,
        "session_id": user_ctx.session_id,
        "workflow_id": user_ctx.workflow_id,
        "end_user_id": user_ctx.end_user_id,
        "step_count": len(ctx.steps),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "policy_tokens_in": policy_tokens_in,
        "policy_tokens_out": policy_tokens_out,
        "cost_usd": cost_usd,
        "latency_ms": latency_ms,
        "verdict": ctx.worst_verdict,
        "prompt_preview": ctx.prompt_text,
        "final_text": ctx.final_text,
        "model": primary_model,
        "error": ctx.error,
    }


# ── Internal helpers ────────────────────────────────────────────────


_VERDICT_RANK = {"allow": 0, "sanitize": 1, "block": 2}


def _worse(a: str, b: str) -> str:
    """Return whichever verdict is stricter."""
    if _VERDICT_RANK.get(b, 0) > _VERDICT_RANK.get(a, 0):
        return b
    return a


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_emit_run_start(ctx: RunContext) -> None:
    """Enqueue ``run.start`` — fail-open per SDK design rule 5."""
    try:
        from egisai._logger import enqueue
        enqueue(_build_run_start_event(ctx))
    except Exception:  # noqa: BLE001
        LOGGER.debug("run.start emit failed", exc_info=True)


def _safe_emit_run_step(ctx: RunContext, step: RunStep) -> None:
    """Enqueue a step event."""
    try:
        from egisai._logger import enqueue
        # The step event carries its own ``kind=run.step`` envelope plus
        # the legacy audit-row fields the backend already knows how to
        # ingest. The backend dispatches on ``kind`` and inserts into
        # ``run_steps`` instead of the legacy single-row path.
        ev = dict(step.event)
        ev["kind"] = "run.step"
        # ``step_kind`` is the kind of step (model_call/tool_call/...);
        # ``kind`` is the wire envelope kind (run.start/run.step/run.end).
        ev["step_kind"] = step.kind
        ev["run_id"] = ctx.run_id
        ev["seq"] = step.seq
        ev["parent_run_id"] = ctx.parent_run_id
        ev["framework"] = ctx.framework
        # ``step.latency_ms`` already honours the patch-stamped value
        # (model-call time only; explicit 0 for input-side blocks) and
        # only falls back to the step timestamps when no value was
        # stamped. Recomputing from timestamps here — as pre-0.41.1
        # code did — clobbered the patch's number with the whole gate's
        # wall clock, so blocked calls that never reached the provider
        # showed the policy-evaluation time as "model latency" on the
        # dashboard.
        ev["latency_ms"] = step.latency_ms
        enqueue(ev)
    except Exception:  # noqa: BLE001
        LOGGER.debug("run.step emit failed", exc_info=True)


def _safe_emit_run_end(ctx: RunContext) -> None:
    """Enqueue ``run.end`` — fail-open."""
    try:
        from egisai._logger import enqueue
        enqueue(_build_run_end_event(ctx))
    except Exception:  # noqa: BLE001
        LOGGER.debug("run.end emit failed", exc_info=True)


# ── Convenience for tests / direct callers ─────────────────────────


def open_run_from_current_identity(
    *,
    framework: str,
    prompt_text: str | None = None,
) -> RunContext:
    """Open a run using whichever identity is currently on the stack.

    Used by ``gate_call`` when it auto-opens an ephemeral run for a
    raw LLM call (no framework wrap above us). Resolves the identity
    once via ``current_identity()`` and locks it for the run.
    """
    return open_run(
        framework=framework,
        identity=current_identity(),
        prompt_text=prompt_text,
    )
