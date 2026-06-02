"""Per-request execution context — survives across nested SDK calls.

Stored on ``ContextVar`` so asyncio tasks and threads inherit cleanly.
Use ``egisai.set_context(user_id=..., agent="Python Developer", ...)``
from your request handler to attach request-level metadata to every
governed call. The first time a new ``agent`` name is seen the SDK
registers it on the platform; subsequent uses are cache hits.
"""

from __future__ import annotations

import contextvars
import logging
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

LOGGER = logging.getLogger("egisai.context")


@dataclass(frozen=True)
class EgisaiContext:
    user_id: str | None = None
    user_role: str | None = None
    agent_id: str | None = None
    agent_name: str | None = None
    session_id: str | None = None
    workflow_id: str | None = None
    # Opaque, operator-supplied identifier for the *end-user* the
    # agent is currently serving. The backend hashes it on intake;
    # the SDK is encouraged to ship a hash already (e.g. a
    # SHA-256 of the customer-id) so a real PII paste never
    # leaves the process. Powers per-end-user behavioral roll-ups
    # inside the Agent Identity modal.
    end_user_id: str | None = None


_ctx: contextvars.ContextVar[EgisaiContext] = contextvars.ContextVar(
    "egisai_ctx", default=EgisaiContext()  # noqa: B039 — frozen dataclass is immutable
)
_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "egisai_trace", default=""
)
_source: contextvars.ContextVar[str] = contextvars.ContextVar(
    "egisai_source", default=""
)
_policy_checked: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "egisai_checked", default=False
)
# Per-call accumulator for tokens consumed by policy-side LLM calls
# (e.g. the ``semantic_guard`` judge).
_policy_usage: contextvars.ContextVar[tuple[int, int]] = contextvars.ContextVar(
    "egisai_policy_usage", default=(0, 0)
)

# Per-call accumulator for **init-time wall-clock** the gate paid that
# was previously bundled into ``policy_latency_ms``. The canonical
# contributor is the one-shot PII NER (Presidio + spaCy) warm-up wait
# inside ``_maybe_wait_for_pii_analyzer`` — it can take up to
# ``EGISAI_PII_WARMUP_TIMEOUT_SECS`` (default 2 s) on call #1 of a
# fresh process. Counting that as "policy enforcement latency"
# misattributes a one-shot library load to per-call governance work
# and makes the dashboard's ``policy_latency_ms`` number look
# permanently inflated for short-lived workloads.
#
# The accumulator is filled by code paths that pay the wait, and the
# gate (``_run_input_phase`` / ``_run_output_phase``) reads + clears
# it BEFORE stamping ``policy_latency_ms`` on the audit event. The
# init time is then surfaced separately on ``ev["init_latency_ms"]``
# so operators can still see the cold-start cost — just not in the
# governance column.
_init_latency_ms: contextvars.ContextVar[int] = contextvars.ContextVar(
    "egisai_init_latency_ms", default=0
)

# Compat alias — the unified cache lives in ``_auto_agent`` now. Tests
# in ``conftest.py`` clear ``_agent_id_cache`` explicitly so we keep
# the symbol bound to the new shared dict to preserve their semantics.
_agent_id_cache: dict[str, str] = {}
_agent_cache_lock = threading.Lock()


def _resolve_agent_id(name: str) -> str | None:
    """Get-or-fetch the platform agent_id for an explicit role name.

    Delegates to the unified resolver in ``_auto_agent`` so explicit
    ``set_context(agent="X")`` calls and auto-detected agents share
    one cache. The old per-name dict (``_agent_id_cache``) is still
    here as an in-process fast path *and* mirrors the unified cache
    so legacy tests that wipe it between cases see fresh state.

    Returns ``None`` on failure or when the SDK isn't initialised yet;
    the call proceeds and the event is attributed to the API-key-bound
    agent (if any) per fail-open philosophy.
    """
    cached = _agent_id_cache.get(name)
    if cached:
        return cached

    with _agent_cache_lock:
        cached = _agent_id_cache.get(name)
        if cached:
            return cached
        try:
            from egisai._auto_agent import _ensure_agent_id, _hash_bundle
            from egisai._config import get_config_optional

            cfg = get_config_optional()
            if cfg is None:
                LOGGER.warning(
                    "egisai.set_context(agent=%r) was called before "
                    "egisai.init() — ignoring. Move the init() call to the "
                    "top of your script.",
                    name,
                )
                return None
            agent_id = _ensure_agent_id(
                display_name=name,
                identity_key=f"explicit:{name}",
                identity_hash=_hash_bundle(("explicit", name)),
                source="explicit",
            )
            if agent_id is not None:
                _agent_id_cache[name] = agent_id
            return agent_id
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "[egisai] could not register sub-agent %r: %s — events "
                "for this call will be attributed to the main agent",
                name,
                exc,
                exc_info=True,
            )
        return None


def set_context(
    *,
    user_id: str | None = None,
    user_role: str | None = None,
    agent: str | None = None,
    agent_id: str | None = None,
    session_id: str | None = None,
    workflow_id: str | None = None,
    end_user_id: str | None = None,
) -> None:
    """Attach request-level metadata to all subsequent governed calls.

    ``None`` values leave existing fields intact. ``agent`` is a
    friendly role name registered with the platform on first use.
    ``agent_id`` is an escape hatch for callers that already know
    the UUID.

    ``end_user_id`` (added in 0.13.0) is an opaque identifier for
    the *end-user* the agent is currently serving. The platform
    hashes it on intake; the SDK already encourages callers to
    ship a hash (e.g. ``hashlib.sha256(customer_id.encode()).hexdigest()``)
    so a real customer-id never leaves the process. Powers per-end-user
    behavioral roll-ups inside the Agent Identity modal.
    """
    cur = _ctx.get()
    resolved_id = agent_id
    resolved_name = cur.agent_name
    if agent is not None:
        resolved_name = agent
        resolved_id = _resolve_agent_id(agent) or resolved_id
    _ctx.set(
        EgisaiContext(
            user_id=user_id if user_id is not None else cur.user_id,
            user_role=user_role if user_role is not None else cur.user_role,
            agent_id=resolved_id if resolved_id is not None else cur.agent_id,
            agent_name=resolved_name,
            session_id=session_id if session_id is not None else cur.session_id,
            workflow_id=workflow_id if workflow_id is not None else cur.workflow_id,
            end_user_id=(
                end_user_id if end_user_id is not None else cur.end_user_id
            ),
        )
    )


def get_context() -> EgisaiContext:
    return _ctx.get()


@contextmanager
def agent(name: str) -> Iterator[str]:
    """Pin an explicit agent identity for the duration of the block.

    The block's identity wins outright over every auto-detection tier
    (OTEL spans, framework patches, system-prompt hashing, …). Use
    when you have a long-lived agent role that the SDK can't infer
    from the call payload — typically because you're using a framework
    we haven't shipped a patch for, or because your prompt is
    generated dynamically and you want one stable label::

        with egisai.agent("Triage"):
            client.chat.completions.create(...)

    Re-entrant: nested ``with`` blocks form an identity stack and the
    innermost one wins per call. The pushed identity is also visible
    to ``_auto_agent.current_identity()`` so framework patches inside
    the block inherit it instead of re-deriving.
    """
    # Resolve the backend agent_id first so policy attribution inside
    # the block knows the id. Failures are swallowed (fail-open) and
    # the block still pins the display name.
    from egisai._auto_agent import (
        IdentityRecord,
        _hash_bundle,
        push_identity,
        reset_identity,
    )

    resolved_id = _resolve_agent_id(name)
    record = IdentityRecord(
        agent_id=resolved_id,
        display_name=name,
        identity_key=f"explicit:{name}",
        identity_hash=_hash_bundle(("explicit", name)),
        source="explicit",
        push_to_stack=True,
    )
    token = push_identity(record)
    try:
        yield resolved_id or ""
    finally:
        reset_identity(token)


def register_agent(name: str) -> str | None:
    """Eagerly register an agent and return its platform agent_id.

    Equivalent to ``set_context(agent=…)`` without mutating the
    current ``EgisaiContext``. Useful when an operator wants to
    pre-create the dashboard row at startup (so it appears under
    Agents the moment the app boots, before any traffic flows).
    Returns ``None`` if the SDK isn't initialised or the backend
    is unreachable — never raises.
    """
    return _resolve_agent_id(name)


def ensure_trace_id() -> str:
    tid = _trace_id.get()
    if tid:
        return tid
    tid = uuid.uuid4().hex
    _trace_id.set(tid)
    return tid


def reset_trace() -> str:
    tid = uuid.uuid4().hex
    _trace_id.set(tid)
    _policy_checked.set(False)
    return tid


def get_source() -> str:
    return _source.get()


def set_source(s: str) -> None:
    _source.set(s)


def get_policy_checked() -> bool:
    return _policy_checked.get()


def set_policy_checked(v: bool) -> None:
    _policy_checked.set(v)


def reset_policy_usage() -> None:
    """Zero out the per-call policy-step token accumulator."""
    _policy_usage.set((0, 0))


def add_policy_usage(tokens_in: int, tokens_out: int) -> None:
    """Add tokens to the per-call policy-step accumulator."""
    cur_in, cur_out = _policy_usage.get()
    _policy_usage.set((cur_in + max(0, int(tokens_in or 0)),
                       cur_out + max(0, int(tokens_out or 0))))


def get_policy_usage() -> tuple[int, int]:
    """Return ``(tokens_in, tokens_out)`` consumed by the policy step."""
    return _policy_usage.get()


def reset_init_latency() -> None:
    """Zero out the per-call init-time accumulator.

    Called at the top of the gate's policy step so a previous turn's
    accumulator (or a stray contextvar default) doesn't bleed into
    this call's measurement.
    """
    _init_latency_ms.set(0)


def add_init_latency(elapsed_ms: int) -> None:
    """Add ``elapsed_ms`` to the per-call init-time accumulator.

    Code paths that pay one-shot warm-up wall-clock (PII NER loader
    today; future analyzers may add to this) call this helper with
    however many milliseconds they actually waited. The gate later
    pops the total via :func:`get_init_latency` and surfaces it on
    ``ev["init_latency_ms"]`` instead of bundling it into
    ``policy_latency_ms``.
    """
    if elapsed_ms <= 0:
        return
    _init_latency_ms.set(_init_latency_ms.get() + int(elapsed_ms))


def get_init_latency() -> int:
    """Return ms accumulated in the per-call init-time accumulator."""
    return _init_latency_ms.get()
