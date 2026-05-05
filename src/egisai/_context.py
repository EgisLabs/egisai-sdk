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

_agent_id_cache: dict[str, str] = {}
_agent_cache_lock = threading.Lock()


def _resolve_agent_id(name: str) -> str | None:
    """Get-or-fetch the platform agent_id for a role name.

    Returns ``None`` on failure or when the SDK isn't initialised
    yet; events are still logged, just without ``agent_id``.
    """
    cached = _agent_id_cache.get(name)
    if cached:
        return cached

    with _agent_cache_lock:
        cached = _agent_id_cache.get(name)
        if cached:
            return cached
        try:
            from egisai._backend import ensure_agent
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
            payload = ensure_agent(name=name)
            agent_id = payload.get("id")
            if isinstance(agent_id, str) and agent_id:
                _agent_id_cache[name] = agent_id
                created = bool(payload.get("created"))
                if created:
                    LOGGER.info(
                        "[egisai] registered sub-agent %r (id=%s…) — "
                        "visible on dashboard now",
                        name,
                        agent_id[:8],
                    )
                else:
                    LOGGER.debug(
                        "[egisai] resolved sub-agent %r (id=%s…)",
                        name,
                        agent_id[:8],
                    )
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
) -> None:
    """Attach request-level metadata to all subsequent governed calls.

    ``None`` values leave existing fields intact. ``agent`` is a
    friendly role name registered with the platform on first use.
    ``agent_id`` is an escape hatch for callers that already know
    the UUID.
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
        )
    )


def get_context() -> EgisaiContext:
    return _ctx.get()


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
