"""Shared helper for framework-identity patches.

Every agentic framework patch follows the same recipe:

1. Detect the framework's entry point (the function that "starts an
   agent's invocation"). Examples: ``Runner.run`` for OpenAI Agents,
   ``Pregel.invoke`` for LangGraph, ``AgentExecutor.invoke`` for
   LangChain.
2. From the entry-point's arguments, derive an identity bundle —
   either an explicit ``name`` (Tier 2A) or a composite hash of the
   agent's full definition (Tier 2B).
3. Push the resolved :class:`IdentityRecord` onto the identity stack.
4. Run the wrapped framework call.
5. Pop on exit.

This module encapsulates the boilerplate so each framework patch is
just a *describe-the-framework* function plus a handful of wiring
imports. Adding a new framework is intentionally a small, isolated
diff — the surface area for a regression is one new file under
``_patches/`` instead of a tweak to the gate.

All patches are import-guarded: ``apply()`` silently returns ``False``
when the framework isn't installed.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from contextvars import copy_context
from typing import Any

from egisai._auto_agent import (
    IdentityRecord,
    _ensure_agent_id,
    _hash_bundle,
    identity_scope,
)

LOGGER = logging.getLogger("egisai.patches.framework")


# ── Identity construction helpers ───────────────────────────────────


def make_identity(
    *,
    source: str,
    display_name: str,
    bundle: tuple[Any, ...],
) -> IdentityRecord | None:
    """Build an IdentityRecord from a framework patch's bundle.

    ``source`` is a controlled-vocabulary token from
    ``egisai._auto_agent.IdentitySource`` (e.g.
    ``"framework:openai_agents"``). ``bundle`` is the tuple of values
    we hash to produce the ``identity_hash`` — typically
    ``(framework_name, agent_name)`` for Tier 2A or
    ``(framework_name, system_prompt, sorted_tools, model)`` for 2B.

    Returns ``None`` when the backend can't be reached so the caller
    falls through to the next tier (the gate's resolver re-tries).
    The fail-open semantics mirror the rest of the SDK.
    """
    name = (display_name or "").strip() or "agent"
    if len(name) > 80:
        name = name[:77].rstrip() + "…"
    digest = _hash_bundle(bundle)
    identity_key = f"{source}:{digest}"
    agent_id = _ensure_agent_id(
        display_name=name,
        identity_key=identity_key,
        identity_hash=digest,
        source=source,
    )
    if agent_id is None:
        return None
    return IdentityRecord(
        agent_id=agent_id,
        display_name=name,
        identity_key=identity_key,
        identity_hash=digest,
        source=source,  # type: ignore[arg-type]
        push_to_stack=True,
    )


# ── Generic entry-point patch ───────────────────────────────────────


def wrap_sync_entrypoint(
    orig: Callable[..., Any],
    derive: Callable[..., IdentityRecord | None],
) -> Callable[..., Any]:
    """Wrap a sync framework entry point so it pushes identity first.

    ``derive(self, *args, **kwargs)`` returns the IdentityRecord or
    ``None``. The wrapper pushes the record and runs the original;
    if ``derive`` raises, the original still runs (fail-open).
    """

    @functools.wraps(orig)
    def wrapped(self_or_first: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            record = derive(self_or_first, *args, **kwargs)
        except Exception:  # noqa: BLE001
            LOGGER.debug(
                "framework identity derive failed", exc_info=True,
            )
            record = None
        if record is None:
            return orig(self_or_first, *args, **kwargs)
        with identity_scope(record):
            return orig(self_or_first, *args, **kwargs)

    setattr(wrapped, "__egisai_wrapped__", True)
    return wrapped


def wrap_async_entrypoint(
    orig: Callable[..., Awaitable[Any]],
    derive: Callable[..., IdentityRecord | None],
) -> Callable[..., Awaitable[Any]]:
    """Async sibling of :func:`wrap_sync_entrypoint`."""

    @functools.wraps(orig)
    async def wrapped(self_or_first: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            record = derive(self_or_first, *args, **kwargs)
        except Exception:  # noqa: BLE001
            LOGGER.debug(
                "framework identity derive (async) failed", exc_info=True,
            )
            record = None
        if record is None:
            return await orig(self_or_first, *args, **kwargs)
        with identity_scope(record):
            return await orig(self_or_first, *args, **kwargs)

    setattr(wrapped, "__egisai_wrapped__", True)
    return wrapped


def wrap_async_iter_entrypoint(
    orig: Callable[..., AsyncIterator[Any]],
    derive: Callable[..., IdentityRecord | None],
) -> Callable[..., AsyncIterator[Any]]:
    """Wrap an async-generator framework entry point.

    Critical: async generators run yields on the asyncio task that
    advances them, which is NOT the task that called ``__anext__``
    in the general case. We use ``copy_context`` so the identity
    stack carries across each yield's task boundary. Without this,
    ``current_identity()`` reads empty inside the inner LLM call
    and Tier 5 fingerprinting fires (double-counting the agent).
    """

    @functools.wraps(orig)
    async def wrapped(self_or_first: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        try:
            record = derive(self_or_first, *args, **kwargs)
        except Exception:  # noqa: BLE001
            LOGGER.debug(
                "framework identity derive (async-iter) failed", exc_info=True,
            )
            record = None
        if record is None:
            async for item in orig(self_or_first, *args, **kwargs):
                yield item
            return
        ctx = copy_context()
        with identity_scope(record):
            it = orig(self_or_first, *args, **kwargs)
            while True:
                try:
                    # Resume the generator in the captured context so
                    # the identity scope inherits per yield.
                    item = await ctx.run(it.__anext__)
                except StopAsyncIteration:
                    break
                yield item

    setattr(wrapped, "__egisai_wrapped__", True)
    return wrapped


def wrap_sync_iter_entrypoint(
    orig: Callable[..., Iterator[Any]],
    derive: Callable[..., IdentityRecord | None],
) -> Callable[..., Iterator[Any]]:
    """Wrap a sync-generator framework entry point (e.g. .stream())."""

    @functools.wraps(orig)
    def wrapped(self_or_first: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
        try:
            record = derive(self_or_first, *args, **kwargs)
        except Exception:  # noqa: BLE001
            record = None
        if record is None:
            yield from orig(self_or_first, *args, **kwargs)
            return
        with identity_scope(record):
            yield from orig(self_or_first, *args, **kwargs)

    setattr(wrapped, "__egisai_wrapped__", True)
    return wrapped


# ── Class-method patcher ────────────────────────────────────────────


def patch_method(
    module_path: str,
    class_name: str,
    method_name: str,
    *,
    derive: Callable[..., IdentityRecord | None],
    kind: str = "sync",
) -> bool:
    """Install a wrapped method on a third-party class.

    ``kind`` is one of ``"sync"``, ``"async"``, ``"async_iter"``, or
    ``"sync_iter"``. Returns ``True`` when the patch lands, ``False``
    when the target isn't importable / doesn't have the method (we
    silently degrade so the framework being absent never breaks the
    SDK).
    """
    try:
        module = __import__(module_path, fromlist=[class_name])
    except Exception:  # noqa: BLE001
        return False
    cls = getattr(module, class_name, None)
    if cls is None:
        return False
    orig = getattr(cls, method_name, None)
    if not callable(orig):
        return False
    if getattr(orig, "__egisai_wrapped__", False):
        return True
    if kind == "sync":
        wrapped: Any = wrap_sync_entrypoint(orig, derive)
    elif kind == "async":
        wrapped = wrap_async_entrypoint(orig, derive)
    elif kind == "async_iter":
        wrapped = wrap_async_iter_entrypoint(orig, derive)
    elif kind == "sync_iter":
        wrapped = wrap_sync_iter_entrypoint(orig, derive)
    else:
        raise ValueError(f"unknown kind: {kind!r}")
    setattr(cls, method_name, wrapped)
    return True
