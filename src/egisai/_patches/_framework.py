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
import inspect
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
from egisai._run import close_run, open_run

LOGGER = logging.getLogger("egisai.patches.framework")


# ── Run lifecycle helpers ───────────────────────────────────────────


def _framework_name(orig: Callable[..., Any]) -> str:
    """Derive a short framework token from the wrapped callable.

    Used to stamp the ``framework`` field on the run event. Falls back
    to the qualified name's first segment when the module path doesn't
    obviously map to a framework token.
    """
    mod = getattr(orig, "__module__", "") or ""
    if mod.startswith("agents."):
        return "openai_agents"
    if mod.startswith("claude_agent_sdk"):
        return "claude_agent_sdk"
    if mod.startswith("langgraph"):
        return "langgraph"
    if mod.startswith("langchain"):
        return "langchain"
    if mod.startswith("crewai"):
        return "crewai"
    if mod.startswith("autogen"):
        return "autogen"
    if mod.startswith("agno"):
        return "agno"
    if mod.startswith("strands"):
        return "strands"
    if mod.startswith("smolagents"):
        return "smolagents"
    if mod.startswith("llama_index"):
        return "llamaindex"
    if mod.startswith("google_adk") or mod.startswith("google.adk"):
        return "google_adk"
    if mod.startswith("pydantic_ai"):
        return "pydantic_ai"
    return mod.split(".")[0] or "framework"


class _RunScope:
    """Open a run on enter, close on exit — identity-record aware."""

    def __init__(self, framework: str, record: IdentityRecord | None) -> None:
        self.framework = framework
        self.record = record
        self.opened = False

    def __enter__(self) -> _RunScope:
        if self.record is not None:
            open_run(framework=self.framework, identity=self.record)
            self.opened = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self.opened:
            close_run(error=repr(exc_val) if exc_val else None)


def _safe_derive(
    derive: Callable[..., IdentityRecord | None],
    self_or_first: Any,
    *args: Any,
    **kwargs: Any,
) -> IdentityRecord | None:
    """Run ``derive`` and swallow errors per fail-open philosophy."""
    try:
        return derive(self_or_first, *args, **kwargs)
    except Exception:  # noqa: BLE001
        LOGGER.debug("framework identity derive failed", exc_info=True)
        return None


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
    """Wrap a sync framework entry point so it opens a Run.

    On entry, derive the identity, open a Run (which locks the agent
    for every inner LLM call), and run the original under
    ``identity_scope``. On exit, close the Run — emits a single
    ``run.end`` audit event with aggregated tokens/latency/verdict
    across all the inner steps the gate captured.

    Fail-open: if derive raises or returns None, the original still
    runs (no run opened) — the SDK never breaks the user's call.
    """
    framework = _framework_name(orig)

    @functools.wraps(orig)
    def wrapped(self_or_first: Any, *args: Any, **kwargs: Any) -> Any:
        record = _safe_derive(derive, self_or_first, *args, **kwargs)
        if record is None:
            return orig(self_or_first, *args, **kwargs)
        with _RunScope(framework, record), identity_scope(record):
            return orig(self_or_first, *args, **kwargs)

    wrapped.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


def wrap_async_entrypoint(
    orig: Callable[..., Awaitable[Any]],
    derive: Callable[..., IdentityRecord | None],
) -> Callable[..., Awaitable[Any]]:
    """Async sibling of :func:`wrap_sync_entrypoint`."""
    framework = _framework_name(orig)

    @functools.wraps(orig)
    async def wrapped(self_or_first: Any, *args: Any, **kwargs: Any) -> Any:
        record = _safe_derive(derive, self_or_first, *args, **kwargs)
        if record is None:
            return await orig(self_or_first, *args, **kwargs)
        with _RunScope(framework, record), identity_scope(record):
            return await orig(self_or_first, *args, **kwargs)

    wrapped.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


def wrap_async_iter_entrypoint(
    orig: Callable[..., AsyncIterator[Any]],
    derive: Callable[..., IdentityRecord | None],
) -> Callable[..., AsyncIterator[Any]]:
    """Wrap an async-generator framework entry point.

    Critical: async generators run yields on the asyncio task that
    advances them, which is NOT the task that called ``__anext__``
    in the general case. We use ``copy_context`` so the identity
    stack AND the run scope carry across each yield's task boundary.
    Without this, ``current_identity()`` and ``current_run()`` both
    read empty inside the inner LLM call and Tier 5 fingerprinting
    fires (double-counting the agent).

    The Run is closed when the iterator exhausts OR when the caller
    breaks out of the loop early (Python invokes the generator's
    aclose on garbage collection).
    """
    framework = _framework_name(orig)

    @functools.wraps(orig)
    async def wrapped(self_or_first: Any, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        record = _safe_derive(derive, self_or_first, *args, **kwargs)
        if record is None:
            async for item in orig(self_or_first, *args, **kwargs):
                yield item
            return
        with _RunScope(framework, record), identity_scope(record):
            ctx = copy_context()
            it = orig(self_or_first, *args, **kwargs)
            try:
                while True:
                    try:
                        item = await ctx.run(it.__anext__)
                    except StopAsyncIteration:
                        break
                    yield item
            finally:
                aclose = getattr(it, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:  # noqa: BLE001
                        pass

    wrapped.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


def wrap_sync_iter_entrypoint(
    orig: Callable[..., Iterator[Any]],
    derive: Callable[..., IdentityRecord | None],
) -> Callable[..., Iterator[Any]]:
    """Wrap a sync-generator framework entry point (e.g. .stream())."""
    framework = _framework_name(orig)

    @functools.wraps(orig)
    def wrapped(self_or_first: Any, *args: Any, **kwargs: Any) -> Iterator[Any]:
        record = _safe_derive(derive, self_or_first, *args, **kwargs)
        if record is None:
            yield from orig(self_or_first, *args, **kwargs)
            return
        with _RunScope(framework, record), identity_scope(record):
            yield from orig(self_or_first, *args, **kwargs)

    wrapped.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


def wrap_polymorphic_entrypoint(
    orig: Callable[..., Any],
    derive: Callable[..., IdentityRecord | None],
) -> Callable[..., Any]:
    """Wrap a polymorphic framework entry point.

    Some upstream entry points are *plain* ``def`` functions that
    inspect their kwargs at runtime and return one of:

    - a **coroutine** to ``await``
      (e.g. ``agno.Agent.arun(stream=False)``)
    - an **async iterator** to ``async for``
      (e.g. ``agno.Agent.arun(stream=True)``,
       ``claude_agent_sdk.query(...)``)
    - a **sync iterator** to ``for``
      (e.g. ``agno.Agent.run(stream=True)``,
       ``smolagents.MultiStepAgent.run(stream=True)``)
    - a **plain value**
      (e.g. ``agno.Agent.run(stream=False)``,
       ``llamaindex.FunctionAgent.run()`` → ``WorkflowHandler``)

    The wrapper opens a Run that spans the *full lifetime* of the
    returned value — coroutine, async-gen, sync-gen, or plain value.
    For the plain-value case (LlamaIndex handlers, futures, etc.) we
    close the Run after the call returns; downstream awaits of the
    returned handle are out-of-scope for v1 (a follow-up will wrap
    those handles individually).
    """
    framework = _framework_name(orig)

    @functools.wraps(orig)
    def wrapped(self_or_first: Any, *args: Any, **kwargs: Any) -> Any:
        record = _safe_derive(derive, self_or_first, *args, **kwargs)
        if record is None:
            return orig(self_or_first, *args, **kwargs)
        # Open the run upfront — must close on every code path below.
        scope = _RunScope(framework, record)
        scope.__enter__()
        try:
            with identity_scope(record):
                result = orig(self_or_first, *args, **kwargs)
        except BaseException as exc:
            scope.__exit__(type(exc), exc, exc.__traceback__)
            raise

        if inspect.iscoroutine(result):
            async def _coro_scope() -> Any:
                try:
                    with identity_scope(record):
                        return await result
                finally:
                    scope.__exit__(None, None, None)
            return _coro_scope()
        if inspect.isasyncgen(result):
            async def _ag_scope() -> AsyncIterator[Any]:
                ctx = copy_context()
                try:
                    with identity_scope(record):
                        while True:
                            try:
                                item = await ctx.run(result.__anext__)
                            except StopAsyncIteration:
                                return
                            yield item
                finally:
                    aclose = getattr(result, "aclose", None)
                    if aclose is not None:
                        try:
                            await aclose()
                        except Exception:  # noqa: BLE001
                            pass
                    scope.__exit__(None, None, None)
            return _ag_scope()
        if inspect.isgenerator(result):
            def _gen_scope() -> Iterator[Any]:
                try:
                    with identity_scope(record):
                        yield from result
                finally:
                    scope.__exit__(None, None, None)
            return _gen_scope()
        # Plain value — close the run now. Downstream awaitable handles
        # (LlamaIndex WorkflowHandler) will be wrapped by a dedicated
        # proxy in v2; for v1 they emit their inner LLM steps under the
        # Tier 5 prompt-hash identity (back-compat with 0.17.x).
        scope.__exit__(None, None, None)
        return result

    wrapped.__egisai_wrapped__ = True  # type: ignore[attr-defined]
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

    ``kind`` is one of ``"sync"``, ``"async"``, ``"async_iter"``,
    ``"sync_iter"``, or ``"polymorphic"``. Use ``"polymorphic"`` when
    the upstream is a plain ``def`` whose return type depends on its
    kwargs (e.g. agno's ``stream=`` toggle, Claude Agent SDK's
    module-level ``query``) — see :func:`wrap_polymorphic_entrypoint`
    for the matrix.

    Returns ``True`` when the patch lands, ``False`` when the target
    isn't importable / doesn't have the method (we silently degrade
    so the framework being absent never breaks the SDK).
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
    elif kind == "polymorphic":
        wrapped = wrap_polymorphic_entrypoint(orig, derive)
    else:
        raise ValueError(f"unknown kind: {kind!r}")
    setattr(cls, method_name, wrapped)
    return True
