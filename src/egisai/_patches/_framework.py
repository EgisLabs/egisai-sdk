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
    push_identity,
    reset_identity,
)
from egisai._run import (
    _current_run,
    close_run,
    finalize_run_in_place,
    open_run,
)

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
    """Open a run on enter, close on exit — identity-record aware.

    Re-entry guard: if a Run is already open in the current
    ContextVar AND its identity matches what we'd otherwise open
    with (same ``identity_hash``), this scope is a no-op and the
    inner orig call inherits the parent's Run. This is the contract
    that keeps frameworks whose user-facing entry point dispatches
    through *another* wrapped entry point (e.g. LangGraph's
    ``Pregel.invoke`` calls ``self.stream`` internally; LlamaIndex's
    ``AgentWorkflow.run`` may invoke other wrapped Workflow
    methods) from producing a duplicate, empty parent Run alongside
    the real one. A *different* identity (sub-agent / handoff) still
    opens a child Run with ``parent_run_id`` wired up — that's the
    legitimate parent→child topology the dashboard renders.

    The same shape can hit users when a single agent invocation
    re-enters the framework's wrapped surface multiple times (e.g.
    ``Pregel.stream`` is consumed from ``Pregel.invoke`` AND from
    user code that wraps ``invoke``). Without the guard each layer
    materialises its own empty Run row that shares ``trace_id``
    but reports zero steps / zero tokens / empty ``prompt_text``,
    confusing every downstream aggregation (the dashboard's
    "average step count" tile, the billing token roll-up, the
    SOC 2 "what actually happened" investigation flow). The guard
    is the cheapest fix: identity equality is a single string
    compare, and any patch that wants its outer scope to dominate
    can simply ensure its derive() returns the same identity for
    every nested call.
    """

    def __init__(self, framework: str, record: IdentityRecord | None) -> None:
        self.framework = framework
        self.record = record
        self.opened = False

    def __enter__(self) -> _RunScope:
        if self.record is None:
            return self
        parent = _current_run.get()
        if (
            parent is not None
            and not parent.closed
            and parent.identity is not None
            and parent.identity.identity_hash == self.record.identity_hash
        ):
            # Already inside an open Run for the same logical agent —
            # the outer wrap (e.g. Pregel.invoke) owns the Run; the
            # inner wrap (e.g. Pregel.stream) just rides along. We
            # set ``opened=False`` so ``__exit__`` doesn't pop the
            # parent's Run out from under it on the inner unwind.
            return self
        open_run(framework=self.framework, identity=self.record)
        self.opened = True
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if not self.opened:
            return
        # SDK-raised block: ``_block_response`` in ``_patches._common``
        # has already dispatched a step with ``verdict='block'`` AND
        # stamped the matched-policy reason on the audit row before
        # raising ``PermissionError`` up the call stack. The Run's
        # rolled-up audit signature therefore carries the full block
        # context via ``run.verdict``/``prompt_decision`` already; if
        # we additionally stamp the PermissionError's repr on
        # ``run.error`` we'd falsely flag the Run as a runtime crash
        # (the dashboard treats non-NULL ``run.error`` as an
        # uncaught-exception marker, and the agents-test validator's
        # allowed-list reserves the field for NULL or one of the
        # canonical short-strings claude_agent_sdk uses on its own
        # close_run sites — ``"input policy block"`` etc.). Mirror
        # that shape here so every framework wrap that bottoms out
        # in the SDK's PermissionError ends up with the same audit
        # signature on a refused turn. Real unexpected exceptions
        # (framework crashes, network errors, programming bugs)
        # still propagate their full repr per the legacy contract.
        err: str | None
        if (
            isinstance(exc_val, PermissionError)
            and isinstance(getattr(exc_val, "args", (None,))[0], str)
            and exc_val.args[0].startswith("[egisai]")
        ):
            err = "policy block"
        elif exc_val is not None:
            err = repr(exc_val)
        else:
            err = None
        close_run(error=err)


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


def wrap_handler_entrypoint(
    orig: Callable[..., Any],
    derive: Callable[..., IdentityRecord | None],
) -> Callable[..., Any]:
    """Wrap a sync entry point that returns a long-lived awaitable handle.

    The canonical case is LlamaIndex's modern agent surface
    (``FunctionAgent.run``, ``ReActAgent.run``, ``CodeActAgent.run``,
    ``AgentWorkflow.run``) — all plain ``def`` methods that return a
    ``WorkflowHandler``. The handle is awaitable AND streamable via
    ``handle.stream_events()``; the actual workflow execution runs
    on ``handle._result_task`` which is an ``asyncio.Task`` created
    inside ``WorkflowHandler.__init__`` (i.e. inside this wrap's
    ``orig()`` call).

    Two things have to happen for inner LLM calls inside the
    workflow to attribute correctly:

    1. The workflow's internal asyncio tasks must capture our
       open ``RunContext`` and pushed identity on the
       ``_current_run`` / ``_identity_stack`` ContextVars at the
       moment they are created. ``open_run`` + ``push_identity``
       before ``orig()`` arranges that — the create_task() call
       captures the surrounding context.
    2. The ``RunContext`` must stay ``closed=False`` while those
       inner tasks actually run, which happens AFTER this wrap
       returns. ``wrap_sync_entrypoint`` would close the run as
       soon as the orig call returns the handle — far too early.

    So we open the run, run ``orig`` (which both creates the
    workflow's tasks and constructs the handle), and then schedule
    the run-finalisation on the handle's ``_result_task`` completion.
    The done-callback captures the current ContextVar state at
    ``add_done_callback`` time (containing our open run), so when
    the callback fires it can finalise that specific
    ``RunContext`` directly. The parent task that called us has
    its contextvars restored before we return, so user code after
    ``agent.run(...)`` sees a clean stack.

    If the returned value lacks ``_result_task`` (older LlamaIndex,
    fakes used in unit tests, or a future shape change), we fall
    back to the sync semantics ``wrap_sync_entrypoint`` provides —
    open + close around the call. That keeps the
    ``test_llamaindex_function_agent_returns_handle`` contract
    intact and never leaves a dangling open run.
    """
    framework = _framework_name(orig)

    @functools.wraps(orig)
    def wrapped(self_or_first: Any, *args: Any, **kwargs: Any) -> Any:
        record = _safe_derive(derive, self_or_first, *args, **kwargs)
        if record is None:
            return orig(self_or_first, *args, **kwargs)

        parent_run = _current_run.get()
        identity_token = push_identity(record)
        run_ctx = open_run(framework=framework, identity=record)
        try:
            result = orig(self_or_first, *args, **kwargs)
        except BaseException as exc:
            finalize_run_in_place(run_ctx, error=repr(exc))
            _current_run.set(parent_run)
            reset_identity(identity_token)
            raise

        result_task = getattr(result, "_result_task", None)
        if result_task is None or not hasattr(result_task, "add_done_callback"):
            # No long-lived task to hook — close immediately, same
            # as ``wrap_sync_entrypoint``. Preserves the contract
            # exercised by ``test_llamaindex_function_agent_returns_handle``
            # where the fake handle has no ``_result_task``.
            close_run()
            reset_identity(identity_token)
            return result

        # ``add_done_callback`` captures ``contextvars.copy_context()``
        # right now — at this point ``_current_run`` is ``run_ctx``
        # and ``_identity_stack`` has our record pushed. The captured
        # snapshot is independent of the parent task's state, so we
        # can safely restore the parent's pointers below without
        # affecting what the callback sees.
        def _on_done(t: Any) -> None:
            err: str | None = None
            try:
                if t.cancelled():
                    err = "cancelled"
                else:
                    exc = t.exception()
                    if exc is not None:
                        err = repr(exc)
            except BaseException:  # noqa: BLE001 — fail-open
                pass
            finalize_run_in_place(run_ctx, error=err)

        try:
            result_task.add_done_callback(_on_done)
        except Exception:  # noqa: BLE001
            # If hooking fails for any reason, close synchronously
            # to avoid leaving a dangling open run forever.
            finalize_run_in_place(run_ctx)

        # Restore parent contextvar state so the user's code after
        # ``agent.run(...)`` sees the same identity stack and run
        # pointer it had before the wrap. The workflow's inner
        # tasks already captured the pre-restore state when ``orig``
        # created them, so their LLM calls keep attributing to
        # ``run_ctx``.
        _current_run.set(parent_run)
        reset_identity(identity_token)
        return result

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
    ``"sync_iter"``, ``"polymorphic"``, or ``"handler"``. Use
    ``"polymorphic"`` when the upstream is a plain ``def`` whose
    return type depends on its kwargs (e.g. agno's ``stream=``
    toggle, Claude Agent SDK's module-level ``query``) — see
    :func:`wrap_polymorphic_entrypoint` for the matrix. Use
    ``"handler"`` for entry points that return a long-lived
    awaitable handle whose inner work runs on a separate task
    (LlamaIndex's ``WorkflowHandler`` is the canonical case) —
    see :func:`wrap_handler_entrypoint`.

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
    elif kind == "handler":
        wrapped = wrap_handler_entrypoint(orig, derive)
    else:
        raise ValueError(f"unknown kind: {kind!r}")
    setattr(cls, method_name, wrapped)
    return True
