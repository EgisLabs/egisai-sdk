"""Optional audit gate that exercises patches against *real* upstream packages.

Disabled by default (the regular test matrix doesn't depend on 14
external SDKs being installed in CI). When ``EGIS_AUDIT_REAL_FRAMEWORKS=1``
is set, this file picks up every installed framework that our patches
target and asserts the wrapped attributes still pass
``inspect.iscoroutinefunction`` / ``isasyncgenfunction`` /
``isgeneratorfunction`` against the *real* upstream's known shape.

This is the gate that would have caught the 0.17.2 Claude regression
**without** us having to ship to PyPI first. To run it locally:

  pip install claude-agent-sdk agno autogen-agentchat crewai \\
              langgraph langchain llama-index-core \\
              openai-agents pydantic-ai-slim smolagents \\
              strands-agents google-adk
  EGIS_AUDIT_REAL_FRAMEWORKS=1 pytest tests/test_framework_audit.py -v

CI can opt in by exporting the env var for the SDK gate.
"""

from __future__ import annotations

import importlib
import inspect
import os
from typing import Any

import pytest

REAL_AUDIT_ENABLED = os.environ.get("EGIS_AUDIT_REAL_FRAMEWORKS") == "1"

pytestmark = pytest.mark.skipif(
    not REAL_AUDIT_ENABLED,
    reason="set EGIS_AUDIT_REAL_FRAMEWORKS=1 to run the real-libraries audit",
)


def _init_sdk() -> None:
    """Initialise the SDK in a way that's safe even without a backend
    — the audit only inspects the patched methods, never calls them.
    """
    import egisai

    egisai.init(
        api_key="egis_live_audit",
        app="audit",
        env="t",
        base_url="http://fake-audit",
        enable_sse=False,
    )


def _resolve(module_path: str, qual: str) -> Any | None:
    """Return the attribute or ``None`` if any segment is missing.

    ``qual`` is dot-separated (e.g. ``Runner.run``). We unwrap
    ``classmethod`` / ``staticmethod`` descriptors so the inspect
    helpers see the underlying function.
    """
    try:
        mod = importlib.import_module(module_path)
    except ImportError:
        return None
    obj: Any = mod
    for part in qual.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    if isinstance(obj, (classmethod, staticmethod)):
        obj = obj.__func__
    return obj


# (label, module_path, qualified_name, expected: (coro, agen, sgen))
CASES: list[tuple[str, str, str, tuple[bool, bool, bool]]] = [
    ("openai_agents Runner.run",         "agents",                   "Runner.run",                  (True,  False, False)),
    ("openai_agents Runner.run_sync",    "agents",                   "Runner.run_sync",             (False, False, False)),
    ("openai_agents Runner.run_streamed","agents",                   "Runner.run_streamed",         (False, False, False)),

    ("langgraph Pregel.invoke",          "langgraph.pregel",         "Pregel.invoke",               (False, False, False)),
    ("langgraph Pregel.ainvoke",         "langgraph.pregel",         "Pregel.ainvoke",              (True,  False, False)),
    ("langgraph Pregel.stream",          "langgraph.pregel",         "Pregel.stream",               (False, False, True)),
    ("langgraph Pregel.astream",         "langgraph.pregel",         "Pregel.astream",              (False, True,  False)),

    ("autogen AssistantAgent.run",       "autogen_agentchat.agents", "AssistantAgent.run",          (True,  False, False)),
    ("autogen AssistantAgent.run_stream","autogen_agentchat.agents", "AssistantAgent.run_stream",   (False, True,  False)),

    ("crewai Agent.execute_task",        "crewai",                   "Agent.execute_task",          (False, False, False)),

    # Agno's run/arun are polymorphic plain ``def``s.
    ("agno Agent.run",                   "agno.agent",               "Agent.run",                   (False, False, False)),
    ("agno Agent.arun",                  "agno.agent",               "Agent.arun",                  (False, False, False)),
    ("agno Agent.print_response",        "agno.agent",               "Agent.print_response",        (False, False, False)),

    ("strands Agent.__call__",           "strands",                  "Agent.__call__",              (False, False, False)),
    ("strands Agent.invoke_async",       "strands",                  "Agent.invoke_async",          (True,  False, False)),

    ("smolagents MultiStepAgent.run",    "smolagents",               "MultiStepAgent.run",          (False, False, False)),
    ("smolagents ToolCallingAgent.run",  "smolagents",               "ToolCallingAgent.run",        (False, False, False)),
    ("smolagents CodeAgent.run",         "smolagents",               "CodeAgent.run",               (False, False, False)),

    ("pydantic_ai Agent.run",            "pydantic_ai",              "Agent.run",                   (True,  False, False)),
    ("pydantic_ai Agent.run_sync",       "pydantic_ai",              "Agent.run_sync",              (False, False, False)),

    ("llamaindex FunctionAgent.run",     "llama_index.core.agent",   "FunctionAgent.run",           (False, False, False)),
    ("llamaindex ReActAgent.run",        "llama_index.core.agent",   "ReActAgent.run",              (False, False, False)),
    ("llamaindex CodeActAgent.run",      "llama_index.core.agent",   "CodeActAgent.run",            (False, False, False)),
    ("llamaindex AgentWorkflow.run",     "llama_index.core.agent",   "AgentWorkflow.run",           (False, False, False)),

    ("google.adk Runner.run",            "google.adk.runners",       "Runner.run",                  (False, False, True)),
    ("google.adk Runner.run_async",      "google.adk.runners",       "Runner.run_async",            (False, True,  False)),

    ("claude_agent_sdk.query",           "claude_agent_sdk",         "query",                       (False, True,  False)),
    ("claude_agent_sdk.ClaudeSDKClient.query", "claude_agent_sdk",   "ClaudeSDKClient.query",       (True,  False, False)),
]


@pytest.mark.parametrize(
    "label,mod_path,qual,expected",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_signature_audit_real(
    label: str,
    mod_path: str,
    qual: str,
    expected: tuple[bool, bool, bool],
) -> None:
    """The wrapped method must preserve the upstream's call shape."""
    raw = _resolve(mod_path, qual)
    if raw is None:
        pytest.skip(f"{label}: framework not installed")
    # First baseline: confirm the EXPECTED shape matches the actual
    # upstream's shape (catches drift when an upstream library
    # changes its signature).
    coro_actual = inspect.iscoroutinefunction(raw)
    agen_actual = inspect.isasyncgenfunction(raw)
    sgen_actual = inspect.isgeneratorfunction(raw)
    assert (coro_actual, agen_actual, sgen_actual) == expected, (
        f"upstream drift: {label} now reports "
        f"(coro={coro_actual}, agen={agen_actual}, sgen={sgen_actual}), "
        f"but our patch table expects {expected!r}. "
        f"Update CASES + the matching patch's ``kind=`` if the upstream changed."
    )

    # Initialise the SDK so init() runs every framework patch's apply().
    _init_sdk()

    # Re-resolve in case ``apply()`` rebound the descriptor.
    wrapped = _resolve(mod_path, qual)
    assert wrapped is not None
    coro_w = inspect.iscoroutinefunction(wrapped)
    agen_w = inspect.isasyncgenfunction(wrapped)
    sgen_w = inspect.isgeneratorfunction(wrapped)
    assert (coro_w, agen_w, sgen_w) == expected, (
        f"{label}: wrapped shape diverged from upstream after apply()! "
        f"wrapped=(coro={coro_w}, agen={agen_w}, sgen={sgen_w}) vs "
        f"expected={expected!r}. This is the same bug class as the "
        f"0.17.2 Claude regression — pick the correct ``kind=`` in the "
        f"framework patch."
    )
