"""Identity patch for LangChain's classic ``AgentExecutor``.

Targets ``AgentExecutor.invoke`` / ``ainvoke`` / ``stream``.
Legacy LangChain agents typically lack an explicit ``name``;
identity is the composite of the agent's prompt template +
tools — Tier 2B.

This module patches the classic ``AgentExecutor`` **wherever it
lives on the user's installation**:

* **LangChain 0.x** — ``from langchain.agents import AgentExecutor``
  (the original home; still in use at many shops).
* **LangChain 1.x** — ``from langchain_classic.agents import AgentExecutor``
  The LangChain team removed ``AgentExecutor`` from
  ``langchain.agents`` in 1.0 and shipped a dedicated
  back-compat package ``langchain-classic`` that re-exports the
  pre-1.0 agentic surface. Users who install ``langchain-classic``
  are explicitly opting into the classic API and expect the
  egisai langchain patch to fire on their executors.

The two targets are tried independently. A user can have either,
both, or neither installed — every ``patch_method`` call returns
``False`` for missing targets, so the patch chain never breaks.

NOTE on the **LangChain 1.x ``create_agent`` entrypoint**:
``langchain.agents.create_agent`` returns a
``langgraph.graph.state.CompiledStateGraph``, which **inherits
from** ``langgraph.pregel.Pregel`` (verified empirically via
``mro()``). The existing ``langgraph`` patches (``Pregel.invoke`` /
``ainvoke`` / ``stream`` / ``astream``) transparently cover
``create_agent(...)`` calls via MRO. No
``langchain.agents.create_agent`` patch is needed — LangGraph
picks up the slack. Users who want the *classic* agentic surface
on 1.x install ``langchain-classic``, and this module covers
that.
"""

from __future__ import annotations

from typing import Any

from egisai._auto_agent import (
    IdentityRecord,
    _derive_identity_from_system,
)
from egisai._patches import has_module
from egisai._patches._framework import make_identity, patch_method

FRAMEWORK_SOURCE = "framework:langchain"

# Module paths that may host ``AgentExecutor`` depending on the
# user's LangChain pin. Order matters only for documentation —
# both are attempted on every ``apply()`` call.
_AGENT_EXECUTOR_HOMES: tuple[str, ...] = (
    "langchain.agents",          # LangChain 0.x — original location
    "langchain_classic.agents",  # LangChain 1.x — back-compat package
)


def _derive(self_or_executor: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    executor = self_or_executor
    name = str(getattr(executor, "name", "") or "")
    inner_agent = getattr(executor, "agent", None)
    prompt = ""
    if inner_agent is not None:
        llm_chain = getattr(inner_agent, "llm_chain", None)
        if llm_chain is not None:
            template = getattr(getattr(llm_chain, "prompt", None), "template", "")
            if isinstance(template, str):
                prompt = template
        if not prompt:
            prompt = str(getattr(inner_agent, "system_message", "") or "")
    tools = getattr(executor, "tools", []) or []
    tool_names: list[str] = []
    for t in tools:
        tn = getattr(t, "name", None)
        if isinstance(tn, str):
            tool_names.append(tn)
    tool_names.sort()
    if not name and prompt:
        _, name = _derive_identity_from_system(prompt)
    if not name:
        name = "LangChain Agent"
    return make_identity(
        source=FRAMEWORK_SOURCE,
        display_name=name,
        bundle=("langchain", prompt, tuple(tool_names)),
    )


def apply() -> bool:
    # Gate on ANY known home being importable. ``langchain-classic``
    # depends on ``langchain``, so on a 1.x install both modules are
    # present; on a 0.x install only ``langchain`` is. Users with
    # neither shouldn't run this patch — gating prevents spurious
    # imports on systems that don't have LangChain at all.
    if not any(has_module(name) for name in _AGENT_EXECUTOR_HOMES):
        return False
    any_patched = False
    for module_path in _AGENT_EXECUTOR_HOMES:
        if patch_method(
            module_path, "AgentExecutor", "invoke",
            derive=_derive, kind="sync",
        ):
            any_patched = True
        if patch_method(
            module_path, "AgentExecutor", "ainvoke",
            derive=_derive, kind="async",
        ):
            any_patched = True
        if patch_method(
            module_path, "AgentExecutor", "stream",
            derive=_derive, kind="sync_iter",
        ):
            any_patched = True
    return any_patched
