"""Identity patch for LangChain legacy AgentExecutor.

Targets ``langchain.agents.AgentExecutor.invoke`` / ``ainvoke`` /
``stream``. Legacy LangChain agents typically lack an explicit
``name``; identity is the composite of the agent's prompt template
+ tools — Tier 2B.
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
    if not has_module("langchain"):
        return False
    any_patched = False
    if patch_method(
        "langchain.agents", "AgentExecutor", "invoke", derive=_derive, kind="sync"
    ):
        any_patched = True
    if patch_method(
        "langchain.agents", "AgentExecutor", "ainvoke",
        derive=_derive, kind="async",
    ):
        any_patched = True
    if patch_method(
        "langchain.agents", "AgentExecutor", "stream",
        derive=_derive, kind="sync_iter",
    ):
        any_patched = True
    return any_patched
