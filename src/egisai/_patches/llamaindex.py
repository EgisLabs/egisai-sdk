"""Identity patch for LlamaIndex agents.

Targets ``llama_index.core.agent.FunctionAgent.run`` and
``AgentRunner.run``. LlamaIndex agents typically lack an explicit
``name``; we hash the agent's ``system_prompt`` + tool list — Tier 2B.
"""

from __future__ import annotations

from typing import Any

from egisai._auto_agent import (
    IdentityRecord,
    _derive_identity_from_system,
)
from egisai._patches import has_module
from egisai._patches._framework import make_identity, patch_method

FRAMEWORK_SOURCE = "framework:llamaindex"


def _derive(self_or_agent: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    agent = self_or_agent
    name = str(getattr(agent, "name", "") or "")
    system_prompt = str(getattr(agent, "system_prompt", "") or "")
    tools = getattr(agent, "tools", []) or []
    tool_names: list[str] = []
    for t in tools:
        metadata = getattr(t, "metadata", None)
        tn = getattr(metadata, "name", None) if metadata is not None else None
        if not tn:
            tn = getattr(t, "name", None)
        if isinstance(tn, str):
            tool_names.append(tn)
    tool_names.sort()
    if not name and system_prompt:
        _, name = _derive_identity_from_system(system_prompt)
    if not name:
        name = "LlamaIndex Agent"
    return make_identity(
        source=FRAMEWORK_SOURCE,
        display_name=name,
        bundle=("llamaindex", system_prompt, tuple(tool_names)),
    )


def apply() -> bool:
    if not has_module("llama_index"):
        return False
    any_patched = False
    if patch_method(
        "llama_index.core.agent", "FunctionAgent", "run",
        derive=_derive, kind="async",
    ):
        any_patched = True
    if patch_method(
        "llama_index.core.agent", "AgentRunner", "run",
        derive=_derive, kind="sync",
    ):
        any_patched = True
    return any_patched
