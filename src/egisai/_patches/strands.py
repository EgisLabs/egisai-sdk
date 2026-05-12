"""Identity patch for AWS Strands Agents.

Targets ``strands.Agent.__call__`` / ``invoke_async``. Strands agents
carry ``name``, ``system_prompt``, and ``tools`` — Tier 2A.
"""

from __future__ import annotations

from typing import Any

from egisai._auto_agent import IdentityRecord
from egisai._patches import has_module
from egisai._patches._framework import make_identity, patch_method

FRAMEWORK_SOURCE = "framework:strands"


def _derive(self_or_agent: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    agent = self_or_agent
    name = str(getattr(agent, "name", "") or "Strands Agent")
    system_prompt = str(getattr(agent, "system_prompt", "") or "")
    tools = getattr(agent, "tools", []) or []
    tool_names: list[str] = []
    for t in tools:
        tn = getattr(t, "name", None) or (
            getattr(t, "__name__", None)
            or (t.get("name") if isinstance(t, dict) else None)
        )
        if isinstance(tn, str):
            tool_names.append(tn)
    return make_identity(
        source=FRAMEWORK_SOURCE,
        display_name=name,
        bundle=("strands", name, system_prompt, tuple(sorted(tool_names))),
    )


def apply() -> bool:
    if not has_module("strands"):
        return False
    any_patched = False
    if patch_method("strands", "Agent", "__call__", derive=_derive, kind="sync"):
        any_patched = True
    if patch_method(
        "strands", "Agent", "invoke_async", derive=_derive, kind="async"
    ):
        any_patched = True
    return any_patched
