"""Identity patch for CrewAI.

Targets ``crewai.Agent.execute_task``. CrewAI agents carry ``role``,
``goal``, and ``backstory`` strings — a Tier 2A bundle keyed on
``role`` (which CrewAI treats as the agent's identity by convention).
"""

from __future__ import annotations

from typing import Any

from egisai._auto_agent import IdentityRecord
from egisai._patches import has_module
from egisai._patches._framework import make_identity, patch_method

FRAMEWORK_SOURCE = "framework:crewai"


def _derive(self_or_agent: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    agent = self_or_agent
    role = str(getattr(agent, "role", "") or "CrewAI Agent")
    goal = str(getattr(agent, "goal", "") or "")
    backstory = str(getattr(agent, "backstory", "") or "")
    tools = getattr(agent, "tools", []) or []
    tool_names: list[str] = []
    for t in tools:
        tn = getattr(t, "name", None) or (
            t.get("name") if isinstance(t, dict) else None
        )
        if isinstance(tn, str):
            tool_names.append(tn)
    return make_identity(
        source=FRAMEWORK_SOURCE,
        display_name=role,
        bundle=("crewai", role, goal, backstory, tuple(sorted(tool_names))),
    )


def apply() -> bool:
    if not has_module("crewai"):
        return False
    any_patched = False
    if patch_method(
        "crewai", "Agent", "execute_task", derive=_derive, kind="sync"
    ):
        any_patched = True
    return any_patched
