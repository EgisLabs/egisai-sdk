"""Identity patch for Agno (formerly Phidata).

Targets ``agno.agent.Agent.run`` / ``arun``. Agno agents have an
explicit ``name`` and a ``description`` — Tier 2A.
"""

from __future__ import annotations

from typing import Any

from egisai._auto_agent import IdentityRecord
from egisai._patches import has_module
from egisai._patches._framework import make_identity, patch_method

FRAMEWORK_SOURCE = "framework:agno"


def _derive(self_or_agent: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    agent = self_or_agent
    name = str(getattr(agent, "name", "") or "Agno Agent")
    description = str(getattr(agent, "description", "") or "")
    instructions = getattr(agent, "instructions", None) or ""
    if isinstance(instructions, list):
        instructions = "\n".join(str(i) for i in instructions)
    return make_identity(
        source=FRAMEWORK_SOURCE,
        display_name=name,
        bundle=("agno", name, description, str(instructions)),
    )


def apply() -> bool:
    if not has_module("agno"):
        return False
    any_patched = False
    if patch_method("agno.agent", "Agent", "run", derive=_derive, kind="sync"):
        any_patched = True
    if patch_method("agno.agent", "Agent", "arun", derive=_derive, kind="async"):
        any_patched = True
    if patch_method(
        "agno.agent", "Agent", "print_response", derive=_derive, kind="sync"
    ):
        any_patched = True
    return any_patched
