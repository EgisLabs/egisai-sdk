"""Identity patch for Google's Agent Development Kit (ADK).

Targets ``google.adk.runners.Runner.run_async``. ADK agents have an
explicit ``name`` and a configured ``instruction`` — a Tier 2A
patch with the (name, instruction, sorted_tools) bundle for the
identity hash so two same-named agents with different toolsets
deduplicate correctly.
"""

from __future__ import annotations

from typing import Any

from egisai._auto_agent import IdentityRecord
from egisai._patches import has_module
from egisai._patches._framework import make_identity, patch_method

FRAMEWORK_SOURCE = "framework:adk"


def _derive(self_or_runner: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    agent = getattr(self_or_runner, "agent", None) or kwargs.get("agent")
    if agent is None:
        return None
    name = str(getattr(agent, "name", "") or "ADK Agent")
    instruction = str(getattr(agent, "instruction", "") or "")
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
        display_name=name,
        bundle=("adk", name, instruction, tuple(sorted(tool_names))),
    )


def apply() -> bool:
    if not has_module("google.adk"):
        return False
    any_patched = False
    if patch_method(
        "google.adk.runners", "Runner", "run_async",
        derive=_derive, kind="async_iter",
    ):
        any_patched = True
    if patch_method(
        "google.adk.runners", "Runner", "run",
        derive=_derive, kind="sync_iter",
    ):
        any_patched = True
    return any_patched
