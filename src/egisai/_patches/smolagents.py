"""Identity patch for HuggingFace smolagents.

Targets ``smolagents.MultiStepAgent.run``. Smolagents agents expose
``name`` (set by the user at construction) and the planning model id.
"""

from __future__ import annotations

from typing import Any

from egisai._auto_agent import IdentityRecord
from egisai._patches import has_module
from egisai._patches._framework import make_identity, patch_method

FRAMEWORK_SOURCE = "framework:smolagents"


def _derive(self_or_agent: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    agent = self_or_agent
    name = str(
        getattr(agent, "name", "") or type(agent).__name__ or "smolagents Agent"
    )
    model_id = ""
    model = getattr(agent, "model", None)
    if model is not None:
        model_id = str(
            getattr(model, "model_id", "") or getattr(model, "model_name", "") or ""
        )
    tools = getattr(agent, "tools", {}) or {}
    tool_names: list[str] = []
    if isinstance(tools, dict):
        tool_names = sorted(str(k) for k in tools.keys())
    elif isinstance(tools, (list, tuple)):
        for t in tools:
            tn = getattr(t, "name", None)
            if isinstance(tn, str):
                tool_names.append(tn)
        tool_names.sort()
    return make_identity(
        source=FRAMEWORK_SOURCE,
        display_name=name,
        bundle=("smolagents", name, model_id, tuple(tool_names)),
    )


def apply() -> bool:
    if not has_module("smolagents"):
        return False
    any_patched = False
    for class_name in ("MultiStepAgent", "ToolCallingAgent", "CodeAgent"):
        if patch_method(
            "smolagents", class_name, "run", derive=_derive, kind="sync"
        ):
            any_patched = True
    return any_patched
