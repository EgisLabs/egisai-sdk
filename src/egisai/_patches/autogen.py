"""Identity patch for Microsoft AutoGen / autogen-agentchat.

Targets ``autogen_agentchat.agents.BaseChatAgent.run`` /
``run_stream``. AutoGen agents have explicit ``name`` and
``system_message`` — a Tier 2A bundle.
"""

from __future__ import annotations

from typing import Any

from egisai._auto_agent import IdentityRecord
from egisai._patches import has_module
from egisai._patches._framework import make_identity, patch_method

FRAMEWORK_SOURCE = "framework:autogen"


def _derive(self_or_agent: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    agent = self_or_agent
    name = str(getattr(agent, "name", "") or "AutoGen Agent")
    sys_msg = str(
        getattr(agent, "system_message", "") or getattr(agent, "description", "") or ""
    )
    return make_identity(
        source=FRAMEWORK_SOURCE,
        display_name=name,
        bundle=("autogen", name, sys_msg),
    )


def apply() -> bool:
    if not has_module("autogen_agentchat"):
        return False
    any_patched = False
    for class_name in ("AssistantAgent", "UserProxyAgent", "BaseChatAgent"):
        if patch_method(
            "autogen_agentchat.agents", class_name, "run",
            derive=_derive, kind="async",
        ):
            any_patched = True
        if patch_method(
            "autogen_agentchat.agents", class_name, "run_stream",
            derive=_derive, kind="async_iter",
        ):
            any_patched = True
    return any_patched
