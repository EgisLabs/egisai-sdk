"""Identity patch for Microsoft AutoGen / autogen-agentchat.

Targets ``autogen_agentchat.agents.BaseChatAgent.run`` /
``run_stream``. AutoGen agents have explicit ``name`` and
``system_message`` — a Tier 2A bundle.
"""

from __future__ import annotations

from typing import Any

from egisai._auto_agent import IdentityRecord, canonicalize_identity_text
from egisai._patches import has_module
from egisai._patches._framework import make_identity, patch_method

FRAMEWORK_SOURCE = "framework:autogen"


def _derive(self_or_agent: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    agent = self_or_agent
    explicit_name = str(getattr(agent, "name", "") or "")
    name = explicit_name or "AutoGen Agent"
    sys_msg = str(
        getattr(agent, "system_message", "") or getattr(agent, "description", "") or ""
    )
    # Identity v2 — anchor-dominant: AutoGen's explicit ``name`` is
    # the identity; a system-message edit on a named agent is a
    # revision of the same agent. Nameless agents key on the
    # canonical system message. The v1 bundle ships as legacy hash.
    if explicit_name:
        bundle: tuple = ("autogen", explicit_name)
    else:
        bundle = ("autogen", name, canonicalize_identity_text(sys_msg))
    return make_identity(
        source=FRAMEWORK_SOURCE,
        display_name=name,
        bundle=bundle,
        legacy_bundle=("autogen", name, sys_msg),
        prompt_text=sys_msg or None,
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
