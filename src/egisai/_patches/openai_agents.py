"""Identity patch for the OpenAI Agents Python SDK.

Targets ``agents.Runner.run`` and its async/streaming siblings. Each
of those is the SDK's "kick off an agent invocation" surface — the
inner LLM calls during the same invocation should inherit the same
identity. We push an :class:`IdentityRecord` at the entry point and
let the existing ``_patches/openai.py`` patch govern the per-LLM
call (it sees ``current_identity()`` via the resolver and skips
re-deriving from the system prompt).

Tier 2A — explicit ``agent.name``. Identity v2 (anchor-dominant):
the declared name IS the identity — ``("openai_agents", name)`` is
the whole bundle. Editing a named agent's instructions or tool set
is a *revision* of the same agent (the backend records a
prompt-version event), never a fork. Two deployments that reuse one
name intentionally coalesce; operators who want them separate give
them distinct names — a name is a declaration of identity. Nameless
agents fall back to a content bundle (canonical instructions +
tools).

Import-guarded: if ``agents`` isn't installed, ``apply()`` returns
``False`` and the SDK runs without it. Fail-open everywhere.
"""

from __future__ import annotations

import logging
from typing import Any

from egisai._auto_agent import IdentityRecord
from egisai._patches import has_module
from egisai._patches._framework import make_identity, patch_method

LOGGER = logging.getLogger("egisai.patches.openai_agents")

FRAMEWORK_SOURCE = "framework:openai_agents"


def _agent_fields(agent: Any) -> tuple[str, str, list[str]]:
    """Pluck (name, instructions, tool_names) off an Agent object.

    Handles both the ``Agent`` dataclass shape and dict-shaped
    configs that some user code passes in. Returns empty strings /
    list when a field is missing — those still produce a stable
    identity hash for the partial bundle.
    """
    if isinstance(agent, dict):
        name = str(agent.get("name") or "")
        instructions_v = agent.get("instructions") or ""
        instructions = (
            instructions_v if isinstance(instructions_v, str) else str(instructions_v)
        )
        tools = agent.get("tools") or []
    else:
        name = str(getattr(agent, "name", "") or "")
        instructions = str(getattr(agent, "instructions", "") or "")
        tools = getattr(agent, "tools", []) or []
    tool_names: list[str] = []
    if isinstance(tools, (list, tuple)):
        for t in tools:
            tn = getattr(t, "name", None) or (
                t.get("name") if isinstance(t, dict) else None
            )
            if isinstance(tn, str):
                tool_names.append(tn)
    return name, instructions, tool_names


def _derive(self_or_runner: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    """Resolve identity from a ``Runner.run(agent, …)`` call.

    Handles three call shapes:
      1. ``Runner.run(agent, …)``       — staticmethod call (current SDK shape).
         ``self_or_runner = agent``, ``args = ()``.
      2. ``runner.run(agent, …)``       — instance call.
         ``self_or_runner = runner``, ``args = (agent, …)``.
      3. Keyword form ``Runner.run(agent=…)`` or ``starting_agent=…``.
    """
    agent = None
    # Treat first positional as the agent if it walks like an Agent
    # (has either ``name`` or ``instructions``). This is how we tell
    # ``Runner.run(agent, …)`` (staticmethod) apart from
    # ``runner.run(agent, …)`` (instance method).
    if self_or_runner is not None and (
        hasattr(self_or_runner, "name") or hasattr(self_or_runner, "instructions")
    ) and not hasattr(self_or_runner, "run"):
        agent = self_or_runner
    elif args:
        agent = args[0]
    elif "agent" in kwargs:
        agent = kwargs["agent"]
    elif "starting_agent" in kwargs:
        agent = kwargs["starting_agent"]
    if agent is None:
        return None
    name, instructions, tools = _agent_fields(agent)
    if not name and not instructions:
        return None
    # Identity v2 — anchor-dominant: the explicit ``agent.name`` IS
    # the identity. Editing instructions (prompt optimization) or
    # adding a tool to a named agent is a revision of the SAME agent,
    # recorded server-side as a prompt-version event — not a fork.
    # Nameless agents key on the model-masked canonical instructions
    # + tools. The v1 bundle ships as the legacy hash so existing
    # rows re-stamp in place.
    from egisai._auto_agent import canonicalize_identity_text

    if name:
        bundle: tuple = ("openai_agents", name)
    else:
        bundle = (
            "openai_agents",
            canonicalize_identity_text(instructions),
            tuple(sorted(tools)),
        )
    return make_identity(
        source=FRAMEWORK_SOURCE,
        display_name=name or "openai-agents-agent",
        bundle=bundle,
        legacy_bundle=(
            "openai_agents", name, instructions, tuple(sorted(tools)),
        ),
        prompt_text=instructions or None,
        tool_names=tools,
    )


def apply() -> bool:
    if not has_module("agents"):
        return False
    any_patched = False
    if patch_method("agents", "Runner", "run", derive=_derive, kind="async"):
        any_patched = True
    if patch_method("agents", "Runner", "run_sync", derive=_derive, kind="sync"):
        any_patched = True
    if patch_method(
        "agents", "Runner", "run_streamed", derive=_derive, kind="sync"
    ):
        any_patched = True
    return any_patched
