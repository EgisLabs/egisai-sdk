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
    # All modern LlamaIndex agents (FunctionAgent, ReActAgent,
    # CodeActAgent, AgentWorkflow) expose ``run`` as a plain ``def``
    # that returns a ``WorkflowHandler`` — an awaitable handle whose
    # ``.stream_events()`` is the streaming API and whose internal
    # ``_result_task`` is the asyncio Task on which the workflow's
    # steps (including inner LLM + tool calls) actually run.
    #
    # ``handler`` kind is the only correct wrap here:
    #
    # * ``async`` swallowed the handle inside our coroutine and broke
    #   the supported ``async for ev in agent.run(...).stream_events()``
    #   pattern (0.17.0–0.17.4 regression).
    # * ``sync`` (0.17.5–0.17.x) returned the handle as-is and so kept
    #   that pattern working, BUT closed the Run scope before the
    #   workflow's inner LLM calls executed — inner steps then fell
    #   through to the legacy ephemeral-run path with no framework
    #   attribution. The dashboard saw a phantom empty
    #   ``framework=llamaindex`` run plus a real
    #   ``framework=legacy`` run for the same task.
    # * ``handler`` keeps the Run scope open from the moment ``run()``
    #   returns the handle until the handle's ``_result_task``
    #   completes, so every inner LLM call records under the correct
    #   ``framework=llamaindex`` run with the right identity.
    for class_name in (
        "FunctionAgent",
        "ReActAgent",
        "CodeActAgent",
        "AgentWorkflow",
    ):
        if patch_method(
            "llama_index.core.agent", class_name, "run",
            derive=_derive, kind="handler",
        ):
            any_patched = True
    # ``AgentRunner`` was removed in modern LlamaIndex but we keep the
    # call for older installations — ``patch_method`` returns ``False``
    # silently when the target is gone. The legacy AgentRunner.run is
    # a fully synchronous call (no handle) so ``sync`` is the right
    # wrap for it.
    if patch_method(
        "llama_index.core.agent", "AgentRunner", "run",
        derive=_derive, kind="sync",
    ):
        any_patched = True
    return any_patched
