"""Identity patch for PydanticAI.

Targets ``pydantic_ai.Agent.run`` / ``run_sync``. PydanticAI agents
have ``name`` (optional, often unset) and ``system_prompt`` —
Tier 2B because name isn't reliable.
"""

from __future__ import annotations

from typing import Any

from egisai._auto_agent import (
    IdentityRecord,
    _derive_identity_from_system,
    canonicalize_identity_text,
)
from egisai._patches import has_module
from egisai._patches._framework import make_identity, patch_method

FRAMEWORK_SOURCE = "framework:pydantic_ai"


def _derive(self_or_agent: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    agent = self_or_agent
    name = str(getattr(agent, "name", "") or "")
    system_prompt = ""
    # PydanticAI's system_prompt can be a string, a callable, or a
    # tuple of either. We try the simple cases only.
    sp = getattr(agent, "_system_prompts", None) or getattr(
        agent, "system_prompt", None
    )
    if isinstance(sp, str):
        system_prompt = sp
    elif isinstance(sp, (list, tuple)):
        chunks = [s for s in sp if isinstance(s, str)]
        system_prompt = "\n".join(chunks)
    model = getattr(agent, "model", None)
    model_name = ""
    if model is not None:
        model_name = str(getattr(model, "name", "") or getattr(model, "model_name", "") or "")
    explicit_name = name
    if not name and system_prompt:
        _, name = _derive_identity_from_system(system_prompt)
    if not name:
        name = "PydanticAI Agent"
    # Identity v2 — anchor-dominant: an explicit ``name`` IS the
    # identity (prompt/model changes on a named agent are revisions,
    # not new agents). Unnamed agents key on the model-masked
    # canonical prompt; the model id is never in the bundle so
    # swapping models keeps the same agent. The v1 bundle (prompt +
    # model) ships as the legacy hash for in-place migration.
    if explicit_name:
        bundle: tuple = ("pydantic_ai", explicit_name)
    else:
        bundle = ("pydantic_ai", canonicalize_identity_text(system_prompt))
    return make_identity(
        source=FRAMEWORK_SOURCE,
        display_name=name,
        bundle=bundle,
        legacy_bundle=("pydantic_ai", system_prompt, model_name),
        prompt_text=system_prompt or None,
        model=model_name or None,
    )


def apply() -> bool:
    if not has_module("pydantic_ai"):
        return False
    any_patched = False
    if patch_method(
        "pydantic_ai", "Agent", "run", derive=_derive, kind="async"
    ):
        any_patched = True
    if patch_method(
        "pydantic_ai", "Agent", "run_sync", derive=_derive, kind="sync"
    ):
        any_patched = True
    return any_patched
