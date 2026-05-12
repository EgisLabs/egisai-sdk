"""Identity patch for LangGraph (per-node agent identity).

Targets ``langgraph.graph.Pregel.invoke`` / ``ainvoke`` / ``stream``.
Each compiled graph is treated as one logical agent (display name =
the graph's name attribute), and nested node calls inherit. A
follow-up release can split this per-node by hooking
``Pregel.stream`` to push a fresh identity at every node visit; for
0.17.0 we ship the graph-level pin to avoid double-counting nested
LLM calls.

Tier 2A — ``graph.name`` if set, otherwise ``graph.config_specs`` is
hashed for a structural id.

Import-guarded; fail-open.
"""

from __future__ import annotations

import logging
from typing import Any

from egisai._auto_agent import IdentityRecord
from egisai._patches import has_module
from egisai._patches._framework import make_identity, patch_method

LOGGER = logging.getLogger("egisai.patches.langgraph")

FRAMEWORK_SOURCE = "framework:langgraph"


def _derive(self_or_pregel: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    pregel = self_or_pregel
    name = (
        str(getattr(pregel, "name", "") or "")
        or str(getattr(pregel, "graph_id", "") or "")
        or "LangGraph Agent"
    )
    nodes: list[str] = []
    try:
        nodes_attr = getattr(pregel, "nodes", {}) or {}
        if isinstance(nodes_attr, dict):
            nodes = sorted(str(k) for k in nodes_attr.keys())
    except Exception:  # noqa: BLE001
        nodes = []
    return make_identity(
        source=FRAMEWORK_SOURCE,
        display_name=name,
        bundle=("langgraph", name, tuple(nodes)),
    )


def apply() -> bool:
    if not has_module("langgraph"):
        return False
    any_patched = False
    if patch_method(
        "langgraph.pregel", "Pregel", "invoke", derive=_derive, kind="sync"
    ):
        any_patched = True
    if patch_method(
        "langgraph.pregel", "Pregel", "ainvoke", derive=_derive, kind="async"
    ):
        any_patched = True
    if patch_method(
        "langgraph.pregel", "Pregel", "stream", derive=_derive, kind="sync_iter"
    ):
        any_patched = True
    if patch_method(
        "langgraph.pregel", "Pregel", "astream",
        derive=_derive, kind="async_iter",
    ):
        any_patched = True
    return any_patched
