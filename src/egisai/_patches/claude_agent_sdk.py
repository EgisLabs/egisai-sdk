"""Identity patch for the Claude Agent SDK (Anthropic's agentic stack).

Targets ``claude_agent_sdk.query`` and ``ClaudeSDKClient.query``. The
Claude Agent SDK doesn't surface an explicit ``name`` on its
``ClaudeAgentOptions`` — agents are identified by their *composite
definition*. So this is a Tier 2B patch: we hash
``(system_prompt, sorted(allowed_tools), permission_mode,
sorted(mcp_server_names), model)`` to produce a stable identity that
survives prompt-rendering noise but flips when the operator actually
changes the agent's permissions or tool set.

Display name comes from NER on the system prompt (NER-first plan),
with a ``claude-agent-<hash[:8]>`` last-resort fallback.

Import-guarded; fail-open.
"""

from __future__ import annotations

import logging
from typing import Any

from egisai._auto_agent import (
    IdentityRecord,
    _derive_identity_from_system,
)
from egisai._patches import has_module
from egisai._patches._framework import make_identity, patch_method

LOGGER = logging.getLogger("egisai.patches.claude_agent_sdk")

FRAMEWORK_SOURCE = "framework:claude_agent_sdk"


def _bundle_from_options(options: Any) -> tuple[str, str, tuple[Any, ...]]:
    """Extract (display_name, system_prompt, bundle_tuple) from options."""
    if options is None:
        return ("Claude Agent", "", ("claude_agent_sdk",))
    sp_raw = getattr(options, "system_prompt", None) or (
        options.get("system_prompt") if isinstance(options, dict) else None
    )
    system_prompt = str(sp_raw or "").strip()
    allowed_tools = getattr(options, "allowed_tools", None) or (
        options.get("allowed_tools") if isinstance(options, dict) else None
    ) or []
    permission_mode = str(
        getattr(options, "permission_mode", "") or (
            options.get("permission_mode") if isinstance(options, dict) else ""
        ) or ""
    )
    model = str(
        getattr(options, "model", "") or (
            options.get("model") if isinstance(options, dict) else ""
        ) or ""
    )
    mcp_servers = getattr(options, "mcp_servers", None) or (
        options.get("mcp_servers") if isinstance(options, dict) else None
    ) or []
    mcp_names: list[str] = []
    if isinstance(mcp_servers, dict):
        mcp_names = sorted(str(k) for k in mcp_servers.keys())
    elif isinstance(mcp_servers, (list, tuple)):
        for s in mcp_servers:
            sn = getattr(s, "name", None) or (
                s.get("name") if isinstance(s, dict) else None
            )
            if isinstance(sn, str):
                mcp_names.append(sn)
        mcp_names.sort()

    tool_names = sorted(
        str(t) for t in (allowed_tools or []) if isinstance(t, (str, bytes))
    )

    display_name: str
    if system_prompt:
        _, display_name = _derive_identity_from_system(system_prompt)
    else:
        display_name = "Claude Agent"
    bundle = (
        "claude_agent_sdk",
        system_prompt,
        tuple(tool_names),
        permission_mode,
        model,
        tuple(mcp_names),
    )
    return display_name, system_prompt, bundle


def _derive(self_or_first: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    """Pluck options from ``query(prompt, options=…)`` and build identity."""
    options = kwargs.get("options")
    if options is None:
        # ``ClaudeSDKClient.query`` carries options on the instance,
        # not on the per-call kwargs. ``self_or_first`` is the
        # client; pull its options attribute (set at construction
        # time) if available.
        options = getattr(self_or_first, "options", None) or getattr(
            self_or_first, "_options", None
        )
    if options is None:
        return None
    display_name, _, bundle = _bundle_from_options(options)
    return make_identity(
        source=FRAMEWORK_SOURCE,
        display_name=display_name,
        bundle=bundle,
    )


def apply() -> bool:
    if not has_module("claude_agent_sdk"):
        return False
    any_patched = False
    # Module-level ``query`` is the canonical entrypoint. The SDK
    # exposes it as both an async generator (streaming) and a
    # coroutine depending on options — wrap as async-iter.
    try:
        import claude_agent_sdk as _sdk  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return False
    if hasattr(_sdk, "query") and callable(_sdk.query):
        orig = _sdk.query
        if not getattr(orig, "__egisai_wrapped__", False):
            from egisai._patches._framework import wrap_async_iter_entrypoint

            _sdk.query = wrap_async_iter_entrypoint(orig, _derive)
            any_patched = True
    # NOTE: ``ClaudeSDKClient.query`` is a **coroutine** (``async def``),
    # not an async generator — even though the module-level ``query``
    # above IS an async generator. They share a name but have different
    # call shapes:
    #   - Module-level: ``async for msg in claude_agent_sdk.query(...)``
    #   - Instance:     ``await client.query(prompt)`` then
    #                   ``async for msg in client.receive_response()``
    # Wrapping the instance method as ``async_iter`` (which we did in
    # 0.17.0–0.17.4) replaced the coroutine with an async-generator
    # function, so ``await client.query(prompt)`` raised
    # ``TypeError: object async_generator can't be used in 'await'
    # expression``. Fixed in 0.17.5.
    if patch_method(
        "claude_agent_sdk", "ClaudeSDKClient", "query",
        derive=_derive, kind="async",
    ):
        any_patched = True
    return any_patched
