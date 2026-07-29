"""egisai — runtime governance SDK for AI agents.

    import egisai
    egisai.init(api_key="egis_live_...", app="my-agent", env="prod")

    import openai
    openai.OpenAI().chat.completions.create(...)   # gated automatically

After ``init()``, supported AI libraries are patched in place and
every model call is governed by your platform-defined policies.
Supported integrations: OpenAI, Anthropic, Google Generative AI,
plus an httpx / requests fallback.
"""

from __future__ import annotations

__version__ = "0.50.0"

# Identity v2 helpers — public for advanced callers (and the
# platform's gateway) that need to compute the same model-invariant
# canonical identity the SDK computes: e.g. re-stamping an agent
# after platform-driven prompt optimization.
from egisai._auto_agent import (
    PromptIdentity,
    canonicalize_identity_text,
    derive_prompt_identity,
    simhash64_hex,
    tool_bundle_hash_from_names,
)
from egisai._client import AsyncClient, Client
from egisai._context import agent, register_agent, set_context
from egisai._init import diagnostics, init, shutdown
from egisai.policy import (
    OutputPolicyContext,
    PolicyContext,
    PolicyDecision,
    PolicyRule,
    evaluate_output_policies,
    evaluate_policies,
)

__all__ = [
    "AsyncClient",
    "Client",
    "OutputPolicyContext",
    "PolicyContext",
    "PolicyDecision",
    "PolicyRule",
    "PromptIdentity",
    "__version__",
    "agent",
    "canonicalize_identity_text",
    "derive_prompt_identity",
    "diagnostics",
    "evaluate_output_policies",
    "evaluate_policies",
    "init",
    "register_agent",
    "set_context",
    "shutdown",
    "simhash64_hex",
    "tool_bundle_hash_from_names",
]
