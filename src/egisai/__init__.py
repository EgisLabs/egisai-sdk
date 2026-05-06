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

__version__ = "0.12.5"

from egisai._context import set_context
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
    "OutputPolicyContext",
    "PolicyContext",
    "PolicyDecision",
    "PolicyRule",
    "__version__",
    "diagnostics",
    "evaluate_output_policies",
    "evaluate_policies",
    "init",
    "set_context",
    "shutdown",
]
