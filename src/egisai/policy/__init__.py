"""egisai.policy — runtime policy primitives for AI agent governance.

Pure-Python rule engine with no web/db dependencies:

- ``engine`` — dataclasses + ``evaluate_policies`` / ``evaluate_output_policies``
- ``pii`` — multi-signal PII detection (Luhn, mod-97, entropy, …)
- ``semantic`` — client for the platform's intent judge
"""

from __future__ import annotations

from egisai.policy.engine import (
    MatchedPolicyRecord,
    OutputPolicyContext,
    PolicyContext,
    PolicyDecision,
    PolicyRule,
    evaluate_output_policies,
    evaluate_policies,
)
from egisai.policy.pii import Sanitization, label_redact

__all__ = [
    "MatchedPolicyRecord",
    "OutputPolicyContext",
    "PolicyContext",
    "PolicyDecision",
    "PolicyRule",
    "Sanitization",
    "evaluate_output_policies",
    "evaluate_policies",
    "label_redact",
]
