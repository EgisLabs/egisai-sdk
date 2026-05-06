"""Pure-Python policy rule engine.

Evaluates ``PolicyRule`` objects against an input or output
``PolicyContext`` and returns a ``PolicyDecision``. No I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from egisai.policy import pii as pii_scanner
from egisai.policy._regex_safe import safe_search
from egisai.policy.semantic import SemanticBlocker


@dataclass(frozen=True)
class PolicyRule:
    """One active rule.

    ``type`` selects the evaluator (``pii_scan``, ``semantic_guard``,
    ``deny_regex``, …); ``config`` carries the type-specific knobs.
    ``agent_ids`` scopes the rule to specific agents — empty means
    "applies to every agent".

    ``phase`` selects which side of the call the rule runs on:

    - ``"pre_model"``  — evaluated against the user prompt before the
      model is called (default for input-side detectors).
    - ``"post_model"`` — evaluated against the model's response after
      it returns (default for output-side detectors).
    - ``"both"`` — runs on both sides; only meaningful for rule types
      that support it (e.g. ``semantic_guard``).
    """

    id: str | None
    name: str
    type: str
    tenant: str | None
    config: dict[str, Any]
    agent_ids: tuple[str, ...] = field(default=())
    phase: str = "both"


@dataclass(frozen=True)
class PolicyContext:
    """Inputs for evaluating *input-side* policies (before the LLM call)."""

    tenant: str
    model: str
    prompt_text: str
    prompt_chars: int
    stream: bool


@dataclass(frozen=True)
class OutputPolicyContext:
    """Inputs for evaluating *output-side* policies (after the LLM responds)."""

    tenant: str
    model: str
    text: str
    tool_names: list[str]
    tool_calls: list[dict[str, str]]
    mcp_targets: list[str]
    stream: bool


@dataclass(frozen=True)
class MatchedPolicyRecord:
    """One policy that fired during evaluation.

    ``verdict`` is what this rule would have returned in isolation
    (``'block'`` or ``'sanitize'``); the final ``PolicyDecision.verdict``
    is computed across all matches. ``sanitize_kinds`` and
    ``sanitize_mask_char`` are only meaningful for sanitize matches.
    """
    name: str
    type: str
    verdict: str
    reason_code: str
    message: str
    sanitize_kinds: tuple[str, ...] = ()
    sanitize_mask_char: str = "#"


@dataclass(frozen=True)
class PolicyDecision:
    """Outcome of running the policy engine on one call.

    ``verdict``:

    - ``"allow"``     forward the call as-is
    - ``"sanitize"``  forward with masked PII (the raw value never
                      reaches the model)
    - ``"block"``     refuse the call

    ``matched_policy`` is the primary matched rule's name;
    ``matched_policies`` is the full ordered list of rules that fired.
    """
    verdict: str
    reason_code: str | None
    message: str | None
    matched_policy: str | None
    matched_policies: tuple[MatchedPolicyRecord, ...] = ()
    sanitize_kinds: list[str] = field(default_factory=list)
    sanitize_mask_char: str = "#"

    @classmethod
    def allow(cls) -> PolicyDecision:
        return cls(
            verdict="allow",
            reason_code=None,
            message=None,
            matched_policy=None,
        )

    @classmethod
    def deny(
        cls,
        *,
        reason_code: str,
        message: str,
        matched_policy: str,
        matched_policies: tuple[MatchedPolicyRecord, ...] = (),
    ) -> PolicyDecision:
        return cls(
            verdict="block",
            reason_code=reason_code,
            message=message,
            matched_policy=matched_policy,
            matched_policies=matched_policies,
        )

    @classmethod
    def sanitize(
        cls,
        *,
        kinds: list[str],
        reason_code: str,
        message: str,
        matched_policy: str,
        mask_char: str = "#",
        matched_policies: tuple[MatchedPolicyRecord, ...] = (),
    ) -> PolicyDecision:
        """The call should forward, but with these PII kinds masked."""
        return cls(
            verdict="sanitize",
            reason_code=reason_code,
            message=message,
            matched_policy=matched_policy,
            matched_policies=matched_policies,
            sanitize_kinds=list(kinds),
            sanitize_mask_char=mask_char or "#",
        )


# Deterministic, local-only checks. Adding a new policy kind here
# means it must not issue any network request.
_DETERMINISTIC_KINDS = frozenset(
    {"allow_model", "deny_regex", "max_prompt_chars", "pii_scan"}
)

# Network-issuing checks (LLM judges, embedding lookups, …).
_LLM_BACKED_KINDS = frozenset({"semantic_guard"})


def _runs_pre_model(rule: PolicyRule) -> bool:
    """Return True when this rule should fire on the prompt side."""
    return rule.phase in ("pre_model", "both")


def _runs_post_model(rule: PolicyRule) -> bool:
    """Return True when this rule should fire on the response side."""
    return rule.phase in ("post_model", "both")


def evaluate_policies(
    policies: list[PolicyRule],
    context: PolicyContext,
    semantic_blocker: SemanticBlocker | None = None,
) -> PolicyDecision:
    """Evaluate input-side policies in two phases.

    Phase 1 runs deterministic local checks (PII / regex / size /
    model allow-list). If any of them block, the call is refused and
    Phase 2 is skipped entirely — raw prompt content never leaves
    the SDK process. If Phase 1 sanitizes, the prompt is masked
    locally before Phase 2 sees it.

    Phase 2 runs LLM-backed checks (``semantic_guard``) against the
    possibly-masked prompt. A Phase 2 block overrides a Phase 1
    sanitize, but both records are kept on the decision.

    Rules whose ``phase`` is ``"post_model"`` are skipped entirely
    on this side — they only run during ``evaluate_output_policies``.

    The verdict precedence across all matches is
    ``block > sanitize > allow``. ``semantic_blocker`` is optional;
    when ``None``, ``semantic_guard`` rules become no-ops.
    """
    pre_model = [p for p in policies if _runs_pre_model(p)]
    phase1 = [p for p in pre_model if p.type in _DETERMINISTIC_KINDS]
    phase2 = [p for p in pre_model if p.type in _LLM_BACKED_KINDS]

    phase1_matches = _collect_input_matches(phase1, context, semantic_blocker=None)

    if phase1_matches.has_block:
        return _synthesize_decision(phase1_matches.records)

    text_for_phase2 = context.prompt_text
    if phase1_matches.has_sanitize:
        text_for_phase2, _ = pii_scanner.sanitize(
            text_for_phase2,
            kinds=phase1_matches.sanitize_kinds or None,
            mask_char=phase1_matches.sanitize_mask_char,
        )

    if not phase2:
        return _synthesize_decision(phase1_matches.records)

    phase2_ctx = PolicyContext(
        tenant=context.tenant,
        model=context.model,
        prompt_text=text_for_phase2,
        prompt_chars=len(text_for_phase2),
        stream=context.stream,
    )
    phase2_matches = _collect_input_matches(
        phase2, phase2_ctx, semantic_blocker=semantic_blocker
    )

    return _synthesize_decision(phase1_matches.records + phase2_matches.records)


# ── Internal: phase-walking + decision synthesis ───────────────────────


@dataclass
class _PhaseMatches:
    """Mutable accumulator for a single-phase walk."""
    records: list[MatchedPolicyRecord] = field(default_factory=list)
    sanitize_kinds: list[str] = field(default_factory=list)  # union, ordered
    sanitize_mask_char: str = "#"

    @property
    def has_block(self) -> bool:
        return any(r.verdict == "block" for r in self.records)

    @property
    def has_sanitize(self) -> bool:
        return any(r.verdict == "sanitize" for r in self.records)

    def add(self, rec: MatchedPolicyRecord) -> None:
        self.records.append(rec)
        if rec.verdict == "sanitize":
            for k in rec.sanitize_kinds:
                if k not in self.sanitize_kinds:
                    self.sanitize_kinds.append(k)
            if not any(
                r.verdict == "sanitize" for r in self.records[:-1]
            ):
                self.sanitize_mask_char = rec.sanitize_mask_char


def _collect_input_matches(
    policies: list[PolicyRule],
    context: PolicyContext,
    semantic_blocker: SemanticBlocker | None,
) -> _PhaseMatches:
    out = _PhaseMatches()
    for policy in policies:
        rec = _evaluate_one_input_policy(policy, context, semantic_blocker)
        if rec is not None:
            out.add(rec)
    return out


def _evaluate_one_input_policy(
    policy: PolicyRule,
    context: PolicyContext,
    semantic_blocker: SemanticBlocker | None,
) -> MatchedPolicyRecord | None:
    if policy.type == "allow_model":
        allowed_models = policy.config.get("models", [])
        if isinstance(allowed_models, list) and context.model not in allowed_models:
            return MatchedPolicyRecord(
                name=policy.name,
                type=policy.type,
                verdict="block",
                reason_code="model_not_allowed",
                message=policy.config.get(
                    "message",
                    f"Model '{context.model}' is not allowed for tenant "
                    f"'{context.tenant}'.",
                ),
            )
        return None

    if policy.type == "deny_regex":
        pattern = policy.config.get("pattern")
        if isinstance(pattern, str):
            flags = 0 if policy.config.get("case_sensitive") else re.IGNORECASE
            if safe_search(pattern, context.prompt_text, flags):
                return MatchedPolicyRecord(
                    name=policy.name,
                    type=policy.type,
                    verdict="block",
                    reason_code="prompt_blocked",
                    message=policy.config.get(
                        "message",
                        "Prompt content matched a blocked pattern.",
                    ),
                )
        return None

    if policy.type == "max_prompt_chars":
        max_chars = policy.config.get("max_chars")
        if isinstance(max_chars, int) and context.prompt_chars > max_chars:
            return MatchedPolicyRecord(
                name=policy.name,
                type=policy.type,
                verdict="block",
                reason_code="prompt_too_large",
                message=policy.config.get(
                    "message",
                    f"Prompt size exceeds the allowed limit of {max_chars} characters.",
                ),
            )
        return None

    if policy.type == "pii_scan":
        threshold = policy.config.get("threshold", 0.5)
        enabled_kinds = policy.config.get("kinds")
        action = policy.config.get("action", "block")
        mask_char_cfg = policy.config.get("mask_char", "#")
        mask_char = (
            mask_char_cfg if isinstance(mask_char_cfg, str) and mask_char_cfg
            else "#"
        )
        findings = pii_scanner.scan(context.prompt_text)
        if enabled_kinds and isinstance(enabled_kinds, list):
            findings = [f for f in findings if f.kind in enabled_kinds]
        if not findings:
            return None
        risk = pii_scanner.compute_risk_score(findings)
        if risk < threshold:
            return None
        detected_kinds = sorted({f.kind for f in findings})
        if action == "sanitize":
            return MatchedPolicyRecord(
                name=policy.name,
                type=policy.type,
                verdict="sanitize",
                reason_code="pii_sanitized",
                message=policy.config.get(
                    "message",
                    f"PII redacted before forwarding ({', '.join(detected_kinds)}).",
                ),
                sanitize_kinds=tuple(detected_kinds),
                sanitize_mask_char=mask_char,
            )
        labels = ", ".join(
            f"{f.kind}({f.value_redacted})" for f in findings[:5]
        )
        return MatchedPolicyRecord(
            name=policy.name,
            type=policy.type,
            verdict="block",
            reason_code="pii_detected",
            message=policy.config.get(
                "message",
                f"PII detected (risk={risk:.2f}): {labels}",
            ),
        )

    if policy.type == "semantic_guard":
        return _semantic_guard_match(
            policy=policy,
            text=context.prompt_text,
            semantic_blocker=semantic_blocker,
            side="prompt",
        )

    return None


def _synthesize_decision(
    records: list[MatchedPolicyRecord],
) -> PolicyDecision:
    """Roll a list of matches up into a single ``PolicyDecision``.

    Verdict precedence is ``block > sanitize > allow``. The first
    record at the winning precedence is the primary; the full list
    is carried on ``matched_policies``.
    """
    if not records:
        return PolicyDecision.allow()

    blocks = [r for r in records if r.verdict == "block"]
    if blocks:
        primary = blocks[0]
        return PolicyDecision.deny(
            reason_code=primary.reason_code,
            message=primary.message,
            matched_policy=primary.name,
            matched_policies=tuple(records),
        )

    sanitizes = [r for r in records if r.verdict == "sanitize"]
    if sanitizes:
        primary = sanitizes[0]
        union_kinds: list[str] = []
        for r in sanitizes:
            for k in r.sanitize_kinds:
                if k not in union_kinds:
                    union_kinds.append(k)
        return PolicyDecision.sanitize(
            kinds=union_kinds,
            mask_char=primary.sanitize_mask_char,
            reason_code=primary.reason_code,
            message=primary.message,
            matched_policy=primary.name,
            matched_policies=tuple(records),
        )

    return PolicyDecision.allow()


def _semantic_guard_match(
    *,
    policy: PolicyRule,
    text: str,
    semantic_blocker: SemanticBlocker | None,
    side: str,
) -> MatchedPolicyRecord | None:
    """Returns a block record when the judge flags the text, else ``None``.

    With no live ``SemanticBlocker`` the rule is a no-op — there's
    no keyword fallback, so an unconfigured judge never produces
    false matches.
    """
    if not text or semantic_blocker is None:
        return None
    match = semantic_blocker.check(text, policy.config)
    if match is None:
        return None
    return MatchedPolicyRecord(
        name=policy.name,
        type=policy.type,
        verdict="block",
        reason_code="semantic_blocked",
        message=policy.config.get(
            "message",
            f"{side.capitalize()} matches blocked intent: '{match.intent}'",
        ),
    )


# ── Output-side evaluator ───────────────────────────────────────────────────


def evaluate_output_policies(
    policies: list[PolicyRule],
    context: OutputPolicyContext,
    semantic_blocker: SemanticBlocker | None = None,
) -> PolicyDecision:
    """Evaluate output-side policies and return a ``PolicyDecision``.

    Rules whose ``phase`` is ``"pre_model"`` are skipped — they only
    fire during ``evaluate_policies``. Same precedence rule as the
    input evaluator (``block > sanitize > allow``).
    """
    records: list[MatchedPolicyRecord] = []
    for policy in policies:
        if not _runs_post_model(policy):
            continue
        rec = _evaluate_one_output_policy(policy, context, semantic_blocker)
        if rec is not None:
            records.append(rec)
    return _synthesize_decision(records)


def _evaluate_one_output_policy(
    policy: PolicyRule,
    context: OutputPolicyContext,
    semantic_blocker: SemanticBlocker | None,
) -> MatchedPolicyRecord | None:
    if policy.type == "deny_output_regex":
        pattern = policy.config.get("pattern")
        if isinstance(pattern, str):
            flags = 0 if policy.config.get("case_sensitive") else re.IGNORECASE
            if safe_search(pattern, context.text, flags):
                return MatchedPolicyRecord(
                    name=policy.name,
                    type=policy.type,
                    verdict="block",
                    reason_code="output_blocked",
                    message=policy.config.get(
                        "message",
                        "Model output matched a blocked pattern.",
                    ),
                )
        return None

    if policy.type == "deny_tool_call":
        patterns = policy.config.get("patterns", [])
        if isinstance(patterns, list):
            tool_names = list(context.tool_names)
            tool_names.extend(
                tool_call.get("name", "")
                for tool_call in context.tool_calls
                if isinstance(tool_call.get("name"), str)
            )
            for tool_name in tool_names:
                for pattern in patterns:
                    if isinstance(pattern, str) and safe_search(
                        pattern, tool_name, re.IGNORECASE
                    ):
                        return MatchedPolicyRecord(
                            name=policy.name,
                            type=policy.type,
                            verdict="block",
                            reason_code="tool_call_blocked",
                            message=policy.config.get(
                                "message",
                                f"Tool call '{tool_name}' was blocked.",
                            ),
                        )
        return None

    if policy.type == "deny_bash_command":
        command_patterns = policy.config.get("command_patterns", [])
        tool_patterns = policy.config.get(
            "tool_patterns", [r"^bash$", r"^shell$"]
        )
        if isinstance(command_patterns, list) and isinstance(tool_patterns, list):
            for tool_call in context.tool_calls:
                tool_name = tool_call.get("name", "")
                arguments = tool_call.get("arguments", "")
                if not isinstance(tool_name, str) or not isinstance(arguments, str):
                    continue
                if not any(
                    isinstance(tp, str) and safe_search(tp, tool_name, re.IGNORECASE)
                    for tp in tool_patterns
                ):
                    continue
                for pattern in command_patterns:
                    if isinstance(pattern, str) and safe_search(
                        pattern, arguments, re.IGNORECASE
                    ):
                        return MatchedPolicyRecord(
                            name=policy.name,
                            type=policy.type,
                            verdict="block",
                            reason_code="bash_command_blocked",
                            message=policy.config.get(
                                "message",
                                f"Bash command in tool call '{tool_name}' was blocked.",
                            ),
                        )
        return None

    if policy.type == "deny_mcp_call":
        patterns = policy.config.get("patterns", [])
        if isinstance(patterns, list):
            for target in context.mcp_targets:
                for pattern in patterns:
                    if isinstance(pattern, str) and safe_search(
                        pattern, target, re.IGNORECASE
                    ):
                        return MatchedPolicyRecord(
                            name=policy.name,
                            type=policy.type,
                            verdict="block",
                            reason_code="mcp_call_blocked",
                            message=policy.config.get(
                                "message",
                                f"MCP call '{target}' was blocked.",
                            ),
                        )
        return None

    if policy.type == "semantic_guard":
        return _semantic_guard_match(
            policy=policy,
            text=context.text,
            semantic_blocker=semantic_blocker,
            side="output",
        )

    return None
