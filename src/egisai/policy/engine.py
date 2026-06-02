"""Pure-Python policy rule engine.

Evaluates ``PolicyRule`` objects against an input or output
``PolicyContext`` and returns a ``PolicyDecision``. No I/O.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
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
    """Inputs for evaluating *output-side* policies (after the LLM responds).

    ``allow_sanitize`` controls whether ``pii_scan`` rules with
    ``action="sanitize"`` are honored on this side. Default ``False``
    because the typical output surface (a streamed assistant text
    response) can't be safely rewritten in flight — once the bytes
    have left the provider we have no atomic mutation point. The
    ``claude_agent_sdk`` ``PostToolUse`` hook is the exception:
    the SDK exposes ``updatedToolOutput`` / ``updatedMCPToolOutput``
    which let us swap the tool result before Claude is shown it, so
    a sanitize verdict there is actually enforceable. Patches that
    have such a mutation point flip ``allow_sanitize=True``; every
    other output-side caller leaves it at the default so PII
    detected in a model response still blocks (the conservative
    SOC 2 / GDPR posture — better refuse than silently let it
    through).
    """

    tenant: str
    model: str
    text: str
    tool_names: list[str]
    tool_calls: list[dict[str, str]]
    mcp_targets: list[str]
    stream: bool
    allow_sanitize: bool = False


@dataclass(frozen=True)
class MatchedPolicyRecord:
    """One policy that fired during evaluation.

    ``verdict`` is what this rule would have returned in isolation
    (``'block'`` or ``'sanitize'``); the final ``PolicyDecision.verdict``
    is computed across all matches. ``sanitize_types`` and
    ``sanitize_mask_char`` are only meaningful for sanitize matches.
    """
    name: str
    type: str
    verdict: str
    reason_code: str
    message: str
    # Operator-facing PII type ids (``"ssn"``, ``"credit_card"``, …)
    # to mask before forwarding. Renamed from ``sanitize_kinds`` —
    # the SDK now consistently uses ``type`` to refer to PII
    # categories. ``sanitize_kinds`` is exposed via a backward-
    # compat property on ``PolicyDecision`` for one release.
    sanitize_types: tuple[str, ...] = ()
    sanitize_mask_char: str = "#"

    @property
    def sanitize_kinds(self) -> tuple[str, ...]:
        """Deprecated alias for ``sanitize_types``."""
        return self.sanitize_types


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
    sanitize_types: list[str] = field(default_factory=list)
    sanitize_mask_char: str = "#"

    @property
    def sanitize_kinds(self) -> list[str]:
        """Deprecated alias for ``sanitize_types``.

        Removed in a future release; kept for one release while
        consumers (the audit-event serializer, the framework
        patches) migrate to ``sanitize_types``.
        """
        return self.sanitize_types

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
        types: list[str] | None = None,
        reason_code: str,
        message: str,
        matched_policy: str,
        mask_char: str = "#",
        matched_policies: tuple[MatchedPolicyRecord, ...] = (),
        kinds: list[str] | None = None,
    ) -> PolicyDecision:
        """The call should forward, but with these PII types masked.

        ``types`` is the canonical operator-facing list. ``kinds`` is
        accepted as a deprecated alias for one release so older
        callers keep compiling — passing both raises.
        """
        if types is None and kinds is not None:
            types = kinds
        elif types is not None and kinds is not None:
            raise ValueError(
                "PolicyDecision.sanitize: pass either types= or kinds=, not both."
            )
        return cls(
            verdict="sanitize",
            reason_code=reason_code,
            message=message,
            matched_policy=matched_policy,
            matched_policies=matched_policies,
            sanitize_types=list(types or []),
            sanitize_mask_char=mask_char or "#",
        )


# Deterministic, local-only checks. Adding a new policy kind here
# means it must not issue any network request.
#
# Includes every output-side detector too, because operators can
# now target any rule type on the pre-model phase: when an
# output-typed rule lands here it routes through phase 1 (still
# fully deterministic) and either fires (``deny_output_regex``
# matches prompt text) or silently no-ops (tool / bash / MCP /
# database / financial rules don't have prompt-side signals to
# evaluate against).
_DETERMINISTIC_KINDS = frozenset(
    {
        "allow_model",
        "deny_regex",
        "deny_output_regex",
        "max_prompt_chars",
        "pii_scan",
        "deny_tool_call",
        "deny_bash_command",
        "deny_mcp_call",
        "deny_db_query",
        "deny_financial_action",
    }
)

# Network-issuing checks (LLM judges, embedding lookups, …).
_LLM_BACKED_KINDS = frozenset({"semantic_guard"})


# ── Curated defaults for the runtime-governance policies ────────────
#
# These are battle-tested seed patterns that block the most common
# classes of agentic damage. Operators turn them on by setting
# ``block_dangerous_defaults: true`` in the rule config; they can
# still add their own ``command_patterns`` / ``query_patterns`` /
# ``action_patterns`` on top. The defaults are deliberately
# conservative — false-positives are easier to debug than the
# alternative.

# Bash / shell command patterns that almost always indicate
# destructive intent. Used by ``deny_bash_command`` when
# ``block_dangerous_defaults`` is set.
_DEFAULT_DANGEROUS_BASH_PATTERNS: tuple[str, ...] = (
    # Recursive force-deletes — the textbook agent footgun.
    r"\brm\s+(-\w*r\w*\s+)+",
    r"\brm\s+-rf?\b",
    # Disk-wreckers.
    r"\bdd\s+if=",
    r"\bmkfs(\.\w+)?\b",
    r"\bshred\b",
    # Fork-bombs and unbounded background loops.
    r":\(\)\s*\{\s*:\|:&\s*\}\s*;",
    # "Pipe a script from the internet straight into a shell."
    r"\bcurl\s+[^|]*\|\s*(bash|sh|zsh)\b",
    r"\bwget\s+[^|]*\|\s*(bash|sh|zsh)\b",
    # Privilege escalation + remote code exec primitives.
    r"\bsudo\s+",
    r"\bchmod\s+(?:\+s|[0-7]?7[0-7][0-7])\b",
    r"\beval\s+\$",
    # Common lateral-movement / credential-leak verbs.
    r"\bnetcat\b|\bnc\s+-",
    r"\b(scp|rsync)\s+.*@",
)

# SQL operations that mutate or destroy data at scale. Used by
# ``deny_db_query`` when ``dangerous_operations`` isn't set
# explicitly.
_DEFAULT_DANGEROUS_DB_OPERATIONS: tuple[str, ...] = (
    "DROP",
    "TRUNCATE",
    "DELETE",
    "ALTER",
    "GRANT",
    "REVOKE",
    "CREATE USER",
    "DROP USER",
)

# Financial / money-movement verbs. The default list of action
# patterns scanned against tool names by ``deny_financial_action``
# when the rule's ``action_patterns`` is empty. Conservative
# enough to fire on real money flows but not generic CRUD.
#
# We use *letter* boundaries (``(?<![a-zA-Z])`` / ``(?![a-zA-Z])``)
# instead of regex ``\b`` because tool names commonly use
# ``snake_case`` (``stripe_payout``, ``acme_transfer``) and
# ``camelCase`` (``transferFunds``); ``\b`` treats ``_`` as a
# word character, so ``\btransfer\b`` would NOT match
# ``stripe_transfer`` — the most common real-world naming.
# Letter boundaries match all four conventions while still
# rejecting partial matches like ``transferred``.
_DEFAULT_FINANCIAL_VERBS: tuple[str, ...] = (
    r"(?<![a-zA-Z])transfer(?![a-zA-Z])",
    r"(?<![a-zA-Z])charge(?![a-zA-Z])",
    r"(?<![a-zA-Z])refund(?![a-zA-Z])",
    r"(?<![a-zA-Z])payout(?![a-zA-Z])",
    r"(?<![a-zA-Z])withdraw(?![a-zA-Z])",
    r"(?<![a-zA-Z])wire(?![a-zA-Z])",
    r"(?<![a-zA-Z])ach(?![a-zA-Z])",
    r"(?<![a-zA-Z])debit(?![a-zA-Z])",
    r"send[_\s-]*money",
    r"(?<![a-zA-Z])purchase(?![a-zA-Z])",
    r"initiate[_\s-]*payment",
)

# JSON-argument-shaped fields most financial APIs use to carry
# the amount. Operators can override via ``amount_field`` in the
# rule config; we walk the parsed JSON for any of these on a
# best-effort basis.
_DEFAULT_AMOUNT_FIELD_NAMES: tuple[str, ...] = (
    "amount",
    "amount_usd",
    "amount_cents",
    "value",
    "total",
    "sum",
    "price",
)


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
            types=phase1_matches.sanitize_types or None,
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
    sanitize_types: list[str] = field(default_factory=list)  # union, ordered
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
            for t in rec.sanitize_types:
                if t not in self.sanitize_types:
                    self.sanitize_types.append(t)
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
    """Evaluate one rule on the prompt side.

    The dispatcher handles every type the engine knows about. Types
    that have no meaningful prompt-side signal (``deny_tool_call``,
    ``deny_bash_command``, ``deny_mcp_call``) silently return
    ``None`` — operators can freely target them on the pre-model
    phase without breaking the call, but the rule simply doesn't
    fire here. ``deny_output_regex`` runs on prompt text the same
    way ``deny_regex`` does so an operator who picked it on the
    pre-model side still gets prompt-pattern enforcement.
    """
    if policy.type == "allow_model":
        return _allow_model_match(policy, context.model, context.tenant)

    if policy.type in ("deny_regex", "deny_output_regex"):
        return _deny_pattern_match(
            policy,
            text=context.prompt_text,
            reason_code="prompt_blocked",
            default_message="Prompt content matched a blocked pattern.",
        )

    if policy.type == "max_prompt_chars":
        return _max_chars_match(
            policy,
            chars=context.prompt_chars,
            reason_code="prompt_too_large",
            default_message_template=(
                "Prompt size exceeds the allowed limit of "
                "{max_chars} characters."
            ),
        )

    if policy.type == "pii_scan":
        return _pii_scan_match(
            policy,
            text=context.prompt_text,
            allow_sanitize=True,
            block_reason_code="pii_detected",
        )

    if policy.type == "semantic_guard":
        return _semantic_guard_match(
            policy=policy,
            text=context.prompt_text,
            semantic_blocker=semantic_blocker,
            side="prompt",
        )

    # Tool / bash / MCP rules need response-side signals
    # (tool_names, tool_calls, mcp_targets) that ``PolicyContext``
    # does not carry today. Operators may still target them on the
    # pre-model phase via the open phase picker; the rule silently
    # no-ops here so the call isn't broken. They fire normally
    # when ``phase`` includes ``post_model``.
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
        union_types: list[str] = []
        for r in sanitizes:
            for t in r.sanitize_types:
                if t not in union_types:
                    union_types.append(t)
        return PolicyDecision.sanitize(
            types=union_types,
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
    tool_calls: list[dict[str, Any]] | None = None,
) -> MatchedPolicyRecord | None:
    """Returns a block record when the judge flags the call, else ``None``.

    Operates against any subset of the available signals selected by
    ``policy.config["targets"]``:

    - ``["text"]`` (the default — preserves pre-0.24 behavior) — the
      judge receives ``text``: the user prompt on the input side or
      the model's accumulated assistant text on the output side.
    - ``["tool_calls"]`` — for each entry in ``tool_calls``, the
      matcher synthesizes a one-sentence description ("The agent is
      requesting to invoke tool 'X' with arguments {...}") and asks
      the judge whether that matches any operator intent. This is
      what closes the "agent makes a mistake" gap: a ``deny_tool_call``
      rule needs to enumerate every dangerous tool name by hand,
      whereas a ``semantic_guard`` rule with ``targets=["tool_calls"]``
      lets the operator describe forbidden behavior in plain English
      ("delete all users", "wipe the production database") and the
      judge decides whether THIS call matches THAT intent.
    - ``["text", "tool_calls"]`` — both, ``text`` first.

    With no live ``SemanticBlocker`` the rule is a no-op — there's
    no keyword fallback, so an unconfigured judge never produces
    false matches.

    Privacy contract (security-and-compliance.mdc §1) — tool args
    are PII-label-redacted via ``pii_scanner.label_redact`` BEFORE
    they reach the judge. The judge is the platform's own endpoint,
    but the rule that "PII never leaves the SDK boundary in raw
    form, including our own LLM-based policy judges" still applies.
    Intent classification accuracy is preserved because the judge
    cares about the verb/noun shape ("the agent is deleting users"),
    not the exact identifier values.

    Each tool call counts as one judge round-trip. The matcher
    short-circuits on the first match for cost control. Operators
    who want to scan many tools per turn can mitigate cost by
    setting tighter ``deny_tool_call`` rules in Phase 1 — those
    fire deterministically and prevent the judge call entirely on
    the obviously-blocked tools.
    """
    if semantic_blocker is None:
        return None

    targets_raw = policy.config.get("targets")
    if isinstance(targets_raw, list) and targets_raw:
        targets = [str(t) for t in targets_raw if isinstance(t, str)]
    else:
        # Backwards compat: a ``semantic_guard`` rule without an
        # explicit ``targets`` field behaves exactly the same as
        # every released version of this SDK — judge the text.
        targets = ["text"]

    # Phase A — text target. Kept verbatim from the pre-0.24 path
    # so existing rules (no ``targets`` field) cannot regress.
    if "text" in targets and text:
        match = semantic_blocker.check(text, policy.config)
        if match is not None:
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

    # Phase B — tool_calls target. Each tool gets its own judge
    # call so the audit row's ``matched_policy`` message can name
    # the specific tool that tripped the rule. First-match-by-input-
    # order wins.
    #
    # Parallelism (BUG 3 fix): pre-fix this loop was strictly
    # sequential — N tool calls in a turn meant N round-trips back-
    # to-back to ``/v1/sdk/judge``. With a P50 judge latency of
    # ~600 ms, a turn that calls 6 tools paid ~3.6 s in policy
    # latency on its own. Now each tool's judge call runs on its
    # own thread (ThreadPoolExecutor), so wall-clock latency for
    # the whole batch collapses toward the *single slowest* call
    # rather than the *sum*.
    #
    # Why threads, not asyncio: ``_semantic_guard_match`` is a
    # synchronous helper called from synchronous patch paths
    # (sync OpenAI/Anthropic users) AND from worker threads spawned
    # by ``asyncio.to_thread`` in the async patches. Either way the
    # inner unit of work — ``semantic_blocker.check`` — is a
    # blocking ``httpx.Client`` POST. Threads parallelize cleanly
    # without forcing the engine to be async-aware.
    #
    # Why bounded: the platform's judge endpoint enforces a per-
    # tenant rate limit. Spawning 100 parallel calls for a 100-tool
    # turn would just rate-limit ourselves into ``Retry-After``
    # storms (now bounded by ``judge_retry_after_max_secs`` — see
    # BUG 8 — but still wasteful). 8 is the empirical sweet spot:
    # most turns have ≤ 4 tools so we never hit the ceiling, and
    # outlier turns with many tools degrade gracefully into
    # serialized batches of 8.
    #
    # First-match-by-input-order semantics are preserved: we
    # collect every result, then walk the input order and return
    # the first match. Cost trade-off: in the rare case where the
    # *first* tool would have matched and short-circuited the rest,
    # we still pay all N judge calls under parallelism. That's
    # acceptable — the *common* case (no match across N tools) is
    # exactly the case that benefits most from parallelism.
    if "tool_calls" in targets and tool_calls:
        normalized: list[tuple[str, str]] = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name")
            if not isinstance(name, str) or not name:
                continue
            args = tc.get("input")
            if args is None:
                args = tc.get("arguments")
            synthesized = _synthesize_tool_call_text(name, args)
            if not synthesized:
                continue
            normalized.append((name, synthesized))

        if not normalized:
            return None

        # Single-tool: skip the executor overhead entirely. Most
        # turns hit this path.
        if len(normalized) == 1:
            name, synthesized = normalized[0]
            match = semantic_blocker.check(synthesized, policy.config)
            if match is None:
                return None
            return MatchedPolicyRecord(
                name=policy.name,
                type=policy.type,
                verdict="block",
                reason_code="semantic_blocked_tool",
                message=policy.config.get(
                    "message",
                    (
                        f"Tool call '{name}' matches blocked intent: "
                        f"'{match.intent}'"
                    ),
                ),
            )

        # Parallel path. ``max_workers`` is min(8, batch size) so a
        # 3-tool turn doesn't spawn 8 idle threads.
        max_workers = min(_TOOL_JUDGE_MAX_WORKERS, len(normalized))
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="egisai-judge",
        ) as pool:
            futures = [
                pool.submit(semantic_blocker.check, synthesized, policy.config)
                for _, synthesized in normalized
            ]
            results: list[Any] = []
            for f in futures:
                try:
                    results.append(f.result())
                except Exception:  # noqa: BLE001
                    # A single judge call failed; treat as no-match
                    # for that tool and continue. Honours the
                    # platform-level fail-open contract — one dead
                    # tool eval doesn't poison the whole batch.
                    results.append(None)

        for (name, _synthesized), match in zip(normalized, results, strict=True):
            if match is None:
                continue
            return MatchedPolicyRecord(
                name=policy.name,
                type=policy.type,
                verdict="block",
                reason_code="semantic_blocked_tool",
                message=policy.config.get(
                    "message",
                    (
                        f"Tool call '{name}' matches blocked intent: "
                        f"'{match.intent}'"
                    ),
                ),
            )

    return None


# Cap on the per-call thread pool the parallel tool_calls judge
# matcher uses. Tuning notes live above the call site; this
# constant is module-level so tests / advanced operators could
# monkeypatch it temporarily without forking the engine.
_TOOL_JUDGE_MAX_WORKERS = 8


def _synthesize_tool_call_text(name: str, args: Any) -> str:
    """Render a tool call as a sentence the judge can intent-classify.

    Shape: ``"The agent is requesting to invoke tool 'X' with
    arguments: {...}"``. Natural-language form is deliberate — the
    judge prompt on the platform side is tuned for intent
    classification of free-text agent behavior descriptions, not
    for arbitrary code-shaped strings.

    PII in the arguments is replaced with typed labels (``<EMAIL>``,
    ``<SSN>``, ``<CREDIT_CARD>``, …) via ``pii_scanner.label_redact``.
    This satisfies security-and-compliance.mdc §1 — raw PII never
    leaves the SDK boundary, including on its way to our own
    LLM-based judges. The judge keeps enough structural context to
    decide intent ("the agent is deleting <NAME>" still classifies
    as a destructive user operation) without ever holding the real
    value.
    """
    import json

    try:
        rendered_args = json.dumps(args, default=str, ensure_ascii=False)
    except Exception:  # noqa: BLE001
        rendered_args = repr(args)
    safe_args = pii_scanner.label_redact(rendered_args)
    return (
        f"The agent is requesting to invoke tool '{name}' "
        f"with arguments: {safe_args}"
    )


# ── Shared per-type evaluators (phase-symmetric) ────────────────────────
#
# Each helper takes the rule's ``config`` plus whichever signals it
# needs (text, char count, model name) and returns a match record
# or ``None``. Both ``_evaluate_one_input_policy`` and
# ``_evaluate_one_output_policy`` call into these so a rule
# behaves identically on either phase, with the only side-specific
# difference being the ``reason_code`` (``prompt_blocked`` vs
# ``output_blocked``, etc.) — which downstream copy templates use
# to phrase the audit narrative correctly.


def _allow_model_match(
    policy: PolicyRule,
    model: str,
    tenant: str,
) -> MatchedPolicyRecord | None:
    """Block when the call's model isn't on the operator's allow-list."""
    allowed_models = policy.config.get("models", [])
    if isinstance(allowed_models, list) and model not in allowed_models:
        return MatchedPolicyRecord(
            name=policy.name,
            type=policy.type,
            verdict="block",
            reason_code="model_not_allowed",
            message=policy.config.get(
                "message",
                f"Model '{model}' is not allowed for tenant '{tenant}'.",
            ),
        )
    return None


def _deny_pattern_match(
    policy: PolicyRule,
    *,
    text: str,
    reason_code: str,
    default_message: str,
) -> MatchedPolicyRecord | None:
    """Block when ``text`` matches the operator's regex pattern."""
    pattern = policy.config.get("pattern")
    if not isinstance(pattern, str):
        return None
    flags = 0 if policy.config.get("case_sensitive") else re.IGNORECASE
    if not safe_search(pattern, text, flags):
        return None
    return MatchedPolicyRecord(
        name=policy.name,
        type=policy.type,
        verdict="block",
        reason_code=reason_code,
        message=policy.config.get("message", default_message),
    )


def _max_chars_match(
    policy: PolicyRule,
    *,
    chars: int,
    reason_code: str,
    default_message_template: str,
) -> MatchedPolicyRecord | None:
    """Block when the relevant text exceeds the configured cap."""
    max_chars = policy.config.get("max_chars")
    if not isinstance(max_chars, int) or chars <= max_chars:
        return None
    return MatchedPolicyRecord(
        name=policy.name,
        type=policy.type,
        verdict="block",
        reason_code=reason_code,
        message=policy.config.get(
            "message",
            default_message_template.format(max_chars=max_chars),
        ),
    )


def _pii_scan_match(
    policy: PolicyRule,
    *,
    text: str,
    allow_sanitize: bool,
    block_reason_code: str,
) -> MatchedPolicyRecord | None:
    """Scan ``text`` for PII and translate the operator's action.

    ``allow_sanitize`` controls whether ``action="sanitize"`` from
    the rule's config is honored (the prompt side wires
    sanitization through to the patched provider call) or coerced
    to block (the response side has no sanitization plumbing yet).
    ``block_reason_code`` lets each side stamp its own reason code
    so audit narratives can phrase the outcome correctly.

    Config field names: ``types`` is the canonical operator-facing
    list; ``kinds`` is accepted as a deprecated alias for one
    release. When neither is set we run every detector. We also
    surface a single warning to stderr if the operator listed a
    type the engine doesn't know how to detect — that's the bug
    that used to silently no-op when ``"passport"`` was typed into
    the legacy free-text ``kinds`` field.
    """
    threshold = policy.config.get("threshold", 0.5)
    enabled_types_raw = policy.config.get("types") or policy.config.get("kinds")
    # Default action is ``sanitize`` — the less-destructive choice.
    # Sanitize forwards the call to the model with the PII masked, so
    # the user's experience continues unaffected while the regulated
    # values never leave the SDK boundary. Operators who need a hard
    # refusal (e.g. an explicit "no SSNs in prompts" compliance bar)
    # opt into ``action: "block"`` in the policy config; the
    # dashboard's checkbox grid surfaces both options. On the
    # response side ``allow_sanitize`` is ``False`` and the engine
    # automatically falls through to block — we can't safely rewrite
    # provider responses, so a detected leak in the response is
    # always refused.
    action = policy.config.get("action", "sanitize")
    mask_char_cfg = policy.config.get("mask_char", "#")
    mask_char = (
        mask_char_cfg if isinstance(mask_char_cfg, str) and mask_char_cfg
        else "#"
    )

    enabled_types: list[str] | None = None
    if enabled_types_raw and isinstance(enabled_types_raw, list):
        # We deliberately do NOT validate the configured types against
        # the canonical taxonomy here. The platform's policy
        # create/update endpoint rejects unknown types at write time
        # (see ``backend/app/schemas/policy.py::_normalize_pii_config``),
        # so live policies on the wire are already vetted. Legacy
        # rows from before the rename window may still carry display
        # labels or stray strings — the membership filter below
        # silently drops anything that doesn't match a real finding
        # type without printing a per-call warning on the hot path.
        enabled_types = [str(t) for t in enabled_types_raw if isinstance(t, str)]

    findings = pii_scanner.scan(text)
    if enabled_types is not None:
        findings = [f for f in findings if f.type in enabled_types]
    if not findings:
        return None
    risk = pii_scanner.compute_risk_score(findings)
    if risk < threshold:
        return None
    detected_types = sorted({f.type for f in findings})
    if action == "sanitize" and allow_sanitize:
        return MatchedPolicyRecord(
            name=policy.name,
            type=policy.type,
            verdict="sanitize",
            reason_code="pii_sanitized",
            message=policy.config.get(
                "message",
                f"PII redacted before forwarding ({', '.join(detected_types)}).",
            ),
            sanitize_types=tuple(detected_types),
            sanitize_mask_char=mask_char,
        )
    labels = ", ".join(
        f"{f.type}({f.value_redacted})" for f in findings[:5]
    )
    return MatchedPolicyRecord(
        name=policy.name,
        type=policy.type,
        verdict="block",
        reason_code=block_reason_code,
        message=policy.config.get(
            "message",
            f"PII detected (risk={risk:.2f}): {labels}",
        ),
    )


# ── Output-side evaluator ───────────────────────────────────────────────────


def evaluate_output_policies(
    policies: list[PolicyRule],
    context: OutputPolicyContext,
    semantic_blocker: SemanticBlocker | None = None,
) -> PolicyDecision:
    """Evaluate output-side policies in two phases.

    Mirrors ``evaluate_policies`` exactly: deterministic local
    checks run first, LLM-backed checks (``semantic_guard``) run
    afterwards — and only when Phase 1 didn't already block. This
    is the same security contract the prompt side honors
    (security-and-compliance.mdc §2): no LLM call, no token spend,
    no chance of forwarding sensitive content to a judge once a
    local rule has already refused the response.

    Rules whose ``phase`` is ``"pre_model"`` are skipped — they
    only fire during ``evaluate_policies``. Verdict precedence
    across all matches is ``block > sanitize > allow``.
    """
    post_model = [p for p in policies if _runs_post_model(p)]
    phase1 = [p for p in post_model if p.type in _DETERMINISTIC_KINDS]
    phase2 = [p for p in post_model if p.type in _LLM_BACKED_KINDS]

    # Phase 1 — every match is deterministic and local. The judge
    # is intentionally not threaded in here so a misclassified type
    # never reaches the network during this phase.
    phase1_matches = _collect_output_matches(
        phase1, context, semantic_blocker=None
    )

    # Hard short-circuit on a Phase 1 block: never call the judge
    # after a local rule has already refused the response.
    # Sanitize on the output side is coerced to block by
    # ``_pii_scan_match`` (the SDK can't safely rewrite provider
    # responses), so a Phase 1 sanitize is impossible by
    # construction — but the ``has_block`` guard here mirrors the
    # prompt side regardless, so the contract reads identically.
    if phase1_matches.has_block:
        return _synthesize_decision(phase1_matches.records)

    if not phase2:
        return _synthesize_decision(phase1_matches.records)

    phase2_matches = _collect_output_matches(
        phase2, context, semantic_blocker=semantic_blocker
    )

    return _synthesize_decision(
        phase1_matches.records + phase2_matches.records
    )


def _collect_output_matches(
    policies: list[PolicyRule],
    context: OutputPolicyContext,
    semantic_blocker: SemanticBlocker | None,
) -> _PhaseMatches:
    """Walk a list of post-model rules and accumulate matches.

    Symmetrical to ``_collect_input_matches``. Used by the
    two-phase ``evaluate_output_policies`` to walk Phase 1 with a
    ``None`` blocker (no network) and Phase 2 with the live
    blocker. Each phase's records are appended to the same
    ``_PhaseMatches`` shape used on the input side, so the
    downstream synthesizer is one path for both evaluators.
    """
    out = _PhaseMatches()
    for policy in policies:
        rec = _evaluate_one_output_policy(policy, context, semantic_blocker)
        if rec is not None:
            out.add(rec)
    return out


def _evaluate_one_output_policy(
    policy: PolicyRule,
    context: OutputPolicyContext,
    semantic_blocker: SemanticBlocker | None,
) -> MatchedPolicyRecord | None:
    """Evaluate one rule on the response side.

    Mirror image of ``_evaluate_one_input_policy``. Every type the
    engine knows about is handled — including the input-side text
    detectors (``pii_scan``, ``deny_regex``, ``max_prompt_chars``,
    ``allow_model``) so operators can target them post-model and
    have the rule actually fire on the response.

    ``pii_scan`` post-model with ``action="sanitize"`` is coerced
    to ``block``: the SDK can mutate prompts before they ship, but
    rewriting a provider's response payload safely across every
    framework is out of scope, so the operator's intent (catch
    leaked PII) is preserved by refusing the response instead of
    silently letting it through.
    """
    if policy.type == "allow_model":
        return _allow_model_match(policy, context.model, context.tenant)

    if policy.type in ("deny_regex", "deny_output_regex"):
        return _deny_pattern_match(
            policy,
            text=context.text,
            reason_code="output_blocked",
            default_message="Model output matched a blocked pattern.",
        )

    if policy.type == "max_prompt_chars":
        return _max_chars_match(
            policy,
            chars=len(context.text),
            reason_code="output_too_large",
            default_message_template=(
                "Response size exceeds the allowed limit of "
                "{max_chars} characters."
            ),
        )

    if policy.type == "pii_scan":
        # Output-side sanitization is wired only through patches
        # that have an atomic mutation point AFTER the model
        # produced the bytes (today: ``claude_agent_sdk``'s
        # PostToolUse hook, which can swap the tool result via
        # ``updatedToolOutput`` / ``updatedMCPToolOutput`` before
        # the model is shown it). On every other output surface
        # (streamed assistant text, finalized model_call responses)
        # we coerce to block: there is no safe in-place rewrite of
        # a response that's already on the wire to the user, and
        # the SOC 2 / GDPR conservative posture is "refuse rather
        # than silently let through". The patch tells us which side
        # of that line we're on via ``context.allow_sanitize``.
        return _pii_scan_match(
            policy,
            text=context.text,
            allow_sanitize=context.allow_sanitize,
            block_reason_code="pii_in_output",
        )

    if policy.type == "deny_tool_call":
        return _deny_tool_call_match(policy, context)

    if policy.type == "deny_bash_command":
        return _deny_bash_command_match(policy, context)

    if policy.type == "deny_mcp_call":
        return _deny_mcp_call_match(policy, context)

    if policy.type == "deny_db_query":
        return _deny_db_query_match(policy, context)

    if policy.type == "deny_financial_action":
        return _deny_financial_action_match(policy, context)

    if policy.type == "semantic_guard":
        return _semantic_guard_match(
            policy=policy,
            text=context.text,
            tool_calls=context.tool_calls,
            semantic_blocker=semantic_blocker,
            side="output",
        )

    return None


# ── Runtime-governance evaluators ───────────────────────────────────────
#
# These four evaluators implement the "runtime control plane" surface
# the platform exposes via the ``deny_tool_call`` / ``deny_bash_command``
# / ``deny_mcp_call`` / ``deny_db_query`` / ``deny_financial_action``
# policy types. They share three properties:
#
# 1. **Local-only.** Pure-Python regex against signals already extracted
#    in ``_output_signals.py``. No network, no LLM judge, no extra
#    state — they fit cleanly in Phase 1 of the two-phase contract.
# 2. **Best-effort.** Each evaluator inspects the structured
#    ``tool_calls`` / ``mcp_targets`` lists the patches collected. When
#    a provider didn't ship those signals (older providers, bare HTTP
#    fallback) the rule silently no-ops — fail-open on availability
#    per the SDK design philosophy.
# 3. **Argument-aware.** Where it makes sense (tool args, SQL query
#    strings, financial amounts) the evaluator parses the
#    JSON-serialized ``arguments`` blob the patches normalize so a
#    rule can introspect *what* the tool was being called with, not
#    just *whether* the tool exists. A tool name allow-list isn't
#    enough on its own — ``send_message(text="DROP TABLE users")``
#    looks innocuous on the name alone.


def _config_str_list(config: dict[str, Any], key: str) -> list[str]:
    """Read a config value that should be ``list[str]``, defensively.

    Returns ``[]`` for any malformed value (None, dict, mixed list,
    string-instead-of-list). Mismatched config never raises here —
    the rule simply does nothing, matching the SDK's fail-open-on-
    availability contract. The same helper is used by every
    runtime-governance evaluator so a typo in a single rule's config
    can't break the whole policy walk.
    """
    raw = config.get(key)
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, str) and item]


def _parse_tool_arguments(arguments: str) -> Any:
    """Parse a tool-call ``arguments`` JSON string into a Python value.

    The patches in ``_output_signals.py`` always coerce arguments to
    a JSON string (sometimes via ``json.dumps`` of a dict the
    provider already structured). Returns ``None`` when the string
    isn't valid JSON — the caller treats that as "no structured
    args available" and skips structural checks.
    """
    if not arguments:
        return None
    try:
        import json as _json

        return _json.loads(arguments)
    except Exception:  # noqa: BLE001
        return None


def _walk_amount_values(obj: Any, field_names: tuple[str, ...]) -> list[float]:
    """Collect every numeric value in ``obj`` keyed by one of
    ``field_names`` (recursive).

    Used by ``deny_financial_action`` to find an amount-shaped value
    inside a tool's arguments without committing to a single schema —
    every payment provider names the field a little differently
    (``amount``, ``amount_cents``, ``value``…). Strings that parse
    as numbers (``"100.00"``) are accepted; non-numeric strings,
    ``None``, and booleans are skipped silently.
    """
    out: list[float] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() in {f.lower() for f in field_names}:
                if isinstance(v, bool):
                    # ``bool`` is a subclass of ``int``; skip it
                    # explicitly so ``True``/``False`` don't read as
                    # 1/0 amounts.
                    continue
                if isinstance(v, int | float):
                    out.append(float(v))
                elif isinstance(v, str):
                    try:
                        out.append(float(v))
                    except ValueError:
                        pass
            else:
                out.extend(_walk_amount_values(v, field_names))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_walk_amount_values(item, field_names))
    return out


def _deny_tool_call_match(
    policy: PolicyRule,
    context: OutputPolicyContext,
) -> MatchedPolicyRecord | None:
    """Block when the model invokes (or registers) a tool that
    matches one of the operator's patterns.

    Three independent matching axes:

    * ``patterns`` — regex against the tool *name* (definition or
      live call). The original behavior, retained verbatim.
    * ``argument_patterns`` — regex against the JSON-serialized
      ``arguments`` blob of each live tool call. Catches dangerous
      usage of an otherwise-legitimate tool (e.g. an allow-listed
      ``http_get`` being pointed at an internal IP). Empty / missing
      list = skipped.
    * ``argument_max_chars`` — integer cap on the size of any
      single tool call's serialized arguments. Stops accidental /
      adversarial dumps from hitting downstream side-effects.
    """
    name_patterns = _config_str_list(policy.config, "patterns")
    arg_patterns = _config_str_list(policy.config, "argument_patterns")
    raw_max_args = policy.config.get("argument_max_chars")
    arg_max_chars: int | None = (
        int(raw_max_args)
        if isinstance(raw_max_args, int) and not isinstance(raw_max_args, bool)
        else None
    )

    # Axis 1: tool name (definition + live call). Walk both lists.
    candidate_names = list(context.tool_names)
    candidate_names.extend(
        tc.get("name", "") for tc in context.tool_calls
        if isinstance(tc.get("name"), str)
    )
    for tool_name in candidate_names:
        for pattern in name_patterns:
            if safe_search(pattern, tool_name, re.IGNORECASE):
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

    # Axes 2 + 3: per-call argument inspection. Only meaningful for
    # *live* tool calls — definitions don't carry arguments.
    if arg_patterns or arg_max_chars is not None:
        for tc in context.tool_calls:
            tool_name = tc.get("name", "") or ""
            arguments = tc.get("arguments", "") or ""
            if not isinstance(arguments, str):
                continue
            if (
                arg_max_chars is not None
                and len(arguments) > arg_max_chars
            ):
                return MatchedPolicyRecord(
                    name=policy.name,
                    type=policy.type,
                    verdict="block",
                    reason_code="tool_call_blocked",
                    message=policy.config.get(
                        "message",
                        f"Tool call '{tool_name}' arguments exceed "
                        f"the {arg_max_chars}-char limit.",
                    ),
                )
            for pattern in arg_patterns:
                if safe_search(pattern, arguments, re.IGNORECASE):
                    return MatchedPolicyRecord(
                        name=policy.name,
                        type=policy.type,
                        verdict="block",
                        reason_code="tool_call_blocked",
                        message=policy.config.get(
                            "message",
                            f"Tool call '{tool_name}' arguments "
                            f"matched a blocked pattern.",
                        ),
                    )
    return None


def _deny_bash_command_match(
    policy: PolicyRule,
    context: OutputPolicyContext,
) -> MatchedPolicyRecord | None:
    """Block shell-shaped tool invocations when their command matches
    a dangerous pattern.

    ``tool_patterns`` (default ``[r"^bash$", r"^shell$"]``) gates
    *which* tools count as a shell. ``command_patterns`` is the
    operator's regex list against each call's argument string.
    Setting ``block_dangerous_defaults: true`` also unions in the
    curated ``_DEFAULT_DANGEROUS_BASH_PATTERNS`` list — the
    "everyone wants this" preset that catches ``rm -rf``, fork
    bombs, ``curl | sh``, sudo, etc., without making the operator
    re-discover the patterns from first principles.
    """
    tool_patterns = _config_str_list(policy.config, "tool_patterns") or [
        r"^bash$", r"^shell$",
    ]
    command_patterns = _config_str_list(policy.config, "command_patterns")
    if policy.config.get("block_dangerous_defaults"):
        # Append the curated defaults; preserve operator additions
        # at the front so explicit patterns still take precedence
        # in the matching order.
        command_patterns = list(command_patterns) + list(
            _DEFAULT_DANGEROUS_BASH_PATTERNS
        )

    if not command_patterns:
        return None

    for tool_call in context.tool_calls:
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", "")
        if not isinstance(tool_name, str) or not isinstance(arguments, str):
            continue
        if not any(
            safe_search(tp, tool_name, re.IGNORECASE) for tp in tool_patterns
        ):
            continue
        for pattern in command_patterns:
            if safe_search(pattern, arguments, re.IGNORECASE):
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


def _deny_mcp_call_match(
    policy: PolicyRule,
    context: OutputPolicyContext,
) -> MatchedPolicyRecord | None:
    """Govern MCP traffic on three independent axes.

    * ``patterns`` — regex blocklist against MCP target strings.
      The original behavior, retained.
    * ``allowed_servers`` — *allowlist* of substring-match server
      identifiers. When non-empty, ANY MCP target that doesn't
      match at least one entry is blocked. This is the "deny by
      default" mode — the safer posture for production agents.
    * ``denied_resources`` — additional regex blocklist scoped to
      MCP resource paths / URIs (a separate axis from server
      identity, useful when one server hosts multiple resources
      with different sensitivity).
    """
    deny_patterns = _config_str_list(policy.config, "patterns")
    allowed_servers = _config_str_list(policy.config, "allowed_servers")
    denied_resources = _config_str_list(policy.config, "denied_resources")

    if not context.mcp_targets:
        return None

    for target in context.mcp_targets:
        # Allowlist pass: when configured, the target MUST match
        # at least one entry. Substring (case-insensitive) is the
        # operator-friendly default; an entry like ``"prod"``
        # allows ``"prod.acme.io/db"`` but blocks
        # ``"staging.acme.io/db"``.
        if allowed_servers:
            target_lc = target.lower()
            if not any(s.lower() in target_lc for s in allowed_servers):
                return MatchedPolicyRecord(
                    name=policy.name,
                    type=policy.type,
                    verdict="block",
                    reason_code="mcp_call_blocked",
                    message=policy.config.get(
                        "message",
                        f"MCP server '{target}' is not on the allowlist.",
                    ),
                )

        # Denylist passes: explicit patterns override.
        for pattern in deny_patterns:
            if safe_search(pattern, target, re.IGNORECASE):
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
        for pattern in denied_resources:
            if safe_search(pattern, target, re.IGNORECASE):
                return MatchedPolicyRecord(
                    name=policy.name,
                    type=policy.type,
                    verdict="block",
                    reason_code="mcp_call_blocked",
                    message=policy.config.get(
                        "message",
                        f"MCP resource '{target}' is on the denied list.",
                    ),
                )
    return None


def _deny_db_query_match(
    policy: PolicyRule,
    context: OutputPolicyContext,
) -> MatchedPolicyRecord | None:
    """Block SQL-shaped tool calls that touch dangerous tables /
    operations.

    Detection is **content-based**, not tool-name-based: agents
    invoke databases under a thousand different tool wrappers
    (``run_sql``, ``execute_query``, ``db.query``, ``snowflake_run``).
    Looking only at tool names misses the long tail. We instead
    scan the arguments blob of every tool call for SQL-like text.

    Three independent matching axes:

    * ``query_patterns`` — operator's full-regex list against the
      argument string of each tool call.
    * ``denied_tables`` — table names. We match
      ``\\b(FROM|UPDATE|INTO|TABLE)\\s+["`]?<table>\\b``.
    * ``dangerous_operations`` — top-level SQL verbs (default:
      DROP / TRUNCATE / DELETE / ALTER / GRANT). Set
      ``block_dangerous_defaults: false`` to disable.

    Operators can scope this to specific tools via ``tool_patterns``
    (default: any tool whose call arguments look SQL-shaped).
    """
    query_patterns = _config_str_list(policy.config, "query_patterns")
    denied_tables = _config_str_list(policy.config, "denied_tables")
    raw_ops = policy.config.get("dangerous_operations")
    if isinstance(raw_ops, list):
        dangerous_ops = [o for o in raw_ops if isinstance(o, str)]
    elif policy.config.get("block_dangerous_defaults", True):
        # Default-on: most operators want the curated list to fire
        # automatically when this rule is created. Opt-out by
        # setting ``dangerous_operations: []`` explicitly.
        dangerous_ops = list(_DEFAULT_DANGEROUS_DB_OPERATIONS)
    else:
        dangerous_ops = []

    if not (query_patterns or denied_tables or dangerous_ops):
        return None

    tool_patterns = _config_str_list(policy.config, "tool_patterns")

    for tool_call in context.tool_calls:
        tool_name = tool_call.get("name", "") or ""
        arguments = tool_call.get("arguments", "") or ""
        if not isinstance(arguments, str) or not arguments:
            continue
        # Optional tool-name scoping; default applies to any tool.
        if tool_patterns and not any(
            safe_search(tp, tool_name, re.IGNORECASE) for tp in tool_patterns
        ):
            continue

        # Axis 1: explicit operator regex.
        for pattern in query_patterns:
            if safe_search(pattern, arguments, re.IGNORECASE):
                return MatchedPolicyRecord(
                    name=policy.name,
                    type=policy.type,
                    verdict="block",
                    reason_code="db_query_blocked",
                    message=policy.config.get(
                        "message",
                        f"Database query in '{tool_name}' was blocked.",
                    ),
                )

        # Axis 2: dangerous operations. We use word-boundary anchors
        # so 'DROP' fires on 'DROP TABLE' but not on 'tear-DROP-shaped'.
        for op in dangerous_ops:
            # Build a tolerant pattern: word-boundary on each side,
            # and treat operator-supplied multi-word strings ("CREATE
            # USER") as literal whitespace runs.
            op_re = r"\b" + r"\s+".join(
                re.escape(part) for part in op.split()
            ) + r"\b"
            if safe_search(op_re, arguments, re.IGNORECASE):
                return MatchedPolicyRecord(
                    name=policy.name,
                    type=policy.type,
                    verdict="block",
                    reason_code="db_query_blocked",
                    message=policy.config.get(
                        "message",
                        f"Dangerous SQL operation '{op}' in tool '{tool_name}' "
                        f"was blocked.",
                    ),
                )

        # Axis 3: denied tables. We look for the table name appearing
        # in a SQL position that mutates / reads from it. Backticks /
        # double-quotes / brackets are tolerated, and so are
        # backslash-escaped quotes that appear when the SQL string
        # arrives JSON-encoded inside the tool's arguments
        # (``"sql": "SELECT * FROM \"users\""``).
        for table in denied_tables:
            tbl_re = (
                r"\b(?:FROM|UPDATE|INTO|TABLE|JOIN)\s+"
                r"\\*['`\"\[]?"
                + re.escape(table)
                + r"\\*['`\"\]]?\b"
            )
            if safe_search(tbl_re, arguments, re.IGNORECASE):
                return MatchedPolicyRecord(
                    name=policy.name,
                    type=policy.type,
                    verdict="block",
                    reason_code="db_query_blocked",
                    message=policy.config.get(
                        "message",
                        f"Database query against table '{table}' "
                        f"in tool '{tool_name}' was blocked.",
                    ),
                )
    return None


def _deny_financial_action_match(
    policy: PolicyRule,
    context: OutputPolicyContext,
) -> MatchedPolicyRecord | None:
    """Block tool calls that look like money movement above the
    operator's risk appetite.

    Three independent matching axes — any one match blocks:

    * ``action_patterns`` — regex against the tool *name*. Default
      list (``transfer``, ``charge``, ``refund``, ``payout``,
      ``withdraw``, …) catches the vast majority of payment
      vendor naming conventions; operator can replace or extend.
    * ``amount_threshold`` — when set, any matching tool call whose
      arguments contain an amount-shaped field above this number
      blocks. Field names default to a curated set
      (``amount``/``amount_cents``/``value``/…) but can be
      narrowed via ``amount_field``.
    * ``denied_destinations`` — regex against destination-shaped
      fields in the arguments (``to_account``, ``recipient``,
      ``destination``, ``iban``).
    * ``allowed_currencies`` — when set, any call whose arguments
      include a ``currency`` field NOT in this list blocks.

    Detection again uses argument introspection (parsed JSON) so
    a generic tool like ``http_post`` to a payments endpoint is
    caught when its body contains the financial primitives.
    """
    action_patterns = _config_str_list(policy.config, "action_patterns")
    if not action_patterns:
        # Default-on if no operator list provided; most operators
        # creating this rule WANT the default list to fire.
        action_patterns = list(_DEFAULT_FINANCIAL_VERBS)

    raw_threshold = policy.config.get("amount_threshold")
    threshold: float | None = None
    if isinstance(raw_threshold, int | float) and not isinstance(
        raw_threshold, bool
    ):
        threshold = float(raw_threshold)

    raw_fields = policy.config.get("amount_field")
    if isinstance(raw_fields, str) and raw_fields:
        amount_fields: tuple[str, ...] = (raw_fields,)
    elif isinstance(raw_fields, list):
        amount_fields = tuple(
            f for f in raw_fields if isinstance(f, str) and f
        ) or _DEFAULT_AMOUNT_FIELD_NAMES
    else:
        amount_fields = _DEFAULT_AMOUNT_FIELD_NAMES

    denied_destinations = _config_str_list(policy.config, "denied_destinations")
    allowed_currencies_raw = _config_str_list(policy.config, "allowed_currencies")
    allowed_currencies = {c.upper() for c in allowed_currencies_raw}

    for tool_call in context.tool_calls:
        tool_name = tool_call.get("name", "") or ""
        arguments = tool_call.get("arguments", "") or ""
        if not isinstance(tool_name, str):
            continue
        # The financial axis ONLY fires for tool calls that look
        # financial — a name match. This prevents the rule from
        # blocking unrelated tools that happen to carry an
        # "amount" field (e.g. an analytics ``track_event`` with
        # ``{"amount": 1}``).
        if not any(
            safe_search(p, tool_name, re.IGNORECASE) for p in action_patterns
        ):
            continue

        # Axis 1: matched on name alone. If neither threshold nor
        # destination filtering is configured, the name match alone
        # is enough — block immediately. Most operators creating a
        # ``deny_financial_action`` rule mean "no money tools."
        no_secondary_filter = (
            threshold is None
            and not denied_destinations
            and not allowed_currencies
        )
        if no_secondary_filter:
            return MatchedPolicyRecord(
                name=policy.name,
                type=policy.type,
                verdict="block",
                reason_code="financial_action_blocked",
                message=policy.config.get(
                    "message",
                    f"Financial action '{tool_name}' was blocked.",
                ),
            )

        parsed = _parse_tool_arguments(arguments) if arguments else None

        # Axis 2: amount threshold. Walk parsed arguments for any
        # amount-shaped field over the configured cap.
        if threshold is not None and parsed is not None:
            amounts = _walk_amount_values(parsed, amount_fields)
            offending = [a for a in amounts if a > threshold]
            if offending:
                return MatchedPolicyRecord(
                    name=policy.name,
                    type=policy.type,
                    verdict="block",
                    reason_code="financial_action_blocked",
                    message=policy.config.get(
                        "message",
                        f"Financial action '{tool_name}' exceeded the "
                        f"amount threshold ({offending[0]} > {threshold}).",
                    ),
                )

        # Axis 3: denied destinations. Matches against the
        # serialized arguments string (operator-supplied regex
        # already encodes the field shape).
        if denied_destinations and isinstance(arguments, str):
            for pattern in denied_destinations:
                if safe_search(pattern, arguments, re.IGNORECASE):
                    return MatchedPolicyRecord(
                        name=policy.name,
                        type=policy.type,
                        verdict="block",
                        reason_code="financial_action_blocked",
                        message=policy.config.get(
                            "message",
                            f"Financial action '{tool_name}' targets a "
                            f"denied destination.",
                        ),
                    )

        # Axis 4: currency allowlist. Walk parsed arguments looking
        # for a ``currency`` field; block when present and not in
        # the allowed set.
        if allowed_currencies and parsed is not None:
            for currency in _walk_currency_values(parsed):
                if currency.upper() not in allowed_currencies:
                    return MatchedPolicyRecord(
                        name=policy.name,
                        type=policy.type,
                        verdict="block",
                        reason_code="financial_action_blocked",
                        message=policy.config.get(
                            "message",
                            f"Financial action '{tool_name}' uses a "
                            f"non-allowed currency '{currency}'.",
                        ),
                    )
    return None


def _walk_currency_values(obj: Any) -> list[str]:
    """Collect every string value keyed by ``currency`` in a parsed
    arguments tree. Used by ``deny_financial_action``'s currency
    allowlist."""
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(k, str) and k.lower() == "currency" and isinstance(v, str):
                out.append(v)
            else:
                out.extend(_walk_currency_values(v))
    elif isinstance(obj, list):
        for item in obj:
            out.extend(_walk_currency_values(item))
    return out
