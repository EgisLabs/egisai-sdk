"""Pure-Python policy rule engine.

Evaluates ``PolicyRule`` objects against an input or output
``PolicyContext`` and returns a ``PolicyDecision``. No I/O.
"""

from __future__ import annotations

import contextvars
import logging
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any

from egisai.policy import _pii_custom, fastpath, injection
from egisai.policy import pii as pii_scanner
from egisai.policy._regex_safe import safe_search
from egisai.policy.semantic import SemanticBlocker

LOGGER = logging.getLogger("egisai.policy.engine")


@dataclass(frozen=True)
class PolicyRule:
    """One active rule.

    ``type`` selects the evaluator (``pii_scan``, ``semantic_guard``,
    ``deny_regex``, …); ``config`` carries the type-specific knobs.
    ``agent_ids`` scopes the rule to specific agents — empty means
    "applies to every agent".

    ``phase`` selects which side of the governed call the rule runs
    on. The names are call-relative (not model-centric) so they read
    correctly for every surface — model calls, tool calls, MCP
    calls, gateway traffic:

    - ``"request"``  — evaluated against the inbound payload (the
      user prompt, tool arguments) before the call is made
      (default for input-side detectors).
    - ``"response"`` — evaluated against the outbound payload (the
      model's completion, a tool result) after the call returns
      (default for output-side detectors).
    - ``"both"`` — runs on both sides; only meaningful for rule types
      that support it (e.g. ``semantic_guard``).

    The legacy spellings ``pre_model`` / ``post_model`` are still
    accepted on the wire (``_policy_cache._to_rule`` normalizes
    them) but the engine only ever sees the canonical names.

    ``applies_to`` scopes the rule to specific call surfaces
    (``"model"``, ``"tool"``, ``"mcp"``). Empty means "all
    surfaces" — the behavior every rule had before surface scoping
    existed. Orthogonal to ``phase``: a rule can fire on the
    request side of tool calls only, the response side of
    everything, etc.
    """

    id: str | None
    name: str
    type: str
    tenant: str | None
    config: dict[str, Any]
    agent_ids: tuple[str, ...] = field(default=())
    phase: str = "both"
    # Surface scoping. Empty tuple ⇒ every surface.
    applies_to: tuple[str, ...] = field(default=())
    # MCP Servers add-on scope. Empty means "not targeted at any
    # specific MCP server" — combined with an empty ``agent_ids`` it
    # is an org-wide rule that applies to both agents and MCP servers.
    # When non-empty, the MCP-server gate only applies this rule to
    # the listed server UUIDs. Has no effect on agent-side evaluation.
    mcp_server_ids: tuple[str, ...] = field(default=())


@dataclass(frozen=True)
class PolicyContext:
    """Inputs for evaluating *input-side* policies (before the LLM call).

    ``agent_id`` is the lower-case UUID of the agent this call is
    attributed to (empty when identity could not be resolved). Added
    for the ``rate_limit`` / ``budget_limit`` kinds, whose counters
    are keyed per agent; every other evaluator ignores it. Appended
    with a default so existing positional/kwarg constructions keep
    compiling unchanged.
    """

    tenant: str
    model: str
    prompt_text: str
    prompt_chars: int
    stream: bool
    agent_id: str = ""


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
    # ── Identity of the end-user this call serves ──────────────────
    # Read from ``EgisaiContext`` (``egisai.set_context(...)``) at gate
    # time and carried here so evaluators stay pure functions of their
    # inputs. Empty string when the caller never set it. Powers
    # identity-aware access rules (``deny_resource_access``): the same
    # tool call is allowed or refused depending on *who* the agent is
    # acting for. Appended with defaults so every existing positional /
    # keyword construction of this dataclass keeps compiling unchanged.
    user_role: str = ""
    end_user_id: str = ""
    user_id: str = ""


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
    - ``"pending_approval"`` hold the call and route it to a human
                      for approval (human-in-the-loop). The call is
                      neither forwarded nor refused yet — the gate
                      pauses, a human approves/rejects, and the call
                      resumes (approved ⇒ forward) or fails
                      (rejected/expired ⇒ block). Precedence is
                      ``block > pending_approval > sanitize > allow``:
                      a hard block always wins over a hold.

    ``matched_policy`` is the primary matched rule's name;
    ``matched_policies`` is the full ordered list of rules that fired.

    ``approval_detail`` is a short human-readable phrase describing
    what needs approving (e.g. ``"$25,000 transfer via wire_send"``);
    only set when ``verdict == "pending_approval"``.
    """
    verdict: str
    reason_code: str | None
    message: str | None
    matched_policy: str | None
    matched_policies: tuple[MatchedPolicyRecord, ...] = ()
    sanitize_types: list[str] = field(default_factory=list)
    sanitize_mask_char: str = "#"
    approval_detail: str | None = None

    @property
    def sanitize_kinds(self) -> list[str]:
        """Deprecated alias for ``sanitize_types``.

        Removed in a future release; kept for one release while
        consumers (the audit-event serializer, the framework
        patches) migrate to ``sanitize_types``.
        """
        return self.sanitize_types

    @classmethod
    def allow(
        cls,
        *,
        matched_policies: tuple[MatchedPolicyRecord, ...] = (),
    ) -> PolicyDecision:
        """The call proceeds.

        ``matched_policies`` may still be non-empty: an *advisory*
        rule (today, ``injection_scan`` with ``action="flag"``) records
        what it saw without changing the verdict. ``matched_policy``
        stays ``None`` on purpose — that field names the rule whose
        verdict the call took, and on an allow no rule did.
        """
        return cls(
            verdict="allow",
            reason_code=None,
            message=None,
            matched_policy=None,
            matched_policies=matched_policies,
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

    @classmethod
    def hold(
        cls,
        *,
        reason_code: str,
        message: str,
        matched_policy: str,
        matched_policies: tuple[MatchedPolicyRecord, ...] = (),
        approval_detail: str | None = None,
    ) -> PolicyDecision:
        """The call must be held for human approval before it proceeds.

        The gate creates an approval request, notifies the configured
        approver(s), and either resumes the call (approved) or refuses
        it (rejected / expired). This is a non-terminal verdict — it
        resolves to ``allow`` or ``block`` once a human decides.
        """
        return cls(
            verdict="pending_approval",
            reason_code=reason_code,
            message=message,
            matched_policy=matched_policy,
            matched_policies=matched_policies,
            approval_detail=approval_detail,
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
        # Identity-aware per-resource access control (the "scalpel"):
        # block one file / record / URI for the wrong end-user while the
        # same tool keeps working for everyone else. Pure-Python — reads
        # the identity carried on ``OutputPolicyContext`` and matches
        # the resource id inside the call's arguments / MCP target.
        "deny_resource_access",
        # Prompt-injection shapes (0.65.0). Compiled regex + two
        # character-class counts over the text — no model, no network,
        # and no ONNX bundle, which is why it is allowed to run on
        # every call. See ``egisai.policy.injection``.
        "injection_scan",
        # Per-agent runtime limits (0.46.1). Pure in-memory counter
        # compares against the backend-synced usage snapshot — no
        # network on the hot path (see ``egisai.policy.limits``).
        "rate_limit",
        "budget_limit",
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


def _runs_request_phase(rule: PolicyRule) -> bool:
    """Return True when this rule should fire on the request side."""
    return rule.phase in ("request", "both")


def _runs_response_phase(rule: PolicyRule) -> bool:
    """Return True when this rule should fire on the response side."""
    return rule.phase in ("response", "both")


def _covers_surface(rule: PolicyRule, surfaces: tuple[str, ...]) -> bool:
    """Return True when the rule's ``applies_to`` scope intersects
    the surfaces this evaluation covers.

    Empty ``applies_to`` ⇒ every surface (legacy behavior). The
    check is set-intersection rather than exact-match because one
    evaluation pass can cover several surfaces at once — e.g. the
    output phase of a model call sees the completion text (model
    surface) AND the model's tool-call requests (tool surface).
    """
    if not rule.applies_to:
        return True
    return any(s in rule.applies_to for s in surfaces)


def evaluate_policies(
    policies: list[PolicyRule],
    context: PolicyContext,
    semantic_blocker: SemanticBlocker | None = None,
    *,
    surfaces: tuple[str, ...] = ("model",),
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

    Rules whose ``phase`` is ``"response"`` are skipped entirely
    on this side — they only run during ``evaluate_output_policies``.
    Rules whose ``applies_to`` doesn't intersect ``surfaces`` are
    skipped too (empty ``applies_to`` matches everything).

    ``surfaces`` names the call surfaces this evaluation covers —
    ``("model",)`` for a model-call prompt (the default),
    ``("tool",)`` for tool arguments, ``("mcp",)`` for an inbound
    MCP ``tools/call``.

    The verdict precedence across all matches is
    ``block > sanitize > allow``. ``semantic_blocker`` is optional;
    when ``None``, ``semantic_guard`` rules become no-ops.
    """
    request_side = [
        p for p in policies
        if _runs_request_phase(p) and _covers_surface(p, surfaces)
    ]
    phase1 = [p for p in request_side if p.type in _DETERMINISTIC_KINDS]
    phase2 = [p for p in request_side if p.type in _LLM_BACKED_KINDS]

    phase1_matches = _collect_input_matches(phase1, context, semantic_blocker=None)

    # A block or a human-approval hold both short-circuit Phase 2:
    # there is no point spending an LLM judge round-trip (or leaking
    # the prompt to it) on a call we are already refusing or pausing.
    if phase1_matches.has_block or phase1_matches.has_pending:
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
        agent_id=context.agent_id,
    )
    # Fast-governance dispatch (see ``egisai.policy.fastpath``).
    # ``off`` keeps the release-tested legacy walk byte-identical.
    # ``shadow`` enforces via the legacy walk while the fast walk
    # runs on a background thread and reports agreement. ``on``
    # enforces via the fast walk. Phase 1 above is untouched in
    # every mode — the dispatch covers ONLY the Phase-2 judges.
    fast_mode = fastpath.mode()
    sem2 = [p for p in phase2 if p.type == "semantic_guard"]
    other2 = [p for p in phase2 if p.type != "semantic_guard"]
    if fast_mode == "on":
        phase2_matches = _collect_semantic_fast(
            sem2,
            text=text_for_phase2,
            tool_calls=None,
            semantic_blocker=semantic_blocker,
            side="prompt",
        )
        # Any future non-mergeable LLM-backed kind still runs, via
        # the legacy walker, in every mode.
        if other2:
            for rec in _collect_input_matches(
                other2, phase2_ctx, semantic_blocker=semantic_blocker
            ).records:
                phase2_matches.add(rec)
    else:
        phase2_matches = _collect_input_matches(
            phase2, phase2_ctx, semantic_blocker=semantic_blocker
        )
        if fast_mode == "shadow":
            _spawn_semantic_shadow(
                sem2,
                text=text_for_phase2,
                tool_calls=None,
                semantic_blocker=semantic_blocker,
                side="prompt",
                legacy_records=phase2_matches.records,
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
    def has_pending(self) -> bool:
        return any(r.verdict == "pending_approval" for r in self.records)

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
    """Walk a list of prompt-side rules and accumulate matches.

    Phase 1 (``semantic_blocker is None``) runs inline: the checks are
    pure-Python regex/math where a thread hand-off would cost more
    than the work itself.

    Phase 2 fans out. Each ``semantic_guard`` policy is an independent
    judge round-trip against the *same* prompt, so N policies used to
    cost N round-trips **serially** — policy latency grew linearly
    with the number of guards an operator configured, which is the
    one thing a runtime governance layer cannot afford. Concurrency
    collapses that to roughly the slowest single call.

    Accuracy is unchanged by construction: every judge call carries
    the byte-identical payload it carried when this loop was serial,
    and results are re-walked in policy order below (``_fan_out``
    guarantees input ordering) so ``_synthesize_decision`` still
    picks the same primary ``matched_policy``. There is no
    short-circuit to lose — this walk never broke early even when an
    earlier policy blocked, so the call count is identical too.
    """
    out = _PhaseMatches()
    if semantic_blocker is None or len(policies) < 2:
        for policy in policies:
            rec = _evaluate_one_input_policy(policy, context, semantic_blocker)
            if rec is not None:
                out.add(rec)
        return out

    results = _fan_out(
        [
            _bind_input_policy(policy, context, semantic_blocker)
            for policy in policies
        ],
        max_workers=_judge_budget.get(),
    )
    for rec in results:
        if rec is not None:
            out.add(rec)
    return out


def _bind_input_policy(
    policy: PolicyRule,
    context: PolicyContext,
    semantic_blocker: SemanticBlocker | None,
) -> Callable[[], MatchedPolicyRecord | None]:
    """Freeze one prompt-side evaluation into a zero-arg callable.

    A named factory rather than an inline ``lambda`` in a
    comprehension — the loop variable would otherwise be captured by
    reference and every task would evaluate the *last* policy.
    """
    def run() -> MatchedPolicyRecord | None:
        return _evaluate_one_input_policy(policy, context, semantic_blocker)

    return run


def _approval_requested(config: dict[str, Any]) -> bool:
    """True when the operator asked this rule to hold for a human.

    Two spellings are accepted so the flag composes with the existing
    ``action`` field some kinds already use:

    * ``config["require_approval"] is True`` — the canonical flag.
    * ``config["action"] == "require_approval"`` — for kinds whose
      operator UI already exposes an ``action`` selector.
    """
    if config.get("require_approval") is True:
        return True
    action = config.get("action")
    return isinstance(action, str) and action.strip().lower() == "require_approval"


def _maybe_require_approval(
    policy: PolicyRule, rec: MatchedPolicyRecord | None
) -> MatchedPolicyRecord | None:
    """Convert a ``block`` record into a ``pending_approval`` hold when
    the policy is configured to require human approval.

    Only a ``block`` is convertible — a ``sanitize`` or ``flag`` was
    never going to refuse the call, so there is nothing to hold. This
    keeps the transform additive: a rule without the flag behaves
    exactly as before.
    """
    if rec is None or rec.verdict != "block":
        return rec
    if not _approval_requested(policy.config or {}):
        return rec
    return replace(rec, verdict="pending_approval")


def _evaluate_one_input_policy(
    policy: PolicyRule,
    context: PolicyContext,
    semantic_blocker: SemanticBlocker | None,
) -> MatchedPolicyRecord | None:
    """Evaluate one prompt-side rule, then apply the approval transform."""
    rec = _dispatch_one_input_policy(policy, context, semantic_blocker)
    return _maybe_require_approval(policy, rec)


def _dispatch_one_input_policy(
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

    if policy.type == "injection_scan":
        return _injection_scan_match(
            policy,
            text=context.prompt_text,
            reason_code="injection_detected",
        )

    if policy.type == "semantic_guard":
        return _semantic_guard_match(
            policy=policy,
            text=context.prompt_text,
            semantic_blocker=semantic_blocker,
            side="prompt",
        )

    if policy.type == "rate_limit":
        return _rate_limit_match(policy, context.agent_id)

    if policy.type == "budget_limit":
        return _budget_limit_match(policy, context.agent_id)

    # Tool / bash / MCP rules need response-side signals
    # (tool_names, tool_calls, mcp_targets) that ``PolicyContext``
    # does not carry today. Operators may still target them on the
    # request phase via the open phase picker; the rule silently
    # no-ops here so the call isn't broken. They fire normally
    # when ``phase`` includes ``response``.
    return None


# ── Per-agent runtime limits (rate_limit / budget_limit) ───────────────
#
# Both kinds are Phase-1 deterministic: a counter compare against the
# in-memory state ``egisai.policy.limits`` keeps (backend-synced
# snapshot + local sliding window). They only have request-side
# semantics — counting happens when a model call is about to leave
# the process — so ``_evaluate_one_output_policy`` intentionally does
# NOT dispatch them (an operator who force-targets one on the
# response phase gets a silent no-op, same as ``deny_tool_call`` on
# the request side).


def _rate_limit_match(
    policy: PolicyRule, agent_id: str
) -> MatchedPolicyRecord | None:
    """Block when the agent (or org) exceeded ``max_requests`` in the
    configured sliding window.

    Fail-open rules:
    * ``max_requests`` missing / non-positive ⇒ rule is inert.
    * ``scope="per_agent"`` with an unresolved agent identity ⇒
      skip (we cannot attribute a per-agent counter; same posture
      as the pause gate for unknown identities).
    """
    from egisai.policy import limits

    config = policy.config or {}
    try:
        max_requests = int(config.get("max_requests") or 0)
    except (TypeError, ValueError):
        return None
    if max_requests <= 0:
        return None

    try:
        window = int(config.get("window_seconds") or 60)
    except (TypeError, ValueError):
        window = 60
    if window not in limits.RATE_WINDOWS:
        # Defensive: backend validation pins the window to the
        # supported trio, but a hand-crafted config could carry
        # anything. Snap UP to the smallest supported window that
        # covers the requested one (strictest honest reading);
        # beyond the largest window, use the largest.
        window = next(
            (w for w in sorted(limits.RATE_WINDOWS) if w >= window),
            max(limits.RATE_WINDOWS),
        )

    scope = str(config.get("scope") or "per_agent")
    if scope not in ("per_agent", "per_org"):
        scope = "per_agent"
    if scope == "per_agent" and not agent_id:
        return None

    current = limits.rate_limit_usage(agent_id, window, scope)
    if current < max_requests:
        return None

    window_label = {60: "minute", 3600: "hour", 86400: "24 hours"}.get(
        window, f"{window}s"
    )
    scope_label = "agent" if scope == "per_agent" else "organization"
    # Operator-set ``config.message`` wins (same contract as every
    # other kind); the dynamic count-carrying text is the default.
    message = str(config.get("message") or "").strip() or (
        f"Rate limit exceeded: {current} of {max_requests} model "
        f"calls in the last {window_label} for this {scope_label}."
    )
    return MatchedPolicyRecord(
        name=policy.name,
        type="rate_limit",
        verdict="block",
        reason_code="rate_limit_exceeded",
        message=message,
    )


def _budget_limit_match(
    policy: PolicyRule, agent_id: str
) -> MatchedPolicyRecord | None:
    """Block when the agent (or org) spent ``max_usd`` in the window.

    Spend is priced by the backend at ingest and arrives via the
    usage snapshot; without a snapshot the rule fails open (the SDK
    cannot observe cost locally — see ``limits.budget_usage_usd``).
    """
    from egisai.policy import limits

    config = policy.config or {}
    try:
        max_usd = float(config.get("max_usd") or 0)
    except (TypeError, ValueError):
        return None
    if max_usd <= 0:
        return None

    window = str(config.get("window") or "monthly")
    if window not in limits.BUDGET_WINDOWS:
        window = "monthly"

    scope = str(config.get("scope") or "per_agent")
    if scope not in ("per_agent", "per_org"):
        scope = "per_agent"
    if scope == "per_agent" and not agent_id:
        return None

    spend = limits.budget_usage_usd(agent_id, window, scope)
    if spend is None or spend < max_usd:
        return None

    scope_label = "agent" if scope == "per_agent" else "organization"
    # Operator-set ``config.message`` wins (same contract as every
    # other kind); the dynamic spend-carrying text is the default.
    message = str(config.get("message") or "").strip() or (
        f"Budget exceeded: ${spend:.4f} of the ${max_usd:.2f} "
        f"{window} budget for this {scope_label} has been spent."
    )
    return MatchedPolicyRecord(
        name=policy.name,
        type="budget_limit",
        verdict="block",
        reason_code="budget_exceeded",
        message=message,
    )


def _synthesize_decision(
    records: list[MatchedPolicyRecord],
) -> PolicyDecision:
    """Roll a list of matches up into a single ``PolicyDecision``.

    Verdict precedence is ``block > sanitize > allow``. The first
    record at the winning precedence is the primary; the full list
    is carried on ``matched_policies``.

    A record whose own verdict is neither ``block`` nor ``sanitize``
    is *advisory* — it saw something worth recording but is not asking
    for the call to change. Those ride along on ``matched_policies``
    of an allow decision so the finding reaches the audit row, which
    is the only reason an operator would set a rule to flag rather
    than block.
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

    # Human-in-the-loop hold. Sits below ``block`` (a hard block
    # always wins) but above ``sanitize`` — a call that needs a human
    # decision should not silently proceed just because some PII was
    # also masked. The finding detail rides on ``approval_detail`` for
    # the approver-facing notification / inbox.
    holds = [r for r in records if r.verdict == "pending_approval"]
    if holds:
        primary = holds[0]
        return PolicyDecision.hold(
            reason_code=primary.reason_code,
            message=primary.message,
            matched_policy=primary.name,
            matched_policies=tuple(records),
            approval_detail=primary.message,
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

    return PolicyDecision.allow(matched_policies=tuple(records))


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

        # Parallel path, via the shared ``_fan_out`` primitive. It
        # owns input-order preservation, per-task context copying (so
        # judge token spend still lands on the audit row — worker
        # threads don't inherit context vars), and per-task fail-open.
        # ``_judge_budget`` is the remaining concurrency this call may
        # use, already divided down if Phase 2 fanned out across
        # policies above us.
        results = _fan_out(
            [
                _bind_tool_judge(semantic_blocker, synthesized, policy.config)
                for _, synthesized in normalized
            ],
            max_workers=_judge_budget.get(),
        )

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

# Remaining concurrency budget for judge round-trips inside the
# current governed call.
#
# Two levels of the engine fan out to the judge: Phase 2 fans out
# across *policies* (``_collect_*_matches``) and each
# ``semantic_guard`` policy fans out across *tool calls*
# (``_semantic_guard_match``). Left uncoordinated, a 3-guard turn
# with 6 tool calls would issue 18 simultaneous round-trips and
# rate-limit itself into ``Retry-After`` storms.
#
# The outer level divides its budget among the tasks it spawns and
# publishes the per-task share here; the inner level reads it as its
# own ceiling. The product therefore stays bounded by
# ``_TOOL_JUDGE_MAX_WORKERS`` no matter how the work is shaped.
#
# A context var (rather than a parameter threaded through five
# signatures) keeps ``_semantic_guard_match``'s public-ish shape
# untouched — tests call it directly with keyword args. Each fan-out
# task runs in its own copied context, so a task writing its share
# here cannot disturb its siblings.
_judge_budget: contextvars.ContextVar[int] = contextvars.ContextVar(
    "egisai_judge_budget", default=_TOOL_JUDGE_MAX_WORKERS
)


def _bind_tool_judge(
    semantic_blocker: SemanticBlocker,
    synthesized: str,
    config: dict[str, Any],
) -> Callable[[], Any]:
    """Freeze one tool-call judge round-trip into a zero-arg callable."""
    def run() -> Any:
        return semantic_blocker.check(synthesized, config)

    return run


def _fan_out(
    tasks: list[Callable[[], Any]],
    *,
    max_workers: int,
) -> list[Any]:
    """Run ``tasks`` concurrently; return results in **input** order.

    The single fan-out primitive for judge round-trips. Contract:

    * **Input order is preserved.** Callers resolve first-match
      semantics by walking the returned list, so completion order
      must never influence which policy a verdict is attributed to.
    * **Each task gets a fresh ``contextvars.copy_context()``.**
      Worker threads start with an *empty* context, so a judge call
      made on one would otherwise lose the gate's per-call token
      accumulator (see ``_PolicyUsageAccumulator``). The copy shares
      value references, so the worker and the gate mutate the same
      accumulator. The copy must be per-task: one ``Context`` cannot
      be entered by two threads at once.
    * **A raising task yields ``None``.** One failed judge call must
      not poison the batch — same fail-open contract the platform
      applies to a judge outage.
    * **Degenerate cases run inline.** A single task, or a budget of
      one, skips executor construction entirely so the common
      one-policy / one-tool turn pays no threading overhead.
    """
    if not tasks:
        return []
    if len(tasks) == 1 or max_workers <= 1:
        results: list[Any] = []
        for task in tasks:
            try:
                results.append(task())
            except Exception:  # noqa: BLE001
                LOGGER.debug("judge task failed; treating as no-match", exc_info=True)
                results.append(None)
        return results

    workers = min(max_workers, len(tasks))
    # Share the remaining budget among the tasks we're about to
    # spawn so any nested fan-out inside them stays inside the cap.
    share = max(1, max_workers // workers)

    def _run(task: Callable[[], Any]) -> Any:
        _judge_budget.set(share)
        return task()

    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="egisai-judge"
    ) as pool:
        futures = [
            pool.submit(contextvars.copy_context().run, _run, task)
            for task in tasks
        ]
        out: list[Any] = []
        for f in futures:
            try:
                out.append(f.result())
            except Exception:  # noqa: BLE001
                LOGGER.debug("judge task failed; treating as no-match", exc_info=True)
                out.append(None)
        return out


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


# ── Fast-governance Phase-2 walk ────────────────────────────────────────
#
# The merged-question evaluator behind ``EGISAI_FAST_GOVERNANCE``.
# Rationale, rollout contract, and compliance notes live in
# ``egisai.policy.fastpath``; the mechanics live here because they
# reuse the engine's private primitives (``_fan_out``, the judge
# budget, ``_synthesize_tool_call_text``).
#
# Question inventory for one turn, legacy vs fast, with G guards that
# all target ["text", "tool_calls"] and share a threshold, and T
# unique tool calls:
#
#     legacy:  G × (1 + T)   judge round-trips
#     fast:    1 + T         judge round-trips
#
# plus the text question is windowed (bounded tokens) and byte-equal
# tool questions are asked once instead of once per policy per
# duplicate. Verdict semantics per question are unchanged: the judge
# receives the same threshold it would have received per-policy
# (grouping is BY threshold), and the union intent list is exactly
# the set of intents that would have been spread across the
# per-policy calls.


def _semantic_intents(policy: PolicyRule) -> list[str]:
    """The rule's operator-authored intent strings (may be empty)."""
    raw = policy.config.get("intents") or []
    if not isinstance(raw, list):
        return []
    return [i for i in raw if isinstance(i, str) and i.strip()]


def _semantic_targets(policy: PolicyRule) -> list[str]:
    """Mirror of ``_semantic_guard_match``'s targets resolution."""
    raw = policy.config.get("targets")
    if isinstance(raw, list) and raw:
        return [str(t) for t in raw if isinstance(t, str)]
    return ["text"]


def _threshold_group_key(policy: PolicyRule) -> str:
    """Policies merge ONLY when their judge threshold is identical.

    The merged call ships one ``threshold`` on the wire, so a shared
    value is what makes merging exact rather than approximate. ``None``
    (policy defers to the platform default) is its own group. ``repr``
    keeps ``0.7`` and ``"0.7"`` apart — coercing operator config here
    would be guessing.
    """
    return repr(policy.config.get("threshold"))


def _merged_judge_config(group: list[PolicyRule]) -> dict[str, Any]:
    """One judge-call config carrying the group's union intent list.

    Order-preserving dedup: policy order then intent order, so the
    judge's prompt lists intents in the same relative order operators
    see on the dashboard. The shared threshold (identical across the
    group by construction) rides along verbatim.
    """
    union: dict[str, None] = {}
    for policy in group:
        for intent in _semantic_intents(policy):
            union.setdefault(intent, None)
    cfg: dict[str, Any] = {"intents": list(union)}
    threshold = group[0].config.get("threshold")
    if threshold is not None:
        cfg["threshold"] = threshold
    return cfg


def _owning_policy(
    intent: str, group: list[PolicyRule]
) -> tuple[PolicyRule, str]:
    """Map the judge's cited intent back to the policy that owns it.

    The backend already resolves the judge's (possibly paraphrased)
    citation to a canonical string from the union list, so the exact
    pass below is the overwhelmingly common case. The fuzzy pass
    mirrors the backend's own containment matching for older backends
    that return the raw citation. The final fallback attributes to the
    group's first policy — the same "trust the verdict, best-effort
    label" posture ``_build_verdict`` takes server-side. A BLOCK is
    never dropped because attribution was ambiguous.
    """
    low = intent.strip().lower()
    for policy in group:
        for candidate in _semantic_intents(policy):
            if candidate.strip().lower() == low:
                return policy, candidate
    for policy in group:
        for candidate in _semantic_intents(policy):
            c = candidate.strip().lower()
            if c and (c in low or low in c):
                return policy, candidate
    return group[0], intent


# The platform's judge endpoint validates ``intents`` at
# ``max_length=16`` (``backend/app/schemas/sdk.py`` — the AI Policy
# Assistant's recommended list size; longer lists hurt judge accuracy
# more than they help coverage). A merged question whose union exceeds
# the cap would be rejected with a 422 and the fast path would fail
# open — silently judging NOTHING (this happened: three real guards
# with 5+8+10 intents merged to 23 and every fast-path call bounced).
# Bin-packing below keeps every merged question at or under the cap.
_MERGE_MAX_INTENTS = 16


def _bin_pack_by_intents(group: list[PolicyRule]) -> list[list[PolicyRule]]:
    """Split one threshold group into bins of ≤ ``_MERGE_MAX_INTENTS``.

    Greedy, order-preserving first-fit: walk the policies in priority
    order and start a new bin when the next policy's intents would
    push the running total over the cap. A policy is never split
    across bins — attribution (``_owning_policy``) works per bin, so
    every intent must sit in the same question as its owner.

    A single policy that alone exceeds the cap still gets its own bin:
    that's byte-for-byte what the legacy walk sends for that policy
    (per-policy intents, no merging), so fast mode inherits exactly
    the legacy behavior for it rather than inventing a new failure
    mode. The sum is computed on raw per-policy counts, not the
    deduped union — conservative: dedupe can only shrink a bin's
    final question, never grow it.
    """
    bins: list[list[PolicyRule]] = []
    current: list[PolicyRule] = []
    current_count = 0
    for policy in group:
        n = len(_semantic_intents(policy))
        if current and current_count + n > _MERGE_MAX_INTENTS:
            bins.append(current)
            current = []
            current_count = 0
        current.append(policy)
        current_count += n
    if current:
        bins.append(current)
    return bins


def _fast_judge_questions(
    active: list[PolicyRule],
    *,
    text: str,
    tool_calls: list[dict[str, Any]] | None,
) -> list[tuple[str, str, list[PolicyRule], str, dict[str, Any]]]:
    """The merged question inventory for one fast-mode evaluation.

    Returns ``(kind, tool_name, group, question_text, merged_config)``
    tuples — text questions first (windowed via
    ``fastpath.window_text``), then one question per *unique*
    synthesized tool sentence per threshold group. Shared by the
    enforcement collector and the shadow DISAGREE diagnostics so the
    two can never drift apart on what was actually asked.

    Groups are keyed by threshold and then bin-packed to the judge
    endpoint's 16-intent cap (see ``_bin_pack_by_intents``), so a
    "group" here is really "one askable question's worth of policies".
    """
    questions: list[tuple[str, str, list[PolicyRule], str, dict[str, Any]]] = []

    def _grouped(candidates: list[PolicyRule]) -> list[list[PolicyRule]]:
        groups: dict[str, list[PolicyRule]] = {}
        for p in candidates:
            groups.setdefault(_threshold_group_key(p), []).append(p)
        packed: list[list[PolicyRule]] = []
        for group in groups.values():
            packed.extend(_bin_pack_by_intents(group))
        return packed

    judge_text = fastpath.window_text(text) if text else ""
    text_policies = [
        p for p in active if "text" in _semantic_targets(p)
    ] if judge_text else []
    for group in _grouped(text_policies):
        questions.append(
            ("text", "", group, judge_text, _merged_judge_config(group))
        )

    tool_policies = [
        p for p in active if "tool_calls" in _semantic_targets(p)
    ] if tool_calls else []
    if tool_policies:
        # Normalize + dedup by synthesized sentence. Two byte-equal
        # questions have one answer; asking twice (per duplicate tool
        # call, and again per policy) was pure latency. First tool
        # name wins the audit label, matching the legacy walk's
        # first-match-by-input-order semantics.
        seen: dict[str, str] = {}
        for tc in tool_calls or []:
            if not isinstance(tc, dict):
                continue
            name = tc.get("name")
            if not isinstance(name, str) or not name:
                continue
            args = tc.get("input")
            if args is None:
                args = tc.get("arguments")
            synthesized = _synthesize_tool_call_text(name, args)
            if synthesized and synthesized not in seen:
                seen[synthesized] = name
        for synthesized, name in seen.items():
            for group in _grouped(tool_policies):
                questions.append(
                    ("tool", name, group, synthesized,
                     _merged_judge_config(group))
                )
    return questions


def _fast_active_policies(policies: list[PolicyRule]) -> list[PolicyRule]:
    """Filter to the ``semantic_guard`` rules the fast path can merge."""
    active: list[PolicyRule] = []
    for policy in policies:
        if policy.type != "semantic_guard":
            # Callers split non-semantic phase-2 kinds off to the
            # legacy walkers BEFORE calling the fast collector (see
            # the dispatch sites in ``evaluate_policies`` /
            # ``evaluate_output_policies``); a stray one here means a
            # new LLM-backed kind was added without teaching either
            # the dispatch or the collector about it. Say so loudly
            # rather than guessing at its semantics.
            LOGGER.warning(
                "fast-governance: policy type %r is not mergeable and was "
                "not routed to the legacy walker — rule %r did not run "
                "this turn; route it at the dispatch site",
                policy.type, policy.name,
            )
            continue
        if not _semantic_intents(policy):
            continue
        if (policy.config.get("engine") or "").lower() == "embedding":
            continue
        active.append(policy)
    return active


def _collect_semantic_fast(
    policies: list[PolicyRule],
    *,
    text: str,
    tool_calls: list[dict[str, Any]] | None,
    semantic_blocker: SemanticBlocker | None,
    side: str,
) -> _PhaseMatches:
    """Merged-question Phase-2 walk. Enforcement path of fast mode.

    Behavior contract relative to the legacy walk:

    * Same verdict inputs. Every question carries the threshold the
      per-policy call would have carried, and the union intent list
      is exactly the intents the per-policy calls would have spread
      over G separate prompts.
    * Same fail-open posture. A ``None`` blocker, empty intents, the
      legacy embedding engine, judge outages — all degrade exactly as
      the legacy walk degrades (outage handling lives inside
      ``SemanticBlocker.check`` and is shared verbatim).
    * At most one record per policy, ``reason_code`` /  message
      templates identical to ``_semantic_guard_match``'s.
    * Text questions are windowed via ``fastpath.window_text``; tool
      questions are deduped by their synthesized sentence (a repeated
      tool call is the same question — asked once).
    """
    out = _PhaseMatches()
    if semantic_blocker is None:
        return out

    active = _fast_active_policies(policies)
    if not active:
        return out

    questions = _fast_judge_questions(
        active, text=text, tool_calls=tool_calls
    )
    if not questions:
        return out

    tasks: list[Callable[[], Any]] = [
        _bind_tool_judge(semantic_blocker, question_text, config)
        for (_kind, _tool_name, _group, question_text, config) in questions
    ]
    meta = [
        (kind, tool_name, group)
        for (kind, tool_name, group, _question_text, _config) in questions
    ]

    results = _fan_out(tasks, max_workers=_judge_budget.get())

    # ── Attribute matches back to owning policies ───────────────────
    # Walk in task order (text first, then tools in input order) and
    # keep at most one record per policy — identical to the legacy
    # walk, where ``_semantic_guard_match`` returns that policy's
    # first match and nothing else.
    claimed: set[int] = set()
    for (kind, tool_name, group), match in zip(meta, results, strict=True):
        if match is None:
            continue
        owner, _canonical = _owning_policy(match.intent, group)
        if id(owner) in claimed:
            continue
        claimed.add(id(owner))
        if kind == "text":
            out.add(
                MatchedPolicyRecord(
                    name=owner.name,
                    type=owner.type,
                    verdict="block",
                    reason_code="semantic_blocked",
                    message=owner.config.get(
                        "message",
                        f"{side.capitalize()} matches blocked intent: "
                        f"'{match.intent}'",
                    ),
                )
            )
        else:
            out.add(
                MatchedPolicyRecord(
                    name=owner.name,
                    type=owner.type,
                    verdict="block",
                    reason_code="semantic_blocked_tool",
                    message=owner.config.get(
                        "message",
                        (
                            f"Tool call '{tool_name}' matches blocked "
                            f"intent: '{match.intent}'"
                        ),
                    ),
                )
            )
    return out


def _spawn_semantic_shadow(
    policies: list[PolicyRule],
    *,
    text: str,
    tool_calls: list[dict[str, Any]] | None,
    semantic_blocker: SemanticBlocker | None,
    side: str,
    legacy_records: list[MatchedPolicyRecord],
) -> None:
    """Kick off one background fast-vs-legacy comparison.

    Never blocks, never raises, never influences the decision the
    caller already made. Inputs are snapshotted before the thread
    starts so a framework mutating its payload after the gate returns
    cannot skew the comparison.
    """
    if semantic_blocker is None or not policies:
        return
    if not fastpath.shadow_sampled():
        return
    tools_snapshot = [
        dict(tc) for tc in (tool_calls or []) if isinstance(tc, dict)
    ]
    legacy_snapshot = list(legacy_records)

    def _run() -> None:
        started = fastpath.now_ms()
        fast = _collect_semantic_fast(
            policies,
            text=text,
            tool_calls=tools_snapshot,
            semantic_blocker=semantic_blocker,
            side=side,
        )
        agree = fastpath.report_shadow(
            side=side,
            legacy_records=legacy_snapshot,
            fast_records=fast.records,
            elapsed_ms=fastpath.now_ms() - started,
        )
        if not agree:
            _diagnose_shadow_disagreement(
                policies,
                text=text,
                tool_calls=tools_snapshot,
                semantic_blocker=semantic_blocker,
                side=side,
            )

    fastpath.spawn_shadow(_run)


def _diagnose_shadow_disagreement(
    policies: list[PolicyRule],
    *,
    text: str,
    tool_calls: list[dict[str, Any]] | None,
    semantic_blocker: SemanticBlocker,
    side: str,
) -> None:
    """Re-ask each merged question and log its raw verdict + score.

    A DISAGREE line alone can't be root-caused: the interesting datum
    is the judge's *confidence* on the merged question (a sub-threshold
    near-miss points at intent-list dilution; a hard ALLOW points at
    something structural), and the enforcement path throws that number
    away. This runs only on disagreement — rare by construction — and
    only inside the shadow thread, so the governed call never pays
    for it.

    Compliance: the lines carry numbers and counts only (confidence,
    intent-list size, question length). Never the prompt text, never
    the intent strings.
    """
    diagnose = getattr(semantic_blocker, "diagnose", None)
    if diagnose is None:
        return
    active = _fast_active_policies(policies)
    questions = _fast_judge_questions(active, text=text, tool_calls=tool_calls)
    for index, (kind, _tool_name, group, question_text, config) in enumerate(
        questions
    ):
        try:
            raw = diagnose(question_text, config)
        except Exception:  # noqa: BLE001
            LOGGER.debug("shadow diagnosis call failed", exc_info=True)
            continue
        if not isinstance(raw, dict):
            continue
        fastpath.report_shadow_diagnosis(
            side=side,
            kind=kind,
            question_index=index,
            policy_count=len(group),
            intent_count=len(config.get("intents") or []),
            question_chars=len(question_text),
            threshold=config.get("threshold"),
            match=bool(raw.get("match")),
            confidence=float(raw.get("confidence") or 0.0),
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


#: What ``injection_scan`` does when the score clears the threshold.
#: ``flag`` is the default and the recommended starting posture — it
#: writes the finding to the audit row and lets the call through, so an
#: operator can watch a week of real traffic before deciding to refuse
#: anything. A detector that blocks on day one gets turned off on day
#: two.
_INJECTION_ACTIONS = ("flag", "block")

#: Default bar. Chosen so a single unambiguous pattern (weight ≥ 0.75)
#: clears it alone while a lone corroborating hint does not.
_INJECTION_DEFAULT_THRESHOLD = 0.75


def _injection_scan_match(
    policy: PolicyRule,
    *,
    text: str,
    reason_code: str,
) -> MatchedPolicyRecord | None:
    """Score ``text`` for prompt-injection shapes and apply the action.

    Config:

    ``threshold`` (float, default 0.75)
        The bar. Below it the rule is silent.
    ``action`` (``"flag"`` | ``"block"``, default ``"flag"``)
        ``flag`` writes the finding to the audit row and lets the call
        through; ``block`` refuses it.
    ``classes`` (list of class ids)
        Narrow the scan. Omit to run all six.
    ``message`` (str)
        Overrides the operator-facing text, same as every other kind.

    A ``flag`` returns a record whose own ``verdict`` is ``"flag"``.
    That is a fourth value in ``MatchedPolicyRecord.verdict``, and it
    is safe by construction: every consumer of that field tests it
    against ``"block"`` or ``"sanitize"`` explicitly, so an unknown
    value is inert everywhere the call's verdict is computed. The
    record still rides along on ``PolicyDecision.matched_policies``,
    which is what puts the finding in front of an operator — the whole
    point of a flag. Returning ``None`` instead would have made the
    default action do nothing at all.
    """
    config = policy.config or {}

    action = str(config.get("action") or "flag").strip().lower()
    if action not in _INJECTION_ACTIONS:
        action = "flag"

    try:
        threshold = float(config.get("threshold", _INJECTION_DEFAULT_THRESHOLD))
    except (TypeError, ValueError):
        threshold = _INJECTION_DEFAULT_THRESHOLD
    # A threshold outside 0–1 is a typo, not an intent. Clamping keeps
    # a fat-fingered ``7.5`` from silently disabling the rule forever.
    threshold = min(1.0, max(0.0, threshold))

    raw_classes = config.get("classes")
    wanted: tuple[str, ...] | None = None
    if isinstance(raw_classes, (list, tuple)):
        picked = tuple(
            str(c).strip() for c in raw_classes if str(c).strip() in injection.CLASSES
        )
        wanted = picked or None

    result = injection.scan(text, classes=wanted)
    if result.score < threshold or not result.findings:
        return None

    primary = result.primary
    label = (primary.cls if primary else "prompt injection").replace("_", " ")
    detail = f"{label}, confidence {result.score:.2f}"
    default_message = (
        f"Content matched a prompt-injection shape ({detail})."
        if action == "block"
        else f"Possible prompt injection in this content ({detail})."
    )
    return MatchedPolicyRecord(
        name=policy.name,
        type="injection_scan",
        verdict="block" if action == "block" else "flag",
        reason_code=reason_code,
        message=str(config.get("message") or "").strip() or default_message,
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
    # ``custom:employee_id`` is the wire id; "Employee ID" is what the
    # operator typed. Messages are read by people, so they get the
    # latter — the id still travels in ``sanitize_types``.
    detected_labels = [
        _pii_custom.label_for(t) if _pii_custom.is_custom(t) else t
        for t in detected_types
    ]
    if action == "sanitize" and allow_sanitize:
        return MatchedPolicyRecord(
            name=policy.name,
            type=policy.type,
            verdict="sanitize",
            reason_code="pii_sanitized",
            message=policy.config.get(
                "message",
                f"PII redacted before forwarding ({', '.join(detected_labels)}).",
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
    *,
    surfaces: tuple[str, ...] = ("model", "tool", "mcp"),
) -> PolicyDecision:
    """Evaluate output-side policies in two phases.

    Mirrors ``evaluate_policies`` exactly: deterministic local
    checks run first, LLM-backed checks (``semantic_guard``) run
    afterwards — and only when Phase 1 didn't already block. This
    is the same security contract the request side honors
    (security-and-compliance.mdc §2): no LLM call, no token spend,
    no chance of forwarding sensitive content to a judge once a
    local rule has already refused the response.

    Rules whose ``phase`` is ``"request"`` are skipped — they
    only fire during ``evaluate_policies``. Rules whose
    ``applies_to`` doesn't intersect ``surfaces`` are skipped too.
    The default covers everything, because the output phase of a
    model call sees the completion text (model surface) *and* the
    model's tool-call requests (tool + mcp surfaces) — narrowing
    happens at call sites that evaluate a single surface, e.g. the
    per-tool hooks pass ``("tool",)`` and the MCP-server gate
    passes ``("mcp",)``.

    Verdict precedence across all matches is
    ``block > sanitize > allow``.
    """
    response_side = [
        p for p in policies
        if _runs_response_phase(p) and _covers_surface(p, surfaces)
    ]
    phase1 = [p for p in response_side if p.type in _DETERMINISTIC_KINDS]
    phase2 = [p for p in response_side if p.type in _LLM_BACKED_KINDS]

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
    if phase1_matches.has_block or phase1_matches.has_pending:
        return _synthesize_decision(phase1_matches.records)

    if not phase2:
        return _synthesize_decision(phase1_matches.records)

    # Fast-governance dispatch — mirror of the input side; see
    # ``evaluate_policies`` and ``egisai.policy.fastpath``.
    fast_mode = fastpath.mode()
    sem2 = [p for p in phase2 if p.type == "semantic_guard"]
    other2 = [p for p in phase2 if p.type != "semantic_guard"]
    if fast_mode == "on":
        phase2_matches = _collect_semantic_fast(
            sem2,
            text=context.text,
            tool_calls=context.tool_calls,
            semantic_blocker=semantic_blocker,
            side="output",
        )
        # Any future non-mergeable LLM-backed kind still runs, via
        # the legacy walker, in every mode.
        if other2:
            for rec in _collect_output_matches(
                other2, context, semantic_blocker=semantic_blocker
            ).records:
                phase2_matches.add(rec)
    else:
        phase2_matches = _collect_output_matches(
            phase2, context, semantic_blocker=semantic_blocker
        )
        if fast_mode == "shadow":
            _spawn_semantic_shadow(
                sem2,
                text=context.text,
                tool_calls=context.tool_calls,
                semantic_blocker=semantic_blocker,
                side="output",
                legacy_records=phase2_matches.records,
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

    Phase 2 fans out across policies for the same reason (and with
    the same accuracy-neutrality argument) as
    ``_collect_input_matches`` — see its docstring.
    """
    out = _PhaseMatches()
    if semantic_blocker is None or len(policies) < 2:
        for policy in policies:
            rec = _evaluate_one_output_policy(policy, context, semantic_blocker)
            if rec is not None:
                out.add(rec)
        return out

    results = _fan_out(
        [
            _bind_output_policy(policy, context, semantic_blocker)
            for policy in policies
        ],
        max_workers=_judge_budget.get(),
    )
    for rec in results:
        if rec is not None:
            out.add(rec)
    return out


def _bind_output_policy(
    policy: PolicyRule,
    context: OutputPolicyContext,
    semantic_blocker: SemanticBlocker | None,
) -> Callable[[], MatchedPolicyRecord | None]:
    """Freeze one response-side evaluation into a zero-arg callable.

    Sibling of ``_bind_input_policy``; see it for why this isn't a
    ``lambda`` inside the comprehension.
    """
    def run() -> MatchedPolicyRecord | None:
        return _evaluate_one_output_policy(policy, context, semantic_blocker)

    return run


def _evaluate_one_output_policy(
    policy: PolicyRule,
    context: OutputPolicyContext,
    semantic_blocker: SemanticBlocker | None,
) -> MatchedPolicyRecord | None:
    """Evaluate one response-side rule, then apply the approval transform."""
    rec = _dispatch_one_output_policy(policy, context, semantic_blocker)
    return _maybe_require_approval(policy, rec)


def _dispatch_one_output_policy(
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

    if policy.type == "deny_resource_access":
        return _deny_resource_access_match(policy, context)

    if policy.type == "injection_scan":
        # The response side is where this earns its keep. A tool
        # result — a fetched web page, a Jira comment, a PDF the agent
        # just read — comes back through here on its way to the next
        # turn, and that is the text an attacker actually controls.
        # The prompt side catches the user typing an override; this
        # catches the document that types it for them.
        return _injection_scan_match(
            policy,
            text=context.text,
            reason_code="injection_in_output",
        )

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


# ── Identity-aware access control (``deny_resource_access``) ─────────
#
# The "scalpel". ``deny_tool_call`` blocks a tool for everyone;
# ``deny_resource_access`` blocks one *resource* — a file id, a record
# id, a path, an MCP resource URI found inside the call's arguments —
# for the wrong *people*, leaving the same tool working for everyone
# else and every other resource. It is the runtime counterpart to a
# data platform's per-file ACL, enforced at the exact seam the SDK
# already sits on (the tool / MCP call), with no re-login and no data
# ever leaving the boundary.
#
# Two things it deliberately is NOT:
#
# * Not a data-source connector. It governs the call the agent already
#   makes; it does not fetch, index, or proxy the file itself.
# * Not a substitute for the source system's own permissions. It is a
#   second, agent-scoped gate an operator controls centrally.


def _tool_call_payload(tc: dict[str, Any]) -> str:
    """Best-effort searchable string for one tool call's arguments.

    The provider patches normalize model-response tool calls to
    ``{"name": ..., "arguments": <json str>}``, but the PreToolUse and
    MCP-client gates hand us ``{"name": ..., "input": <dict>}`` instead
    (see ``_patches/claude_agent_sdk.py`` and ``_patches/mcp_client.py``).
    A resource rule must see the file id / path regardless of which path
    produced the call, so read both shapes and JSON-serialize a dict /
    list ``input`` into the same string the argument matchers expect.
    """
    args = tc.get("arguments")
    if isinstance(args, str) and args:
        return args
    payload = tc.get("input")
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict | list):
        try:
            import json as _json

            return _json.dumps(payload, default=str, sort_keys=True)
        except Exception:  # noqa: BLE001
            return str(payload)
    # A caller that stuffed a dict directly under ``arguments``.
    if isinstance(args, dict | list):
        try:
            import json as _json

            return _json.dumps(args, default=str, sort_keys=True)
        except Exception:  # noqa: BLE001
            return str(args)
    return ""


def _identity_in(value: str, entries: list[str]) -> bool:
    """Case-insensitive membership test for a role / end-user token.

    An empty ``value`` (the caller never set ``user_role`` /
    ``end_user_id``) never matches: ``_config_str_list`` already drops
    empty strings from ``entries``, so an unknown identity is never on
    any list. That single property is what makes the allowlist branch
    fail closed — an un-identified caller is treated as "not permitted"
    rather than "permitted".
    """
    if not value:
        return False
    needle = value.lower()
    return any(needle == entry.lower() for entry in entries)


def _short_resource_label(text: str, limit: int = 120) -> str:
    """Trim a resource string to something quotable in a block message
    without leaking a whole serialized argument blob into the audit."""
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _resource_hit(haystack: str, patterns: list[str]) -> str | None:
    """Is ``haystack`` a resource this rule governs?

    Returns a short label when yes, ``None`` when no. A label of ``""``
    means "in scope but nothing quotable" (an un-patterned rule against
    an empty payload) — callers must test the result with ``is None``,
    never truthiness, so that case isn't mistaken for a miss.

    * No ``patterns`` → every resource is in scope.
    * With ``patterns`` → in scope only when one matches.
    """
    if not patterns:
        return _short_resource_label(haystack) if haystack else ""
    if not haystack:
        return None
    for pattern in patterns:
        if safe_search(pattern, haystack, re.IGNORECASE):
            return _short_resource_label(haystack)
    return None


def _resource_in_scope(
    context: OutputPolicyContext,
    resource_patterns: list[str],
    tool_patterns: list[str],
) -> str | None:
    """First governed resource this call touches, or ``None``.

    ``tool_patterns`` narrows *which* tool calls the rule looks at (by
    name); ``resource_patterns`` matches the resource id *inside* those
    calls' arguments — or inside an MCP target. Returns a short label
    (possibly ``""``) for the first hit, ``None`` when the call touches
    nothing the rule governs.
    """
    for tc in context.tool_calls:
        raw_name = tc.get("name")
        name = raw_name if isinstance(raw_name, str) else ""
        if tool_patterns and not any(
            safe_search(pattern, name, re.IGNORECASE)
            for pattern in tool_patterns
        ):
            continue
        hit = _resource_hit(_tool_call_payload(tc), resource_patterns)
        if hit is not None:
            return hit or name
    # An MCP target ("server/resource-uri") carries no tool name of its
    # own, so a rule narrowed to specific tool names doesn't reach it;
    # an un-narrowed rule does.
    if not tool_patterns:
        for target in context.mcp_targets:
            hit = _resource_hit(target, resource_patterns)
            if hit is not None:
                return hit or target
    return None


def _deny_resource_access_match(
    policy: PolicyRule,
    context: OutputPolicyContext,
) -> MatchedPolicyRecord | None:
    """Block a tool / MCP call that touches a governed resource unless
    the end-user behind the call is permitted it.

    Config (all optional, but at least one identity predicate is
    required for the rule to do anything):

    * ``resource_patterns`` — regex matched against the serialized tool
      arguments AND MCP target strings, i.e. the resource identifier
      (file id, record id, path, URI). Empty ⇒ every resource the call
      touches is in scope.
    * ``tool_patterns`` — regex against the tool *name*, to scope the
      rule to specific tools. Empty ⇒ any tool.
    * ``allow_roles`` / ``allow_end_users`` — allowlists. When set,
      anyone NOT on them — **including an unknown / unset identity** —
      is blocked. Fail-closed: access control refuses when it can't
      prove the caller is permitted.
    * ``deny_roles`` / ``deny_end_users`` — blocklists, evaluated first
      so a denied user can't slip through by also being allow-listed.
    * ``message`` — custom block message.

    Identity comes from ``EgisaiContext`` (``egisai.set_context(
    user_role=..., end_user_id=...)``) carried on the context. A rule
    with no identity predicate no-ops rather than blocking everything —
    that would just be a mislabelled ``deny_tool_call`` — matching the
    fail-open-on-availability contract for a single broken rule. The
    backend rejects such a rule at creation time so the operator learns
    early; this is the runtime safety net.
    """
    resource_patterns = _config_str_list(policy.config, "resource_patterns")
    tool_patterns = _config_str_list(policy.config, "tool_patterns")
    allow_roles = _config_str_list(policy.config, "allow_roles")
    deny_roles = _config_str_list(policy.config, "deny_roles")
    allow_end_users = _config_str_list(policy.config, "allow_end_users")
    deny_end_users = _config_str_list(policy.config, "deny_end_users")

    if not (allow_roles or deny_roles or allow_end_users or deny_end_users):
        return None

    matched_resource = _resource_in_scope(
        context, resource_patterns, tool_patterns
    )
    if matched_resource is None:
        return None

    role = context.user_role or ""
    end_user = context.end_user_id or ""

    blocked = (
        _identity_in(role, deny_roles)
        or _identity_in(end_user, deny_end_users)
        or (bool(allow_roles) and not _identity_in(role, allow_roles))
        or (
            bool(allow_end_users)
            and not _identity_in(end_user, allow_end_users)
        )
    )
    if not blocked:
        return None

    who = role or end_user or "this user"
    default_message = (
        f"Access to '{matched_resource}' is not permitted for {who}."
        if matched_resource
        else f"This resource is not permitted for {who}."
    )
    return MatchedPolicyRecord(
        name=policy.name,
        type=policy.type,
        verdict="block",
        reason_code="resource_access_blocked",
        message=policy.config.get("message", default_message),
    )
