"""Process-wide SDK configuration.

Set once by ``egisai.init()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OnBlock = Literal["raise", "stub"]
# What the SDK does when a call is held for human approval and the
# wait budget elapses before a human decides.
# "raise" (default) — raise ``ApprovalPendingError`` so the caller
#           knows the action is parked; a retry re-attaches to the
#           same hold via the idempotency key and returns instantly
#           once decided. "stub" — return a framework-shaped
#           "pending" response so an agent loop keeps running.
OnPending = Literal["raise", "stub"]
OnOutage = Literal["allow", "block"]
# In-process availability posture: what the SDK does when it cannot
# confirm policy from the control plane (it never completed a
# successful policy sync AND has no cached rules to fall back on).
# "allow" — fail open (default; the SDK's core availability contract
#           — never break the customer's call path). "block" — fail
#           closed; refuse governed calls until the SDK syncs policy
#           at least once. The symmetric opposite of the Gateway's
#           ``gateway_degraded_mode="refuse"`` posture, for customers
#           who want an ungoverned SDK to stop traffic rather than
#           run blind.
InProcessOnOutage = Literal["allow", "block"]
# Behavior when the inline Gateway itself is unreachable in gateway
# mode. "local" — re-run the call on the customer's own provider
# client under in-process governance. "fail" — let the transport
# error propagate so the Gateway stays a hard enforcement boundary.
GatewayOnOutage = Literal["local", "fail"]
# Stack-frame inspection mode for ``_auto_agent`` Tier 3.
# "strict"  — only honor the explicit ``__egisai_agent__`` marker.
# "loose"   — also honor ``agent_name`` / ``egisai_agent`` / string ``agent`` locals.
# "off"     — disable stack inspection entirely.
StackHints = Literal["strict", "loose", "off"]


@dataclass(frozen=True)
class EgisaiConfig:
    api_key: str
    app: str
    env: str
    base_url: str = "https://app.egisai.co"
    on_block: OnBlock = "raise"
    # Human-in-the-loop hold behavior. ``on_pending`` mirrors
    # ``on_block``: what to do when the in-line wait budget elapses
    # before a human decides. ``approval_wait_budget_ms`` bounds how
    # long the SDK blocks the call polling for a decision before it
    # gives up and applies ``on_pending``. A generous default (2 min)
    # covers a human clicking Approve in Slack/email; long approvals
    # fall back to the retry-resumes contract.
    on_pending: OnPending = "raise"
    approval_wait_budget_ms: int = 120_000
    refresh_interval_seconds: float = 10.0
    flush_interval_seconds: float = 1.0
    flush_batch_size: int = 50
    enable_sse: bool = True
    enable_http_fallback: bool = True
    sdk_version: str = "0.60.0"
    timeout_seconds: float = 10.0
    org_id: str | None = None
    agent_id: str | None = None
    # Behavior when the platform's semantic-guard judge is unreachable.
    # "allow"  — fail open (default; matches pre-0.11 behavior).
    # "block"  — fail closed; treat the call as if every semantic_guard
    #            rule fired. Use when the operator considers Phase 2
    #            checks the primary defense for that workload.
    semantic_on_outage: OnOutage = "allow"
    # In-process availability posture (``init(on_outage=…)`` /
    # ``EGISAI_ON_OUTAGE``). Governs the core local-governance path
    # when the SDK has never synced policy from the control plane and
    # has nothing cached to enforce — the in-process equivalent of the
    # Gateway's ``gateway_degraded_mode``.
    #
    # "allow" (default) — fail open. An org with genuinely zero
    #     policies, and an SDK that couldn't reach Egis at startup, both
    #     let calls through. This is the SDK's core contract: it runs
    #     inside the customer's process and must never break their call
    #     path (``sdk-design-philosophy.mdc`` rule 5).
    # "block" — fail closed. Until the SDK confirms policy from the
    #     control plane at least once, governed calls are refused with
    #     ``egis_unavailable``. Once any sync succeeds (even one that
    #     returns zero rules), this posture stops firing — a healthy
    #     org with no policies still allows. Pick this only when a
    #     refused call is genuinely preferable to an ungoverned one.
    #
    # Never affects PII: PII detection is local and always runs
    # regardless of this posture. See ``egisai._evaluator``.
    on_outage: InProcessOnOutage = "allow"
    # Stack-frame inspection mode for Agent Identity v1 Tier 3 — see
    # ``egisai._auto_agent._try_stack_identity``. The default "loose"
    # mode picks up the common ``agent_name`` / ``__egisai_agent__``
    # / string-typed ``agent`` locals; "strict" only honors the
    # explicit ``__egisai_agent__`` marker; "off" disables Tier 3.
    auto_stack_hints: StackHints = "loose"
    # Agent descriptor opt-out. When True (default), the SDK ships a
    # PII-sanitised, truncated excerpt of an agent's system prompt the
    # first time that agent is auto-registered, so the platform can
    # generate a human description + business function in the
    # background. When False, no excerpt ever leaves the process — the
    # agent keeps the local "Auto-detected by SDK …" placeholder and
    # its business function is filled later by the behavioral class
    # judge. Set via ``init(auto_describe=False)`` or the
    # ``EGISAI_AUTO_DESCRIBE=0`` env var for privacy-sensitive
    # deployments that don't want prompt text (even sanitised) to
    # transit to the backend.
    auto_describe: bool = True
    # MCP Servers add-on. Set from the handshake response: ``True``
    # only when the caller's org has the ``mcp_servers`` entitlement
    # enabled by EgisAI staff. When ``False`` (the default for every
    # org that hasn't bought the add-on) the ``mcp_server`` patch
    # stays fully dormant — it never wraps the customer's MCP server,
    # never registers anything, and never emits events. This keeps
    # the add-on a true no-op for everyone who isn't entitled.
    mcp_servers_enabled: bool = False
    # Smart Model Routing. Set from the handshake response's
    # ``features.smart_model_routing`` flag — TRUE only when the org's
    # plan carries the entitlement AND the Model Center master switch
    # is on. When FALSE the routing client stays fully dormant (zero
    # ``/v1/sdk/route`` calls). Live flips are picked up via the
    # ``routing.changed`` SSE event without a process restart.
    smart_routing_enabled: bool = False
    # Identity stamping (``init(stamp_identity=True)`` /
    # ``EGISAI_STAMP_IDENTITY=1``). When enabled, allowlisted tool
    # invocations that create durable artifacts (git commits via the
    # Bash tool, GitHub MCP pull requests / issues / file commits)
    # get an ``On-Behalf-Of: <agent> (egis:<id>)`` trailer appended
    # via the Claude Agent SDK PreToolUse hook's ``updatedInput``
    # rewrite, so agent attribution survives inside the artifact
    # itself. Default False — nothing is ever touched unless the
    # operator opts in. See ``egisai._patches._identity_stamp``.
    stamp_identity: bool = False
    # Gateway mode (``init(gateway=True)`` / ``EGISAI_GATEWAY=1``).
    # When enabled, OpenAI chat-completions calls are rerouted through
    # the platform's inline Gateway (``<base_url>/v1``) with the
    # ``X-Egis-Api-Key`` / ``X-Egis-Agent`` headers injected
    # automatically — enforcement and audit happen server-side, and
    # the local gate is skipped for those calls to avoid double
    # evaluation. Every other endpoint / provider keeps the normal
    # in-process governance path. See ``egisai._gateway``.
    gateway_mode: bool = False
    # What happens in gateway mode when the Gateway is unreachable
    # (connection refused / timeout / 502 / 503 / 504).
    #
    # "local" (default) — the SDK re-runs the call against the
    #     customer's own provider client and governs it in-process
    #     from the last-known-good policy cache. Governance degrades
    #     from server-side to client-side; the customer's call path
    #     survives. Only possible when the client carries a real
    #     provider key (header/passthrough mode) — BYOK-vault callers
    #     have no upstream credential locally, so they always fail.
    # "fail" — the transport error propagates to the caller. Pick
    #     this when the Gateway must remain a hard enforcement
    #     boundary and a refused call is preferable to a locally
    #     governed one.
    #
    # See ``egisai._gateway.should_fall_back``.
    gateway_on_outage: GatewayOnOutage = "local"


_CONFIG: EgisaiConfig | None = None


def set_config(cfg: EgisaiConfig) -> None:
    global _CONFIG
    _CONFIG = cfg


def get_config() -> EgisaiConfig:
    if _CONFIG is None:
        raise RuntimeError("egisai not initialized — call egisai.init(...) first.")
    return _CONFIG


def get_config_optional() -> EgisaiConfig | None:
    return _CONFIG


def update_config(**fields: object) -> EgisaiConfig:
    """Replace the config with a copy carrying the supplied fields."""
    global _CONFIG
    if _CONFIG is None:
        raise RuntimeError("egisai not initialized — call egisai.init(...) first.")
    base = {
        "api_key": _CONFIG.api_key,
        "app": _CONFIG.app,
        "env": _CONFIG.env,
        "base_url": _CONFIG.base_url,
        "on_block": _CONFIG.on_block,
        "on_pending": _CONFIG.on_pending,
        "approval_wait_budget_ms": _CONFIG.approval_wait_budget_ms,
        "refresh_interval_seconds": _CONFIG.refresh_interval_seconds,
        "flush_interval_seconds": _CONFIG.flush_interval_seconds,
        "flush_batch_size": _CONFIG.flush_batch_size,
        "enable_sse": _CONFIG.enable_sse,
        "enable_http_fallback": _CONFIG.enable_http_fallback,
        "sdk_version": _CONFIG.sdk_version,
        "timeout_seconds": _CONFIG.timeout_seconds,
        "org_id": _CONFIG.org_id,
        "agent_id": _CONFIG.agent_id,
        "semantic_on_outage": _CONFIG.semantic_on_outage,
        "on_outage": _CONFIG.on_outage,
        "auto_stack_hints": _CONFIG.auto_stack_hints,
        "auto_describe": _CONFIG.auto_describe,
        "mcp_servers_enabled": _CONFIG.mcp_servers_enabled,
        "smart_routing_enabled": _CONFIG.smart_routing_enabled,
        "stamp_identity": _CONFIG.stamp_identity,
        "gateway_mode": _CONFIG.gateway_mode,
        "gateway_on_outage": _CONFIG.gateway_on_outage,
    }
    base.update(fields)
    new = EgisaiConfig(**base)  # type: ignore[arg-type]
    _CONFIG = new
    return new
