"""``egisai.init()`` and ``egisai.shutdown()``.

Idempotent. Each process maintains its own SDK state.
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
from typing import Any

from egisai import __version__
from egisai import _gateway as _gateway_mod
from egisai._backend import close_client, handshake
from egisai._config import EgisaiConfig, get_config_optional, set_config, update_config
from egisai._logger import start_worker as start_logger
from egisai._logger import stop_worker as stop_logger
from egisai._patches import agno as patch_agno
from egisai._patches import anthropic as patch_anthropic
from egisai._patches import autogen as patch_autogen
from egisai._patches import bedrock_agent as patch_bedrock_agent
from egisai._patches import bedrock_runtime as patch_bedrock_runtime
from egisai._patches import claude_agent_sdk as patch_claude_agent_sdk
from egisai._patches import crewai as patch_crewai
from egisai._patches import decision as patch_decision
from egisai._patches import genai as patch_genai
from egisai._patches import google as patch_google
from egisai._patches import google_adk as patch_google_adk
from egisai._patches import http as patch_http
from egisai._patches import langchain as patch_langchain
from egisai._patches import langgraph as patch_langgraph
from egisai._patches import llamaindex as patch_llamaindex
from egisai._patches import mcp_client as patch_mcp_client
from egisai._patches import mcp_server as patch_mcp_server
from egisai._patches import openai as patch_openai
from egisai._patches import openai_agents as patch_openai_agents
from egisai._patches import pydantic_ai as patch_pydantic_ai
from egisai._patches import smolagents as patch_smolagents
from egisai._patches import strands as patch_strands
from egisai._policy_cache import refresh_now
from egisai._redact import redact_api_key
from egisai._refresher import start_worker as start_refresher
from egisai._refresher import stop_worker as stop_refresher
from egisai.policy import _pii_loader

LOGGER = logging.getLogger("egisai")

_init_lock = threading.RLock()


def init(
    *,
    api_key: str | None = None,
    app: str = "default",
    env: str = "production",
    base_url: str | None = None,
    on_block: str = "raise",
    semantic_on_outage: str = "allow",
    on_outage: str | None = None,
    refresh_interval_seconds: float = 10.0,
    enable_sse: bool = True,
    enable_http_fallback: bool = True,
    auto_stack_hints: str = "loose",
    auto_describe: bool | None = None,
    gateway: bool | None = None,
    gateway_on_outage: str | None = None,
    stamp_identity: bool | None = None,
    cloud_identity: bool | None = None,
    quiet: bool = False,
) -> None:
    """Activate egisai for the current process.

    After this call returns, every supported AI library installed in the
    environment is patched and will route through your platform-defined
    policies before reaching the model. Decisions are logged to the
    platform's audit trail asynchronously.

    Parameters
    ----------
    api_key
        Your platform API key (the one starting with ``egis_live_`` that
        you created on the dashboard's API Keys page). If not provided,
        falls back to the ``EGISAI_API_KEY`` environment variable.
    app
        Logical agent name. The platform auto-creates an Agent matching
        this name in your org if one doesn't exist; subsequent calls
        from this SDK instance are tied to it for auditing.
    env
        Free-form environment label (``"dev"``, ``"staging"``, ``"prod"``).
    base_url
        Override the platform URL. Defaults to ``EGISAI_BASE_URL`` or
        ``https://app.egisai.co``. Only needed for self-hosted /
        regional installs.
    on_block
        ``"raise"`` (default) — raise ``PermissionError`` when a policy
        denies a call. ``"stub"`` — return a framework-shaped "blocked"
        response so the agent keeps running.
    semantic_on_outage
        Behavior when the semantic-guard judge cannot be reached.
        ``"allow"`` (default, fail-open) treats the rule as a no-op so
        availability of the primary call path is preserved. ``"block"``
        (fail-closed) refuses the call when the judge is unreachable —
        appropriate when an operator considers Phase 2 their primary
        defense for that workload.
    on_outage
        In-process availability posture — what the SDK does when it has
        never confirmed policy from the control plane (handshake or the
        first policy fetch failed) and has nothing cached to enforce.
        The client-side counterpart of the Gateway's
        ``gateway_degraded_mode``.

        * ``"allow"`` (default, fail-open) — run the call ungoverned.
          This is the SDK's core contract: it lives inside your process
          and must never break your call path. A genuinely policy-free
          org lands here too.
        * ``"block"`` — fail closed. Governed calls are refused with
          ``egis_unavailable`` until the SDK reaches Egis at least once
          (even a sync that returns zero rules clears it). Choose this
          only when a refused call beats an ungoverned one.

        Never affects PII: PII detection is fully local and always runs
        regardless of this setting. Also settable via
        ``EGISAI_ON_OUTAGE``.
    refresh_interval_seconds
        How often to poll for policy changes if SSE is unavailable.
    enable_sse
        Use Server-Sent Events for instant policy updates. Falls back to
        polling on connection failure.
    enable_http_fallback
        Patch ``httpx`` / ``requests`` for HTTP-level audit visibility.
    auto_stack_hints
        Controls the Agent Identity v1 Tier 3 stack-frame inspector
        (see ``_auto_agent._try_stack_identity``).

        * ``"loose"`` (default) — when the resolver can't find an
          identity any other way, it walks up to 12 stack frames
          looking for ``__egisai_agent__`` / ``agent_name`` /
          ``egisai_agent`` / string-typed ``agent`` locals.
        * ``"strict"`` — only the explicit ``__egisai_agent__``
          marker is recognised. Quieter — won't accidentally pick
          up an enclosing test's ``agent_name`` variable.
        * ``"off"`` — Tier 3 is disabled entirely; the resolver
          skips to Tier 4 (class-name introspection).

        Has no effect on explicit ``egisai.set_context`` /
        ``with egisai.agent(...)`` calls (Tier 0) which always win.
    auto_describe
        Whether to let the platform auto-generate a human description
        and business function for each newly-detected agent. When
        enabled (the default), the SDK ships a PII-sanitised,
        truncated excerpt of the agent's system prompt the *first*
        time that agent is registered; the platform summarises it in
        the background (no added latency on your call path) into the
        identity card you see on the dashboard. When disabled, no
        prompt text — even sanitised — leaves the process: the agent
        keeps the local placeholder description and its business
        function is inferred later from anonymised behavioral
        telemetry instead. Defaults to ``True``; override with
        ``auto_describe=False`` or ``EGISAI_AUTO_DESCRIBE=0``.
    gateway
        Route OpenAI chat-completions calls through the platform's
        inline Gateway instead of evaluating policies in-process.
        The SDK injects ``X-Egis-Api-Key`` (this ``api_key``) and —
        when an explicit identity is active via ``set_context`` /
        ``with egisai.agent(...)`` — ``X-Egis-Agent`` on every call,
        so context works identically in both modes. Enforcement and
        audit happen server-side; the local gate is skipped for
        those calls to avoid double governance. Everything the
        Gateway doesn't carry (Responses API, Anthropic / Google /
        Bedrock SDKs, agent frameworks, MCP) keeps the normal
        in-process path. Requires the org's ``inline_gateway``
        feature. Defaults to ``False``; also settable via the
        ``EGISAI_GATEWAY`` env var.
    gateway_on_outage
        What happens in gateway mode when the Gateway can't be
        reached (connection refused, timeout, HTTP 502 / 503 / 504).

        * ``"local"`` (default) — re-run the call against your own
          provider client and govern it in-process from the
          last-known-good policy cache. Your call path survives an
          Egis outage; governance degrades from server-side to
          client-side and the audit row is written by the SDK. Only
          possible when your client carries a real provider key —
          BYOK-vault callers have no upstream credential locally.
        * ``"fail"`` — let the error propagate, keeping the Gateway a
          hard enforcement boundary.

        A 4xx from the Gateway is never retried locally under either
        setting: it's a decision (policy block, auth, quota), not an
        outage. Also settable via ``EGISAI_GATEWAY_ON_OUTAGE``.
    stamp_identity
        Propagate agent identity into the durable artifacts agents
        create. When enabled, allowlisted tool invocations (git
        commits made through the Claude Agent SDK's Bash tool,
        GitHub MCP pull requests / issues / file commits) get an
        ``On-Behalf-Of: <agent-name> (egis:<agent-id>)`` trailer
        appended before execution, so anyone reading the git
        history or the PR later can attribute it to the specific
        agent. Only the agent's display name and Egis UUID are
        embedded — never prompt text or PII. Defaults to ``False``
        (nothing is touched); also settable via the
        ``EGISAI_STAMP_IDENTITY`` env var.
    cloud_identity
        Report which cloud principal this process runs as — the AWS
        role ARN, GCP service account email, Azure client id, or OAuth
        client ids already present in the environment. The platform
        uses it to link the OAuth grants and service principals it
        inventories from your SaaS tenants to the specific agent using
        them, turning "some app has standing access to Drive" into
        "this agent does". Identity strings only: no credential, token,
        or key is ever read, including from the instance metadata
        service. The probe runs in a background thread with a 200 ms
        budget, so it never delays your first model call. Defaults to
        ``True``; disable with ``cloud_identity=False`` or
        ``EGISAI_CLOUD_IDENTITY=0``.
    quiet
        Suppress the one-line "egisai active" startup log.
    """
    with _init_lock:
        existing = get_config_optional()
        if existing is not None:
            LOGGER.debug("egisai.init() called twice; ignoring (already initialized)")
            return

        api_key = api_key or os.getenv("EGISAI_API_KEY", "")
        if not api_key:
            raise RuntimeError(
                "egisai.init() requires `api_key` or the EGISAI_API_KEY env var."
            )
        if on_block not in ("raise", "stub"):
            raise ValueError(f"on_block must be 'raise' or 'stub', got {on_block!r}")
        if semantic_on_outage not in ("allow", "block"):
            raise ValueError(
                f"semantic_on_outage must be 'allow' or 'block', "
                f"got {semantic_on_outage!r}"
            )
        # ``on_outage`` resolves explicit kwarg → env var → default
        # "allow" (fail open). Validated at init() so a typo can't
        # silently leave the in-process posture in an unintended state.
        resolved_on_outage = (
            on_outage
            or os.getenv("EGISAI_ON_OUTAGE")
            or "allow"
        ).strip().lower()
        if resolved_on_outage not in ("allow", "block"):
            raise ValueError(
                f"on_outage must be 'allow' or 'block', "
                f"got {resolved_on_outage!r}"
            )
        if auto_stack_hints not in ("strict", "loose", "off"):
            raise ValueError(
                f"auto_stack_hints must be 'strict', 'loose', or 'off', "
                f"got {auto_stack_hints!r}"
            )
        # ``gateway_on_outage`` resolves explicit kwarg → env var →
        # default "local". Validated the same way as ``on_block`` so a
        # typo fails at init() rather than silently disabling the
        # fallback during the outage it exists for.
        resolved_gateway_on_outage = (
            gateway_on_outage
            or os.getenv("EGISAI_GATEWAY_ON_OUTAGE")
            or "local"
        ).strip().lower()
        if resolved_gateway_on_outage not in ("local", "fail"):
            raise ValueError(
                f"gateway_on_outage must be 'local' or 'fail', "
                f"got {resolved_gateway_on_outage!r}"
            )

        # ``auto_describe`` resolves explicit kwarg → env var → default
        # True. The env var accepts the usual falsey spellings so an
        # operator can flip it off per-deploy without code changes.
        if auto_describe is None:
            env_describe = os.getenv("EGISAI_AUTO_DESCRIBE")
            if env_describe is None:
                resolved_auto_describe = True
            else:
                resolved_auto_describe = env_describe.strip().lower() not in (
                    "0", "false", "no", "off", "",
                )
        else:
            resolved_auto_describe = bool(auto_describe)

        # ``gateway`` resolves explicit kwarg → env var → default False,
        # mirroring ``auto_describe``'s resolution shape.
        if gateway is None:
            env_gateway = os.getenv("EGISAI_GATEWAY")
            resolved_gateway = (
                env_gateway is not None
                and env_gateway.strip().lower() in ("1", "true", "yes", "on")
            )
        else:
            resolved_gateway = bool(gateway)

        # ``stamp_identity`` resolves explicit kwarg → env var →
        # default False, mirroring ``gateway``'s resolution shape.
        # Default-off is the contract: nothing about a user's tool
        # traffic changes unless they opted in.
        if stamp_identity is None:
            env_stamp = os.getenv("EGISAI_STAMP_IDENTITY")
            resolved_stamp_identity = (
                env_stamp is not None
                and env_stamp.strip().lower() in ("1", "true", "yes", "on")
            )
        else:
            resolved_stamp_identity = bool(stamp_identity)

        # ``cloud_identity`` resolves explicit kwarg → env var →
        # default True, mirroring ``auto_describe``. Default-on because
        # the join it enables is the whole point of the feature and the
        # data is metadata the environment already asserts about
        # itself; opting out is one flag away for anyone who'd rather
        # their role ARNs stay put.
        if cloud_identity is None:
            env_cloud_identity = os.getenv("EGISAI_CLOUD_IDENTITY")
            if env_cloud_identity is None:
                resolved_cloud_identity = True
            else:
                resolved_cloud_identity = (
                    env_cloud_identity.strip().lower()
                    not in ("0", "false", "no", "off", "")
                )
        else:
            resolved_cloud_identity = bool(cloud_identity)

        cfg = EgisaiConfig(
            api_key=api_key,
            app=app,
            env=env,
            base_url=base_url
            or os.getenv("EGISAI_BASE_URL")
            or "https://app.egisai.co",
            on_block=on_block,  # type: ignore[arg-type]
            semantic_on_outage=semantic_on_outage,  # type: ignore[arg-type]
            on_outage=resolved_on_outage,  # type: ignore[arg-type]
            refresh_interval_seconds=refresh_interval_seconds,
            enable_sse=enable_sse,
            enable_http_fallback=enable_http_fallback,
            sdk_version=__version__,
            auto_stack_hints=auto_stack_hints,  # type: ignore[arg-type]
            auto_describe=resolved_auto_describe,
            stamp_identity=resolved_stamp_identity,
            gateway_mode=resolved_gateway,
            gateway_on_outage=resolved_gateway_on_outage,  # type: ignore[arg-type]
        )
        set_config(cfg)
        # A fresh init() starts a fresh fallback tally — the counter
        # describes this process's current gateway health, not a
        # previous config's.
        _gateway_mod.reset_fallback_total()

        # Started before the handshake so the probe overlaps the round
        # trip it would otherwise delay. Whatever it has found by the
        # time the first ``ensure`` goes out ships with it; anything
        # slower rides the next one.
        if resolved_cloud_identity:
            from egisai import _cloud_identity as _cloud_identity_mod

            _cloud_identity_mod.prime_async()

        handshake_ok = False
        try:
            from egisai._runtime import collect_runtime_fingerprint

            rt_blob: dict[str, Any] | None
            try:
                rt_blob = collect_runtime_fingerprint(sdk_version=__version__)
            except Exception:  # noqa: BLE001
                rt_blob = None
            hs = handshake(
                app=app,
                env=env,
                sdk_version=__version__,
                runtime=rt_blob,
            )
            # The handshake response carries an org-level ``features``
            # map (override-aware) so the SDK learns which paid add-ons
            # are enabled. ``mcp_servers`` activates the MCP server
            # patch; absent / older backends leave it ``False`` so the
            # patch stays dormant.
            raw_features = hs.get("features")
            features: dict[str, Any] = (
                raw_features if isinstance(raw_features, dict) else {}
            )
            cfg = update_config(
                org_id=hs.get("org_id"),
                agent_id=hs.get("agent_id"),
                mcp_servers_enabled=bool(features.get("mcp_servers")),
                smart_routing_enabled=bool(
                    features.get("smart_model_routing")
                ),
            )
            # Seed the routing client's enablement hint so entitled
            # orgs route from call #1 and everyone else stays fully
            # dormant (zero /route calls). Live flips arrive later
            # via the ``routing.changed`` SSE event.
            from egisai import _routing as _routing_mod

            _routing_mod.reset()
            _routing_mod.set_enabled_hint(
                bool(features.get("smart_model_routing"))
            )
            # Seed the routing-quality sample rate. ``0.0`` (the value
            # every current backend sends) keeps the post-hoc quality
            # probe fully dormant — no extra HTTP, no judge cost.
            _routing_mod.set_quality_sample(hs.get("routing_quality_sample"))
            handshake_ok = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "[egisai] handshake failed: %s — running in OFFLINE mode "
                "(no policies will be enforced) api_key=%s base_url=%s",
                exc,
                redact_api_key(cfg.api_key),
                cfg.base_url,
            )
            # Offline ⇒ the routing decision service is unreachable
            # anyway; pin the client dormant so the hot path never
            # pays a doomed /route round-trip.
            from egisai import _routing as _routing_mod

            _routing_mod.reset()
            _routing_mod.set_enabled_hint(False)
            if resolved_on_outage == "block":
                LOGGER.warning(
                    "[egisai] on_outage='block' and Egis is unreachable "
                    "— governed calls will be REFUSED until the SDK "
                    "reaches the control plane. Set on_outage='allow' "
                    "to fail open instead."
                )

        rules_count = 0
        if handshake_ok:
            try:
                refresh_now()
                from egisai._policy_cache import get_rules

                rules_count = len(get_rules())
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "[egisai] policy fetch failed: %s — no policies will be "
                    "enforced until next refresh",
                    exc,
                )

        enabled: list[str] = []
        if patch_openai.apply():
            enabled.append("openai")
        if patch_anthropic.apply():
            enabled.append("anthropic")
        if patch_genai.apply():
            enabled.append("google.genai")
        if patch_google.apply():
            enabled.append("google.generativeai")
        # ── Framework identity patches (Agent Identity v1, 0.17.0) ──
        # All import-guarded — silently no-op when the framework
        # isn't installed. Patches run AFTER the raw-provider patches
        # so an inner LLM call (governed by, e.g., the openai patch)
        # sees the framework's pushed identity via ContextVar instead
        # of re-deriving from the (often empty) inner system prompt.
        if patch_openai_agents.apply():
            enabled.append("openai-agents")
        if patch_claude_agent_sdk.apply():
            enabled.append("claude-agent-sdk")
        if patch_langgraph.apply():
            enabled.append("langgraph")
        if patch_bedrock_runtime.apply():
            enabled.append("bedrock-runtime")
        if patch_bedrock_agent.apply():
            enabled.append("bedrock-agent")
        if patch_google_adk.apply():
            enabled.append("google-adk")
        if patch_autogen.apply():
            enabled.append("autogen")
        if patch_crewai.apply():
            enabled.append("crewai")
        if patch_agno.apply():
            enabled.append("agno")
        if patch_strands.apply():
            enabled.append("strands")
        if patch_smolagents.apply():
            enabled.append("smolagents")
        if patch_langchain.apply():
            enabled.append("langchain")
        if patch_llamaindex.apply():
            enabled.append("llamaindex")
        if patch_pydantic_ai.apply():
            enabled.append("pydantic-ai")
        if enable_http_fallback and patch_http.apply():
            enabled.append("httpx/requests")
        # Marks calls this process already decided, so an egress node
        # in front of the workload doesn't govern — or log — the same
        # call a second time. No-op without httpx, and invisible
        # unless a node is actually deployed.
        patch_decision.apply()
        # Outbound MCP — the agent calling somebody else's server.
        # Ordinary agent governance (same shape as the claude_agent_sdk
        # tool hooks), so unlike the server-side patch below it is not
        # gated on an add-on. No-op unless the ``mcp`` package is
        # installed.
        if patch_mcp_client.apply():
            enabled.append("mcp-client")
        # MCP Servers add-on — dormant unless the handshake reported
        # the org is entitled. Patches FastMCP / the official mcp SDK
        # to govern inbound tools/call. No-op for every non-add-on org.
        if patch_mcp_server.apply():
            enabled.append("mcp-server")

        start_logger()
        start_refresher()
        # Kick off Presidio + spaCy NER warm-up in a daemon thread.
        # Idempotent. Until the analyzer is ready (1–3 s after import,
        # plus a one-time ~750 MB download on first install), the PII
        # path falls back to deterministic regex+checksum detection
        # for SSN / credit_card / IBAN / email / phone / api_key. So
        # PII protection is **never** off — only the NER-derived
        # entities (names, addresses, GDPR special-category text)
        # are temporarily unavailable while the model loads.
        _pii_loader.prime_analyzer_async(quiet=quiet)
        atexit.register(shutdown)

        if not quiet:
            integrations = ", ".join(enabled) if enabled else "none"
            mode = (
                f" mode=gateway on_outage={cfg.gateway_on_outage}"
                if cfg.gateway_mode
                else ""
            )
            # Surface a fail-closed in-process posture on the banner —
            # it changes what happens on an Egis outage, so an operator
            # should see it without reading diagnostics().
            outage = (
                " on_outage=block" if cfg.on_outage == "block" else ""
            )
            print(
                f"✓ [egisai] active — app={cfg.app} env={cfg.env} "
                f"on_block={cfg.on_block}{outage}{mode} "
                f"integrations=[{integrations}] "
                f"policies={rules_count}",
                flush=True,
            )
            if rules_count == 0 and handshake_ok:
                print(
                    "   ⚠  no enabled policies in this org — every call will be allowed.\n"
                    "      visit your dashboard → Policies → + New policy.",
                    flush=True,
                )


def shutdown() -> None:
    """Stop background workers and flush remaining events. Idempotent."""
    try:
        stop_refresher()
    except Exception:  # noqa: BLE001
        pass
    try:
        stop_logger()
    except Exception:  # noqa: BLE001
        pass
    try:
        close_client()
    except Exception:  # noqa: BLE001
        pass
    try:
        from egisai._evaluator import _close_semantic_blocker

        _close_semantic_blocker()
    except Exception:  # noqa: BLE001
        pass
    try:
        from egisai import _routing as _routing_mod

        _routing_mod.reset()
    except Exception:  # noqa: BLE001
        pass
    try:
        from egisai.policy import limits as _limits_mod

        _limits_mod.stop()
    except Exception:  # noqa: BLE001
        pass


def diagnostics() -> dict[str, object]:
    """Return a JSON-serializable snapshot of SDK runtime health.

    Suitable for surfacing in dashboards or ``/healthz`` endpoints.
    The keys are stable across patch releases:

    * ``initialized`` — whether ``init()`` has run.
    * ``sdk_version`` — version string of this SDK install.
    * ``app`` / ``env`` — process-wide config.
    * ``on_outage`` — in-process availability posture (``allow`` /
      ``block``).
    * ``policy_synced`` — whether the SDK has completed at least one
      successful policy sync from the control plane. ``False`` with
      ``on_outage="block"`` means governed calls are being refused.
    * ``policy_etag`` — opaque cache version of the current rule set.
    * ``policy_rule_count`` — number of cached rules currently active.
    * ``audit_queue_size`` — pending audit events not yet flushed.
    * ``audit_dropped_total`` — events dropped due to queue overflow.
    * ``gateway_fallback_total`` — gateway-mode calls this process
      served under local governance because the Gateway was
      unreachable. Non-zero means governance is currently degrading
      to the local cache; pair it with an alert.
    """
    cfg = get_config_optional()
    if cfg is None:
        return {"initialized": False, "sdk_version": __version__}

    try:
        from egisai._logger import get_dropped_total, queue_size
        from egisai._policy_cache import get_etag, get_rules, has_synced

        return {
            "initialized": True,
            "sdk_version": __version__,
            "app": cfg.app,
            "env": cfg.env,
            "base_url": cfg.base_url,
            "on_block": cfg.on_block,
            "gateway_mode": cfg.gateway_mode,
            "gateway_on_outage": cfg.gateway_on_outage,
            "gateway_fallback_total": _gateway_mod.fallback_total(),
            "semantic_on_outage": cfg.semantic_on_outage,
            "on_outage": cfg.on_outage,
            "policy_synced": has_synced(),
            "policy_etag": get_etag(),
            "policy_rule_count": len(get_rules()),
            "audit_queue_size": queue_size(),
            "audit_dropped_total": get_dropped_total(),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "initialized": True,
            "sdk_version": __version__,
            "diagnostics_error": exc.__class__.__name__,
        }
