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
from egisai._patches import genai as patch_genai
from egisai._patches import google as patch_google
from egisai._patches import google_adk as patch_google_adk
from egisai._patches import http as patch_http
from egisai._patches import langchain as patch_langchain
from egisai._patches import langgraph as patch_langgraph
from egisai._patches import llamaindex as patch_llamaindex
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
    refresh_interval_seconds: float = 10.0,
    enable_sse: bool = True,
    enable_http_fallback: bool = True,
    auto_stack_hints: str = "loose",
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
        if auto_stack_hints not in ("strict", "loose", "off"):
            raise ValueError(
                f"auto_stack_hints must be 'strict', 'loose', or 'off', "
                f"got {auto_stack_hints!r}"
            )

        cfg = EgisaiConfig(
            api_key=api_key,
            app=app,
            env=env,
            base_url=base_url
            or os.getenv("EGISAI_BASE_URL")
            or "https://app.egisai.co",
            on_block=on_block,  # type: ignore[arg-type]
            semantic_on_outage=semantic_on_outage,  # type: ignore[arg-type]
            refresh_interval_seconds=refresh_interval_seconds,
            enable_sse=enable_sse,
            enable_http_fallback=enable_http_fallback,
            sdk_version=__version__,
            auto_stack_hints=auto_stack_hints,  # type: ignore[arg-type]
        )
        set_config(cfg)

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
            cfg = update_config(org_id=hs.get("org_id"), agent_id=hs.get("agent_id"))
            handshake_ok = True
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "[egisai] handshake failed: %s — running in OFFLINE mode "
                "(no policies will be enforced) api_key=%s base_url=%s",
                exc,
                redact_api_key(cfg.api_key),
                cfg.base_url,
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
            print(
                f"✓ [egisai] active — app={cfg.app} env={cfg.env} "
                f"on_block={cfg.on_block} integrations=[{integrations}] "
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


def diagnostics() -> dict[str, object]:
    """Return a JSON-serializable snapshot of SDK runtime health.

    Suitable for surfacing in dashboards or ``/healthz`` endpoints.
    The keys are stable across patch releases:

    * ``initialized`` — whether ``init()`` has run.
    * ``sdk_version`` — version string of this SDK install.
    * ``app`` / ``env`` — process-wide config.
    * ``policy_etag`` — opaque cache version of the current rule set.
    * ``policy_rule_count`` — number of cached rules currently active.
    * ``audit_queue_size`` — pending audit events not yet flushed.
    * ``audit_dropped_total`` — events dropped due to queue overflow.
    """
    cfg = get_config_optional()
    if cfg is None:
        return {"initialized": False, "sdk_version": __version__}

    try:
        from egisai._logger import get_dropped_total, queue_size
        from egisai._policy_cache import get_etag, get_rules

        return {
            "initialized": True,
            "sdk_version": __version__,
            "app": cfg.app,
            "env": cfg.env,
            "base_url": cfg.base_url,
            "on_block": cfg.on_block,
            "semantic_on_outage": cfg.semantic_on_outage,
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
