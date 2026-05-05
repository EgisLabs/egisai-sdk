"""Zero-touch agent identity detection.

Each in-flight model call is fingerprinted by its system prompt.
Different sub-agents in a multi-agent app use different system
prompts, so each unique system prompt becomes a distinct agent on
the dashboard automatically.

Resolution order:

  1. Explicit ``set_context(agent=…)`` (the user always wins).
  2. Hash of the system prompt; auto-registered with a friendly
     name extracted from the prompt itself.
  3. Fallback to the init-time agent.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
from typing import Any, Optional

LOGGER = logging.getLogger("egisai.auto_agent")

# (identity_hash, source) → agent_id
_id_cache: dict[str, str] = {}
_id_lock = threading.Lock()

_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bYou are\b\s+(?:a\s+|an\s+)?specialist:\s*([^.\n,;:]+)"),
    re.compile(r"^\s*You are\s+([A-Z][^.\n,;:]+?),", re.MULTILINE),
    re.compile(r"^\s*You are\s+(?:a\s+|an\s+)?([^.\n,;:]+?)\s*(?:[.\n;:]|$)", re.MULTILINE),
    re.compile(r"^\s*#+\s*([^\n.]+)$", re.MULTILINE),
)

_FILLER_RE = re.compile(r"\b(an?|the|specialist:?|expert|professional)\b\s*", re.IGNORECASE)


def _normalize_name(raw: str) -> str:
    """Trim, drop filler words, collapse whitespace, cap length."""
    s = raw.strip().rstrip(".:;,")
    s = _FILLER_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > 60:
        s = s[:57].rstrip() + "…"
    return s


def _system_text(payload: Any, messages: Any) -> str:
    """Pull the system prompt out of whatever shape the framework provides."""
    if isinstance(payload, dict):
        sys = payload.get("system") or payload.get("system_instruction")
        if isinstance(sys, str) and sys.strip():
            return sys.strip()
        if isinstance(sys, list):
            chunks = [
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in sys
            ]
            joined = "\n".join(c for c in chunks if c).strip()
            if joined:
                return joined

    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                content = m.get("content")
                if isinstance(content, str) and content.strip():
                    return content.strip()
                if isinstance(content, list):
                    chunks = [
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    ]
                    joined = "\n".join(c for c in chunks if c).strip()
                    if joined:
                        return joined
    return ""


def derive_identity(payload: Any, messages: Any) -> Optional[tuple[str, str]]:
    """Return ``(identity_hash, display_name)``, or ``None`` if no system prompt."""
    system = _system_text(payload, messages)
    if not system:
        return None

    digest = hashlib.sha1(system.encode("utf-8")).hexdigest()  # noqa: S324

    name: str | None = None
    for pat in _NAME_PATTERNS:
        m = pat.search(system)
        if m:
            candidate = _normalize_name(m.group(1))
            if candidate and 2 <= len(candidate) <= 60:
                name = candidate
                break

    if name is None:
        name = f"agent-{digest[:8]}"

    return digest, name


def resolve_agent_id(identity_hash: str, display_name: str) -> Optional[str]:
    """Look up or lazily register the agent id for this identity.

    Cached for the lifetime of the process. Failures degrade
    gracefully: the call proceeds and the event is attributed to
    the init-time agent.
    """
    cached = _id_cache.get(identity_hash)
    if cached:
        return cached

    with _id_lock:
        cached = _id_cache.get(identity_hash)
        if cached:
            return cached
        try:
            from egisai._backend import ensure_agent
            from egisai._config import get_config_optional

            cfg = get_config_optional()
            if cfg is None:
                return None
            payload = ensure_agent(
                name=display_name,
                description=f"Auto-detected by SDK from system prompt fingerprint {identity_hash[:8]}",
            )
            agent_id = payload.get("id")
            if isinstance(agent_id, str) and agent_id:
                _id_cache[identity_hash] = agent_id
                if payload.get("created"):
                    print(
                        f"✓ [egisai] auto-registered agent {display_name!r} "
                        f"(id={agent_id[:8]}…) — now visible on dashboard",
                        flush=True,
                    )
                return agent_id
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "egisai: auto-register for %r failed: %s",
                display_name,
                exc,
                exc_info=True,
            )
        return None
