"""Zero-touch agent identity resolution — Agent Identity v1.

After ``egisai.init()``, every in-flight model call goes through this
module to answer "which agent is making this call?" The answer is
needed in two places:

1. **Policy attribution.** ``_active_agent_id()`` is read inside the
   policy evaluator so scoped rules (``target_agents = […]``) match
   the right agent — even when the user never called ``set_context``.
2. **Audit trail.** The audit event ships ``agent_id`` + ``app`` so the
   dashboard's Requests / Agents / Provenance views render the right
   row. The same identity is recycled across nested calls (a framework
   loop calls the LLM N times for one logical "agent invocation"); we
   never count the same agent N times.

The resolver walks a 7-tier ladder. The **first match wins** and is
pushed onto a process-local ``ContextVar`` identity stack so any inner
call inherits the parent's identity without re-deriving it.

Tier table
----------

==== ======================================== ============================
Tier Source                                   Stable across calls?
==== ======================================== ============================
0    Explicit ``set_context`` / ``agent()``   Yes — user-supplied
0.5  Active OTEL span ``gen_ai.agent.*``      Yes — span-scoped
1    Server-issued stable id                  Yes — OpenAI prompt_id,
                                              Gemini cached_content,
                                              Bedrock InvokeAgent agentId
2A   Framework patch (explicit name)          Yes — OpenAI Agents SDK,
                                              ADK, AutoGen, Agno, Strands,
                                              CrewAI, LangGraph nodes
2B   Framework patch (composite bundle hash)  Yes — Claude Agent SDK,
                                              LlamaIndex, PydanticAI,
                                              legacy LangChain
3    Stack-frame hint                         Per-call — looks for
                                              ``__egisai_agent__`` /
                                              ``agent_name`` locals
4    Class-name introspection                 Per-call — e.g.
                                              ``self.__class__.__name__``
5    System-prompt SHA-256 + spaCy NER name   Yes within process
6    Init-time ``app=`` fallback              Yes within process

Caching
-------
A single unified ``_identity_cache`` maps the resolver's *identity
key* (a structured string like ``framework:openai_agents:Triage`` or
``hash:131a8e6a…``) to the backend's ``agent_id``. Per-process. The
backend's own ``(org_id, identity_hash)`` unique index keeps state
consistent across SDK processes.

Compliance
----------
* Only the SHA-256 *digest* of structural data (system prompt + tool
  names + model id) ever leaves the process boundary, never the raw
  prompt. ``identity_source`` is a controlled-vocabulary token. No
  PII can land in either field.
* When the analyzer is warm, names derived from system prompts use
  spaCy NER (PERSON / ORG / NORP / WORK_OF_ART) — never the prompt's
  raw free text. When the analyzer is cold or fails, we fall through
  to ``agent-<hash[:8]>`` rather than ship a name that might leak
  prompt content.
* Fail-open on availability: if any tier raises, we drop to the next
  one. The user's model call is never blocked by identity resolution.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import threading
import time
import unicodedata
from collections.abc import Iterable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

LOGGER = logging.getLogger("egisai.auto_agent")

# ── Source tokens (mirror backend migration 0036) ───────────────────
# This vocabulary is documented in
# ``backend/alembic/versions/20260530_0000_agent_identity_v1.py``; new
# tokens must be added there too so SOC 2 reviewers can map an audit
# row's ``identity_source`` back to the code path that produced it.
IdentitySource = Literal[
    "explicit",
    "otel",
    "stored_prompt:openai",
    "stored_prompt:gemini",
    "framework:openai_agents",
    "framework:claude_agent_sdk",
    "framework:langgraph",
    "framework:bedrock_runtime",
    "framework:bedrock_agent",
    "framework:adk",
    "framework:autogen",
    "framework:crewai",
    "framework:agno",
    "framework:strands",
    "framework:smolagents",
    "framework:langchain",
    "framework:llamaindex",
    "framework:pydantic_ai",
    "stack",
    "class",
    "hash",
    "app",
]


@dataclass(frozen=True)
class IdentityRecord:
    """Resolved identity for a single in-flight model call.

    ``identity_key`` is a structured string we use to dedup inside the
    SDK process (e.g. ``framework:openai_agents:Triage`` or
    ``hash:131a8e6a…``). ``identity_hash`` is the 64-hex SHA-256 we
    send to the backend so the partial unique index can dedup at the
    org level. ``display_name`` is the human label.
    """

    agent_id: str | None
    display_name: str
    identity_key: str
    identity_hash: str
    source: IdentitySource
    # Tiers 0–2 push themselves onto the identity stack so inner
    # nested calls inherit. Tiers 3–6 are per-call only — they reflect
    # the *current* call's surroundings (stack vars, system prompt)
    # which the next call should re-derive from its own context.
    push_to_stack: bool = field(default=False)


# ── Unified identity cache (replaces _id_cache + _agent_id_cache) ────
#
# Keyed by ``identity_key`` so a hash-derived identity and an
# explicit-name identity for the same agent can NEVER produce two
# rows in the cache for one server-side row. Backend dedups by
# ``(org_id, identity_hash)`` so racing inserts converge to one
# agent_id regardless of which SDK process won.

_identity_cache: dict[str, str] = {}
_identity_lock = threading.Lock()

# ── Registration backoff (negative cache) ───────────────────────────
#
# ``_identity_cache`` only ever holds *successes*. Without a
# companion record of failures, an unreachable backend turns every
# governed call into a fresh ``POST /v1/sdk/agents/ensure`` attempt —
# inline on the customer's call path, serialized behind
# ``_identity_lock``. Connection-refused fails in microseconds, but
# the outage shape that actually hurts is a black-holed load
# balancer: there the customer pays the full HTTP timeout on *every*
# call, one at a time.
#
# So a failed ensure marks its identity key as "don't try again
# before <deadline>". Inside the window the resolver returns ``None``
# immediately (the documented unattributed-but-governed path); after
# it, exactly one call retries. Steady-state cost is one dict lookup.
_ENSURE_BACKOFF_S = 60.0
_ENSURE_BACKOFF_ENV = "EGISAI_AGENT_ENSURE_BACKOFF_SECS"
# Bound on the negative cache. Identity keys are derived from system
# prompts, so a pathological caller could mint many distinct ones;
# capping keeps the dict from growing without limit during a long
# outage. Expired entries are pruned first, then the whole map is
# dropped (which only costs one extra ensure attempt per identity).
_ENSURE_BACKOFF_MAX_ENTRIES = 2048
_ensure_backoff: dict[str, float] = {}


def _ensure_backoff_secs() -> float:
    """How long to wait before retrying a failed registration."""
    raw = os.environ.get(_ENSURE_BACKOFF_ENV)
    if raw is None or not raw.strip():
        return _ENSURE_BACKOFF_S
    try:
        return max(0.0, float(raw.strip()))
    except ValueError:
        return _ENSURE_BACKOFF_S


def _in_ensure_backoff(identity_key: str) -> bool:
    """True while ``identity_key``'s registration is being backed off.

    Reads the plain dict without taking ``_identity_lock`` — a stale
    read only costs one redundant ensure attempt, and keeping the
    fast path lock-free is the whole point.
    """
    deadline = _ensure_backoff.get(identity_key)
    if deadline is None:
        return False
    if time.monotonic() >= deadline:
        _ensure_backoff.pop(identity_key, None)
        return False
    return True


def _note_ensure_failure(identity_key: str) -> None:
    """Record that registering ``identity_key`` just failed."""
    backoff = _ensure_backoff_secs()
    if backoff <= 0:
        return
    now = time.monotonic()
    if len(_ensure_backoff) >= _ENSURE_BACKOFF_MAX_ENTRIES:
        for key in [k for k, v in _ensure_backoff.items() if now >= v]:
            _ensure_backoff.pop(key, None)
        if len(_ensure_backoff) >= _ENSURE_BACKOFF_MAX_ENTRIES:
            _ensure_backoff.clear()
    _ensure_backoff[identity_key] = now + backoff


def reset_ensure_backoff() -> None:
    """Clear the registration backoff map. Used by tests."""
    _ensure_backoff.clear()


# ── Identity stack (ContextVar — async/thread-inherits) ─────────────
#
# Each pushed identity carries the resolver's full IdentityRecord
# so inner calls can read the parent's display name (some patches
# format it into their event description) without going through
# the backend again.

_identity_stack: ContextVar[tuple[IdentityRecord, ...]] = ContextVar(
    "egisai_identity_stack", default=()
)


def push_identity(record: IdentityRecord) -> object:
    """Push an identity onto the stack; return a token for resetting.

    Use the ``identity_scope`` context manager in patch code instead
    of calling this directly — it guarantees the pop happens even
    when the wrapped framework call raises.
    """
    stack = _identity_stack.get()
    return _identity_stack.set(stack + (record,))


def reset_identity(token: object) -> None:
    """Restore the stack to the state captured in ``token``."""
    try:
        _identity_stack.reset(token)  # type: ignore[arg-type]
    except (LookupError, ValueError):
        # ``reset`` is strict about provenance; if a different
        # ContextVar token leaked in we'd rather degrade gracefully
        # than crash the user's call.
        _identity_stack.set(())


@contextmanager
def identity_scope(record: IdentityRecord) -> Any:
    """Push ``record`` for the duration of the ``with`` block."""
    token = push_identity(record)
    try:
        yield record
    finally:
        reset_identity(token)


def current_identity() -> IdentityRecord | None:
    """Return the innermost pushed identity, or ``None`` if empty.

    Read by patches BEFORE running policy + audit so attribution is
    consistent with the framework's outer agent identity even when
    the inner LLM call has a different / no system prompt.
    """
    stack = _identity_stack.get()
    return stack[-1] if stack else None


# ── Tier 0.5: OpenTelemetry GenAI semantic conventions ──────────────
#
# Soft-import. If opentelemetry-api isn't installed (or no span is
# active), we return None and the resolver drops to the next tier.
# We don't bring OTEL in as a hard dependency; this is purely a
# "if you have it, we use it" interop path so apps already
# instrumented via Arize Phoenix / OpenInference / Traceloop / etc.
# get framework-agnostic agent detection for free.


def _try_otel_identity() -> tuple[str, str] | None:
    """Read ``gen_ai.agent.id`` + ``gen_ai.agent.name`` from the
    currently-active OTEL span, if any.

    Returns ``(agent_id_attr, agent_name_attr)`` on hit. The first
    value becomes the identity key; the second is the display name.
    """
    try:
        from opentelemetry import trace  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    try:
        span = trace.get_current_span()
        if span is None or not span.is_recording():
            return None
        # OTEL API doesn't expose attributes on the public surface in
        # a guaranteed way across versions — fall through any errors.
        attrs = getattr(span, "attributes", None) or {}
        if not isinstance(attrs, dict):
            try:
                attrs = dict(attrs)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None
        agent_id = attrs.get("gen_ai.agent.id")
        agent_name = attrs.get("gen_ai.agent.name")
        if not isinstance(agent_id, str) or not agent_id.strip():
            # Name alone is still usable — pad an empty id so the
            # resolver has a deterministic key.
            if isinstance(agent_name, str) and agent_name.strip():
                agent_id = f"otel-name:{agent_name.strip()}"
            else:
                return None
        if not isinstance(agent_name, str) or not agent_name.strip():
            agent_name = agent_id
        return (agent_id.strip(), agent_name.strip())
    except Exception:  # noqa: BLE001
        return None


# ── Tier 3: Stack-frame variable inspection ─────────────────────────
#
# Walks up to ~12 frames looking for the user's per-loop agent
# identifier. Two recognised conventions:
#
# 1. ``__egisai_agent__`` — an opt-in marker the user sets explicitly
#    in their per-agent function. Always wins inside Tier 3 (it's
#    the only way to be 100% sure we found the right variable).
# 2. ``agent_name`` (or ``agent`` if it's a string) — a soft hint.
#    Cheap and useful when the user has a natural variable already
#    holding the role name. ``auto_stack_hints="strict"`` disables
#    the soft variant; ``"off"`` disables Tier 3 entirely.
#
# Frame walking is bounded so we never accidentally pick up an
# enclosing test runner's ``agent_name`` variable or worse.

_STACK_WALK_DEPTH_MAX = 12
_STACK_HINT_STRICT_VARS = ("__egisai_agent__",)
_STACK_HINT_LOOSE_VARS = ("__egisai_agent__", "agent_name", "egisai_agent")


def _try_stack_identity(mode: str = "loose") -> str | None:
    """Walk the call stack for an explicit per-agent identifier.

    ``mode`` matches the ``auto_stack_hints`` init kwarg:
    ``"strict"`` only respects ``__egisai_agent__``; ``"loose"``
    (default) also accepts ``agent_name`` / ``egisai_agent``;
    ``"off"`` disables entirely.
    """
    if mode == "off":
        return None
    targets = (
        _STACK_HINT_STRICT_VARS if mode == "strict" else _STACK_HINT_LOOSE_VARS
    )
    try:
        # ``sys._getframe(2)`` skips this function + its caller (a
        # patch wrapper). We then walk up to ~12 frames. CPython
        # documents ``_getframe`` as available; on alternative
        # interpreters that lack it we fall through with ``None``.
        frame: Any = sys._getframe(2)
    except (ValueError, AttributeError):
        return None
    depth = 0
    while frame is not None and depth < _STACK_WALK_DEPTH_MAX:
        locs = frame.f_locals
        for name in targets:
            v = locs.get(name)
            if isinstance(v, str) and v.strip():
                # 1–80 char hard cap so a buggy iteration variable
                # full of giant text can't become a display name.
                candidate = v.strip()[:80]
                # Skip values that are obviously not agent labels
                # (uuids, file paths, urls). Operators who want
                # those as labels can use the strict marker.
                if mode == "loose" and (
                    candidate.startswith(("http://", "https://", "/"))
                    or "/" in candidate
                ):
                    continue
                return candidate
        # Allow the `agent` (no `_name`) variable but only when it's
        # a string — otherwise an Agent SDK instance object would
        # accidentally match.
        if mode == "loose":
            agent_val = locs.get("agent")
            if isinstance(agent_val, str) and agent_val.strip():
                return agent_val.strip()[:80]
        frame = frame.f_back
        depth += 1
    return None


# ── Tier 4: Class-name introspection ───────────────────────────────
#
# Frameworks that expect users to subclass an Agent class still leak
# the class name onto the call stack via ``self``. We treat any class
# name ending in ``Agent`` / ``Bot`` / ``Worker`` / ``Specialist`` as
# a strong identity signal — the user almost certainly named that
# class as their agent's role.

_CLASS_SUFFIXES = ("Agent", "Bot", "Worker", "Specialist", "Assistant")


def _try_class_identity() -> str | None:
    """Inspect ``self`` on the call stack for an agent-shaped class."""
    try:
        frame: Any = sys._getframe(2)
    except (ValueError, AttributeError):
        return None
    depth = 0
    while frame is not None and depth < _STACK_WALK_DEPTH_MAX:
        locs = frame.f_locals
        self_obj = locs.get("self")
        if self_obj is not None:
            cls_name = type(self_obj).__name__
            if any(cls_name.endswith(suffix) for suffix in _CLASS_SUFFIXES):
                # Don't return obvious internals or test scaffolding.
                if cls_name not in (
                    "Agent",  # too generic — almost certainly the base class
                    "BaseAgent",
                    "MockAgent",
                    "TestAgent",
                    "AbstractAgent",
                ):
                    return _humanize_class_name(cls_name)
        frame = frame.f_back
        depth += 1
    return None


def _humanize_class_name(name: str) -> str:
    """Turn ``CustomerSupportBot`` → ``Customer Support Bot``."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).strip()


# ── Tier 5: System-prompt SHA-256 + NER name ───────────────────────
#
# The defensive last resort for raw chat-style calls. We hash the
# system prompt (NFKC-normalized so identical prompts in different
# encodings collapse to the same digest) and derive a display name
# either from spaCy NER (when warm) or a low-key
# ``agent-<hash[:8]>`` fallback.


def _normalize_text(text: str) -> str:
    """NFKC normalize and collapse whitespace. Mirrors PII engine."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


# ── Identity v2: model-invariant canonicalization ───────────────────
#
# An agent's identity is its *role* (persona + tools), never the model
# engine it happens to run on today. Two churn sources broke that in
# v1:
#
#   1. Framework patches hashed ``model`` into their identity bundles,
#      so switching Claude → GPT forked the agent.
#   2. Tool platforms (Cursor, n8n, …) embed the model name in the
#      system-prompt *text* ("powered by GPT-5"), so a model switch
#      changed the prompt bytes and forked the Tier-5 hash.
#
# ``canonicalize_identity_text`` fixes #2 by masking model-id tokens
# to a fixed ``<model>`` placeholder BEFORE hashing. #1 is fixed in
# the patches themselves (model is no longer part of any v2 bundle).
#
# The v2 hash recipe is versioned (``IDENTITY_ALGO_VERSION``) and the
# SDK ships the v1 hash alongside it (``identity_hash_legacy``) so the
# backend can migrate existing agents in place — no fork on upgrade,
# and the same continuity rail covers any future recipe change.

IDENTITY_ALGO_VERSION = "2"

_MODEL_MASK = "<model>"

# Families that are unambiguous model names even as bare words. A
# bare "claude" / "gpt" in prose is essentially always model chrome
# ("You are Claude, an AI assistant"), never operator-authored agent
# identity. ``o[1-9]`` is safe because it carries its own digit.
# Order matters inside the alternation: longer tokens first so
# "chatgpt" isn't half-eaten by "gpt" and "codellama" by "llama".
_MODEL_STRONG_FAMILIES = (
    "chatgpt|gpt|claude|gemini|gemma|codellama|llama|mixtral|ministral"
    "|mistral|codestral|deepseek|grok|qwen|kimi|davinci|dall-?e"
    "|command-r|o[1-9]"
)
# Families that are also common English words ("write a sonnet",
# "phi coefficient"). Masked ONLY when followed by a version-shaped
# tail containing a digit ("Sonnet 4.5", "opus-4", "nova 2") so we
# never mangle ordinary prose.
_MODEL_WEAK_FAMILIES = "sonnet|opus|haiku|palm|bard|command|titan|jamba|phi|nova|turbo"

# Provider-prefixed ids (Bedrock model ids, OpenRouter-style paths):
# "anthropic.claude-3-haiku-20240307-v1:0", "openai/gpt-4o",
# "us.meta.llama3-1-70b-instruct-v1:0".
_MODEL_PROVIDER_PREFIX = (
    r"(?:(?:us|eu|apac|global)\.)?"
    r"(?:anthropic|openai|google|meta|amazon|mistralai|cohere|ai21|xai|deepseek)"
    r"[./:][\w:-]*\w(?:\.[\w:-]*\w)*"
)

# Version-ish tail shared by strong families: digits directly attached
# ("qwen2.5", "llama3") and/or hyphen/dot-joined components
# ("gpt-4o-mini", "claude-3-5-sonnet-20241022"). Components can never
# end in a bare "." so a sentence period after "GPT-5." survives the
# mask exactly like the one after "Claude." does — otherwise the two
# prompts would canonicalize differently and defeat the whole point.
_MODEL_STRONG_TAIL = r"(?:\d\w*(?:\.\w+)*)?(?:[-.]\w+)*"
# Weak families need at least one digit-led component ("sonnet 4.5",
# "opus-4", "titan 2"), optionally followed by more components.
_MODEL_WEAK_TAIL = r"(?:[ .-]v?\d\w*(?:\.\w+)*)+(?:[-.]\w+)*"

_MODEL_TOKEN_RE = re.compile(
    r"(?<![\w.-])"
    r"(?:"
    + _MODEL_PROVIDER_PREFIX
    + r"|(?:" + _MODEL_STRONG_FAMILIES + r")" + _MODEL_STRONG_TAIL
    + r"|(?:" + _MODEL_WEAK_FAMILIES + r")" + _MODEL_WEAK_TAIL
    + r")"
    r"(?![\w-])",
    re.IGNORECASE,
)

# Adjacent masked tokens ("Claude Sonnet 4.5" → "<model> <model>")
# collapse to one so multi-word and single-word model mentions
# canonicalize identically.
_MODEL_MASK_RUN_RE = re.compile(
    r"<model>(?:[\s,/|:.-]*<model>)+"
)


def canonicalize_identity_text(text: str) -> str:
    """Return the model-invariant canonical form of identity text.

    NFKC-normalizes, collapses whitespace, and masks every model-id
    token to ``<model>``. Deterministic and pure — the SAME function
    runs in the SDK's Tier-5 resolver, the framework patches, and the
    backend gateway (which imports it from this module), so a prompt
    hashes identically no matter which door the traffic entered.

    Known, accepted edges (all deterministic):

    * Over-masking merges two prompts only when they differ *solely*
      in a model-ish token — which is exactly the "same agent,
      different engine" case we want to merge.
    * Bare weak-family words ("write a sonnet") are never masked; a
      weak family is only masked with a digit-led version tail.
    """
    normalized = _normalize_text(text)
    masked = _MODEL_TOKEN_RE.sub(_MODEL_MASK, normalized)
    masked = _MODEL_MASK_RUN_RE.sub(_MODEL_MASK, masked)
    return re.sub(r"\s+", " ", masked).strip()


# ── Identity v2: SimHash (prompt-evolution reconciliation) ──────────
#
# A 64-bit locality-sensitive fingerprint of the canonical prompt.
# Similar prompts land within a small Hamming distance; unrelated
# prompts land ~32 bits apart. The backend uses it — gated on same
# org + same identity_source + same tool bundle — to recognise "same
# agent, edited prompt" instead of forking a new agent when a prompt
# is intentionally revised (operator edit or platform-driven prompt
# optimization).
#
# Privacy: the simhash is a 16-hex digest of 3-token shingle votes.
# Like the SHA-256 identity hash it is non-reversible — no prompt
# text can be reconstructed from it — so shipping it preserves the
# "only digests cross the boundary" contract.

# 2-token shingles: an edited word only flips two shingles, so
# incremental prompt revisions stay within a small Hamming distance
# while unrelated prompts still land ~32 bits apart (binomial mean
# for independent 64-bit hashes). 3-token shingles were measurably
# too edit-sensitive: a 3-word tweak in a 30-word prompt already
# drifted past any safe threshold.
_SIMHASH_SHINGLE = 2
# Below this many word tokens a simhash is statistically useless —
# with so few shingles, two UNRELATED short prompts were measured at
# Hamming distance ~20, well inside any threshold that catches real
# edits. Short prompts therefore ship no simhash and never
# participate in prompt-evolution reconciliation (their exact hash
# stays the sole identity, which is cheap and safe).
_SIMHASH_MIN_TOKENS = 8


def simhash64_hex(text: str) -> str | None:
    """64-bit SimHash over 2-token shingles, as 16 lowercase hex chars.

    Returns ``None`` for texts shorter than ``_SIMHASH_MIN_TOKENS``
    word tokens — a hashable-but-meaningless value would only invite
    bad merges (see the constant's comment for the measurement).
    """
    tokens = re.findall(r"\w+", text.lower())
    if len(tokens) < _SIMHASH_MIN_TOKENS:
        return None
    shingles = {
        " ".join(tokens[i : i + _SIMHASH_SHINGLE])
        for i in range(len(tokens) - _SIMHASH_SHINGLE + 1)
    }
    counts = [0] * 64
    for sh in shingles:
        h = int.from_bytes(
            hashlib.blake2b(sh.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for bit in range(64):
            counts[bit] += 1 if (h >> bit) & 1 else -1
    value = 0
    for bit in range(64):
        if counts[bit] >= 0:
            value |= 1 << bit
    return f"{value:016x}"


def hamming64(a_hex: str, b_hex: str) -> int:
    """Hamming distance between two 16-hex simhashes (0–64)."""
    return bin(int(a_hex, 16) ^ int(b_hex, 16)).count("1")


def tool_bundle_hash_from_names(names: Iterable[Any]) -> str:
    """Order-insensitive SHA-256 of a tool-name set.

    The reconciliation corroborator: the backend only merges a
    changed-prompt identity into an existing agent when the tool
    bundles match exactly. Shared verbatim by SDK patches and the
    backend gateway so both sides compute byte-identical values
    (including the constant empty-set hash when there are no tools).
    """
    canon = sorted({_normalize_text(str(n)) for n in names if str(n).strip()})
    return hashlib.sha256("\x1f".join(canon).encode("utf-8")).hexdigest()


def _payload_tool_names(payload: Any) -> list[str]:
    """Extract declared tool names across provider payload shapes.

    Recognises OpenAI (``{"type": "function", "function": {"name"}}``),
    Anthropic (``{"name": …}``), and Bedrock ``toolConfig`` shapes.
    Metadata only — never reads schemas or arguments.
    """
    if not isinstance(payload, dict):
        return []
    names: list[str] = []
    tools = payload.get("tools")
    if isinstance(tools, list):
        for t in tools:
            if not isinstance(t, dict):
                continue
            fn = t.get("function")
            n = fn.get("name") if isinstance(fn, dict) else t.get("name")
            if isinstance(n, str) and n.strip():
                names.append(n.strip())
    tool_config = payload.get("toolConfig")
    if isinstance(tool_config, dict):
        for t in tool_config.get("tools") or []:
            if isinstance(t, dict):
                spec = t.get("toolSpec") or {}
                n = spec.get("name") if isinstance(spec, dict) else None
                if isinstance(n, str) and n.strip():
                    names.append(n.strip())
    return names


def _system_text(payload: Any, messages: Any) -> str:
    """Extract the system prompt across every framework's payload shape.

    Recognised inputs:

    * ``payload["system"]``           — Anthropic style (str or list[dict])
    * ``payload["system_instruction"]`` — Gemini style
    * ``payload["instructions"]``     — OpenAI Agents / Mastra style
    * ``payload["instruction"]``      — ADK style
    * ``payload["system_prompt"]``    — Claude Agent SDK style
    * Any ``messages`` entry with ``role="system"``

    Returns ``""`` when no system text is present.
    """
    if isinstance(payload, dict):
        for key in (
            "system",
            "system_instruction",
            "instructions",
            "instruction",
            "system_prompt",
        ):
            sys_v = payload.get(key)
            text = _coerce_text(sys_v)
            if text:
                return text

    if isinstance(messages, list):
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                text = _coerce_text(m.get("content"))
                if text:
                    return text
    return ""


def _coerce_text(value: Any) -> str:
    """Flatten str / list[str|dict] / dict-with-text → single string."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        chunks = []
        for p in value:
            if isinstance(p, str):
                chunks.append(p)
            elif isinstance(p, dict):
                text = p.get("text") or p.get("content") or ""
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(c for c in chunks if c).strip()
    if isinstance(value, dict):
        text = value.get("text") or value.get("content") or ""
        if isinstance(text, str):
            return text.strip()
    return ""


def _hash_bundle(parts: Iterable[Any]) -> str:
    """SHA-256 (hex) of a tuple of strings. Stable across calls.

    Inputs are NFKC-normalized + joined with a delimiter so reorderings
    of the bundle don't accidentally collide. Used by framework patches
    that want to fingerprint a composite agent definition.
    """
    pieces: list[str] = []
    for p in parts:
        if p is None:
            pieces.append("")
        elif isinstance(p, str):
            pieces.append(_normalize_text(p))
        elif isinstance(p, (list, tuple)):
            inner = sorted(_normalize_text(str(x)) for x in p)
            pieces.append("\x1f".join(inner))
        else:
            pieces.append(_normalize_text(str(p)))
    return hashlib.sha256("\x1e".join(pieces).encode("utf-8")).hexdigest()


# Legacy name patterns (kept for parity with prior Behavior when NER
# isn't warm). The new tiers prefer NER for novel prompts.
_NAME_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bYou are\b\s+(?:a\s+|an\s+)?specialist:\s*([^.\n,;:]+)"),
    re.compile(r"^\s*You are\s+([A-Z][^.\n,;:]+?),", re.MULTILINE),
    re.compile(
        r"^\s*You are\s+(?:a\s+|an\s+)?([^.\n,;:]+?)\s*(?:[.\n;:]|$)",
        re.MULTILINE,
    ),
    re.compile(r"^\s*#+\s*([^\n.]+)$", re.MULTILINE),
)

_FILLER_RE = re.compile(
    r"\b(an?|the|specialist:?|expert|professional)\b\s*",
    re.IGNORECASE,
)


def _normalize_name(raw: str) -> str:
    """Trim filler words, collapse whitespace, cap length to 60."""
    s = raw.strip().rstrip(".:;,")
    s = _FILLER_RE.sub("", s).strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > 60:
        s = s[:57].rstrip() + "…"
    return s


def _name_from_ner(text: str) -> str | None:
    """Try spaCy NER for a noun-phrase agent name (NER-first plan)."""
    try:
        from egisai.policy import _pii_loader

        analyzer = _pii_loader.try_get_analyzer()
        if analyzer is None:
            return None
        # We pull the spaCy doc out of Presidio's analyzer so we don't
        # have to spin up our own pipeline. The analyzer keeps the
        # nlp engine alive after warm-up.
        engine = getattr(analyzer, "nlp_engine", None)
        if engine is None:
            return None
        nlp = getattr(engine, "nlp", None)
        if isinstance(nlp, dict):
            # Multi-language Presidio engines store the model per-lang.
            nlp = nlp.get("en")
        if nlp is None:
            return None
        doc = nlp(text[:512])  # cap prompt length so NER stays fast
        # Prefer entities (PERSON / ORG / WORK_OF_ART / PRODUCT)
        for ent in getattr(doc, "ents", []):
            label = getattr(ent, "label_", "")
            if label in ("PERSON", "ORG", "WORK_OF_ART", "PRODUCT", "NORP"):
                candidate = _normalize_name(ent.text)
                if 2 <= len(candidate) <= 60:
                    return candidate
        # Fall through to noun phrases (less reliable but useful for
        # "Python Developer" / "Customer Support" style prompts).
        for chunk in getattr(doc, "noun_chunks", []):
            chunk_text = getattr(chunk, "text", "")
            candidate = _normalize_name(chunk_text)
            if 4 <= len(candidate) <= 60 and " " in candidate:
                lowered = candidate.lower()
                # Skip obvious self-reference / chrome words.
                if lowered.startswith(("you ", "your ", "the ", "a ", "an ")):
                    continue
                return candidate
    except Exception:  # noqa: BLE001
        return None
    return None


def _name_from_regex(text: str) -> str | None:
    """Legacy regex chain — used only when NER is cold."""
    for pat in _NAME_PATTERNS:
        m = pat.search(text)
        if m:
            candidate = _normalize_name(m.group(1))
            if candidate and 2 <= len(candidate) <= 60:
                return candidate
    return None


@dataclass(frozen=True)
class PromptIdentity:
    """Full v2 identity derivation for a system prompt.

    ``digest`` is the v2 canonical hash (model tokens masked);
    ``legacy_digest`` is the v1 hash (plain normalized text) shipped
    for backend continuity so pre-v2 agents don't fork on upgrade;
    ``simhash`` feeds prompt-evolution reconciliation server-side.
    """

    digest: str
    legacy_digest: str
    simhash: str | None
    display_name: str


def derive_prompt_identity(system_text: str) -> PromptIdentity:
    """Canonical v2 prompt-identity derivation (hash + name + simhash).

    The display name is derived from the *unmasked* normalized text
    (NER quality is unchanged from v1); only the hashes use the
    model-masked canonical form.
    """
    normalized = _normalize_text(system_text)
    canonical = canonicalize_identity_text(system_text)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    legacy_digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    name = _name_from_ner(normalized)
    if not name:
        name = _name_from_regex(normalized)
    if not name:
        name = f"agent-{digest[:8]}"
    return PromptIdentity(
        digest=digest,
        legacy_digest=legacy_digest,
        simhash=simhash64_hex(canonical),
        display_name=name,
    )


def _derive_identity_from_system(system_text: str) -> tuple[str, str]:
    """NER-first, hash-fallback display name + identity hash.

    Since Identity v2 the digest is computed over the model-masked
    canonical text, so a system prompt that embeds a model name
    ("powered by GPT-5") keeps the same identity when the model
    changes. Callers that need the legacy digest / simhash too use
    :func:`derive_prompt_identity`.
    """
    pid = derive_prompt_identity(system_text)
    return pid.digest, pid.display_name


# Back-compat shim for callers that still want the (hash, name) tuple.
def derive_identity(payload: Any, messages: Any) -> tuple[str, str] | None:
    """Legacy helper preserved for the (hash, name) shape.

    New call sites should use :func:`resolve_identity`, which does
    the full 7-tier walk + cache + ContextVar push. This shim exists
    for tests that pin the (digest, display_name) contract.
    """
    text = _system_text(payload, messages)
    if not text:
        return None
    return _derive_identity_from_system(text)


# ── Hot-path resolver ───────────────────────────────────────────────


def resolve_identity(
    payload: Any,
    *,
    auto_stack_hints: str = "loose",
) -> IdentityRecord | None:
    """Run the 7-tier ladder and return the resolved identity.

    Returns ``None`` only when **every** tier declined — extremely
    rare in practice; means the SDK has no init-time ``app=`` set
    *and* the payload has no system text *and* no framework patch
    found a match. Callers (the gate) then drop ``agent_id`` from
    the event and the backend attributes the row to the API key's
    bound agent if any.

    Tiers 0–2 push their resolved record onto the identity stack so
    inner nested calls inherit. Tiers 3–6 are per-call only.
    """
    # Tier 0: already pushed by an outer scope (egisai.agent() or
    # framework patch). The patch wrapper pushed first and our gate
    # is now inside its scope.
    pushed = current_identity()
    if pushed is not None:
        return pushed

    # Tier 0 cont'd: explicit set_context(agent="…")
    from egisai._context import get_context

    ctx = get_context()
    if ctx.agent_id and ctx.agent_name:
        return IdentityRecord(
            agent_id=ctx.agent_id,
            display_name=ctx.agent_name,
            identity_key=f"explicit:{ctx.agent_name}",
            identity_hash=_hash_bundle(("explicit", ctx.agent_name)),
            source="explicit",
            push_to_stack=False,
        )

    # Tier 0.5: OTEL
    otel = _try_otel_identity()
    if otel is not None:
        otel_id, otel_name = otel
        agent_id = _ensure_agent_id(
            display_name=otel_name,
            identity_key=f"otel:{otel_id}",
            identity_hash=_hash_bundle(("otel", otel_id)),
            source="otel",
        )
        if agent_id is not None:
            return IdentityRecord(
                agent_id=agent_id,
                display_name=otel_name,
                identity_key=f"otel:{otel_id}",
                identity_hash=_hash_bundle(("otel", otel_id)),
                source="otel",
            )

    # Tier 1: stored-prompt ids on the payload itself
    stored = _try_stored_prompt_identity(payload)
    if stored is not None:
        return stored

    # Tier 3: stack-frame hint (only respected when hints are on)
    if auto_stack_hints != "off":
        hint = _try_stack_identity(mode=auto_stack_hints)
        if hint is not None:
            agent_id = _ensure_agent_id(
                display_name=hint,
                identity_key=f"stack:{hint}",
                identity_hash=_hash_bundle(("stack", hint)),
                source="stack",
            )
            if agent_id is not None:
                return IdentityRecord(
                    agent_id=agent_id,
                    display_name=hint,
                    identity_key=f"stack:{hint}",
                    identity_hash=_hash_bundle(("stack", hint)),
                    source="stack",
                )

    # Tier 4: class-name introspection
    cls_hint = _try_class_identity()
    if cls_hint is not None:
        agent_id = _ensure_agent_id(
            display_name=cls_hint,
            identity_key=f"class:{cls_hint}",
            identity_hash=_hash_bundle(("class", cls_hint)),
            source="class",
        )
        if agent_id is not None:
            return IdentityRecord(
                agent_id=agent_id,
                display_name=cls_hint,
                identity_key=f"class:{cls_hint}",
                identity_hash=_hash_bundle(("class", cls_hint)),
                source="class",
            )

    # Tier 5: system-prompt SHA-256 (model-masked canonical) + NER
    messages = payload.get("messages") if isinstance(payload, dict) else None
    system = _system_text(payload, messages)
    if system:
        pid = derive_prompt_identity(system)
        model_v = payload.get("model") if isinstance(payload, dict) else None
        agent_id = _ensure_agent_id(
            display_name=pid.display_name,
            identity_key=f"hash:{pid.digest}",
            identity_hash=pid.digest,
            source="hash",
            system_excerpt=system,
            legacy_identity_hash=(
                pid.legacy_digest if pid.legacy_digest != pid.digest else None
            ),
            simhash=pid.simhash,
            tool_bundle_hash=tool_bundle_hash_from_names(
                _payload_tool_names(payload)
            ),
            model=model_v if isinstance(model_v, str) else None,
        )
        if agent_id is not None:
            return IdentityRecord(
                agent_id=agent_id,
                display_name=pid.display_name,
                identity_key=f"hash:{pid.digest}",
                identity_hash=pid.digest,
                source="hash",
            )

    # Tier 6: init-time app= fallback
    return _try_app_fallback()


def _try_stored_prompt_identity(payload: Any) -> IdentityRecord | None:
    """Pluck a server-issued stable id out of the payload, if any."""
    if not isinstance(payload, dict):
        return None
    # OpenAI Responses API — ``prompt`` can be a stored-prompt
    # reference ``{"id": "pmpt_…", "version": "…"}``.
    prompt_ref = payload.get("prompt")
    if isinstance(prompt_ref, dict):
        pid = prompt_ref.get("id")
        if isinstance(pid, str) and pid.startswith("pmpt_"):
            display = f"prompt:{pid[:16]}"
            agent_id = _ensure_agent_id(
                display_name=display,
                identity_key=f"stored_prompt:openai:{pid}",
                identity_hash=_hash_bundle(("stored_prompt", "openai", pid)),
                source="stored_prompt:openai",
            )
            if agent_id is not None:
                return IdentityRecord(
                    agent_id=agent_id,
                    display_name=display,
                    identity_key=f"stored_prompt:openai:{pid}",
                    identity_hash=_hash_bundle(("stored_prompt", "openai", pid)),
                    source="stored_prompt:openai",
                )
    # Gemini cached_content — string like ``cachedContents/abc-123``.
    cached = payload.get("cached_content") or payload.get("cachedContent")
    if isinstance(cached, str) and cached.strip():
        cid = cached.strip()
        display = f"cache:{cid.split('/')[-1][:16]}"
        agent_id = _ensure_agent_id(
            display_name=display,
            identity_key=f"stored_prompt:gemini:{cid}",
            identity_hash=_hash_bundle(("stored_prompt", "gemini", cid)),
            source="stored_prompt:gemini",
        )
        if agent_id is not None:
            return IdentityRecord(
                agent_id=agent_id,
                display_name=display,
                identity_key=f"stored_prompt:gemini:{cid}",
                identity_hash=_hash_bundle(("stored_prompt", "gemini", cid)),
                source="stored_prompt:gemini",
            )
    return None


def _try_app_fallback() -> IdentityRecord | None:
    """Final fallback: register the init-time ``app=`` as the agent."""
    try:
        from egisai._config import get_config_optional

        cfg = get_config_optional()
        if cfg is None:
            return None
        if cfg.agent_id and cfg.app:
            # API key already bound to an agent on the server side.
            return IdentityRecord(
                agent_id=cfg.agent_id,
                display_name=cfg.app,
                identity_key=f"app:{cfg.app}",
                identity_hash=_hash_bundle(("app", cfg.app)),
                source="app",
            )
        if not cfg.app:
            return None
        agent_id = _ensure_agent_id(
            display_name=cfg.app,
            identity_key=f"app:{cfg.app}",
            identity_hash=_hash_bundle(("app", cfg.app)),
            source="app",
        )
        if agent_id is None:
            return None
        return IdentityRecord(
            agent_id=agent_id,
            display_name=cfg.app,
            identity_key=f"app:{cfg.app}",
            identity_hash=_hash_bundle(("app", cfg.app)),
            source="app",
        )
    except Exception:  # noqa: BLE001
        return None


# ── Backend round-trip ──────────────────────────────────────────────


# Hard cap on the system-prompt excerpt shipped for descriptor
# generation. 2 KB is plenty for an LLM to infer the agent's role
# and keeps the ensure payload small. The backend re-caps
# defensively at 4 KB.
_SYSTEM_EXCERPT_MAX_CHARS = 2000


def _sanitized_excerpt(system_text: str | None) -> str | None:
    """PII-sanitize + truncate a system prompt for backend descriptor.

    Returns ``None`` — meaning "don't ship anything" — when:

    * ``auto_describe`` is disabled (operator opt-out),
    * there's no system text to summarise, or
    * sanitization fails for any reason (fail-open — registration
      must never break because we couldn't scrub a prompt).

    The returned string has been run through the SDK's PII engine so
    no validated PII (SSN, email, API key, …) leaves the process, and
    truncated to :data:`_SYSTEM_EXCERPT_MAX_CHARS`. The backend uses
    it transiently for a single LLM call and never persists it.
    """
    if not system_text:
        return None
    try:
        from egisai._config import get_config_optional

        cfg = get_config_optional()
        if cfg is None or not cfg.auto_describe:
            return None
    except Exception:  # noqa: BLE001
        return None
    try:
        from egisai.policy import pii

        normalized = _normalize_text(system_text)
        if not normalized:
            return None
        masked, _findings = pii.sanitize(normalized)
        excerpt = (masked or "").strip()[:_SYSTEM_EXCERPT_MAX_CHARS]
        return excerpt or None
    except Exception:  # noqa: BLE001
        return None


def _ensure_agent_id(
    *,
    display_name: str,
    identity_key: str,
    identity_hash: str,
    source: str,
    system_excerpt: str | None = None,
    legacy_identity_hash: str | None = None,
    simhash: str | None = None,
    tool_bundle_hash: str | None = None,
    model: str | None = None,
) -> str | None:
    """Get-or-fetch the backend agent_id for an identity.

    Caches by ``identity_key`` so repeated calls for the same agent
    are a dict lookup. The backend dedups by ``(org_id, identity_hash)``
    server-side so concurrent SDK processes converge on one row.

    ``system_excerpt`` (optional) is the agent's raw system prompt.
    When provided AND the agent is being seen for the first time this
    process (cache miss), it's PII-sanitised + truncated locally and
    shipped so the platform can generate a human description +
    business function in the background. Tiers without a system
    prompt (stack / class / app / OTEL / stored-id) pass ``None``.

    Identity v2 continuity fields (all optional; older backends
    ignore them):

    * ``legacy_identity_hash`` — the v1 hash for the same identity,
      so the backend can re-stamp an existing pre-v2 agent instead
      of forking a new one.
    * ``simhash`` / ``tool_bundle_hash`` — feed server-side
      prompt-evolution reconciliation (same agent, edited prompt).
    * ``model`` — observed metadata only (models_seen histogram);
      NEVER part of the identity hash.

    Returns ``None`` on any error — fail-open per
    ``sdk-design-philosophy.mdc`` rule 5: the user's call must not
    break because we can't reach the backend. A failure also arms a
    short backoff for this identity key (see ``_ensure_backoff``) so
    an outage costs one attempt per window instead of one per call.
    """
    cached = _identity_cache.get(identity_key)
    if cached:
        return cached
    if _in_ensure_backoff(identity_key):
        return None

    with _identity_lock:
        cached = _identity_cache.get(identity_key)
        if cached:
            return cached
        # Re-checked under the lock: while this thread waited, the
        # holder may have just failed and armed the backoff. Without
        # this, every thread queued on the lock pays its own timeout.
        if _in_ensure_backoff(identity_key):
            return None
        try:
            from egisai._backend import ensure_agent
            from egisai._config import get_config_optional
            from egisai._runtime import collect_runtime_fingerprint

            cfg = get_config_optional()
            if cfg is None:
                return None
            try:
                runtime = collect_runtime_fingerprint(sdk_version=cfg.sdk_version)
            except Exception:  # noqa: BLE001
                runtime = None
            payload = ensure_agent(
                name=display_name,
                description=(
                    f"Auto-detected by SDK ({source}) "
                    f"identity={identity_hash[:8]}"
                ),
                runtime=runtime,
                identity_hash=identity_hash,
                identity_source=source,
                system_prompt_excerpt=_sanitized_excerpt(system_excerpt),
                identity_version=IDENTITY_ALGO_VERSION,
                identity_hash_legacy=legacy_identity_hash,
                identity_simhash=simhash,
                tool_bundle_hash=tool_bundle_hash,
                model=model,
                init_app=cfg.app,
            )
            agent_id = payload.get("id")
            if isinstance(agent_id, str) and agent_id:
                _identity_cache[identity_key] = agent_id
                _ensure_backoff.pop(identity_key, None)
                if payload.get("created"):
                    LOGGER.info(
                        "[egisai] registered agent %r (id=%s…, source=%s)",
                        display_name, agent_id[:8], source,
                    )
                return agent_id
            # 2xx without a usable id — treat as unavailable rather
            # than retrying it on every subsequent call.
            _note_ensure_failure(identity_key)
        except Exception as exc:  # noqa: BLE001
            _note_ensure_failure(identity_key)
            LOGGER.warning(
                "[egisai] agent ensure failed (%s, source=%s): %s — "
                "proceeding unattributed, retrying in %.0fs",
                display_name, source, exc, _ensure_backoff_secs(),
            )
        return None


# ── Compat: keep _id_cache name for tests ───────────────────────────
# The old test conftest clears ``_id_cache``; we point that to the
# new unified cache so existing tests keep working without changes.
_id_cache = _identity_cache  # noqa: E305 — module-level alias


def resolve_agent_id(identity_hash: str, display_name: str) -> str | None:
    """Legacy helper used by ``_attribute_event`` pre-resolver.

    Now a thin shim onto the unified cache + ``ensure_agent``. Pinned
    so historical tests that import this symbol keep passing through
    the 0.17 transition. New code should use ``resolve_identity``.
    """
    return _ensure_agent_id(
        display_name=display_name,
        identity_key=f"hash:{identity_hash}",
        identity_hash=identity_hash,
        source="hash",
    )
