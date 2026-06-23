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
import re
import sys
import threading
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


def _derive_identity_from_system(system_text: str) -> tuple[str, str]:
    """NER-first, hash-fallback display name + identity hash."""
    normalized = _normalize_text(system_text)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    name = _name_from_ner(normalized)
    if not name:
        name = _name_from_regex(normalized)
    if not name:
        name = f"agent-{digest[:8]}"
    return digest, name


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

    # Tier 5: system-prompt SHA-256 + NER
    messages = payload.get("messages") if isinstance(payload, dict) else None
    system = _system_text(payload, messages)
    if system:
        digest, name = _derive_identity_from_system(system)
        agent_id = _ensure_agent_id(
            display_name=name,
            identity_key=f"hash:{digest}",
            identity_hash=digest,
            source="hash",
            system_excerpt=system,
        )
        if agent_id is not None:
            return IdentityRecord(
                agent_id=agent_id,
                display_name=name,
                identity_key=f"hash:{digest}",
                identity_hash=digest,
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

    Returns ``None`` on any error — fail-open per
    ``sdk-design-philosophy.mdc`` rule 5: the user's call must not
    break because we can't reach the backend.
    """
    cached = _identity_cache.get(identity_key)
    if cached:
        return cached

    with _identity_lock:
        cached = _identity_cache.get(identity_key)
        if cached:
            return cached
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
            )
            agent_id = payload.get("id")
            if isinstance(agent_id, str) and agent_id:
                _identity_cache[identity_key] = agent_id
                if payload.get("created"):
                    LOGGER.info(
                        "[egisai] registered agent %r (id=%s…, source=%s)",
                        display_name, agent_id[:8], source,
                    )
                return agent_id
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "[egisai] agent ensure failed (%s, source=%s): %s",
                display_name, source, exc,
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
