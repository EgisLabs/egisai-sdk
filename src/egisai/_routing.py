"""Smart Model Routing — SDK-side decision client + cross-provider transport.

The platform decides, the SDK applies. For each governed call the gate
asks ``POST /v1/sdk/route`` (through an aggressive in-process cache)
whether a better-fit model should serve the request — cheaper when the
request is simple, smarter when it exceeds the requested model's tier,
or no change at all. The decision engine (catalog / classifier /
selector) lives server-side so it can be tuned without an SDK release.

Application has two shapes:

* **Same-provider swap** — the patch rewrites ``kwargs["model"]``
  before forwarding; the original client library executes the call
  exactly as it would have (same auth, same wire format).
* **Cross-provider swap** — only when the process holds an API key for
  the target provider (env detection) AND the call is simple enough to
  translate faithfully (non-streaming, no tools). The SDK then executes
  the call directly against the target provider's REST API and
  translates the response back into the source framework's shape.

Everything fails open: no key, no network, unknown model, translation
too risky → the call proceeds on the model the caller asked for.

Privacy: the decision request carries the post-sanitization,
label-redacted preview the audit trail already uses — raw prompt text
never rides on ``/v1/sdk/route`` (``security-and-compliance.mdc`` §1).
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger("egisai.routing")

#: Decision cache TTL. Agent loops re-send near-identical prompts; one
#: decision per distinct (model, preview) pair per window keeps the
#: /route call volume negligible.
_DECISION_TTL_S = 300.0
_DECISION_CACHE_MAX = 2048

#: How long an affirmative "routing is disabled" answer is honoured
#: before asking again. The ``routing.changed`` SSE event clears it
#: sooner in practice.
_DISABLED_TTL_S = 300.0

#: Per-call budget for the /route round-trip. Fail-open past this.
_ROUTE_TIMEOUT_S = 3.0

_lock = threading.Lock()
_decision_cache: dict[str, tuple[float, dict[str, Any] | None]] = {}
_disabled_until: float = 0.0
#: ``None`` → unknown (ask the backend); True/False → handshake answer.
_enabled_hint: bool | None = None


def set_enabled_hint(value: bool | None) -> None:
    """Record the handshake's ``smart_model_routing`` feature flag.

    ``False`` keeps the module fully dormant (zero /route calls).
    ``None`` (set by the refresher on a ``routing.changed`` event)
    means "unknown — ask the backend", which re-learns the state
    without a process restart.
    """
    global _enabled_hint
    with _lock:
        _enabled_hint = value


def invalidate() -> None:
    """Drop every cached decision + disabled flag (SSE ``routing.changed``)."""
    global _disabled_until, _enabled_hint
    with _lock:
        _decision_cache.clear()
        _disabled_until = 0.0
        # Re-learn enablement from the backend on the next call —
        # the operator may have just flipped the master switch on.
        if _enabled_hint is False:
            _enabled_hint = None


def reset() -> None:
    """Test / shutdown hook — restore pristine module state."""
    global _disabled_until, _enabled_hint
    with _lock:
        _decision_cache.clear()
        _disabled_until = 0.0
        _enabled_hint = None


# ── Provider detection ───────────────────────────────────────────────

#: Canonical provider ids, mirroring the backend's ``resolve_provider_id``.
_PROVIDER_PREFIXES: tuple[tuple[str, str], ...] = (
    ("claude", "anthropic"),
    ("anthropic", "anthropic"),
    ("gemini", "google"),
    ("models/gemini", "google"),
)

_ENV_KEYS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
}


def provider_for_model(model: str | None) -> str:
    """Best-effort canonical provider id for a wire model name."""
    m = (model or "").strip().lower()
    for prefix, provider in _PROVIDER_PREFIXES:
        if m.startswith(prefix):
            return provider
    return "openai"


def _env_key_for(provider: str) -> str | None:
    for name in _ENV_KEYS.get(provider, ()):
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def detect_available_providers(
    requested_provider: str, *, allow_cross: bool
) -> list[str]:
    """Providers this process can authenticate against.

    The requested model's provider is always reachable (the caller's
    own client is configured for it). Cross-provider candidates are
    added only when the call shape supports translation AND the env
    holds a key for them.
    """
    providers = [requested_provider]
    if allow_cross:
        for provider in ("openai", "anthropic", "google"):
            if provider != requested_provider and _env_key_for(provider):
                providers.append(provider)
    return providers


# ── Decision client ──────────────────────────────────────────────────


def _cache_key(model: str, preview: str, has_tools: bool, providers: list[str]) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update(preview.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(("1" if has_tools else "0").encode())
    h.update(b"\x00")
    h.update(",".join(sorted(providers)).encode())
    return h.hexdigest()


def maybe_route(
    *,
    model: str,
    prompt_preview: str,
    prompt_chars: int,
    has_tools: bool,
    agent_id: str | None,
    allow_cross: bool,
) -> dict[str, Any] | None:
    """Ask the platform for a best-fit model. Never raises.

    Returns the decision dict (``{"model", "provider", "direction",
    "reason", "projected_savings_usd"}``) or ``None`` ("keep the
    requested model"). ``prompt_preview`` MUST already be the
    label-redacted audit preview.
    """
    global _disabled_until
    try:
        with _lock:
            if _enabled_hint is False:
                return None
            if time.monotonic() < _disabled_until:
                return None

        requested_provider = provider_for_model(model)
        providers = detect_available_providers(
            requested_provider, allow_cross=allow_cross
        )
        preview = (prompt_preview or "")[:2000]
        key = _cache_key(model, preview, has_tools, providers)
        now = time.monotonic()
        with _lock:
            hit = _decision_cache.get(key)
            if hit is not None:
                expires_at, cached = hit
                if now < expires_at:
                    return dict(cached) if cached is not None else None
                _decision_cache.pop(key, None)

        from egisai._backend import route as backend_route

        body = backend_route(
            model=model,
            prompt_preview=preview,
            prompt_chars=prompt_chars,
            has_tools=has_tools,
            available_providers=providers,
            agent_id=agent_id,
            timeout_s=_ROUTE_TIMEOUT_S,
        )
        if body is None:
            return None

        if body.get("disabled"):
            with _lock:
                _disabled_until = time.monotonic() + _DISABLED_TTL_S
            return None

        decision: dict[str, Any] | None = None
        if body.get("routed") and body.get("model"):
            decision = {
                "model": str(body["model"]),
                "provider": str(body.get("provider") or ""),
                "direction": str(body.get("direction") or ""),
                "reason": str(body.get("reason") or ""),
                "projected_savings_usd": float(
                    body.get("projected_savings_usd") or 0.0
                ),
                "requested_provider": requested_provider,
            }

        with _lock:
            if len(_decision_cache) >= _DECISION_CACHE_MAX:
                _decision_cache.clear()
            _decision_cache[key] = (
                time.monotonic() + _DECISION_TTL_S,
                dict(decision) if decision is not None else None,
            )
        return decision
    except Exception:  # noqa: BLE001
        LOGGER.debug("route decision failed — keeping requested model", exc_info=True)
        return None


# ── Routing adapter (patch → gate contract) ─────────────────────────


@dataclass(frozen=True)
class RoutingAdapter:
    """Hooks a framework patch hands the gate so a decision can be applied.

    ``apply_same_provider(new_model)`` rewrites the pending call's
    ``model`` in place (the patch closes over its own ``kwargs``) and
    returns ``True`` on success.

    ``build_cross_forward(decision)`` returns a zero-arg callable that
    executes the call on the routed provider and returns a response in
    the SOURCE framework's native shape — or ``None`` when the call
    can't be translated faithfully (the gate then keeps the requested
    model). Patches that don't support cross-provider leave it unset.
    """

    apply_same_provider: Callable[[str], bool]
    build_cross_forward: (
        Callable[[dict[str, Any]], Callable[[], Any] | None] | None
    ) = None

    @property
    def supports_cross(self) -> bool:
        return self.build_cross_forward is not None


# ── Cross-provider transport ─────────────────────────────────────────
#
# Canonical form: OpenAI-style ``[{"role", "content": str}]`` messages.
# Only calls whose every message content is a plain string are
# translated — anything richer (image parts, tool results, documents)
# stays on the requested model. Canonical result:
# ``{"text", "tokens_in", "tokens_out", "model"}``.

_ANTHROPIC_VERSION = "2023-06-01"
_CROSS_TIMEOUT_S = 120.0
_DEFAULT_MAX_TOKENS = 4096


def canonicalize_openai_messages(
    messages: Any,
) -> list[dict[str, str]] | None:
    """Validate/normalise OpenAI-shaped messages into the canonical form.

    Returns ``None`` when any message can't be represented as a plain
    ``{role, content: str}`` pair — the caller must then skip
    cross-provider routing for this call.
    """
    if not isinstance(messages, list) or not messages:
        return None
    out: list[dict[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("system", "developer", "user", "assistant"):
            return None
        if not isinstance(content, str):
            return None
        out.append({"role": "system" if role == "developer" else role, "content": content})
    if not any(m["role"] == "user" for m in out):
        return None
    return out


def canonicalize_anthropic_messages(
    messages: Any, system: Any
) -> list[dict[str, str]] | None:
    """Anthropic ``messages`` + ``system`` → canonical OpenAI-style list."""
    out: list[dict[str, str]] = []
    if isinstance(system, str) and system:
        out.append({"role": "system", "content": system})
    elif system not in (None, ""):
        # Structured system blocks (list of text blocks) — flatten
        # plain-text blocks; bail on anything richer.
        if not isinstance(system, list):
            return None
        parts: list[str] = []
        for block in system:
            text = block.get("text") if isinstance(block, dict) else None
            if not isinstance(text, str):
                return None
            parts.append(text)
        if parts:
            out.append({"role": "system", "content": "\n".join(parts)})
    if not isinstance(messages, list) or not messages:
        return None
    for msg in messages:
        if not isinstance(msg, dict):
            return None
        role = msg.get("role")
        content = msg.get("content")
        if role not in ("user", "assistant"):
            return None
        if isinstance(content, str):
            out.append({"role": role, "content": content})
        elif isinstance(content, list):
            parts = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    return None
                text = block.get("text")
                if not isinstance(text, str):
                    return None
                parts.append(text)
            out.append({"role": role, "content": "\n".join(parts)})
        else:
            return None
    if not any(m["role"] == "user" for m in out):
        return None
    return out


def _http_post(url: str, *, headers: dict[str, str], json_body: dict[str, Any]) -> Any:
    import httpx

    with httpx.Client(timeout=_CROSS_TIMEOUT_S) as client:
        r = client.post(url, headers=headers, json=json_body)
        r.raise_for_status()
        return r.json()


def _cross_call_openai(
    model: str, messages: list[dict[str, str]], params: dict[str, Any]
) -> dict[str, Any]:
    key = _env_key_for("openai")
    if not key:
        raise RuntimeError("no OpenAI key in env")
    body: dict[str, Any] = {"model": model, "messages": messages}
    if params.get("temperature") is not None:
        body["temperature"] = params["temperature"]
    if params.get("max_tokens") is not None:
        body["max_tokens"] = params["max_tokens"]
    data = _http_post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {key}"},
        json_body=body,
    )
    choice = (data.get("choices") or [{}])[0]
    usage = data.get("usage") or {}
    return {
        "text": ((choice.get("message") or {}).get("content")) or "",
        "tokens_in": usage.get("prompt_tokens"),
        "tokens_out": usage.get("completion_tokens"),
        "model": data.get("model") or model,
    }


def _cross_call_anthropic(
    model: str, messages: list[dict[str, str]], params: dict[str, Any]
) -> dict[str, Any]:
    key = _env_key_for("anthropic")
    if not key:
        raise RuntimeError("no Anthropic key in env")
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    convo = [m for m in messages if m["role"] in ("user", "assistant")]
    body: dict[str, Any] = {
        "model": model,
        "max_tokens": int(params.get("max_tokens") or _DEFAULT_MAX_TOKENS),
        "messages": convo,
    }
    if system_parts:
        body["system"] = "\n".join(system_parts)
    if params.get("temperature") is not None:
        body["temperature"] = params["temperature"]
    data = _http_post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": key, "anthropic-version": _ANTHROPIC_VERSION},
        json_body=body,
    )
    text = "".join(
        block.get("text", "")
        for block in (data.get("content") or [])
        if isinstance(block, dict) and block.get("type") == "text"
    )
    usage = data.get("usage") or {}
    return {
        "text": text,
        "tokens_in": usage.get("input_tokens"),
        "tokens_out": usage.get("output_tokens"),
        "model": data.get("model") or model,
    }


def _cross_call_google(
    model: str, messages: list[dict[str, str]], params: dict[str, Any]
) -> dict[str, Any]:
    key = _env_key_for("google")
    if not key:
        raise RuntimeError("no Google key in env")
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    contents = [
        {
            "role": "user" if m["role"] == "user" else "model",
            "parts": [{"text": m["content"]}],
        }
        for m in messages
        if m["role"] in ("user", "assistant")
    ]
    body: dict[str, Any] = {"contents": contents}
    if system_parts:
        body["systemInstruction"] = {"parts": [{"text": "\n".join(system_parts)}]}
    generation_config: dict[str, Any] = {}
    if params.get("temperature") is not None:
        generation_config["temperature"] = params["temperature"]
    if params.get("max_tokens") is not None:
        generation_config["maxOutputTokens"] = params["max_tokens"]
    if generation_config:
        body["generationConfig"] = generation_config
    data = _http_post(
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent",
        headers={"x-goog-api-key": key},
        json_body=body,
    )
    candidates = data.get("candidates") or []
    text = ""
    if candidates:
        parts = ((candidates[0].get("content") or {}).get("parts")) or []
        text = "".join(
            p.get("text", "") for p in parts if isinstance(p, dict)
        )
    usage = data.get("usageMetadata") or {}
    return {
        "text": text,
        "tokens_in": usage.get("promptTokenCount"),
        "tokens_out": usage.get("candidatesTokenCount"),
        "model": model,
    }


_CROSS_EXECUTORS: dict[
    str, Callable[[str, list[dict[str, str]], dict[str, Any]], dict[str, Any]]
] = {
    "openai": _cross_call_openai,
    "anthropic": _cross_call_anthropic,
    "google": _cross_call_google,
}


def execute_cross_call(
    *,
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one canonical chat call against ``provider``. Raises on failure.

    The caller (a patch's ``build_cross_forward`` closure) is expected
    to run inside the gate's ``forward()`` slot, where an exception is
    already handled like any provider error.
    """
    executor = _CROSS_EXECUTORS.get(provider)
    if executor is None:
        raise RuntimeError(f"unknown provider {provider!r}")
    return executor(model, messages, params or {})
