"""Escalation client for the platform's prompt-injection smart tier.

The SDK ships a fast, standard, local pre-filter in
:mod:`egisai.policy.injection` — chat-template delimiters, invisible
Unicode, base64 runs, and the well-known override / exfiltration
shapes. That tier runs on every call, sub-millisecond, offline, and
leaks nothing. It is deliberately the *commodity* half of the design:
everything in it is public knowledge an operator could write
themselves.

The proprietary half — a fine-tuned classifier plus EgisAI's
calibrated pattern/scoring extensions — lives on the platform and
never ships in a public PyPI artefact. This module is the thin HTTP
client that reaches it, mirroring
:class:`egisai.policy.semantic.SemanticBlocker` almost exactly:

* Escalation is an **escalation**, not the default. The engine only
  calls it when the local pre-filter did not already block, so an
  obvious attack is refused instantly with no network.
* By the time this client runs, Phase 1 of the engine has already
  masked PII (same slot the LLM judge runs in), so the text posted
  here is data-clean — see security-and-compliance.mdc rule 1.
* **Fail open.** A judge/classifier outage must never break the
  customer's call path. On any error the client returns ``None`` so
  the rule degrades to the local pre-filter's verdict, unless the
  operator opted into fail-closed mode.
* **Cheap in steady state.** A SHA-256-keyed verdict cache collapses
  the repeated system-prompt turns and byte-identical retries that
  dominate agentic traffic, exactly like the semantic cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

import httpx

LOGGER = logging.getLogger("egisai.policy.injection_client")


def _env_float(name: str, default: float, *, lo: float = 0.0) -> float:
    """Parse a float env var with a clamped fallback (see semantic.py)."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        parsed = float(raw)
    except ValueError:
        LOGGER.warning(
            "%s=%r is not a valid float — falling back to default %.1f",
            name, raw, default,
        )
        return default
    return max(lo, parsed)


# ── HTTP + cache knobs ──────────────────────────────────────────────
#
# The injection smart tier is a classifier + regex pass on the
# platform — much cheaper than the LLM judge, so its default timeout
# is tighter. The cache semantics are identical to the semantic
# cache: the classifier is deterministic over (text, classes,
# threshold), so an identical question has an identical correct
# answer and a memoized reply can never change a verdict. Editing the
# policy changes ``classes``/``threshold`` and therefore the key, so
# a stale rule can't keep firing.
_DEFAULT_TIMEOUT_SECS = 4.0
_DEFAULT_RETRY_AFTER_MAX_SECS = 5.0
_DEFAULT_CACHE_TTL_SECS = 60.0
_CACHE_MAX = 512


@dataclass(frozen=True)
class InjectionMatch:
    """A prompt-injection finding reported by the platform smart tier.

    ``score`` is the classifier's confidence in ``[0.0, 1.0]``.
    ``cls`` is the strongest matching class id (see
    :data:`egisai.policy.injection.CLASSES`). A ``score`` of ``0.0``
    paired with ``cls == "<judge unavailable>"`` indicates the match
    was synthesized on outage under fail-closed mode.
    """

    cls: str
    score: float


# Sentinel returned on outage when the operator opted into fail-closed.
_OUTAGE_MATCH = InjectionMatch(cls="<judge unavailable>", score=0.0)


class InjectionBlocker:
    """Client for the platform's ``/v1/sdk/injection`` smart tier.

    Constructed once per process by ``egisai.init()``. Holds both a
    sync and async HTTP client; use ``check()`` from sync patchers and
    ``acheck()`` from async ones so the event loop is never blocked.
    """

    _RETRY_429_MAX = 3
    _RETRY_429_FALLBACK_S = 1.0

    def __init__(
        self,
        platform_api_key: str,
        platform_base_url: str,
        on_outage: str = "allow",
        timeout_secs: float | None = None,
        retry_after_max_secs: float | None = None,
        cache_ttl_secs: float | None = None,
    ) -> None:
        if on_outage not in ("allow", "block"):
            raise ValueError(
                f"on_outage must be 'allow' or 'block', got {on_outage!r}"
            )
        self._api_key = platform_api_key
        self._base_url = platform_base_url.rstrip("/")
        self._on_outage = on_outage
        if timeout_secs is None:
            timeout_secs = _env_float(
                "EGISAI_INJECTION_TIMEOUT_SECS", _DEFAULT_TIMEOUT_SECS, lo=0.5
            )
        if retry_after_max_secs is None:
            retry_after_max_secs = _env_float(
                "EGISAI_INJECTION_RETRY_AFTER_MAX_SECS",
                _DEFAULT_RETRY_AFTER_MAX_SECS,
                lo=0.1,
            )
        if cache_ttl_secs is None:
            cache_ttl_secs = _env_float(
                "EGISAI_INJECTION_CACHE_TTL_SECS",
                _DEFAULT_CACHE_TTL_SECS,
                lo=0.0,
            )
        self._timeout_secs = float(timeout_secs)
        self._retry_after_max_secs = float(retry_after_max_secs)
        self._cache_ttl_secs = float(cache_ttl_secs)
        self._http_client = httpx.Client(timeout=self._timeout_secs)
        self._async_http_client: httpx.AsyncClient | None = None
        self._cache: dict[str, tuple[float, InjectionMatch | None]] = {}
        self._cache_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────

    def check(
        self, text: str, config: dict[str, Any]
    ) -> InjectionMatch | None:
        """Synchronous check; returns an ``InjectionMatch`` or ``None``."""
        body = self._prepare(text, config)
        if body is None:
            return None
        key = self._cache_key(body)
        hit, found = self._cache_get(key)
        if found:
            return hit
        try:
            response = self._post_with_429_retry(body)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            return self._on_outage_response(exc)
        match = self._interpret(data)
        self._cache_put(key, match)
        return match

    async def acheck(
        self, text: str, config: dict[str, Any]
    ) -> InjectionMatch | None:
        """Async sibling of ``check`` — never blocks the event loop."""
        body = self._prepare(text, config)
        if body is None:
            return None
        key = self._cache_key(body)
        hit, found = self._cache_get(key)
        if found:
            return hit
        try:
            response = await self._apost_with_429_retry(body)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            return self._on_outage_response(exc)
        match = self._interpret(data)
        self._cache_put(key, match)
        return match

    def close(self) -> None:
        """Close both HTTP clients. Idempotent."""
        try:
            self._http_client.close()
        except Exception:  # noqa: BLE001
            pass
        if self._async_http_client is not None:
            try:
                client = self._async_http_client
                self._async_http_client = None
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(client.aclose())
                    else:
                        loop.run_until_complete(client.aclose())
                except RuntimeError:
                    asyncio.run(client.aclose())
            except Exception:  # noqa: BLE001
                pass

    # ── Verdict cache ─────────────────────────────────────────────────

    def _cache_key(self, body: dict[str, Any]) -> str:
        keyed = body
        try:
            from egisai.policy import fastpath

            if fastpath.cache_normalization_enabled():
                keyed = {
                    **body,
                    "prompt_text": fastpath.normalize_for_cache_key(
                        str(body.get("prompt_text") or "")
                    ),
                }
        except Exception:  # noqa: BLE001
            LOGGER.debug("cache-key normalization failed", exc_info=True)
        return hashlib.sha256(
            json.dumps(keyed, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

    def _cache_get(self, key: str) -> tuple[InjectionMatch | None, bool]:
        if self._cache_ttl_secs <= 0:
            return None, False
        now = time.monotonic()
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is None:
                return None, False
            expires_at, verdict = entry
            if expires_at <= now:
                self._cache.pop(key, None)
                return None, False
            return verdict, True

    def _cache_put(self, key: str, verdict: InjectionMatch | None) -> None:
        if self._cache_ttl_secs <= 0:
            return
        with self._cache_lock:
            if len(self._cache) >= _CACHE_MAX:
                self._cache.clear()
            self._cache[key] = (
                time.monotonic() + self._cache_ttl_secs,
                verdict,
            )

    # ── Internals ─────────────────────────────────────────────────────

    def _prepare(
        self, text: str, config: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build the request body, or ``None`` to short-circuit.

        Returns ``None`` when there's nothing to judge (empty text) or
        when the operator explicitly disabled escalation for this rule
        via ``escalate: false`` — some operators want a purely local,
        offline-only injection rule and that must be honoured.
        """
        if not text:
            return None
        if config.get("escalate") is False:
            return None
        body: dict[str, Any] = {"prompt_text": text}
        raw_classes = config.get("classes")
        if isinstance(raw_classes, (list, tuple)):
            classes = [str(c).strip() for c in raw_classes if str(c).strip()]
            if classes:
                body["classes"] = classes
        if config.get("threshold") is not None:
            body["threshold"] = config["threshold"]
        return body

    def _interpret(self, data: dict[str, Any]) -> InjectionMatch | None:
        if not isinstance(data, dict) or not data.get("match"):
            return None
        try:
            score = float(data.get("score") or 1.0)
        except (TypeError, ValueError):
            score = 1.0
        return InjectionMatch(
            cls=str(data.get("cls") or "prompt injection"),
            score=max(0.0, min(1.0, score)),
        )

    def _on_outage_response(self, exc: BaseException) -> InjectionMatch | None:
        detail = exc.__class__.__name__
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            detail = f"HTTP {status}"
            if 400 <= status < 500:
                LOGGER.error(
                    "injection smart tier: request REJECTED by the platform "
                    "(%s) — this is a request problem, not an outage; the "
                    "smart tier is silently not scanning. Check SDK/backend "
                    "version skew.",
                    detail,
                )
        if self._on_outage == "block":
            LOGGER.warning(
                "injection smart tier: call failed (%s) — failing CLOSED "
                "(on_outage='block')",
                detail,
            )
            return _OUTAGE_MATCH
        LOGGER.warning(
            "injection smart tier: call failed (%s) — failing open "
            "(local pre-filter verdict stands)",
            detail,
        )
        return None

    def _ensure_async_client(self) -> httpx.AsyncClient:
        if self._async_http_client is None:
            self._async_http_client = httpx.AsyncClient(
                timeout=self._timeout_secs
            )
        return self._async_http_client

    def _post_with_429_retry(self, body: dict[str, Any]) -> httpx.Response:
        last: httpx.Response | None = None
        for attempt in range(self._RETRY_429_MAX + 1):
            last = self._http_client.post(
                f"{self._base_url}/v1/sdk/injection",
                json=body,
                headers=self._auth_headers(),
            )
            if last.status_code != 429:
                return last
            if attempt >= self._RETRY_429_MAX:
                return last
            time.sleep(self._retry_after_seconds(last))
        return last  # type: ignore[return-value]

    async def _apost_with_429_retry(
        self, body: dict[str, Any]
    ) -> httpx.Response:
        client = self._ensure_async_client()
        last: httpx.Response | None = None
        for attempt in range(self._RETRY_429_MAX + 1):
            last = await client.post(
                f"{self._base_url}/v1/sdk/injection",
                json=body,
                headers=self._auth_headers(),
            )
            if last.status_code != 429:
                return last
            if attempt >= self._RETRY_429_MAX:
                return last
            await asyncio.sleep(self._retry_after_seconds(last))
        return last  # type: ignore[return-value]

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _retry_after_seconds(self, response: httpx.Response) -> float:
        retry_after_raw = response.headers.get("Retry-After")
        if not retry_after_raw:
            return self._RETRY_429_FALLBACK_S
        try:
            parsed = float(retry_after_raw)
        except ValueError:
            return self._RETRY_429_FALLBACK_S
        return max(0.1, min(parsed, self._retry_after_max_secs))
