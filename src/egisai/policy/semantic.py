"""Semantic intent enforcement for ``semantic_guard`` policies.

The SDK doesn't host the judge model. When a ``semantic_guard``
rule fires, the SDK calls the EgisAI platform with the already-
redacted prompt (Phase 1 of the engine has run by then, so PII is
masked) and a list of operator-authored intent strings, and the
platform returns a verdict.

Outage behavior is operator-configurable via
``init(semantic_on_outage=...)``:

- ``"allow"`` (default) — return ``None`` so the rule becomes a
  no-op for that one call. This preserves availability of the
  primary call path.
- ``"block"`` — return a synthetic ``SemanticMatch`` so the engine
  produces a ``block`` verdict. Use when the operator considers
  Phase 2 the primary defense for that workload.

Async-aware: ``acheck()`` is the non-blocking sibling of ``check()``
and is invoked from async patchers (e.g. ``AsyncOpenAI``). The
synchronous ``check()`` retains identical semantics for
``OpenAI`` / ``Anthropic`` / ``GenAI`` / ``httpx.Client`` paths.
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

LOGGER = logging.getLogger("egisai.policy.semantic")


def _env_float(name: str, default: float, *, lo: float = 0.0) -> float:
    """Parse a float env var with a sane fallback, clamped at ``lo``.

    Returns ``default`` when the env var is missing, blank, or
    unparseable. Operators tuning these knobs in production
    typically set them in seconds; we clamp at ``lo`` so a stray
    "0" doesn't make every judge call instantly time out (which
    would silently turn ``semantic_guard`` into a no-op under the
    default ``on_outage="allow"`` mode — exactly the kind of
    governance regression compliance auditors hate).
    """
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


# ── SDK-side judge HTTP knobs ────────────────────────────────────────
#
# The judge endpoint is the only network call inside the policy
# evaluation hot path. Its tail latency directly inflates the
# dashboard's ``policy_latency_ms`` column, so the SDK gives
# operators two knobs they can tighten without re-installing:
#
#   * ``EGISAI_JUDGE_TIMEOUT_SECS`` — maximum wall-clock the SDK
#     waits on a single round-trip to ``/v1/sdk/judge``. The
#     backend's own OpenAI-judge timeout is 15 s, so anything
#     much higher than that just lets a slow backend silently
#     widen the SDK's stall window. The default lands at 8.0 s
#     to give operators a healthier worst-case while still
#     leaving comfortable headroom for normal P50 traffic
#     (~0.5–2 s round-trip).
#
#   * ``EGISAI_JUDGE_RETRY_AFTER_MAX_SECS`` — clamp on the
#     ``Retry-After`` header value the SDK honors on HTTP 429.
#     A misconfigured upstream proxy can return ``Retry-After:
#     90`` and freeze the call for 90 s per attempt × 3 attempts.
#     The clamp guarantees a single retry costs at most this
#     many seconds even when the upstream tells us to wait
#     longer. Default 5.0 s is plenty of breathing room for a
#     real bursty rate limit while keeping a misbehaving header
#     from holding governance hostage.
#
#   * ``EGISAI_JUDGE_CACHE_TTL_SECS`` — how long an identical
#     judge question may be answered from memory instead of the
#     network. See the verdict-cache notes below. ``0`` disables
#     caching entirely.
#
# Both are read every time a ``SemanticBlocker`` is constructed
# (i.e. once per ``egisai.init()``), so changing the env var
# requires re-initialising the SDK — same lifecycle as every
# other knob (api_key, app, etc.). This is intentional: per-call
# tunables would be a footgun for compliance.
_DEFAULT_JUDGE_TIMEOUT_SECS = 8.0
_DEFAULT_JUDGE_RETRY_AFTER_MAX_SECS = 5.0

# ── Verdict cache ────────────────────────────────────────────────────
#
# Agentic workloads re-ask the judge the same question constantly: a
# multi-turn loop resends a system prompt that dominates the text, and
# a retried tool call is byte-identical to its first attempt. Every one
# of those was a fresh network round-trip on the policy hot path.
#
# Why this is accuracy-neutral rather than a speed/precision trade:
# the judge is a temperature-0 classifier over exactly three inputs —
# the prompt text, the intent list, and the threshold. The cache key is
# all three. Identical inputs therefore have an identical correct
# answer, and returning it from memory cannot change a verdict. Editing
# a policy changes its intents (or threshold), which changes the key,
# so a stale rule can't keep firing — invalidation is structural, not
# time-based.
#
# What is deliberately NOT cached:
#
#   * Outage results. A synthesized fail-open ``None`` or fail-closed
#     ``_OUTAGE_MATCH`` reflects the judge being unreachable at one
#     instant, not a verdict about the text. Caching those would let a
#     two-second blip govern traffic for the whole TTL window.
#   * Token spend. A cache hit consumed no tokens, so it books none.
#     ``policy_tokens_*`` keeps meaning "tokens this call actually
#     paid for", which is what the cost columns are built on.
#
# The key is a SHA-256 digest, so no prompt text is retained in the
# cache — only a fingerprint of it. (Phase 1 has already
# label-redacted PII by the time text reaches the judge; hashing is
# belt-and-braces on top of that.)
#
# TTL is short by default: long enough to cover an agent loop or a
# retry storm, short enough that an operator watching the dashboard
# sees their policy edit take effect promptly even in the pathological
# case where the edit somehow preserved the key.
_DEFAULT_JUDGE_CACHE_TTL_SECS = 60.0

# Entry ceiling. Judge verdicts are tiny (a bool, a short string, a
# float) so this is kilobytes, but an unbounded dict in a long-lived
# server process is a leak. On overflow the cache is cleared outright
# rather than LRU-evicted: the access pattern is bursty-then-idle, a
# full clear is O(1), and the cost of a miss is one round-trip.
_JUDGE_CACHE_MAX = 512


# ── Public surface ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SemanticMatch:
    """A blocked intent reported by the judge.

    ``similarity`` is the judge's confidence in ``[0.0, 1.0]``. A
    similarity of ``0.0`` paired with the sentinel intent
    ``"<judge unavailable>"`` indicates the match was synthesized by
    the SDK on outage under fail-closed mode.
    """

    intent: str
    similarity: float


# Sentinel returned by ``check()`` / ``acheck()`` on outage when
# the operator has opted into fail-closed mode.
_OUTAGE_MATCH = SemanticMatch(intent="<judge unavailable>", similarity=0.0)


class SemanticBlocker:
    """Client for the platform's ``semantic_guard`` judge.

    Constructed once per process by ``egisai.init()``. Each instance
    holds both a synchronous and asynchronous HTTP client; use
    ``check()`` from sync code paths and ``acheck()`` from async
    code paths so we never block the event loop.
    """

    _RETRY_429_MAX = 3
    _RETRY_429_FALLBACK_S = 1.0

    def __init__(
        self,
        platform_api_key: str,
        platform_base_url: str,
        on_outage: str = "allow",
        judge_timeout_secs: float | None = None,
        judge_retry_after_max_secs: float | None = None,
        judge_cache_ttl_secs: float | None = None,
    ) -> None:
        if on_outage not in ("allow", "block"):
            raise ValueError(
                f"on_outage must be 'allow' or 'block', got {on_outage!r}"
            )
        self._api_key = platform_api_key
        self._base_url = platform_base_url.rstrip("/")
        self._on_outage = on_outage
        # Resolution order for each knob: explicit constructor
        # arg → env var override → tuned default. Constructor args
        # exist primarily for tests; production callers go through
        # ``egisai.init()`` which lets the env vars win.
        if judge_timeout_secs is None:
            judge_timeout_secs = _env_float(
                "EGISAI_JUDGE_TIMEOUT_SECS",
                _DEFAULT_JUDGE_TIMEOUT_SECS,
                lo=0.5,
            )
        if judge_retry_after_max_secs is None:
            judge_retry_after_max_secs = _env_float(
                "EGISAI_JUDGE_RETRY_AFTER_MAX_SECS",
                _DEFAULT_JUDGE_RETRY_AFTER_MAX_SECS,
                lo=0.1,
            )
        if judge_cache_ttl_secs is None:
            judge_cache_ttl_secs = _env_float(
                "EGISAI_JUDGE_CACHE_TTL_SECS",
                _DEFAULT_JUDGE_CACHE_TTL_SECS,
                lo=0.0,
            )
        self._judge_timeout_secs = float(judge_timeout_secs)
        self._judge_retry_after_max_secs = float(judge_retry_after_max_secs)
        self._judge_cache_ttl_secs = float(judge_cache_ttl_secs)
        self._http_client = httpx.Client(timeout=self._judge_timeout_secs)
        self._async_http_client: httpx.AsyncClient | None = None
        # Verdict cache. Guarded by its own lock because Phase 2 fans
        # judge calls out across worker threads, so concurrent
        # get/put on the same blocker instance is the normal case.
        self._cache: dict[str, tuple[float, SemanticMatch | None]] = {}
        self._cache_lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────

    def check(
        self, prompt_text: str, config: dict[str, Any]
    ) -> SemanticMatch | None:
        """Synchronous check; returns a ``SemanticMatch`` or ``None``."""
        prepared = self._prepare(prompt_text, config)
        if prepared is None:
            return None
        body = prepared

        key = self._cache_key(body)
        hit, found = self._cache_get(key)
        if found:
            return hit

        try:
            response = self._post_with_429_retry(body)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            # Not cached: an outage is a fact about the network at one
            # instant, not a verdict about this text.
            return self._on_outage_response(exc)

        match = self._interpret(data, body)
        self._cache_put(key, match)
        return match

    async def acheck(
        self, prompt_text: str, config: dict[str, Any]
    ) -> SemanticMatch | None:
        """Async sibling of ``check`` — never blocks the event loop."""
        prepared = self._prepare(prompt_text, config)
        if prepared is None:
            return None
        body = prepared

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

        match = self._interpret(data, body)
        self._cache_put(key, match)
        return match

    def diagnose(
        self, prompt_text: str, config: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Raw judge verdict for one question — shadow diagnostics only.

        Same wire call as ``check`` but returns the platform's full
        decision dict (``match`` / ``intent`` / ``confidence`` /
        usage) instead of collapsing it to ``SemanticMatch | None``.
        The enforcement path throws the confidence away on a no-match;
        the shadow harness needs it to distinguish a sub-threshold
        near-miss (intent-list dilution) from a hard ALLOW (structural
        bug). Deliberately skips the verdict cache — a diagnosis wants
        the judge's answer *now* — and deliberately skips token
        accounting: it only ever runs on the shadow thread, whose
        spend is not booked to the governed call.

        Fail-quiet: any error returns ``None``. This path must never
        matter to enforcement.
        """
        prepared = self._prepare(prompt_text, config)
        if prepared is None:
            return None
        try:
            response = self._post_with_429_retry(prepared)
            response.raise_for_status()
            data = response.json()
        except Exception:  # noqa: BLE001
            LOGGER.debug("judge diagnosis call failed", exc_info=True)
            return None
        return data if isinstance(data, dict) else None

    def close(self) -> None:
        """Close both HTTP clients. Idempotent."""
        try:
            self._http_client.close()
        except Exception:  # noqa: BLE001
            pass
        if self._async_http_client is not None:
            try:
                # An async client must be closed inside a running loop.
                # ``aclose()`` requires await, so we schedule it on the
                # current loop if there is one and otherwise spin up a
                # short-lived loop solely for the close.
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
        """Fingerprint the judge question.

        Covers every input the verdict depends on — prompt text, the
        intent list, and the threshold — and nothing else. ``body`` is
        exactly ``_prepare``'s output, so the key can never drift out
        of sync with what gets posted. Sorted keys make the digest
        stable across dict ordering.

        Fast-governance mode ``on`` additionally canonicalises opaque
        identifiers (UUIDs, hex run ids, ``CUST-9374731``-style
        prefixed ids) before hashing, so two questions that differ
        only in an id share one verdict. The normalisation is applied
        to the KEY only — the judge always receives the raw text —
        and never touches bare numbers (amounts). See
        ``egisai.policy.fastpath.normalize_for_cache_key``.
        """
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

    def _cache_get(self, key: str) -> tuple[SemanticMatch | None, bool]:
        """Return ``(verdict, found)``.

        The two-value shape is load-bearing: ``None`` is a perfectly
        good cached verdict ("the judge saw this and did not match"),
        so it cannot double as the sentinel for a miss.
        """
        if self._judge_cache_ttl_secs <= 0:
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

    def _cache_put(self, key: str, verdict: SemanticMatch | None) -> None:
        """Memoize a genuine judge verdict."""
        if self._judge_cache_ttl_secs <= 0:
            return
        with self._cache_lock:
            if len(self._cache) >= _JUDGE_CACHE_MAX:
                self._cache.clear()
            self._cache[key] = (
                time.monotonic() + self._judge_cache_ttl_secs,
                verdict,
            )

    # ── Internals ─────────────────────────────────────────────────────

    def _prepare(
        self, prompt_text: str, config: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Build the judge-call payload, or ``None`` to short-circuit."""
        if not prompt_text:
            return None
        intents = config.get("intents") or []
        if not isinstance(intents, list) or not intents:
            return None

        if (config.get("engine") or "").lower() == "embedding":
            _warn_legacy_embedding_engine_once()
            return None

        body: dict[str, Any] = {
            "prompt_text": prompt_text,
            "intents": list(intents),
        }
        if config.get("threshold") is not None:
            body["threshold"] = config["threshold"]
        # ``judge_model`` used to be forwarded here so an operator
        # could swap the underlying judge model per policy. That
        # escape hatch has been removed: the platform's judge
        # SYSTEM_PROMPT is calibrated against a single model and a
        # silent swap would break the calibrated ``threshold``
        # semantics every other knob assumes. Existing policies
        # that still carry the field stay valid (no schema error);
        # we just stop forwarding it and emit a one-time warning so
        # the operator knows to clean it up.
        if config.get("judge_model"):
            _warn_judge_model_ignored_once()
        return body

    def _interpret(
        self, data: dict[str, Any], body: dict[str, Any]
    ) -> SemanticMatch | None:
        """Account tokens then translate the judge response."""
        try:
            from egisai._context import add_policy_usage

            add_policy_usage(
                tokens_in=int(data.get("tokens_in") or 0),
                tokens_out=int(data.get("tokens_out") or 0),
            )
        except Exception:  # noqa: BLE001
            LOGGER.debug("policy usage accounting failed", exc_info=True)

        if not data.get("match"):
            return None

        intents = body.get("intents") or []
        return SemanticMatch(
            intent=str(data.get("intent") or (intents[0] if intents else "")),
            similarity=float(data.get("confidence") or 1.0),
        )

    def _on_outage_response(self, exc: BaseException) -> SemanticMatch | None:
        """Decide what to return when the judge call raised."""
        detail = exc.__class__.__name__
        if isinstance(exc, httpx.HTTPStatusError):
            # A 4xx is NOT an outage — it's this client sending a
            # request the platform rejects (schema validation, auth,
            # payload size). Fail-open still applies (availability
            # first), but the log must say exactly what bounced so a
            # 422 can never masquerade as a transient blip. Status
            # code only — the response body can echo request fields
            # and never belongs in a log line.
            status = exc.response.status_code
            detail = f"HTTP {status}"
            if 400 <= status < 500:
                LOGGER.error(
                    "semantic_guard: judge request REJECTED by the "
                    "platform (%s) — this is a request problem, not an "
                    "outage; the guard is silently not judging. Check "
                    "SDK/backend version skew and policy config.",
                    detail,
                )
        if self._on_outage == "block":
            LOGGER.warning(
                "semantic_guard: judge call failed (%s) — failing CLOSED "
                "(semantic_on_outage='block'); call will be refused",
                detail,
            )
            return _OUTAGE_MATCH
        LOGGER.warning(
            "semantic_guard: judge call failed (%s) — failing open "
            "(semantic_on_outage='allow')",
            detail,
        )
        return None

    def _ensure_async_client(self) -> httpx.AsyncClient:
        if self._async_http_client is None:
            self._async_http_client = httpx.AsyncClient(
                timeout=self._judge_timeout_secs,
            )
        return self._async_http_client

    def _post_with_429_retry(self, body: dict[str, Any]) -> httpx.Response:
        import time

        last: httpx.Response | None = None
        for attempt in range(self._RETRY_429_MAX + 1):
            last = self._http_client.post(
                f"{self._base_url}/v1/sdk/judge",
                json=body,
                headers=self._auth_headers(),
            )
            if last.status_code != 429:
                return last
            if attempt >= self._RETRY_429_MAX:
                return last
            delay = self._retry_after_seconds(last)
            LOGGER.info(
                "semantic_guard: rate-limited (HTTP 429) — retrying in %.1fs "
                "(attempt %d/%d)",
                delay,
                attempt + 1,
                self._RETRY_429_MAX,
            )
            time.sleep(delay)
        return last  # type: ignore[return-value]

    async def _apost_with_429_retry(
        self, body: dict[str, Any]
    ) -> httpx.Response:
        client = self._ensure_async_client()
        last: httpx.Response | None = None
        for attempt in range(self._RETRY_429_MAX + 1):
            last = await client.post(
                f"{self._base_url}/v1/sdk/judge",
                json=body,
                headers=self._auth_headers(),
            )
            if last.status_code != 429:
                return last
            if attempt >= self._RETRY_429_MAX:
                return last
            delay = self._retry_after_seconds(last)
            LOGGER.info(
                "semantic_guard: rate-limited (HTTP 429) — retrying in %.1fs "
                "(attempt %d/%d)",
                delay,
                attempt + 1,
                self._RETRY_429_MAX,
            )
            await asyncio.sleep(delay)
        return last  # type: ignore[return-value]

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _retry_after_seconds(self, response: httpx.Response) -> float:
        """Parse + clamp the ``Retry-After`` header.

        Clamp is two-sided:

        * Lower bound 0.1 s — sleeping below 100 ms hammers the
          upstream and is rarely what an operator actually wanted
          when they set the header to ``"0"``.
        * Upper bound ``self._judge_retry_after_max_secs`` (default
          5 s, env-overridable) — guards against a misconfigured
          upstream proxy or rogue 429 emitter shipping a header
          like ``Retry-After: 90`` and freezing every governed
          call for 90 s × ``_RETRY_429_MAX`` attempts. With three
          retries that turns a single misconfigured upstream into
          a 270-second policy stall — completely unacceptable for
          something that runs inside the SDK's hot path.

        Note: the HTTP/1.1 spec also allows ``Retry-After`` to be
        an HTTP-date instead of a delta-seconds integer. The SDK
        does not honor that variant — production observation shows
        it's used by zero major LLM platforms / CDNs in front of
        our judge endpoint, and parsing dates here is more risk
        than reward (timezone sources of error, clock skew). On a
        non-numeric header we fall back to the fixed
        ``_RETRY_429_FALLBACK_S`` and proceed with the next attempt.
        """
        retry_after_raw = response.headers.get("Retry-After")
        if not retry_after_raw:
            return self._RETRY_429_FALLBACK_S
        try:
            parsed = float(retry_after_raw)
        except ValueError:
            return self._RETRY_429_FALLBACK_S
        return max(0.1, min(parsed, self._judge_retry_after_max_secs))


_legacy_warning_emitted = False
_judge_model_warning_emitted = False


def _warn_legacy_embedding_engine_once() -> None:
    global _legacy_warning_emitted
    if _legacy_warning_emitted:
        return
    _legacy_warning_emitted = True
    LOGGER.warning(
        "semantic_guard policy uses ``engine: \"embedding\"`` — that path is "
        "no longer supported. Remove the ``engine`` field from the policy "
        "config to use the LLM judge (the default).",
    )


def _warn_judge_model_ignored_once() -> None:
    """Loud-but-non-fatal warning the first time a semantic_guard
    rule's ``judge_model`` field is seen at evaluation time.

    The platform's judge SYSTEM_PROMPT is calibrated against a
    single model, so an operator-supplied override would silently
    skew the ``threshold`` semantics every other rule assumes. The
    SDK stops forwarding the field as of 0.27.0; this warning
    nudges operators to remove the dead key from their policy
    config so audits don't show a knob that does nothing.
    """
    global _judge_model_warning_emitted
    if _judge_model_warning_emitted:
        return
    _judge_model_warning_emitted = True
    LOGGER.warning(
        "semantic_guard policy has a ``judge_model`` field; that knob is "
        "ignored — the platform always uses its calibrated judge model so "
        "the ``threshold`` field behaves as documented. Remove "
        "``judge_model`` from the policy config to clean this up.",
    )
