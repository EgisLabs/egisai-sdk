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
import logging
from dataclasses import dataclass
from typing import Any

import httpx

LOGGER = logging.getLogger("egisai.policy.semantic")


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
    ) -> None:
        if on_outage not in ("allow", "block"):
            raise ValueError(
                f"on_outage must be 'allow' or 'block', got {on_outage!r}"
            )
        self._api_key = platform_api_key
        self._base_url = platform_base_url.rstrip("/")
        self._on_outage = on_outage
        self._http_client = httpx.Client(timeout=20.0)
        self._async_http_client: httpx.AsyncClient | None = None

    # ── Public API ────────────────────────────────────────────────────

    def check(
        self, prompt_text: str, config: dict[str, Any]
    ) -> SemanticMatch | None:
        """Synchronous check; returns a ``SemanticMatch`` or ``None``."""
        prepared = self._prepare(prompt_text, config)
        if prepared is None:
            return None
        body = prepared

        try:
            response = self._post_with_429_retry(body)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            return self._on_outage_response(exc)

        return self._interpret(data, body)

    async def acheck(
        self, prompt_text: str, config: dict[str, Any]
    ) -> SemanticMatch | None:
        """Async sibling of ``check`` — never blocks the event loop."""
        prepared = self._prepare(prompt_text, config)
        if prepared is None:
            return None
        body = prepared

        try:
            response = await self._apost_with_429_retry(body)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            return self._on_outage_response(exc)

        return self._interpret(data, body)

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
        if config.get("judge_model"):
            body["judge_model"] = config["judge_model"]
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
        if self._on_outage == "block":
            LOGGER.warning(
                "semantic_guard: judge call failed (%s) — failing CLOSED "
                "(semantic_on_outage='block'); call will be refused",
                exc.__class__.__name__,
            )
            return _OUTAGE_MATCH
        LOGGER.warning(
            "semantic_guard: judge call failed (%s) — failing open "
            "(semantic_on_outage='allow')",
            exc.__class__.__name__,
        )
        return None

    def _ensure_async_client(self) -> httpx.AsyncClient:
        if self._async_http_client is None:
            self._async_http_client = httpx.AsyncClient(timeout=20.0)
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
        retry_after_raw = response.headers.get("Retry-After")
        if not retry_after_raw:
            return self._RETRY_429_FALLBACK_S
        try:
            return max(0.1, float(retry_after_raw))
        except ValueError:
            return self._RETRY_429_FALLBACK_S


_legacy_warning_emitted = False


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
