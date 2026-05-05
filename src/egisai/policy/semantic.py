"""Semantic intent enforcement for ``semantic_guard`` policies.

The SDK doesn't host the judge model. When a ``semantic_guard``
rule fires, the SDK calls the EgisAI platform with the already-
redacted prompt (Phase 1 of the engine has run by then, so PII
is masked) and a list of operator-authored intent strings, and
the platform returns a verdict.

The judge call fails open: a network error or auth failure
returns "no match" so governance never breaks the call path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

LOGGER = logging.getLogger("egisai.policy.semantic")


# ── Public surface ────────────────────────────────────────────────────


@dataclass(frozen=True)
class SemanticMatch:
    """A blocked intent reported by the judge.

    ``similarity`` is the judge's confidence in ``[0.0, 1.0]``.
    """
    intent: str
    similarity: float


class SemanticBlocker:
    """Client for the platform's ``semantic_guard`` judge.

    Constructed once per process by ``egisai.init()``.
    """

    def __init__(self, platform_api_key: str, platform_base_url: str):
        self._api_key = platform_api_key
        self._base_url = platform_base_url.rstrip("/")
        self._http_client = httpx.Client(timeout=20.0)

    def check(self, prompt_text: str, config: dict[str, Any]) -> SemanticMatch | None:
        """Return a ``SemanticMatch`` if the prompt hits a blocked intent.

        Returns ``None`` to allow, or on any error (fail open).
        """
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

        try:
            response = self._post_with_429_retry(body)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "semantic_guard: judge call failed (%s) — failing open",
                exc.__class__.__name__,
            )
            return None

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

        return SemanticMatch(
            intent=str(data.get("intent") or (intents[0] if intents else "")),
            similarity=float(data.get("confidence") or 1.0),
        )

    _RETRY_429_MAX = 3
    _RETRY_429_FALLBACK_S = 1.0

    def _post_with_429_retry(self, body: dict[str, Any]) -> httpx.Response:
        import time
        last: httpx.Response | None = None
        for attempt in range(self._RETRY_429_MAX + 1):
            last = self._http_client.post(
                f"{self._base_url}/v1/sdk/judge",
                json=body,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
            )
            if last.status_code != 429:
                return last
            if attempt >= self._RETRY_429_MAX:
                return last
            retry_after_raw = last.headers.get("Retry-After")
            delay = self._RETRY_429_FALLBACK_S
            if retry_after_raw:
                try:
                    delay = max(0.1, float(retry_after_raw))
                except ValueError:
                    pass
            LOGGER.info(
                "semantic_guard: rate-limited (HTTP 429) — retrying in %.1fs "
                "(attempt %d/%d)",
                delay, attempt + 1, self._RETRY_429_MAX,
            )
            time.sleep(delay)
        return last  # type: ignore[return-value]

    def close(self) -> None:
        """Close the underlying HTTP client. Idempotent."""
        try:
            self._http_client.close()
        except Exception:  # noqa: BLE001
            pass


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
