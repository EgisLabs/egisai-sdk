"""HTTP PII engine that offloads detection to the Egis control plane.

On an **on-prem** build the SDK must never download or run the heavy NER
model in the customer's own process (security-and-compliance.mdc rule 1 +
gap #35: no outbound model download, no silent regex-only degrade). Instead
it posts text to the operator's OWN control plane
(``{EGISAI_BASE_URL}/v1/sdk/pii/*``), authenticated with the agent's Egis
API key. The control plane runs ``egisai.policy.pii`` which — inside the same
customer network — offloads the heavy pass to the dedicated ``egisai-ner``
worker. Raw text therefore travels only to operator-owned infrastructure,
never to a third party (rule 1's stated exception).

This engine is auto-installed by :func:`egisai._init.init` on on-prem builds
after a successful handshake — no manual ``set_remote_engine`` call is
needed. Cloud keeps the fully-local on-device engine and never installs this.

Fail-closed on-prem: every method raises on any transport error, timeout, or
non-2xx. ``egisai.policy.pii`` catches that and marks the request as
remote-degraded (:func:`egisai.policy.pii.note_remote_failure`); the on-prem
evaluator then refuses the governed call with ``pii_engine_unavailable``
rather than fall back to regex-only detection. The timeouts below bound how
long a struggling control plane can slow one governance pass.
"""
from __future__ import annotations

import httpx

from egisai.policy.pii import PIIFinding, Sanitization

# Fast connect, a few seconds of read headroom for a large-payload scan.
# A failure (timeout or error) raises → the SDK fails closed on-prem, so
# this is a ceiling on how long one governance pass can wait, not a hard
# availability dependency.
_PII_TIMEOUT = httpx.Timeout(connect=2.0, read=10.0, write=5.0, pool=2.0)


class HttpRemotePIIEngine:
    """Drop-in :class:`egisai.policy.pii.RemotePIIEngine` over HTTP.

    Posts to ``{base_url}/v1/sdk/pii/*`` using the shared backend client
    (which already carries ``Authorization: Bearer <api_key>`` and the
    SDK User-Agent). Return values match the local ``pii`` functions
    exactly so callers can't tell the work ran elsewhere.
    """

    def __init__(self, path_prefix: str = "/v1/sdk/pii") -> None:
        self._prefix = path_prefix.rstrip("/")

    def _post(self, path: str, payload: dict) -> dict:
        # Reuse the SDK's shared, authenticated client so base_url + auth
        # + UA + connection pooling are all inherited. Bounded timeout on
        # the hot path; any error propagates to the caller.
        from egisai._backend import get_client

        resp = get_client().post(
            f"{self._prefix}{path}", json=payload, timeout=_PII_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):  # malformed reply → fail closed
            raise ValueError("remote PII engine returned a non-object body")
        return data

    def scan(self, text: str) -> list[PIIFinding]:
        data = self._post("/scan", {"text": text})
        # A malformed body must raise (KeyError), not silently return "no
        # PII" — that would be fail-open on detection (a leak on-prem).
        findings = data["findings"]
        return [
            PIIFinding(
                type=f["type"],
                value_redacted=f["value_redacted"],
                confidence=float(f["confidence"]),
                method=f["method"],
            )
            for f in findings
        ]

    def sanitize(
        self,
        text: str,
        types: list[str] | None = None,
        mask_char: str = "#",
    ) -> tuple[str, list[Sanitization]]:
        data = self._post(
            "/sanitize",
            {"text": text, "types": types, "mask_char": mask_char},
        )
        # ``masked_text`` MUST be present — falling back to the raw input
        # on a malformed reply would forward unmasked PII. Let the
        # KeyError propagate so the on-prem SDK fails closed.
        masked = data["masked_text"]
        records = [
            Sanitization(
                type=s["type"], count=int(s["count"]), pattern=s["pattern"]
            )
            for s in data.get("sanitizations", [])
        ]
        return masked, records

    def label_redact(
        self, text: str, types: list[str] | None = None
    ) -> str:
        data = self._post("/label_redact", {"text": text, "types": types})
        return str(data["redacted"])


_installed: HttpRemotePIIEngine | None = None


def install() -> bool:
    """Install the HTTP remote PII engine into ``egisai.policy.pii``.

    Idempotent. Returns ``True`` once an engine is installed so the caller
    can skip the local model warm-up (the remote engine fully covers NER).
    """
    global _installed
    from egisai.policy import pii

    if _installed is None:
        _installed = HttpRemotePIIEngine()
    pii.set_remote_engine(_installed)
    return True
