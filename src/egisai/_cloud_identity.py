"""Which cloud principal is this agent running as?

The problem this solves is a join. The platform inventories OAuth
grants, service principals, and installed apps from the customer's
SaaS tenants — every one of them keyed by a client id or a service
account. Separately it sees agents at runtime, keyed by a prompt
fingerprint. Nothing connected the two, so "which agent is using this
app's standing access to Drive?" had no answer that wasn't a guess.

A cloud identity is the join key. An agent running as
``arn:aws:iam::123:role/support-bot`` and a Bedrock grant issued to
that same role are the same actor, and matching them is a string
compare rather than an inference.

What this reports
-----------------
Identity strings only, and only ones the environment already asserts:

* AWS — the assumed role ARN, from ``AWS_ROLE_ARN`` or IMDSv2.
* GCP — the service account email, from the metadata server or the
  ``client_email`` field of the file ``GOOGLE_APPLICATION_CREDENTIALS``
  points at.
* Azure — ``AZURE_CLIENT_ID`` / ``MSI_CLIENT_ID``.
* Generic OAuth — ``OAUTH_CLIENT_ID`` and the common vendor spellings.

What it will never report
-------------------------
**No credentials, ever.** IMDS will hand out temporary access keys to
anyone who asks it; this module reads the role name and stops. The GCP
service account file contains a private key; this module reads one
field out of it and never touches ``private_key``. Nothing here reads a
token, a secret, or a password, and there is no code path that could —
the extraction functions name the fields they want.

Reporting a role ARN is still infrastructure metadata leaving the
customer's environment, which is why it is opt-out via
``init(cloud_identity=False)`` or ``EGISAI_CLOUD_IDENTITY=0``, and why
``SECURITY.md`` documents it.

Why it runs in a thread
-----------------------
:mod:`egisai._runtime` is deliberately network-free, and ``ensure_agent``
runs inline on the customer's first model call with a two-second budget.
IMDS is normally a single-digit-millisecond hop, but on a host where it
is firewalled rather than absent the connection hangs until something
times out — and that something must not be the customer's first
request. So the probe runs in a daemon thread started from ``init()``
with a short deadline, exactly like the PII model loader, and whatever
it has found by the time an ``ensure`` goes out is what ships. The
first call may carry env-only data; the next one carries the rest.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

LOGGER = logging.getLogger("egisai.cloud_identity")

#: Per-request deadline for a metadata hop. Deliberately tight: the
#: metadata service is link-local and answers in single-digit
#: milliseconds when it exists, so a slow answer means something is
#: wrong (firewall blackhole, proxy interception) and waiting longer
#: only delays the same empty result.
PROBE_TIMEOUT_SECONDS = 0.2

#: Link-local addresses, hard-coded rather than read from the
#: environment. ``AWS_EC2_METADATA_SERVICE_ENDPOINT`` exists and is
#: honoured by the AWS SDKs, but honouring it here would let anything
#: that can set an env var in the agent's process point this probe at
#: a host of its choosing. The address is a constant of the platform;
#: treat it as one.
_AWS_IMDS = "http://169.254.169.254"
_GCP_METADATA = "http://metadata.google.internal"

#: Env vars that carry an OAuth client id in the wild. Values are
#: reported as-is; a client id is a public identifier by design (it
#: travels in every authorization URL), which is what makes it safe to
#: report and useful as a join key.
_OAUTH_CLIENT_ID_VARS = (
    "OAUTH_CLIENT_ID",
    "OAUTH2_CLIENT_ID",
    "CLIENT_ID",
    "OKTA_CLIENT_ID",
    "AUTH0_CLIENT_ID",
    "GOOGLE_CLIENT_ID",
    "MICROSOFT_CLIENT_ID",
    "SALESFORCE_CLIENT_ID",
)

#: Anything longer than this is not an identifier — it is a token, a
#: certificate, or a paste accident. Refusing it is cheap insurance
#: against a misconfigured env var turning this into an exfiltration
#: channel for something that was never meant to leave.
_MAX_VALUE_CHARS = 512


def _clean(value: Any) -> str | None:
    """A reportable identity string, or ``None``.

    Total by design — every caller is reading something it does not
    control (an env var, a JSON field, an HTTP body).
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped or len(stripped) > _MAX_VALUE_CHARS:
        return None
    return stripped


def _http_get(url: str, headers: dict[str, str] | None = None) -> str | None:
    """One metadata request, or ``None``. Never raises.

    ``urllib`` rather than ``httpx`` because this must work in a bare
    install with no optional dependencies, and because the SDK must not
    reuse a customer-visible HTTP client for a probe they did not ask
    for.
    """
    try:
        req = urlrequest.Request(url, headers=headers or {})  # noqa: S310
        with urlrequest.urlopen(  # noqa: S310
            req, timeout=PROBE_TIMEOUT_SECONDS
        ) as response:
            if response.status != 200:
                return None
            return response.read(_MAX_VALUE_CHARS * 4).decode(
                "utf-8", errors="replace"
            )
    except (urlerror.URLError, OSError, ValueError):
        # Not on this cloud, firewalled, or answering something we
        # can't parse. All three mean the same thing here.
        return None


# ── AWS ─────────────────────────────────────────────────────────────


def _aws_role_arn_via_imds() -> str | None:
    """IMDSv2: PUT for a token, then read the instance profile name.

    IMDSv2 only. v1's unauthenticated GET is the one an SSRF bug in the
    customer's own app can reach, and reaching for it here would be
    this SDK depending on a configuration AWS itself recommends
    disabling.

    Returns the *profile* ARN rather than the assumed-role ARN, which
    is what the instance metadata actually exposes. They differ in the
    ``:instance-profile/`` versus ``:assumed-role/`` segment, and the
    backend's matcher normalizes on the trailing role name for exactly
    this reason.

    ``AWS_ROLE_ARN`` is checked before this is ever called — EKS IRSA
    and web identity federation both set it, and it is the more precise
    of the two answers.
    """
    try:
        token_req = urlrequest.Request(  # noqa: S310
            f"{_AWS_IMDS}/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urlrequest.urlopen(  # noqa: S310
            token_req, timeout=PROBE_TIMEOUT_SECONDS
        ) as response:
            token = response.read(4096).decode("ascii", errors="replace")
    except (urlerror.URLError, OSError, ValueError):
        return None

    body = _http_get(
        f"{_AWS_IMDS}/latest/meta-data/iam/info",
        headers={"X-aws-ec2-metadata-token": token},
    )
    if not body:
        return None
    try:
        info = json.loads(body)
    except ValueError:
        return None
    if not isinstance(info, dict):
        return None
    return _clean(info.get("InstanceProfileArn"))


# ── GCP ─────────────────────────────────────────────────────────────


def _gcp_service_account() -> str | None:
    """The service account email, metadata server then key file.

    The key file path is the fallback because it is what a local
    developer has, and reading one field out of it is strictly better
    than reporting nothing — but note the ordering: on a real GCP host
    the metadata server is authoritative and the file may be stale or
    absent.
    """
    email = _clean(
        _http_get(
            f"{_GCP_METADATA}/computeMetadata/v1/instance/"
            "service-accounts/default/email",
            headers={"Metadata-Flavor": "Google"},
        )
    )
    if email:
        return email

    path = _clean(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    if not path:
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            # Bounded read. A service account file is under 3 KB; if
            # the variable points somewhere else entirely we want to
            # fail on a parse, not load an arbitrary file into memory.
            payload = json.loads(handle.read(64 * 1024))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    # ``client_email`` and nothing else. The same file holds
    # ``private_key``; naming the field explicitly is what guarantees
    # it is never in reach.
    return _clean(payload.get("client_email"))


# ── Azure and generic OAuth ─────────────────────────────────────────


def _azure_client_id() -> str | None:
    for name in ("AZURE_CLIENT_ID", "MSI_CLIENT_ID"):
        value = _clean(os.environ.get(name))
        if value:
            return value
    return None


def _oauth_client_ids() -> list[str]:
    """Every OAuth client id visible in the environment, deduplicated.

    A list rather than one value: an agent that talks to three SaaS
    APIs legitimately has three, and any of them may be the one a
    grant is keyed by.
    """
    seen: list[str] = []
    for name in _OAUTH_CLIENT_ID_VARS:
        value = _clean(os.environ.get(name))
        if value and value not in seen:
            seen.append(value)
    return seen


# ── Collection ──────────────────────────────────────────────────────


def collect() -> dict[str, Any]:
    """Probe every provider once. Never raises; may return ``{}``.

    Runs the network hops only where an env signal suggests the cloud
    is plausible, so a laptop pays nothing: no AWS env var means no
    IMDS call, and the 169.254 hop that would otherwise hang behind a
    corporate firewall never happens.
    """
    found: dict[str, Any] = {}

    aws = _clean(os.environ.get("AWS_ROLE_ARN"))
    if not aws and _looks_like("aws"):
        aws = _aws_role_arn_via_imds()
    if aws:
        found["aws_role_arn"] = aws

    if _looks_like("gcp") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        gcp = _gcp_service_account()
        if gcp:
            found["gcp_service_account"] = gcp

    azure = _azure_client_id()
    if azure:
        found["azure_client_id"] = azure

    oauth = _oauth_client_ids()
    if oauth:
        found["oauth_client_ids"] = oauth

    return found


def _looks_like(provider: str) -> bool:
    """Is a metadata hop to this provider worth attempting?

    Reuses the env-var heuristics :mod:`egisai._runtime` already
    applies rather than duplicating them, so the two can't disagree
    about which cloud a host is on.
    """
    from egisai._runtime import _detect_cloud_provider

    return _detect_cloud_provider() == provider


# ── Background probe ────────────────────────────────────────────────

_result: dict[str, Any] = {}
_started = False
_settled = threading.Event()
_lock = threading.Lock()


def prime_async() -> None:
    """Start the probe in a daemon thread, idempotently.

    Called once from ``egisai.init()``. Returns immediately — the
    caller's first model call must not wait on a metadata service.
    """
    global _started
    with _lock:
        if _started:
            return
        _started = True

    thread = threading.Thread(
        target=_probe,
        name="egisai-cloud-identity",
        daemon=True,
    )
    thread.start()


def _probe() -> None:
    global _result
    try:
        found = collect()
    except BaseException as exc:  # noqa: BLE001 — fail open, always
        LOGGER.debug(
            "cloud identity probe failed (%s); reporting nothing",
            exc.__class__.__name__,
        )
        found = {}
    with _lock:
        _result = found
    _settled.set()


def snapshot() -> dict[str, Any]:
    """What the probe has found so far. ``{}`` until it settles.

    Deliberately non-blocking. An ``ensure`` that goes out before the
    probe finishes ships without the identity and the next one carries
    it — one round trip of latency is not worth adding a wait to the
    customer's first model call.
    """
    return dict(_result)


def is_settled() -> bool:
    """``True`` once the probe has finished, however it finished."""
    return _settled.is_set()


def wait(timeout_secs: float) -> bool:
    """Block for the probe, for tests and for callers that can afford it."""
    if timeout_secs <= 0:
        return _settled.is_set()
    return _settled.wait(timeout_secs)


def reset_for_tests() -> None:
    """Drop probe state so a test can drive a fresh run."""
    global _result, _started
    with _lock:
        _result = {}
        _started = False
    _settled.clear()


__all__ = [
    "PROBE_TIMEOUT_SECONDS",
    "collect",
    "is_settled",
    "prime_async",
    "reset_for_tests",
    "snapshot",
    "wait",
]
