"""Local runtime fingerprint for agent identity / provenance.

Collected once at ``init()`` time and shipped with every
``ensure_agent`` call so the platform can:

  * Bind the agent's first-seen environment to its identity (Python
    version, OS, framework). The Provenance card on the dashboard
    renders these directly.
  * Detect "the same agent name now lives on a different host"
    (anomaly: ``runtime_change``) without polluting the audit log.

Privacy contract
----------------
This module returns ONLY platform-side fingerprint info — nothing
in the user's process memory, no environment variables, no file
contents, no network discovery. Specifically:

  * Python version (``sys.version_info``).
  * OS family + kernel string (``platform.system`` / ``platform.release``).
  * Optional ``container`` flag derived from ``/proc`` / env hints
    (read locally, never shipped raw).
  * Framework names + versions of the supported integrations
    (``openai``, ``anthropic``, ``google.genai``) — what's importable.
  * SDK version.

What it does NOT collect:

  * Hostname, IP, MAC.
  * Username, home directory, working directory.
  * Environment variables.
  * Any user-defined config.

The platform never receives a raw IP — the backend hashes the IP
of the request itself for its provenance row. This module ships
no IP at all.
"""

from __future__ import annotations

import importlib.metadata as md
import logging
import os
import platform
import sys
import threading
from typing import Any

LOGGER = logging.getLogger("egisai.runtime")


def _safe_distribution_version(name: str) -> str | None:
    try:
        return md.version(name)
    except Exception:  # noqa: BLE001
        return None


def _detect_container() -> bool:
    """Best-effort container detection.

    Looks for the standard signals that work on every common
    Linux container runtime without needing root or any
    syscalls. Returns False (not None) on non-Linux so the
    payload shape stays simple.
    """
    if sys.platform != "linux":
        return False
    if os.environ.get("DOCKER_CONTAINER") or os.environ.get("KUBERNETES_SERVICE_HOST"):
        return True
    try:
        with open("/proc/1/cgroup", encoding="utf-8") as f:
            cg = f.read()
        if "docker" in cg or "kubepods" in cg or "containerd" in cg:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _detect_serverless_hint() -> str | None:
    """Cheap serverless-runtime detector via env vars.

    Each major serverless platform sets a distinct env var; we
    surface the platform name when we recognise one. The backend
    uses this to choose the ``host_class`` badge.
    """
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return "lambda"
    if os.environ.get("FUNCTION_TARGET") and os.environ.get("FUNCTION_NAME"):
        return "cloud_functions"
    if os.environ.get("K_SERVICE"):
        return "cloud_run"
    if os.environ.get("VERCEL"):
        return "vercel"
    if os.environ.get("NETLIFY"):
        return "netlify"
    return None


def _detect_cloud_provider() -> str | None:
    """Best-effort cloud-provider detector via env vars.

    Distinct from :func:`_detect_serverless_hint` — that one
    answers "is the agent on a serverless runtime, and which?"
    and drives the ``host_class`` badge. This one answers
    "which cloud provider owns this network?" and drives the
    ``ASN`` field on the agent's Provenance card. The two
    signals are orthogonal: a Lambda function is both
    ``serverless = "lambda"`` AND ``cloud = "aws"``; a bare
    EC2 box is ``serverless = None`` but still ``cloud = "aws"``.

    Detection is purely env-var-based — no metadata-service
    HTTP calls, no socket probes, no DNS. The SDK design
    philosophy bans network calls in ``init()``; the IMDS
    endpoint that would otherwise give us the most authoritative
    answer is intentionally off the table for that reason.

    Returns a stable, low-cardinality token (``aws``, ``gcp``,
    ``azure``, ``vercel``, …). Tokens MUST stay in sync with
    ``backend/app/services/asn_lookup.py::_RUNTIME_HINT_TO_ASN``.
    """
    env = os.environ.get

    if (
        env("AWS_LAMBDA_FUNCTION_NAME")
        or env("AWS_EXECUTION_ENV")
        or env("ECS_CONTAINER_METADATA_URI")
        or env("ECS_CONTAINER_METADATA_URI_V4")
        or env("AWS_BATCH_JOB_ID")
    ):
        return "aws"
    if (
        env("K_SERVICE")
        or env("FUNCTION_TARGET")
        or env("GOOGLE_CLOUD_PROJECT")
        or env("GCLOUD_PROJECT")
    ):
        return "gcp"
    if (
        env("WEBSITE_SITE_NAME")
        or env("AZURE_FUNCTIONS_ENVIRONMENT")
        or env("MSI_ENDPOINT")
    ):
        return "azure"

    # PaaS providers — surface them by name even though they
    # lease IPs from the underlying clouds. The Provenance card
    # is more useful when it says "Vercel" than "AWS" for an app
    # the operator deployed via ``vercel deploy``.
    if env("VERCEL"):
        return "vercel"
    if env("NETLIFY"):
        return "netlify"
    if env("FLY_APP_NAME"):
        return "fly"
    if env("RAILWAY_ENVIRONMENT"):
        return "railway"
    if env("RENDER"):
        return "render"
    if env("DYNO"):
        return "heroku"
    if env("DIGITALOCEAN_APP_NAME"):
        return "digitalocean"
    return None


# Process-lifetime cache. The fingerprint is built from values that
# CAN'T change inside a running process (Python version, OS, machine,
# container/serverless-platform env, installed framework versions).
# Walking ``importlib.metadata`` for four framework names + reading
# ``/proc/1/cgroup`` on every per-prompt agent registration would
# burn measurable CPU on a multi-agent app — for a value that's
# byte-for-byte identical between calls. Cache once.
#
# The cache is guarded by a lock so two threads racing to populate
# it on first init produce one collected blob (rather than two
# concurrent ``importlib.metadata`` walks). The cache value itself
# is read lock-free via the double-checked pattern in the body —
# the lock only matters on the first miss.
_CACHED: dict[str, Any] | None = None
_CACHED_SDK_VERSION: str | None = None
_CACHE_LOCK = threading.Lock()


def collect_runtime_fingerprint(*, sdk_version: str) -> dict[str, Any]:
    """Return the JSON-friendly runtime blob shipped to the backend.

    Cached for the lifetime of the SDK process. Subsequent calls
    return the same dict (defensively copied so callers can't mutate
    the cache).
    """
    cached = _CACHED
    if cached is not None and _CACHED_SDK_VERSION == sdk_version:
        return dict(cached)
    with _CACHE_LOCK:
        # Double-check after acquiring the lock — another thread may
        # have populated the cache while we were waiting.
        if _CACHED is not None and _CACHED_SDK_VERSION == sdk_version:
            return dict(_CACHED)

        framework_versions: dict[str, str] = {}
        for name in (
            "openai", "anthropic", "google.genai", "google-generativeai",
        ):
            v = _safe_distribution_version(name)
            if v:
                framework_versions[name] = v

        container = _detect_container()
        serverless = _detect_serverless_hint()
        cloud = _detect_cloud_provider()

        blob: dict[str, Any] = {
            "sdk_version": sdk_version,
            "python": ".".join(str(p) for p in sys.version_info[:3]),
            "implementation": platform.python_implementation(),
            "os": platform.system(),
            "platform": platform.release(),
            "machine": platform.machine(),
            "container": container,
            "serverless": serverless,
            # Backend uses ``cloud`` to populate the agent's
            # ASN (Provenance card). Cheap to ship — single
            # short string — and dramatically more reliable
            # than guessing from the request IP, which mis-
            # attributes PaaS workloads to the underlying
            # cloud they happen to be running on.
            "cloud": cloud,
            "frameworks": framework_versions,
        }
        _set_cache(blob, sdk_version)
        return dict(blob)


def _set_cache(blob: dict[str, Any], sdk_version: str) -> None:
    """Internal helper to mutate module-level cache state.

    Pulled out so the lock-protected critical section reads cleanly
    and so :func:`reset_runtime_cache` doesn't reach in directly.
    """
    global _CACHED, _CACHED_SDK_VERSION
    _CACHED = blob
    _CACHED_SDK_VERSION = sdk_version


def reset_runtime_cache() -> None:
    """Test hook — drop the cached fingerprint so a fresh collect runs."""
    global _CACHED, _CACHED_SDK_VERSION
    with _CACHE_LOCK:
        _CACHED = None
        _CACHED_SDK_VERSION = None
