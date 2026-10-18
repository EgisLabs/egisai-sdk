"""Resolve the platform base URL — cloud default vs on-prem hard require."""

from __future__ import annotations

import os

_CLOUD_DEFAULT = "https://app.egisai.co"
_CLOUD_HOSTS = frozenset(
    {
        "app.egisai.co",
        "www.app.egisai.co",
        # Madrid / EU deployment — a separate stack sharing this
        # codebase. Listed so an on-prem build refuses it too.
        "es.app.egisai.co",
    }
)


def is_onprem_build() -> bool:
    try:
        from egisai import __onprem_build__
    except Exception:  # noqa: BLE001
        return False
    return bool(__onprem_build__)


def resolve_platform_base_url(base_url: str | None) -> str:
    """Return the platform root URL (no ``/v1`` suffix).

    Cloud (public PyPI) builds default to ``https://app.egisai.co``.

    On-prem builds (``egisai.__onprem_build__ is True``) **require**
    ``base_url=…`` or ``EGISAI_BASE_URL`` and refuse the cloud host so a
    dependency-confused PyPI install cannot silently talk to SaaS.
    """
    explicit = (base_url or "").strip()
    env = (os.getenv("EGISAI_BASE_URL") or "").strip()
    resolved = explicit or env

    if is_onprem_build():
        if not resolved:
            raise RuntimeError(
                "This egisai install is the on-prem SDK "
                "(egisai.__onprem_build__ is True). Pass base_url=… to "
                "egisai.init() / egisai.Client(), or set EGISAI_BASE_URL "
                "to your control plane (e.g. https://egis.internal:8008). "
                "It will not default to https://app.egisai.co. "
                "If you installed from public PyPI by mistake, uninstall "
                "and install the private on-prem wheel "
                "(see on-prem sdk-offline-install.md)."
            )
        host = _url_host(resolved)
        if host in _CLOUD_HOSTS:
            raise RuntimeError(
                "This egisai install is the on-prem SDK and must not use "
                f"the cloud host {host!r}. Set base_url / EGISAI_BASE_URL "
                "to your on-prem control plane."
            )
        return resolved.rstrip("/")

    return (resolved or _CLOUD_DEFAULT).rstrip("/")


def _url_host(url: str) -> str:
    # Avoid importing urllib for a tiny parse; strip scheme + path.
    rest = url.split("://", 1)[-1]
    return rest.split("/", 1)[0].split(":", 1)[0].lower()
