"""On-prem SDK must not default to the cloud URL."""

from __future__ import annotations

import pytest

import egisai
from egisai._base_url import resolve_platform_base_url


def test_cloud_build_defaults_to_app_egisai_co(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(egisai, "__onprem_build__", False)
    monkeypatch.delenv("EGISAI_BASE_URL", raising=False)
    assert resolve_platform_base_url(None) == "https://app.egisai.co"


def test_onprem_build_requires_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    monkeypatch.delenv("EGISAI_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="on-prem SDK"):
        resolve_platform_base_url(None)


def test_onprem_build_accepts_explicit_control_plane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    monkeypatch.delenv("EGISAI_BASE_URL", raising=False)
    assert (
        resolve_platform_base_url("https://egis.internal:8008")
        == "https://egis.internal:8008"
    )


def test_onprem_build_refuses_cloud_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    with pytest.raises(RuntimeError, match="must not use the cloud host"):
        resolve_platform_base_url("https://app.egisai.co")


def test_onprem_build_refuses_eu_cloud_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The EU deployment host counts as a cloud host too."""
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    with pytest.raises(RuntimeError, match="must not use the cloud host"):
        resolve_platform_base_url("https://es.app.egisai.co")


def test_eu_host_is_recognized_as_cloud() -> None:
    from egisai._base_url import _CLOUD_HOSTS

    assert "es.app.egisai.co" in _CLOUD_HOSTS


def test_onprem_build_reads_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    monkeypatch.setenv("EGISAI_BASE_URL", "http://cp:8008/")
    assert resolve_platform_base_url(None) == "http://cp:8008"


def test_public_sdk_marker_is_false_by_default() -> None:
    """Public PyPI wheels must keep ``__onprem_build__ = False``."""
    assert egisai.__onprem_build__ is False


def test_public_semantic_on_outage_default_is_allow() -> None:
    from egisai._config import EgisaiConfig

    cfg = EgisaiConfig(api_key="egis_live_x", app="t", env="prod")
    assert cfg.semantic_on_outage == "allow"
