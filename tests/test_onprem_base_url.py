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


# ── on_outage default is build-dependent (gap #34) ───────────────────


def test_onprem_build_defaults_on_outage_to_block(
    monkeypatch: pytest.MonkeyPatch, fake_backend
) -> None:
    """On-prem fails CLOSED by default: no handshake ⇒ refuse governed calls."""
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    monkeypatch.delenv("EGISAI_ON_OUTAGE", raising=False)
    egisai.init(
        api_key="egis_live_x",
        app="t",
        env="prod",
        base_url="http://fake",
        enable_sse=False,
    )
    assert egisai.diagnostics()["on_outage"] == "block"


def test_cloud_build_defaults_on_outage_to_allow(
    monkeypatch: pytest.MonkeyPatch, fake_backend
) -> None:
    """Cloud keeps its fail-OPEN default — unchanged by gap #34."""
    monkeypatch.setattr(egisai, "__onprem_build__", False)
    monkeypatch.delenv("EGISAI_ON_OUTAGE", raising=False)
    egisai.init(
        api_key="egis_live_x",
        app="t",
        env="prod",
        base_url="http://fake",
        enable_sse=False,
    )
    assert egisai.diagnostics()["on_outage"] == "allow"


def test_onprem_on_outage_explicit_allow_opts_out(
    monkeypatch: pytest.MonkeyPatch, fake_backend
) -> None:
    """The documented opt-out: on_outage='allow' overrides the on-prem default."""
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    monkeypatch.delenv("EGISAI_ON_OUTAGE", raising=False)
    egisai.init(
        api_key="egis_live_x",
        app="t",
        env="prod",
        base_url="http://fake",
        enable_sse=False,
        on_outage="allow",
    )
    assert egisai.diagnostics()["on_outage"] == "allow"


def test_onprem_on_outage_env_allow_opts_out(
    monkeypatch: pytest.MonkeyPatch, fake_backend
) -> None:
    """EGISAI_ON_OUTAGE=allow is the env-var opt-out on on-prem."""
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    monkeypatch.setenv("EGISAI_ON_OUTAGE", "allow")
    egisai.init(
        api_key="egis_live_x",
        app="t",
        env="prod",
        base_url="http://fake",
        enable_sse=False,
    )
    assert egisai.diagnostics()["on_outage"] == "allow"
