"""Process-wide SDK configuration.

Set once by ``egisai.init()``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

OnBlock = Literal["raise", "stub"]
OnOutage = Literal["allow", "block"]


@dataclass(frozen=True)
class EgisaiConfig:
    api_key: str
    app: str
    env: str
    base_url: str = "https://app.egisai.co"
    on_block: OnBlock = "raise"
    refresh_interval_seconds: float = 10.0
    flush_interval_seconds: float = 1.0
    flush_batch_size: int = 50
    enable_sse: bool = True
    enable_http_fallback: bool = True
    sdk_version: str = "0.11.0"
    timeout_seconds: float = 10.0
    org_id: str | None = None
    agent_id: str | None = None
    # Behavior when the platform's semantic-guard judge is unreachable.
    # "allow"  — fail open (default; matches pre-0.11 behavior).
    # "block"  — fail closed; treat the call as if every semantic_guard
    #            rule fired. Use when the operator considers Phase 2
    #            checks the primary defense for that workload.
    semantic_on_outage: OnOutage = "allow"


_CONFIG: EgisaiConfig | None = None


def set_config(cfg: EgisaiConfig) -> None:
    global _CONFIG
    _CONFIG = cfg


def get_config() -> EgisaiConfig:
    if _CONFIG is None:
        raise RuntimeError("egisai not initialized — call egisai.init(...) first.")
    return _CONFIG


def get_config_optional() -> EgisaiConfig | None:
    return _CONFIG


def update_config(**fields: object) -> EgisaiConfig:
    """Replace the config with a copy carrying the supplied fields."""
    global _CONFIG
    if _CONFIG is None:
        raise RuntimeError("egisai not initialized — call egisai.init(...) first.")
    base = {
        "api_key": _CONFIG.api_key,
        "app": _CONFIG.app,
        "env": _CONFIG.env,
        "base_url": _CONFIG.base_url,
        "on_block": _CONFIG.on_block,
        "refresh_interval_seconds": _CONFIG.refresh_interval_seconds,
        "flush_interval_seconds": _CONFIG.flush_interval_seconds,
        "flush_batch_size": _CONFIG.flush_batch_size,
        "enable_sse": _CONFIG.enable_sse,
        "enable_http_fallback": _CONFIG.enable_http_fallback,
        "sdk_version": _CONFIG.sdk_version,
        "timeout_seconds": _CONFIG.timeout_seconds,
        "org_id": _CONFIG.org_id,
        "agent_id": _CONFIG.agent_id,
        "semantic_on_outage": _CONFIG.semantic_on_outage,
    }
    base.update(fields)
    new = EgisaiConfig(**base)  # type: ignore[arg-type]
    _CONFIG = new
    return new
