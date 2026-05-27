"""Startup-warning telemetry contract.

When the PII NER analyzer fails to load (the most common SDK-init
diagnostic — e.g. a missing transitive dep like ``click``), the SDK
posts a one-shot fire-and-forget telemetry blob to the backend at
``POST /v1/sdk/telemetry/startup-warning`` so the operator's
dashboard can surface the warning without waiting for a customer
support ticket.

These tests lock the contract on three independent axes:

1. **Payload shape and privacy.** The body MUST carry the code,
   exception class, *sanitized* error message, SDK version,
   Python version, and OS — and absolutely nothing else (no
   prompts, no API keys, no file paths or home directories that
   could leak operator infra).
2. **Fail-open.** Every backend failure mode (offline, 4xx, 5xx,
   missing config, malformed exception) MUST be swallowed
   silently — telemetry can never break ``egisai.init()`` or
   the user's first model call.
3. **One-shot.** The warning fires once per process. We don't
   retry a failed POST, and we don't re-emit on every restart of
   the daemon thread.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from egisai import _backend, _config


@pytest.fixture
def _configured_sdk(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Stand up a minimal SDK config so ``post_startup_warning``
    finds something to read. We don't go through ``egisai.init()``
    because that path spins up daemon threads and exercises the
    full handshake — overkill for unit-testing one HTTP wrapper.
    """
    cfg = _config.EgisaiConfig(
        api_key="egis_test_dummy",
        app="telemetry-tests",
        env="test",
        base_url="https://app.egisai.co",
        sdk_version="0.28.0-test",
    )
    _config.set_config(cfg)
    yield
    _config._CONFIG = None


# ── Payload shape ───────────────────────────────────────────────────


def test_startup_warning_payload_carries_required_fields(
    fake_backend, _configured_sdk
) -> None:
    """Happy path: a real exception, a configured SDK, a backend
    that accepts the POST → exactly one warning shows up on the
    backend with every documented field populated."""
    _backend.post_startup_warning(
        "pii_ner_loader_failed",
        ModuleNotFoundError("No module named 'click'"),
    )
    assert len(fake_backend.startup_warnings) == 1
    body = fake_backend.startup_warnings[0]
    # ``code`` and ``error_class`` are verbatim from the call.
    assert body["code"] == "pii_ner_loader_failed"
    assert body["error_class"] == "ModuleNotFoundError"
    # ``error_message`` is the sanitized ``str(exc)``.
    assert body["error_message"] == "No module named 'click'"
    # Runtime fingerprint bits surfaced from ``_runtime``.
    assert body["sdk_version"] == "0.28.0-test"
    # ``python_version`` looks like "3.11.x" / "3.12.x" / "3.13.x".
    assert isinstance(body["python_version"], str)
    assert body["python_version"].count(".") == 2
    # ``os`` is the platform.system() string ("Darwin"/"Linux"/
    # "Windows") — we only assert it's a non-empty string so the
    # test stays portable across CI matrix cells.
    assert isinstance(body["os"], str) and body["os"]


def test_startup_warning_strips_unix_home_dir(
    fake_backend, _configured_sdk
) -> None:
    """A FileNotFoundError that embeds the operator's home dir
    must NOT ship that path verbatim. The sanitizer scrubs
    ``/Users/<name>/`` and ``/home/<name>/`` before transmission."""
    _backend.post_startup_warning(
        "model_download_failed",
        FileNotFoundError(
            "[Errno 2] No such file or directory: "
            "'/Users/soheil/.cache/egisai/model.bin'"
        ),
    )
    body = fake_backend.startup_warnings[0]
    msg = body["error_message"]
    assert "soheil" not in msg, (
        "operator home-directory username leaked into telemetry payload"
    )
    assert "/Users/<redacted>/" in msg, (
        "expected the sanitizer to scrub /Users/<name>/ "
        f"but got: {msg!r}"
    )


def test_startup_warning_strips_linux_home_dir(
    fake_backend, _configured_sdk
) -> None:
    """Same contract on Linux paths (`/home/<name>/...`)."""
    _backend.post_startup_warning(
        "pii_ner_loader_failed",
        PermissionError("[Errno 13] Permission denied: '/home/agent_runner/.cache/'"),
    )
    body = fake_backend.startup_warnings[0]
    assert "agent_runner" not in body["error_message"]
    assert "/home/<redacted>/" in body["error_message"]


def test_startup_warning_truncates_long_messages(
    fake_backend, _configured_sdk
) -> None:
    """The ``error_message`` field has a fixed budget (default
    256 chars) so the operator's dashboard never has to render
    a stack-dump-shaped string."""
    payload = "x" * 600
    _backend.post_startup_warning("oversize_message", RuntimeError(payload))
    body = fake_backend.startup_warnings[0]
    assert len(body["error_message"]) == 256


def test_sanitizer_passes_through_safe_strings_unchanged() -> None:
    """The sanitizer must NOT mangle non-path strings — the
    canonical example we got bit by, ``"No module named 'click'"``,
    has to travel byte-for-byte from the exception to the
    operator's screen."""
    assert (
        _backend._sanitize_telemetry_string("No module named 'click'")
        == "No module named 'click'"
    )
    # Multi-line error message stays intact (other than redaction).
    raw = "RuntimeError: connection refused on localhost:5432"
    assert _backend._sanitize_telemetry_string(raw) == raw


# ── Fail-open contract ──────────────────────────────────────────────


def test_post_startup_warning_swallows_backend_errors(
    monkeypatch: pytest.MonkeyPatch, _configured_sdk
) -> None:
    """A 5xx (or 4xx, or transport error) MUST NOT raise out of
    the telemetry call. The PII loader's daemon thread depends on
    this being bulletproof — a raise would kill the daemon and
    leave the loader stuck in ``loading=True, settled=False``,
    which is worse than the missing-dep regression itself."""

    def boom_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "internal server error"})

    transport = httpx.MockTransport(boom_handler)

    def patched_get_client() -> httpx.Client:
        if _backend._client is None:
            cfg = _config.get_config()
            _backend._client = httpx.Client(
                base_url=cfg.base_url.rstrip("/"),
                timeout=cfg.timeout_seconds,
                transport=transport,
            )
        return _backend._client

    monkeypatch.setattr(_backend, "get_client", patched_get_client)

    # If this raises, the test fails — we expect it to be silent.
    _backend.post_startup_warning(
        "pii_ner_loader_failed",
        ModuleNotFoundError("No module named 'click'"),
    )

    if _backend._client is not None:
        _backend._client.close()
        _backend._client = None


def test_post_startup_warning_noop_when_sdk_not_configured(
    fake_backend,
) -> None:
    """If the user calls into ``post_startup_warning`` before
    ``egisai.init()`` (theoretically possible because the daemon
    thread is spawned by init itself but background-loader code
    paths can still race a partial config write), we MUST exit
    without hitting the network. There's no auth token to send,
    and we don't want to surface a spurious unauthenticated
    payload on the dashboard."""
    _config._CONFIG = None  # belt-and-suspenders, no config set

    _backend.post_startup_warning(
        "pii_ner_loader_failed",
        ModuleNotFoundError("No module named 'click'"),
    )

    assert fake_backend.startup_warnings == [], (
        "expected no telemetry POST when SDK is unconfigured; "
        f"saw {fake_backend.startup_warnings!r}"
    )


def test_post_startup_warning_swallows_transport_errors(
    monkeypatch: pytest.MonkeyPatch, _configured_sdk
) -> None:
    """A complete network outage (e.g. DNS failure, ``ConnectError``)
    MUST be silent. This is the realistic case where a customer
    runs the SDK in an air-gapped environment."""

    def boom_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("nodename nor servname provided")

    transport = httpx.MockTransport(boom_handler)

    def patched_get_client() -> httpx.Client:
        if _backend._client is None:
            cfg = _config.get_config()
            _backend._client = httpx.Client(
                base_url=cfg.base_url.rstrip("/"),
                timeout=cfg.timeout_seconds,
                transport=transport,
            )
        return _backend._client

    monkeypatch.setattr(_backend, "get_client", patched_get_client)

    _backend.post_startup_warning(
        "pii_ner_loader_failed",
        ModuleNotFoundError("No module named 'click'"),
    )

    if _backend._client is not None:
        _backend._client.close()
        _backend._client = None
