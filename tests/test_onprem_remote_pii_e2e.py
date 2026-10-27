"""On-prem direct-mode PII runs through the control plane, never the model.

This is the end-to-end proof for the on-prem remote-NER path (gap #35):

  * In **direct mode** (the in-process patched-provider path, NOT the
    Gateway), a ``pii_scan`` policy masks a NAME and an ADDRESS using the
    REMOTE engine — the SDK posts raw text to the operator's own control
    plane (``{EGISAI_BASE_URL}/v1/sdk/pii/*``) and uses the masked reply.
  * With the NER tier STOPPED (the control plane returns 503), the governed
    call is REFUSED on-prem (``pii_engine_unavailable``) rather than
    silently masking regex-only.
  * ZERO outbound connections leave the customer network — the only socket
    the SDK opens is to the loopback control-plane stub.

The control plane is a real loopback HTTP server here (stdlib, no uvicorn,
no ``openai`` dependency, no network). It stands in for the in-network
control plane + ``egisai-ner`` worker. We drive the SAME ``evaluate()`` /
``pii`` code path the OpenAI patch invokes in direct mode, so this covers
the real governed flow without pulling the provider SDK into the test.
"""
from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import egisai
from egisai import _backend, _config, _context, _evaluator, _policy_cache
from egisai._evaluator import InputCall, evaluate
from egisai.policy import _pii_loader, _remote_pii_http
from egisai.policy import pii as pii_mod

_API_KEY = "egis_live_test"
_NAME = "John Smith"
_ADDRESS = "742 Evergreen Terrace"
_PROMPT = f"Hi, my name is {_NAME} and I live at {_ADDRESS}."


# ── Loopback control-plane stub (in-network) ─────────────────────────


class _ControlPlaneState:
    """Shared, mutable state for the stub across requests."""

    def __init__(self) -> None:
        self.ner_up = True
        self.saw_auth: list[str] = []
        self.saw_raw_text = False
        self.base_url = ""


def _mask(text: str) -> tuple[str, list[dict]]:
    """Mimic the worker masking a name + address (NER-only entities)."""
    out = text
    records: list[dict] = []
    if _NAME in out:
        out = out.replace(_NAME, "#" * len(_NAME))
        records.append(
            {"type": "person_name", "count": 1, "pattern": "#" * len(_NAME)}
        )
    if _ADDRESS in out:
        out = out.replace(_ADDRESS, "#" * len(_ADDRESS))
        records.append(
            {"type": "address", "count": 1, "pattern": "#" * len(_ADDRESS)}
        )
    return out, records


def _findings(text: str) -> list[dict]:
    out: list[dict] = []
    if _NAME in text:
        out.append(
            {
                "type": "person_name",
                "value_redacted": "J*** S****",
                "confidence": 0.95,
                "method": "remote",
            }
        )
    if _ADDRESS in text:
        out.append(
            {
                "type": "address",
                "value_redacted": "7** E******* T******",
                "confidence": 0.9,
                "method": "remote",
            }
        )
    return out


def _make_handler(state: _ControlPlaneState) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *a: object) -> None:  # silence test noise
            pass

        def _reply(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self) -> None:  # noqa: N802 — stdlib naming
            state.saw_auth.append(self.headers.get("Authorization", ""))
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            text = payload.get("text", "")
            if text:
                state.saw_raw_text = True
            # NER stopped → fail closed, exactly like the real endpoint.
            if not state.ner_up:
                self._reply(503, {"detail": "pii_engine_unavailable"})
                return
            if self.path.endswith("/pii/scan"):
                self._reply(200, {"findings": _findings(text)})
            elif self.path.endswith("/pii/sanitize"):
                masked, records = _mask(text)
                self._reply(
                    200, {"masked_text": masked, "sanitizations": records}
                )
            elif self.path.endswith("/pii/label_redact"):
                masked, _ = _mask(text)
                self._reply(200, {"redacted": masked})
            else:  # pragma: no cover — defensive
                self._reply(404, {"detail": "not found"})

    return _Handler


@pytest.fixture
def control_plane() -> Iterator[_ControlPlaneState]:
    state = _ControlPlaneState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    state.base_url = f"http://{host}:{port}"
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def _reset_sdk_state() -> Iterator[None]:
    _pii_loader.reset_for_tests()
    _evaluator._reset_warmup_gate_for_tests()
    pii_mod.set_remote_engine(None)
    pii_mod.reset_remote_failure()
    _policy_cache.clear()
    _backend.close_client()
    _config._CONFIG = None
    _context._ctx.set(_context.EgisaiContext())
    yield
    _pii_loader.reset_for_tests()
    _evaluator._reset_warmup_gate_for_tests()
    pii_mod.set_remote_engine(None)
    pii_mod.reset_remote_failure()
    _policy_cache.clear()
    _backend.close_client()
    _config._CONFIG = None
    _context._ctx.set(_context.EgisaiContext())


def _wire_onprem(
    monkeypatch: pytest.MonkeyPatch, state: _ControlPlaneState
) -> None:
    """On-prem build + config pointed at the loopback control plane +
    the real HTTP remote engine installed (as init() would on-prem)."""
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    monkeypatch.delenv("EGISAI_ONPREM_PII_REGEX_ONLY", raising=False)
    _config.set_config(
        _config.EgisaiConfig(
            api_key=_API_KEY,
            app="test",
            env="dev",
            base_url=state.base_url,
            on_outage="block",
        )
    )
    _remote_pii_http.install()
    # Mirror init(): a remote engine short-circuits the local model load.
    _pii_loader.prime_analyzer_async(quiet=True)
    _evaluator._reset_warmup_gate_for_tests()


def _pii_rule() -> dict:
    return {
        "id": "pii-onprem",
        "name": "mask PII",
        "type": "pii_scan",
        "tenant": None,
        "config": {
            "types": ["person_name", "address"],
            "action": "sanitize",
        },
    }


def _input_call() -> InputCall:
    return InputCall(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4o",
        prompt_text=_PROMPT,
        stream=False,
    )


# ── 1. Direct-mode masking via the remote engine ─────────────────────


def test_direct_mode_masks_name_and_address_via_remote(
    monkeypatch: pytest.MonkeyPatch, control_plane: _ControlPlaneState
) -> None:
    _wire_onprem(monkeypatch, control_plane)

    masked, records = pii_mod.sanitize(_PROMPT)

    # The name AND the address were masked by the REMOTE control plane.
    assert _NAME not in masked
    assert _ADDRESS not in masked
    types = {r.type for r in records}
    assert "person_name" in types
    assert "address" in types
    # Proof the offload actually happened: the stub saw the raw text and
    # the Egis API key rode the request.
    assert control_plane.saw_raw_text is True
    assert control_plane.saw_auth[-1] == f"Bearer {_API_KEY}"


def test_evaluate_sanitizes_when_control_plane_healthy(
    monkeypatch: pytest.MonkeyPatch, control_plane: _ControlPlaneState
) -> None:
    _wire_onprem(monkeypatch, control_plane)
    _policy_cache.replace_rules('"v1"', [_pii_rule()])
    _policy_cache.mark_synced()

    decision = evaluate(_input_call())

    # A healthy remote tier → the pii_scan rule fires a sanitize verdict,
    # NOT a fail-closed block.
    assert decision.verdict == "sanitize"
    assert decision.reason_code != "pii_engine_unavailable"


# ── 2. NER stopped → the call is refused ─────────────────────────────


def test_evaluate_fails_closed_when_ner_stopped(
    monkeypatch: pytest.MonkeyPatch, control_plane: _ControlPlaneState
) -> None:
    _wire_onprem(monkeypatch, control_plane)
    _policy_cache.replace_rules('"v1"', [_pii_rule()])
    _policy_cache.mark_synced()

    control_plane.ner_up = False  # stop the NER tier

    decision = evaluate(_input_call())

    assert decision.verdict == "block"
    assert decision.reason_code == "pii_engine_unavailable"


def test_regex_only_optout_allows_when_ner_stopped(
    monkeypatch: pytest.MonkeyPatch, control_plane: _ControlPlaneState
) -> None:
    """The documented escape hatch still works: regex-only opt-in does
    NOT fail closed even when the remote tier is down."""
    _wire_onprem(monkeypatch, control_plane)
    monkeypatch.setenv("EGISAI_ONPREM_PII_REGEX_ONLY", "1")
    _policy_cache.replace_rules('"v1"', [_pii_rule()])
    _policy_cache.mark_synced()

    control_plane.ner_up = False

    decision = evaluate(_input_call())

    assert decision.reason_code != "pii_engine_unavailable"


# ── 3. Zero outbound connections outside the customer network ────────


def test_zero_outbound_outside_customer_network(
    monkeypatch: pytest.MonkeyPatch, control_plane: _ControlPlaneState
) -> None:
    _wire_onprem(monkeypatch, control_plane)

    seen_hosts: list[str] = []
    real_connect = socket.socket.connect

    def _guarded_connect(self: socket.socket, address, *a, **k):  # noqa: ANN001
        host = address[0] if isinstance(address, tuple) else str(address)
        seen_hosts.append(host)
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(
                f"on-prem SDK attempted an outbound connection to {host}"
            )
        return real_connect(self, address, *a, **k)

    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)

    # A full detect + mask round trip.
    pii_mod.scan(_PROMPT)
    masked, _ = pii_mod.sanitize(_PROMPT)

    assert _NAME not in masked
    # Every socket we opened went to the loopback control plane only.
    assert seen_hosts  # we did make at least one connection
    assert all(h in ("127.0.0.1", "::1", "localhost") for h in seen_hosts)
