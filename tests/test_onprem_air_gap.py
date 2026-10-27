"""On-prem SDK must NEVER reach the internet (gap #35).

The customer's app process runs the on-prem SDK inside a sealed network.
On a fresh install the SDK used to download the spaCy ``en_core_web_lg``
model (~400–750 MB) from github.com, and the optional Ettin ONNX engine
pulled weights from huggingface.co. Both are forbidden on-prem.

These tests prove:
  * on-prem model loaders raise instead of downloading;
  * no outbound socket is opened while attempting a load;
  * a remote NER engine short-circuits local loading entirely;
  * with no NER source available, on-prem fails CLOSED on pii_scan
    (rather than silently degrading to regex-only), with a documented
    opt-out.
"""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

import egisai
from egisai import _evaluator
from egisai.policy import _onnx_ner, _pii_loader
from egisai.policy import pii as pii_mod
from egisai.policy.engine import PolicyRule


@pytest.fixture(autouse=True)
def _reset_state():
    _pii_loader.reset_for_tests()
    _evaluator._reset_warmup_gate_for_tests()
    pii_mod.set_remote_engine(None)
    yield
    _pii_loader.reset_for_tests()
    _evaluator._reset_warmup_gate_for_tests()
    pii_mod.set_remote_engine(None)


def _pii_rule() -> PolicyRule:
    return PolicyRule(
        id="pii-test",
        name="PII test",
        type="pii_scan",
        tenant=None,
        config={"types": ["person_name"], "action": "sanitize"},
        agent_ids=(),
        phase="both",
    )


# ── Never download on-prem ──────────────────────────────────────────


def test_onprem_spacy_loader_refuses_to_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    # Force the "model missing" branch and fail loudly if download is tried.
    with patch("spacy.util.is_package", return_value=False), patch(
        "spacy.cli.download.download",
        side_effect=AssertionError("on-prem must not download the spaCy model"),
    ):
        with pytest.raises(RuntimeError, match="will not download the spaCy model"):
            _pii_loader._ensure_spacy_model_present(quiet=True)


def test_onprem_onnx_loader_refuses_to_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    # Empty dir ⇒ all model files "missing". httpx must never be touched.
    with patch("httpx.stream", side_effect=AssertionError("on-prem must not fetch")):
        with pytest.raises(RuntimeError, match="will not download NER model files"):
            _onnx_ner.ensure_model_files(tmp_path)


def test_onprem_model_load_opens_no_socket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hard air-gap proof: attempting a load opens zero outbound sockets."""
    monkeypatch.setattr(egisai, "__onprem_build__", True)

    def _blocked_connect(self, *a, **k):  # noqa: ANN001, ANN002, ANN003
        raise AssertionError("on-prem SDK attempted an outbound connection")

    with patch("spacy.util.is_package", return_value=False), patch.object(
        socket.socket, "connect", _blocked_connect
    ):
        with pytest.raises(RuntimeError):
            _pii_loader._ensure_spacy_model_present(quiet=True)


# ── Remote engine short-circuits local load ─────────────────────────


class _FakeRemoteEngine:
    def scan(self, text):  # noqa: ANN001, ANN201
        return []

    def sanitize(self, text, types=None, mask_char="#"):  # noqa: ANN001, ANN201
        return text, []

    def label_redact(self, text, types=None):  # noqa: ANN001, ANN201
        return text


def test_remote_engine_skips_local_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    pii_mod.set_remote_engine(_FakeRemoteEngine())

    # If the loader tried to build a local analyzer it would call this;
    # the remote short-circuit must prevent that entirely.
    with patch.object(
        _pii_loader,
        "_build_analyzer",
        side_effect=AssertionError("must not load a local model when remote is set"),
    ):
        _pii_loader.prime_analyzer_async(quiet=True)

    assert _pii_loader.is_settled() is True
    assert _pii_loader.try_get_analyzer() is None


# ── Fail closed when no NER source available ────────────────────────


def _simulate_ner_permanently_failed() -> None:
    """Drive the loader into the settled-with-error state (model missing)."""
    _pii_loader._state.analyzer = None
    _pii_loader._state.settled = True
    _pii_loader._state.loading = False
    _pii_loader._state.error = RuntimeError("on-prem: spaCy model unavailable")
    _pii_loader._state.settle_event.set()


def test_onprem_fails_closed_when_ner_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    monkeypatch.delenv("EGISAI_ONPREM_PII_REGEX_ONLY", raising=False)
    _simulate_ner_permanently_failed()

    assert _evaluator._onprem_pii_fail_closed([_pii_rule()]) is True
    decision = _evaluator._pii_engine_unavailable_decision()
    assert decision.verdict == "block"
    assert decision.reason_code == "pii_engine_unavailable"


def test_onprem_regex_only_optout_disables_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    monkeypatch.setenv("EGISAI_ONPREM_PII_REGEX_ONLY", "1")
    _simulate_ner_permanently_failed()

    assert _evaluator._onprem_pii_fail_closed([_pii_rule()]) is False


def test_onprem_remote_engine_disables_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    monkeypatch.delenv("EGISAI_ONPREM_PII_REGEX_ONLY", raising=False)
    _simulate_ner_permanently_failed()
    pii_mod.set_remote_engine(_FakeRemoteEngine())

    assert _evaluator._onprem_pii_fail_closed([_pii_rule()]) is False


def test_cloud_never_fails_closed_on_ner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cloud keeps its fail-open regex fallback — unchanged by gap #35."""
    monkeypatch.setattr(egisai, "__onprem_build__", False)
    _simulate_ner_permanently_failed()

    assert _evaluator._onprem_pii_fail_closed([_pii_rule()]) is False


def test_onprem_no_pii_rule_does_not_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-closed only applies when a pii_scan rule is actually in scope."""
    monkeypatch.setattr(egisai, "__onprem_build__", True)
    _simulate_ner_permanently_failed()

    semantic = PolicyRule(
        id="sem",
        name="sem",
        type="semantic_guard",
        tenant=None,
        config={"intents": ["x"]},
        agent_ids=(),
        phase="both",
    )
    assert _evaluator._onprem_pii_fail_closed([semantic]) is False
