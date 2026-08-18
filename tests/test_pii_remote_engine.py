"""The optional remote PII engine hook.

A trusted caller (our control plane) can offload the heavy Presidio +
spaCy pass to an in-network service via ``set_remote_engine``. These
tests pin the contract that makes that safe to ship:

* when no engine is installed, detection runs locally (the default
  every ``pip install`` user gets, untouched);
* when one is installed, ``scan`` / ``sanitize`` / ``label_redact``
  delegate to it;
* custom operator patterns still run locally on top of the remote
  result — the worker never sees them, so offloading must never drop a
  customer's own ``custom:*`` types;
* the engine fails **open**: if it raises, the SDK falls back to the
  local engine rather than erroring on the user's call path.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from egisai.policy import _pii_custom
from egisai.policy import pii as pii_scanner
from egisai.policy.pii import PIIFinding, Sanitization


@pytest.fixture(autouse=True)
def _clean_state():
    # The remote engine and the custom-pattern set are both process
    # globals; leaking either across tests would make failures depend
    # on ordering.
    pii_scanner.set_remote_engine(None)
    _pii_custom.reset_for_tests()
    yield
    pii_scanner.set_remote_engine(None)
    _pii_custom.reset_for_tests()


def _rule(*patterns: dict[str, object]):
    return SimpleNamespace(
        type="pii_scan", config={"custom_types": list(patterns)}
    )


class _FakeEngine:
    """Records calls and returns canned built-in results."""

    def __init__(self) -> None:
        self.scan_calls: list[str] = []
        self.sanitize_calls: list[tuple[str, list[str] | None, str]] = []
        self.label_calls: list[tuple[str, list[str] | None]] = []

    def scan(self, text: str) -> list[PIIFinding]:
        self.scan_calls.append(text)
        return [
            PIIFinding(
                type="email",
                value_redacted="a***@x.com",
                confidence=0.99,
                method="remote",
            )
        ]

    def sanitize(self, text, types=None, mask_char="#"):
        self.sanitize_calls.append((text, types, mask_char))
        return "masked-by-worker", [
            Sanitization(type="email", count=1, pattern="####")
        ]

    def label_redact(self, text, types=None):
        self.label_calls.append((text, types))
        return "labelled-by-worker <EMAIL>"


class _BrokenEngine:
    """Every method raises — exercises the fail-open fallback."""

    def scan(self, text: str) -> list[PIIFinding]:
        raise RuntimeError("worker down")

    def sanitize(self, text, types=None, mask_char="#"):
        raise RuntimeError("worker down")

    def label_redact(self, text, types=None):
        raise RuntimeError("worker down")


class TestDelegation:
    def test_scan_delegates_to_the_installed_engine(self) -> None:
        engine = _FakeEngine()
        pii_scanner.set_remote_engine(engine)

        findings = pii_scanner.scan("email me at alice@example.com")

        assert engine.scan_calls == ["email me at alice@example.com"]
        assert [f.type for f in findings] == ["email"]
        assert findings[0].method == "remote"

    def test_sanitize_delegates_and_returns_worker_masked_text(self) -> None:
        engine = _FakeEngine()
        pii_scanner.set_remote_engine(engine)

        masked, records = pii_scanner.sanitize("alice@example.com")

        assert masked == "masked-by-worker"
        assert [r.type for r in records] == ["email"]
        assert engine.sanitize_calls and engine.sanitize_calls[0][0] == (
            "alice@example.com"
        )

    def test_label_redact_delegates(self) -> None:
        engine = _FakeEngine()
        pii_scanner.set_remote_engine(engine)

        assert (
            pii_scanner.label_redact("alice@example.com")
            == "labelled-by-worker <EMAIL>"
        )


class TestCustomPatternsStayLocal:
    def test_scan_overlays_local_custom_patterns_on_remote_result(
        self,
    ) -> None:
        # The worker returns only its built-in "email"; the custom
        # employee-id pattern is registered in THIS process and must
        # still be detected.
        _pii_custom.notify_rules(
            [_rule({"id": "employee_id", "label": "Emp", "pattern": r"EMP-\d{4}"})]
        )
        pii_scanner.set_remote_engine(_FakeEngine())

        findings = pii_scanner.scan("ping alice@example.com about EMP-1234")

        types = {f.type for f in findings}
        assert "email" in types  # from the (fake) worker
        assert "custom:employee_id" in types  # applied locally

    def test_sanitize_overlays_local_custom_masking(self) -> None:
        _pii_custom.notify_rules(
            [_rule({"id": "employee_id", "label": "Emp", "pattern": r"EMP-\d{4}"})]
        )

        class _EchoEngine:
            # Returns the input unmasked so we can prove the custom
            # overlay is what masks EMP-1234 locally.
            def scan(self, text):  # pragma: no cover - unused here
                return []

            def sanitize(self, text, types=None, mask_char="#"):
                return text, []

            def label_redact(self, text, types=None):  # pragma: no cover
                return text

        pii_scanner.set_remote_engine(_EchoEngine())

        masked, records = pii_scanner.sanitize("badge EMP-1234 here")

        assert "EMP-1234" not in masked
        assert any(r.type == "custom:employee_id" for r in records)


class TestFailOpen:
    def test_scan_falls_back_to_local_when_engine_raises(self) -> None:
        pii_scanner.set_remote_engine(_BrokenEngine())

        # Must not raise; local engine handles it. A credit card is
        # caught by the regex+checksum fallback even when the NER model
        # isn't loaded (an email on a reserved test domain would be
        # ignored by design, so use a checksum-backed type here).
        findings = pii_scanner.scan("pay with 4111 1111 1111 1111")

        assert any(f.type == "credit_card" for f in findings)

    def test_sanitize_falls_back_and_never_returns_raw_pii(self) -> None:
        pii_scanner.set_remote_engine(_BrokenEngine())

        masked, _ = pii_scanner.sanitize("card 4111 1111 1111 1111")

        # The local fallback still masks — a broken worker degrades
        # detection, it never leaks the raw value through.
        assert "4111 1111 1111 1111" not in masked

    def test_label_redact_falls_back_when_engine_raises(self) -> None:
        pii_scanner.set_remote_engine(_BrokenEngine())

        # Should not raise; returns a locally-redacted string.
        out = pii_scanner.label_redact("pay with 4111 1111 1111 1111")
        assert "4111 1111 1111 1111" not in out


def test_clearing_the_engine_restores_local_behavior() -> None:
    engine = _FakeEngine()
    pii_scanner.set_remote_engine(engine)
    pii_scanner.set_remote_engine(None)

    pii_scanner.scan("alice@example.com")

    assert engine.scan_calls == []  # never called after clear
    assert pii_scanner.get_remote_engine() is None
