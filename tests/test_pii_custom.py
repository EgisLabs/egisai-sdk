"""Operator-defined PII patterns.

The canonical taxonomy covers shapes that are the same everywhere. The
shapes that actually leak are often local — an employee number, an
internal case id — and no shipped detector can know them. These tests
pin the contract for the escape hatch: it runs beside the built-ins,
it obeys the same audit rules, and a bad pattern degrades to nothing
rather than taking the engine down with it.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from egisai.policy import _pii_custom
from egisai.policy import pii as pii_scanner
from egisai.policy.engine import PolicyContext, PolicyRule, evaluate_policies


def _ctx(prompt_text: str) -> PolicyContext:
    return PolicyContext(
        tenant=None,
        model="gpt-4o",
        prompt_text=prompt_text,
        prompt_chars=len(prompt_text),
        stream=False,
    )


@pytest.fixture(autouse=True)
def _clean_registry():
    """The active set is process-global, like the policy cache it
    follows. Leaking one test's patterns into the next would make
    failures depend on ordering."""
    _pii_custom.reset_for_tests()
    yield
    _pii_custom.reset_for_tests()


def _rule(*patterns: dict[str, object], types: list[str] | None = None):
    config: dict[str, object] = {"custom_types": list(patterns)}
    if types is not None:
        config["types"] = types
    return SimpleNamespace(type="pii_scan", config=config)


EMPLOYEE = {"id": "employee_id", "label": "Employee ID", "pattern": r"EMP-\d{6}"}


class TestRegistration:
    def test_a_pattern_from_a_policy_becomes_active(self) -> None:
        _pii_custom.notify_rules([_rule(EMPLOYEE)])

        active = _pii_custom.active()
        assert [p.type_id for p in active] == ["custom:employee_id"]
        assert active[0].label == "Employee ID"

    def test_the_id_is_namespaced_so_it_cannot_shadow_a_real_type(
        self,
    ) -> None:
        """An operator naming their pattern "ssn" must not be able to
        override the checksum-backed detector of the same name."""
        _pii_custom.notify_rules(
            [_rule({"id": "ssn", "label": "Our SSN", "pattern": r"X\d{4}"})]
        )

        assert [p.type_id for p in _pii_custom.active()] == ["custom:ssn"]

    def test_an_id_is_derived_from_the_label_when_absent(self) -> None:
        _pii_custom.notify_rules(
            [_rule({"label": "Case Number!", "pattern": r"C-\d+"})]
        )

        assert _pii_custom.active()[0].type_id == "custom:case_number"

    def test_a_dangerous_pattern_is_dropped_not_raised(self) -> None:
        """One catastrophic regex must not disable PII detection for
        every other type the operator configured."""
        _pii_custom.notify_rules(
            [
                _rule(
                    {"label": "Bad", "pattern": r"(a+)+$"},
                    EMPLOYEE,
                )
            ]
        )

        assert [p.type_id for p in _pii_custom.active()] == [
            "custom:employee_id"
        ]

    def test_an_invalid_regex_is_dropped_not_raised(self) -> None:
        _pii_custom.notify_rules(
            [_rule({"label": "Broken", "pattern": "[unclosed"}, EMPLOYEE)]
        )

        assert [p.type_id for p in _pii_custom.active()] == [
            "custom:employee_id"
        ]

    def test_rules_that_are_not_pii_scan_are_ignored(self) -> None:
        _pii_custom.notify_rules(
            [SimpleNamespace(type="deny_regex", config={"custom_types": [EMPLOYEE]})]
        )

        assert _pii_custom.active() == ()


class TestDetection:
    def test_a_custom_shape_is_found(self) -> None:
        _pii_custom.notify_rules([_rule(EMPLOYEE)])

        findings = pii_scanner.scan("ticket raised by EMP-004182 today")

        assert [f.type for f in findings] == ["custom:employee_id"]
        assert findings[0].method == "custom_pattern"

    def test_a_finding_never_carries_the_matched_value(self) -> None:
        """security-and-compliance.mdc rule 1: the raw value must not
        survive onto anything that gets logged."""
        _pii_custom.notify_rules([_rule(EMPLOYEE)])

        findings = pii_scanner.scan("EMP-004182")

        assert "004182" not in repr(findings)

    def test_built_in_detection_still_runs_alongside(self) -> None:
        """A custom pattern adds a shape; it must never narrow what
        Egis looks for."""
        _pii_custom.notify_rules([_rule(EMPLOYEE)])

        findings = pii_scanner.scan("EMP-004182 ssn 123-45-6789")
        types = {f.type for f in findings}

        assert "custom:employee_id" in types
        assert "ssn" in types


class TestSanitize:
    def test_it_masks_and_preserves_shape(self) -> None:
        _pii_custom.notify_rules([_rule(EMPLOYEE)])

        masked, records = pii_scanner.sanitize("ref EMP-004182 ok")

        assert masked == "ref ###-###### ok"
        assert [r.type for r in records] == ["custom:employee_id"]
        assert records[0].count == 1

    def test_the_audit_record_carries_no_original_value(self) -> None:
        _pii_custom.notify_rules([_rule(EMPLOYEE)])

        _, records = pii_scanner.sanitize("EMP-004182")

        assert "004182" not in repr(records)
        assert records[0].pattern == "###-######"

    def test_repeats_are_counted(self) -> None:
        _pii_custom.notify_rules([_rule(EMPLOYEE)])

        _, records = pii_scanner.sanitize("EMP-004182 and EMP-119900")

        assert records[0].count == 2

    def test_a_type_filter_excludes_it(self) -> None:
        """A rule scoped to ``ssn`` must not mask more than the
        operator asked for."""
        _pii_custom.notify_rules([_rule(EMPLOYEE)])

        masked, _ = pii_scanner.sanitize(
            "EMP-004182 ssn 123-45-6789", types=["ssn"]
        )

        assert "EMP-004182" in masked

    def test_a_type_filter_including_it_masks_it(self) -> None:
        _pii_custom.notify_rules([_rule(EMPLOYEE)])

        masked, _ = pii_scanner.sanitize(
            "EMP-004182", types=["custom:employee_id"]
        )

        assert masked == "###-######"

    def test_a_custom_id_is_not_reported_as_an_unknown_type(
        self, caplog
    ) -> None:
        _pii_custom.notify_rules([_rule(EMPLOYEE)])

        with caplog.at_level("WARNING"):
            pii_scanner.sanitize("EMP-004182", types=["custom:employee_id"])

        assert "unknown PII types" not in caplog.text

    def test_label_redact_uses_the_operator_name(self) -> None:
        _pii_custom.notify_rules([_rule(EMPLOYEE)])

        assert (
            pii_scanner.label_redact("ref EMP-004182") == "ref <EMPLOYEE_ID>"
        )


class TestThroughTheEngine:
    """The part that decides whether any of the above matters: a policy
    carrying a custom pattern has to produce a sanitize decision whose
    ``sanitize_types`` the downstream masking call sites can resolve."""

    @staticmethod
    def _policy() -> PolicyRule:
        return PolicyRule(
            id="1",
            name="Internal identifiers",
            type="pii_scan",
            tenant=None,
            config={
                "action": "sanitize",
                "types": ["custom:employee_id"],
                "custom_types": [EMPLOYEE],
            },
        )

    def test_a_custom_match_sanitizes(self) -> None:
        rule = self._policy()
        _pii_custom.notify_rules([rule])

        decision = evaluate_policies(
            [rule],
            _ctx("raised by EMP-004182"),
        )

        assert decision.verdict == "sanitize"
        assert decision.sanitize_types == ["custom:employee_id"]

    def test_the_message_names_it_the_way_the_operator_did(self) -> None:
        """``custom:employee_id`` is the wire id. Nobody should have to
        read that in a policy message."""
        rule = self._policy()
        _pii_custom.notify_rules([rule])

        decision = evaluate_policies(
            [rule],
            _ctx("raised by EMP-004182"),
        )

        assert "Employee ID" in (decision.message or "")
        assert "custom:" not in (decision.message or "")

    def test_the_decision_masks_what_it_says_it_masked(self) -> None:
        """The end-to-end invariant. The decision travels to a call
        site that knows nothing about policies and masks by type id
        alone; if that lookup missed, Egis would report a sanitize it
        never performed."""
        rule = self._policy()
        _pii_custom.notify_rules([rule])
        decision = evaluate_policies(
            [rule],
            _ctx("raised by EMP-004182"),
        )

        masked, records = pii_scanner.sanitize(
            "raised by EMP-004182",
            types=decision.sanitize_types,
            mask_char=decision.sanitize_mask_char,
        )

        assert "EMP-004182" not in masked
        assert records and records[0].type == "custom:employee_id"

    def test_no_match_leaves_the_call_alone(self) -> None:
        rule = self._policy()
        _pii_custom.notify_rules([rule])

        decision = evaluate_policies(
            [rule],
            _ctx("nothing to see"),
        )

        assert decision.verdict == "allow"
