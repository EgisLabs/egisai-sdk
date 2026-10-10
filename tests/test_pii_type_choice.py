"""Which detector keeps the label when several claim the same digits.

The customer-visible failures this locks:

* "my phone number is …" labeled as a UK health number or a bank account
* "my credit card is …" labeled as a phone, or a real 2-series Mastercard
  labeled as a bank account
* a weaker hit winning only because it starts one character earlier
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from egisai.policy import pii as pii_mod
from egisai.policy._pii_recognizers import MastercardRangeRecognizer


def _hit(
    entity: str,
    start: int,
    end: int,
    score: float,
    context: str = "",
) -> SimpleNamespace:
    return SimpleNamespace(
        entity_type=entity,
        start=start,
        end=end,
        score=score,
        analysis_explanation=SimpleNamespace(supportive_context_word=context),
    )


def _label(text: str, results: list[SimpleNamespace]) -> str:
    masked, _records = pii_mod._apply_results(
        text,
        results,
        type_filter=None,
        mask_char="#",
        label_mode=True,
    )
    return masked


def test_phone_words_beat_health_number_checksum() -> None:
    text = "My phone number is 4155552671."
    start = text.index("4155552671")
    end = start + len("4155552671")
    labeled = _label(
        text,
        [
            _hit("UK_NHS", start, end, 1.0),
            _hit("PHONE_NUMBER", start, end, 0.75, context="phone"),
            _hit("US_BANK_NUMBER", start, end, 0.05),
        ],
    )
    assert labeled == "My phone number is <PHONE>."


def test_card_checksum_beats_account_word() -> None:
    """A real card next to the word account stays a card."""
    text = "account 4111111111111111"
    start = text.index("4111111111111111")
    end = start + 16
    labeled = _label(
        text,
        [
            _hit("CREDIT_CARD", start, end, 1.0),
            _hit("US_BANK_NUMBER", start, end, 0.4, context="account"),
        ],
    )
    assert labeled == "account <CREDIT_CARD>"


def test_failed_card_check_is_not_invented_from_the_sentence() -> None:
    text = "my credit card is 5551234567."
    start = text.index("5551234567")
    end = start + 10
    labeled = _label(
        text,
        [
            _hit("PHONE_NUMBER", start, end, 0.4),
            _hit("US_BANK_NUMBER", start, end, 0.05),
        ],
    )
    assert labeled == "my credit card is <PHONE>."


def test_earlier_weaker_span_does_not_win_or_leak_a_digit() -> None:
    text = "94155552671"
    labeled = _label(
        text,
        [
            _hit("US_BANK_NUMBER", 0, 11, 0.05),
            _hit("PHONE_NUMBER", 1, 11, 0.75, context="phone"),
        ],
    )
    assert labeled == "<PHONE>"
    assert "9" not in labeled


def test_bare_digit_run_is_hidden_but_not_called_a_bank_account() -> None:
    text = "My phone number is 4111111111."
    start = text.index("4111111111")
    hits = [_hit("US_BANK_NUMBER", start, start + 10, 0.05)]
    labeled = _label(text, hits)
    assert "4111111111" not in labeled
    assert "<BANK_ACCOUNT>" not in labeled
    assert "##########" in labeled

    # The policy path still hears about the hit, so the digits are
    # masked on the way to the model and not only in the preview.
    masked, records = pii_mod._apply_results(
        text, hits, type_filter=None, mask_char="#", label_mode=False
    )
    assert "4111111111" not in masked
    assert [record.type for record in records] == ["bank_account"]
    decisions = pii_mod._resolve_detections(text, hits, None)
    assert [(d.op_type, d.confirmed) for d in decisions] == [("bank_account", False)]


def test_account_words_keep_the_bank_label() -> None:
    text = "account number 123456789012"
    start = text.index("123456789012")
    labeled = _label(
        text,
        [_hit("US_BANK_NUMBER", start, start + 12, 0.4, context="account")],
    )
    assert labeled == "account number <BANK_ACCOUNT>"


def test_separate_values_are_both_kept() -> None:
    text = "bob@acme.com and 4155552671"
    email = _hit("EMAIL_ADDRESS", 0, 12, 0.8)
    phone_at = text.index("4155552671")
    phone = _hit(
        "PHONE_NUMBER",
        phone_at,
        phone_at + 10,
        0.75,
        context="phone",
    )
    labeled = _label(text, [email, phone])
    assert labeled == "<EMAIL> and <PHONE>"


def test_scan_reports_the_same_winner_as_the_label() -> None:
    text = "My phone number is 4155552671."
    start = text.index("4155552671")
    end = start + 10
    results = [
        _hit("UK_NHS", start, end, 1.0),
        _hit("PHONE_NUMBER", start, end, 0.75, context="phone"),
    ]
    decisions = pii_mod._resolve_detections(text, results, None)
    assert [d.op_type for d in decisions] == ["phone"]
    assert _label(text, results) == "My phone number is <PHONE>."


def test_two_series_mastercard_passes_the_card_check() -> None:
    assert pii_mod._is_card_number("2223000048400011")
    assert pii_mod._is_card_number("4111111111111111")
    assert not pii_mod._is_card_number("4111111111111112")
    assert not pii_mod._is_card_number("2000000000000000")


def test_fallback_labels_two_series_mastercard_as_a_card() -> None:
    labeled = pii_mod._label_redact_with_fallback(
        "my credit card is 2223000048400011.",
        type_filter=None,
    )
    assert labeled == "my credit card is <CREDIT_CARD>."


def test_recognizer_rejects_a_two_prefix_that_is_not_a_card() -> None:
    recognizer = MastercardRangeRecognizer()
    assert recognizer.validate_result("2223000048400011")
    assert not recognizer.validate_result("4111111111111111")
    assert not recognizer.validate_result("2223000048400012")


@pytest.fixture(scope="module")
def analyzer():  # type: ignore[no-untyped-def]
    from presidio_analyzer import AnalyzerEngine

    from egisai.policy._fast_nlp import FastBlankNlpEngine
    from egisai.policy._pii_recognizers import register_custom_recognizers

    nlp = FastBlankNlpEngine()
    nlp.load()
    engine = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
    register_custom_recognizers(engine.registry)
    return engine


def _live(analyzer, text: str) -> str:  # type: ignore[no-untyped-def]
    return pii_mod._label_redact_with_presidio(text, analyzer, None)


def test_live_phone_sentence_is_a_phone(analyzer) -> None:  # type: ignore[no-untyped-def]
    labeled = _live(analyzer, "My phone number is 4155552671. can you repeat it?")
    assert labeled.startswith("My phone number is <PHONE>.")
    assert "4155552671" not in labeled
    assert "<BANK_ACCOUNT>" not in labeled
    assert "<MEDICAL_LICENSE>" not in labeled


def test_live_visa_is_a_card(analyzer) -> None:  # type: ignore[no-untyped-def]
    labeled = _live(
        analyzer,
        "my credit card is 4111111111111111. can you send it to the customer.",
    )
    assert "my credit card is <CREDIT_CARD>." in labeled
    assert "<PHONE>" not in labeled
    assert "<BANK_ACCOUNT>" not in labeled


def test_live_two_series_mastercard_is_a_card(analyzer) -> None:  # type: ignore[no-untyped-def]
    labeled = _live(
        analyzer,
        "my credit card is 2223000048400011. can you send it to the customer.",
    )
    assert "my credit card is <CREDIT_CARD>." in labeled
    assert "<BANK_ACCOUNT>" not in labeled


def test_live_unconfirmed_digits_stay_hidden(analyzer) -> None:  # type: ignore[no-untyped-def]
    labeled = _live(analyzer, "My phone number is 4111111111. can you repeat it?")
    assert "4111111111" not in labeled
    assert "<BANK_ACCOUNT>" not in labeled
