"""Unit tests for the European national-identifier checksum validators.

These run without Presidio (the validators are pure Python), so they
stay fast and also guard the air-gapped / warm-up code path. Valid
vectors are real public test numbers or checksum-consistent values;
invalid vectors flip exactly one digit so a broken checksum is caught.
"""
from __future__ import annotations

import pytest

from egisai.policy import _pii_eu as eu

# (validator, valid-sample, invalid-sample)
_CASES = [
    (eu.is_valid_italian_codice_fiscale, "RSSMRA85T10A562S", "RSSMRA85T10A562T"),
    (eu.is_valid_italian_tessera_sanitaria, "80380800301234567895", "12345678901234567890"),
    (eu.is_valid_swedish_personnummer, "811218-9876", "811218-9875"),
    (eu.is_valid_dutch_bsn, "111222333", "111222334"),
    (eu.is_valid_french_nir, "2550833100005" + "33", "2550833100005" + "34"),
    (eu.is_valid_german_tax_id, "86095742719", "86095742718"),
    (eu.is_valid_belgian_rrn, "93051822361", "93051822362"),
    (eu.is_valid_finnish_hetu, "131052-308T", "131052-308U"),
    (eu.is_valid_polish_pesel, "44051401359", "44051401358"),
    (eu.is_valid_polish_nip, "8567346215", "8567346216"),
    (eu.is_valid_portuguese_nif, "501442600", "501442601"),
    (eu.is_valid_greek_amka, "01019000007", "01019000008"),
    (eu.is_valid_greek_afm, "090000045", "090000046"),
    (eu.is_valid_norwegian_fnr, "01019000083", "01019000084"),
    (eu.is_valid_danish_cpr, "010190-1234", "320190-1234"),
    (eu.is_valid_irish_pps, "1234567T", "1234567A"),
    (eu.is_valid_austrian_svnr, "1237 010180", "1230 010180"),
    (eu.is_valid_swiss_ahv, "7569217076985", "7569217076984"),
]


@pytest.mark.parametrize("fn,good,bad", _CASES, ids=[c[0].__name__ for c in _CASES])
def test_valid_accepts_invalid_rejects(fn, good, bad) -> None:  # type: ignore[no-untyped-def]
    assert fn(good) is True, f"{fn.__name__} rejected a valid value {good!r}"
    assert fn(bad) is False, f"{fn.__name__} accepted an invalid value {bad!r}"


@pytest.mark.parametrize("fn", [c[0] for c in _CASES], ids=[c[0].__name__ for c in _CASES])
def test_empty_and_garbage_are_rejected(fn) -> None:  # type: ignore[no-untyped-def]
    assert fn("") is False
    assert fn("not an id") is False


def test_formatting_variants_validate_identically() -> None:
    # Spaces, dots and hyphens are normalised away before the checksum.
    assert eu.is_valid_swiss_ahv("756.9217.0769.85")
    assert eu.is_valid_swiss_ahv("756 9217 0769 85")
    assert eu.is_valid_belgian_rrn("93.05.18-223.61")
    assert eu.is_valid_german_tax_id("86 095 742 719")
    assert eu.is_valid_swedish_personnummer("198112189876")  # 12-digit form


def test_french_nir_handles_corsica_departments() -> None:
    # 2A -> 19 (-1,000,000), 2B -> 18 (-2,000,000) per INSEE.
    body = "185012A123456"
    key = 97 - ((int(body.replace("2A", "19")) - 1_000_000) % 97)
    assert eu.is_valid_french_nir(body + f"{key:02d}")


def test_swedish_coordination_number_day_offset() -> None:
    # samordningsnummer adds 60 to the day field (DD 18 -> 78); the
    # nine significant digits are still Luhn-checked.
    from egisai.policy._pii_helpers import luhn_check

    core = "811278987"  # YY=81 MM=12 DD=78 (18+60) + individual 987
    check = next(c for c in range(10) if luhn_check(core + str(c)))
    assert eu.is_valid_swedish_personnummer(core + str(check))
