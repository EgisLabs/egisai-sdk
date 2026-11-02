"""Live-analyzer tests for the European national-identifier recognizers.

Builds a Presidio analyzer on the fast blank NLP engine (no 750 MB
spaCy download) with our custom recognizers registered — the same
fixture shape ``test_pii_type_choice.py`` uses — and exercises the
end-to-end path: regex match -> national checksum -> context gate ->
operator-facing label.
"""
from __future__ import annotations

import pytest

from egisai.policy import _pii_taxonomy as tax
from egisai.policy import pii as pii_mod
from egisai.policy._pii_recognizers import EU_ID_ENTITIES


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


def _label(analyzer, text: str, types: list[str] | None) -> str:  # type: ignore[no-untyped-def]
    return pii_mod._label_redact_with_presidio(text, analyzer, types)


# One row per new type: (operator type id, a prompt that should mask).
# The prompt carries a country keyword so the context-gated numeric
# schemes fire too — distinctive formats don't need it but it's
# harmless to include.
_MASK_CASES = [
    ("it_codice_fiscale", "codice fiscale RSSMRA85T10A562S"),
    ("it_tessera_sanitaria", "tessera sanitaria 80380800301234567895"),
    ("it_passport", "passport YA1234567"),
    ("it_id_card", "carta d'identità CA00000AA"),
    ("fr_nir", "sécurité sociale 2 55 08 33 100 005 33"),
    ("de_tax_id", "Steuer-ID 86 095 742 719"),
    ("nl_bsn", "BSN 111222333"),
    ("be_rrn", "rijksregister 93.05.18-223.61"),
    ("pt_nif", "NIF 501442600"),
    ("pl_pesel", "PESEL 44051401359"),
    ("pl_nip", "NIP 856-734-62-15"),
    ("gr_amka", "AMKA 01019000007"),
    ("gr_afm", "AFM 090000045"),
    ("se_personnummer", "personnummer 811218-9876"),
    ("no_fodselsnummer", "fødselsnummer 01019000083"),
    ("dk_cpr", "CPR 010190-1234"),
    ("fi_hetu", "HETU 131052-308T"),
    ("at_svnr", "Sozialversicherung 1237 010180"),
    ("ch_ahv", "AHV 756.9217.0769.85"),
    ("ie_pps", "PPS 1234567T"),
]


@pytest.mark.parametrize("type_id,prompt", _MASK_CASES, ids=[c[0] for c in _MASK_CASES])
def test_each_type_masks_when_selected(analyzer, type_id, prompt) -> None:  # type: ignore[no-untyped-def]
    labeled = _label(analyzer, prompt, [type_id])
    assert f"<{type_id.upper()}>" in labeled, labeled
    # The raw identifier must be gone from the output.
    identifier = prompt.split(maxsplit=1)[1] if " " in prompt else prompt
    tail = identifier.rsplit(" ", 1)[-1]
    assert tail not in labeled, labeled


def test_weak_checksum_type_needs_context(analyzer) -> None:
    # A checksum-valid Dutch BSN with no BSN/Netherlands context must
    # NOT be labelled a BSN — otherwise every 9-digit number that
    # happens to pass the 11-test gets masked in a scan.
    labeled = _label(analyzer, "reference number 111222333 approved", ["nl_bsn"])
    assert "<NL_BSN>" not in labeled
    assert "111222333" in labeled


def test_distinctive_type_fires_without_context(analyzer) -> None:
    # The Swiss 756 prefix + EAN-13 checksum is distinctive enough to
    # trust with no surrounding keyword.
    labeled = _label(analyzer, "transfer to 756.9217.0769.85 today", ["ch_ahv"])
    assert "<CH_AHV>" in labeled


def test_failed_checksum_is_not_masked(analyzer) -> None:
    # One digit off -> checksum fails -> no mask, even with context.
    labeled = _label(analyzer, "my codice fiscale is RSSMRA85T10A562T here", ["it_codice_fiscale"])
    assert "<IT_CODICE_FISCALE>" not in labeled
    assert "RSSMRA85T10A562T" in labeled


def test_default_scan_does_not_overmask_ordinary_numbers(analyzer) -> None:
    # No EU country selected and no context: an ordinary 9-digit order
    # number must not be labelled as any European identifier. (It may
    # still be caught by the pre-existing structural SSN recognizer —
    # that is unchanged behavior — so we only assert no EU label.)
    labeled = _label(analyzer, "order 501442600 shipped", None)
    for type_id, _ in _MASK_CASES:
        assert f"<{type_id.upper()}>" not in labeled, (type_id, labeled)


def test_taxonomy_maps_every_eu_entity() -> None:
    # Every entity the recognizer can emit resolves to exactly one
    # operator type, and that type lists exactly that entity back.
    for entity in EU_ID_ENTITIES:
        type_id = tax.type_for_entity(entity)
        assert type_id is not None, f"{entity} has no operator type"
        spec = tax.get_spec(type_id)
        assert spec is not None
        assert entity in spec.presidio_entities


def test_catalog_exposes_country_categories() -> None:
    cats = {spec.category for spec in tax.all_types()}
    for country in ("Italy", "France", "Germany", "Sweden", "Netherlands"):
        assert country in cats
