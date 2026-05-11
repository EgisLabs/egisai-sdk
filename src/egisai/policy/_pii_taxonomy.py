"""Canonical PII type taxonomy for the SDK and the platform UI.

Exactly one place defines:

* the operator-facing **type names** (``"ssn"``, ``"passport"``, …) that
  appear in policy ``config.types`` JSON, in audit ``sanitizations``
  records, and on the dashboard's checkbox grid;
* which **Presidio entity types** each operator-facing type maps to —
  e.g. ``"national_id"`` covers ``IN_AADHAAR``, ``ES_NIF``, ``US_ITIN``
  and a dozen more checksum-validated detectors;
* the **catalog metadata** (label, category, description, example) that
  the backend exposes at ``GET /v1/sdk/pii-types`` so the dashboard
  policy modal renders the same taxonomy without duplicating it.

Why centralise this:

* the SDK turns operator config (``types``) into a Presidio entity
  filter at scan time;
* the backend echoes the catalog to the frontend so the UI never
  references a type the SDK doesn't know how to detect (the bug that
  used to silently no-op when an operator typed ``"passport"`` in a
  free-form text box);
* compliance audits trace a sanitization record back to an operator
  type and from there to the exact recognizers that fired — no detour
  through opaque names.

Adding a new type is a 4-line change in this file:

    1. add a ``PiiTypeSpec`` to ``CANONICAL_TYPES``,
    2. list its Presidio entity names in ``presidio_entities``,
    3. give it a clear ``description`` + ``example`` for the UI,
    4. (optional) register a custom Presidio recognizer in
       ``_pii_recognizers.py`` if Presidio doesn't ship one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# ── Categories — the visual grouping operators see in the UI ─────────
#
# Adding a new category is fine; the UI groups checkboxes by this
# label and rendering is alphabetical inside a group. Keep the labels
# short and operator-facing — these strings ship to the dashboard.

CATEGORY_IDENTITY = "Identity"
CATEGORY_CONTACT = "Contact"
CATEGORY_FINANCIAL = "Financial"
CATEGORY_MEDICAL = "Medical"
CATEGORY_CREDENTIALS = "Credentials"
CATEGORY_NETWORK = "Network"
CATEGORY_VEHICLE = "Vehicle"


@dataclass(frozen=True)
class PiiTypeSpec:
    """One operator-facing PII type and how the SDK detects it."""

    # Stable identifier shipped on the wire. Lower-snake-case. Operators
    # see this in policy JSON, audit records, the events bus payload.
    # Renaming this is a breaking schema change.
    id: str
    # Human-readable label rendered on the dashboard checkbox grid.
    label: str
    # Visual grouping bucket — see CATEGORY_* constants above.
    category: str
    # Short operator-facing prose explaining what this catches. No
    # implementation jargon ("Luhn", "regex", …) — describe the
    # outcome, not the mechanism. Industry-standard tooltip length.
    description: str
    # One realistic-looking example so the operator can match it
    # against the data they expect to flow through their app.
    example: str
    # Set of Presidio entity strings this maps to. The SDK passes
    # exactly this set to ``AnalyzerEngine.analyze(entities=...)``.
    # Custom recognizers (defined in ``_pii_recognizers.py``) emit
    # entity names that begin with ``EGIS_`` so they're easy to spot.
    presidio_entities: frozenset[str] = field(default_factory=frozenset)


# ── Canonical taxonomy ──────────────────────────────────────────────
#
# The order in this list is the order the UI defaults to within each
# category. We also use this list to drive the catalog endpoint, so
# adding an entry here is the single change required to get the new
# type onto the dashboard.
#
# Every entry below is backed by Presidio's checksum-validated
# detectors (Luhn for ``CREDIT_CARD``; mod-97 for ``IBAN_CODE``;
# country-specific Verhoeff / mod-11 / mod-10 for various national
# IDs) or by a custom recognizer in ``_pii_recognizers.py``. We don't
# accept "best-effort with no validation" detectors at this layer —
# those go via the semantic-judge path, not pii_scan.

CANONICAL_TYPES: tuple[PiiTypeSpec, ...] = (
    # ── Identity ───────────────────────────────────────────────────
    PiiTypeSpec(
        id="person_name",
        label="Person name",
        category=CATEGORY_IDENTITY,
        description=(
            "First, middle, and last names of individuals. Detected via "
            "named-entity recognition; works on natural-language prose."
        ),
        example="Jane Doe",
        presidio_entities=frozenset({"PERSON"}),
    ),
    PiiTypeSpec(
        id="date_of_birth",
        label="Date of birth",
        category=CATEGORY_IDENTITY,
        description=(
            "Birth dates in any common format. Distinguished from generic "
            "dates by surrounding context (\"DOB\", \"born\", \"birth\")."
        ),
        example="1985-04-12",
        # ``EGIS_DOB`` is the custom recognizer that adds the DOB
        # context filter on top of Presidio's ``DATE_TIME`` matches.
        presidio_entities=frozenset({"EGIS_DOB"}),
    ),
    PiiTypeSpec(
        id="ssn",
        label="US Social Security Number",
        category=CATEGORY_IDENTITY,
        description=(
            "9-digit US SSN with area-number validation. Catches plain, "
            "dashed, and English word-form (\"one two three…\") variants."
        ),
        example="123-45-6789",
        presidio_entities=frozenset(
            {"US_SSN", "EGIS_US_SSN", "EGIS_WORD_FORM_SSN"}
        ),
    ),
    PiiTypeSpec(
        id="passport",
        label="Passport number",
        category=CATEGORY_IDENTITY,
        description=(
            "Passport numbers from the US, UK, Italy, India, and Korea, "
            "matched against each country's format rules."
        ),
        example="P12345678",
        presidio_entities=frozenset({
            "US_PASSPORT",
            "UK_PASSPORT",
            "IT_PASSPORT",
            "IN_PASSPORT",
            "KR_PASSPORT",
        }),
    ),
    PiiTypeSpec(
        id="drivers_license",
        label="Driver's license",
        category=CATEGORY_IDENTITY,
        description=(
            "Driver's-license numbers from the US, Italy, and Korea. "
            "Format rules + nearby keyword context."
        ),
        example="D1234562",
        presidio_entities=frozenset({
            "US_DRIVER_LICENSE",
            "IT_DRIVER_LICENSE",
            "KR_DRIVER_LICENSE",
        }),
    ),
    PiiTypeSpec(
        id="national_id",
        label="National / tax ID",
        category=CATEGORY_IDENTITY,
        description=(
            "Government-issued personal IDs across 14 countries — Aadhaar, "
            "PESEL, NIF/NIE, ITIN, NINO, RRN, TFN, fiscal codes, more. "
            "Each is checksum-validated where the country defines one."
        ),
        example="AB123456C",
        presidio_entities=frozenset({
            "US_ITIN",
            "IN_AADHAAR",
            "IN_PAN",
            "IN_VOTER",
            "ES_NIF",
            "ES_NIE",
            "PL_PESEL",
            "FI_PERSONAL_IDENTITY_CODE",
            "KR_RRN",
            "KR_FRN",
            "TH_TNIN",
            "NG_NIN",
            "AU_TFN",
            "SG_NRIC_FIN",
            "IT_FISCAL_CODE",
            "IT_IDENTITY_CARD",
            "UK_NINO",
        }),
    ),
    PiiTypeSpec(
        id="nationality_or_religion",
        label="Nationality / religion",
        category=CATEGORY_IDENTITY,
        description=(
            "Mentions of a person's nationality, religion, or political "
            "affiliation (a GDPR special category of data)."
        ),
        example="French citizen",
        presidio_entities=frozenset({"NRP"}),
    ),
    # ── Contact ────────────────────────────────────────────────────
    PiiTypeSpec(
        id="email",
        label="Email address",
        category=CATEGORY_CONTACT,
        description=(
            "RFC-822 email addresses. Reserved-domain placeholders "
            "(``user@example.com``, ``*.test``, ``*.invalid``) are skipped "
            "automatically to avoid noise on docs and code samples."
        ),
        example="alice@acme.com",
        presidio_entities=frozenset({"EMAIL_ADDRESS"}),
    ),
    PiiTypeSpec(
        id="phone",
        label="Phone number",
        category=CATEGORY_CONTACT,
        description=(
            "Phone numbers in international and national formats. "
            "Validated by length + nearby keyword context."
        ),
        example="+1 (415) 555-0124",
        presidio_entities=frozenset({"PHONE_NUMBER"}),
    ),
    PiiTypeSpec(
        id="address",
        label="Physical address / location",
        category=CATEGORY_CONTACT,
        description=(
            "Street addresses, cities, regions, and other geographic "
            "locations identified by named-entity recognition."
        ),
        example="221B Baker Street, London",
        presidio_entities=frozenset({"LOCATION"}),
    ),
    PiiTypeSpec(
        id="url",
        label="URL",
        category=CATEGORY_CONTACT,
        description=(
            "Fully-qualified URLs. Useful when a prompt may carry "
            "session-bearing or pre-signed URLs you don't want to leak."
        ),
        example="https://share.acme.com/abc123",
        presidio_entities=frozenset({"URL"}),
    ),
    # ── Financial ─────────────────────────────────────────────────
    PiiTypeSpec(
        id="credit_card",
        label="Credit / debit card number",
        category=CATEGORY_FINANCIAL,
        description=(
            "Card numbers from Visa, Mastercard, Amex, Discover. Validated "
            "with the Luhn checksum so test placeholders don't false-fire."
        ),
        example="4111-1111-1111-1111",
        presidio_entities=frozenset({"CREDIT_CARD", "EGIS_WORD_FORM_CC"}),
    ),
    PiiTypeSpec(
        id="iban",
        label="IBAN (international bank account)",
        category=CATEGORY_FINANCIAL,
        description=(
            "International Bank Account Numbers. Validated with the "
            "ISO 7064 mod-97 checksum."
        ),
        example="GB82WEST12345698765432",
        presidio_entities=frozenset({"IBAN_CODE"}),
    ),
    PiiTypeSpec(
        id="bank_account",
        label="Bank / business account number",
        category=CATEGORY_FINANCIAL,
        description=(
            "Domestic bank account numbers, business registration "
            "numbers, and tax codes (US bank, Australian ABN, "
            "Singapore UEN, Korean BRN, Indian GSTIN, Italian VAT)."
        ),
        example="123456789",
        presidio_entities=frozenset({
            "US_BANK_NUMBER",
            "AU_ABN",
            "AU_ACN",
            "SG_UEN",
            "KR_BRN",
            "IN_GSTIN",
            "IT_VAT_CODE",
        }),
    ),
    PiiTypeSpec(
        id="crypto_wallet",
        label="Cryptocurrency wallet",
        category=CATEGORY_FINANCIAL,
        description=(
            "Bitcoin and other crypto-wallet addresses. Validated with "
            "the network-specific checksum to avoid false positives on "
            "look-alike base58 strings."
        ),
        example="1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2",
        presidio_entities=frozenset({"CRYPTO"}),
    ),
    # ── Medical ────────────────────────────────────────────────────
    PiiTypeSpec(
        id="medical_license",
        label="Medical / healthcare ID",
        category=CATEGORY_MEDICAL,
        description=(
            "Provider and patient identifiers for HIPAA-regulated flows: "
            "US NPI, Medicare Beneficiary IDs, UK NHS, Australian Medicare, "
            "and generic medical license numbers."
        ),
        example="1234567893",
        presidio_entities=frozenset({
            "US_NPI",
            "MEDICAL_LICENSE",
            "US_MBI",
            "UK_NHS",
            "AU_MEDICARE",
        }),
    ),
    # ── Credentials ───────────────────────────────────────────────
    PiiTypeSpec(
        id="api_key",
        label="API key / secret token",
        category=CATEGORY_CREDENTIALS,
        description=(
            "API keys, bearer tokens, GitHub PATs, AWS access keys, JWTs. "
            "Detected by known prefix patterns plus a Shannon-entropy "
            "test to catch novel high-entropy credentials."
        ),
        example="sk-proj-AbC123…XyZ789",
        presidio_entities=frozenset({"EGIS_API_KEY"}),
    ),
    PiiTypeSpec(
        id="password",
        label="Password",
        category=CATEGORY_CREDENTIALS,
        description=(
            "Plaintext passwords adjacent to context keywords like "
            "``password:``, ``passwd``, or ``pwd``. Best-effort — strong "
            "passwords without nearby keywords are not always catchable."
        ),
        example="password: hunter2!",
        presidio_entities=frozenset({"EGIS_PASSWORD"}),
    ),
    # ── Network ────────────────────────────────────────────────────
    PiiTypeSpec(
        id="ip_address",
        label="IP address",
        category=CATEGORY_NETWORK,
        description=(
            "IPv4 and IPv6 addresses. Useful to keep customer source IPs "
            "from leaving your perimeter via prompts or logs."
        ),
        example="203.0.113.42",
        presidio_entities=frozenset({"IP_ADDRESS"}),
    ),
    PiiTypeSpec(
        id="mac_address",
        label="MAC address",
        category=CATEGORY_NETWORK,
        description="Network-interface MAC addresses in colon or dash form.",
        example="aa:bb:cc:dd:ee:ff",
        presidio_entities=frozenset({"MAC_ADDRESS"}),
    ),
    # ── Vehicle ────────────────────────────────────────────────────
    PiiTypeSpec(
        id="vehicle_registration",
        label="Vehicle registration",
        category=CATEGORY_VEHICLE,
        description=(
            "Vehicle plate / registration numbers from the UK, India, "
            "and Nigeria, matched against each country's modern format."
        ),
        example="AB12 CDE",
        presidio_entities=frozenset({
            "UK_VEHICLE_REGISTRATION",
            "IN_VEHICLE_REGISTRATION",
            "NG_VEHICLE_REGISTRATION",
        }),
    ),
)


# ── Lookup tables built once at import ──────────────────────────────

_BY_ID: dict[str, PiiTypeSpec] = {spec.id: spec for spec in CANONICAL_TYPES}

# Reverse map: Presidio entity → operator-facing type. One Presidio
# entity always belongs to exactly one operator type, but the same
# operator type can fan out into many Presidio entities (e.g.
# ``national_id`` covers 17 country-specific recognizers).
_ENTITY_TO_TYPE: dict[str, str] = {}
for _spec in CANONICAL_TYPES:
    for _entity in _spec.presidio_entities:
        if _entity in _ENTITY_TO_TYPE and _ENTITY_TO_TYPE[_entity] != _spec.id:
            # Same Presidio entity claimed by two operator types is a
            # taxonomy bug; surface loudly rather than silently merge.
            raise RuntimeError(
                f"PII taxonomy conflict: Presidio entity {_entity!r} is "
                f"mapped to both {_ENTITY_TO_TYPE[_entity]!r} and {_spec.id!r}."
            )
        _ENTITY_TO_TYPE[_entity] = _spec.id


# ── Public lookup API ───────────────────────────────────────────────


def all_types() -> tuple[PiiTypeSpec, ...]:
    """Return the full canonical taxonomy in display order."""
    return CANONICAL_TYPES


def type_ids() -> frozenset[str]:
    """Set of every operator-facing type id the SDK can detect."""
    return frozenset(_BY_ID.keys())


def get_spec(type_id: str) -> PiiTypeSpec | None:
    """Look up a single ``PiiTypeSpec`` by its operator-facing id."""
    return _BY_ID.get(type_id)


def entities_for(type_ids_subset: list[str] | None) -> frozenset[str] | None:
    """Translate a list of operator types into Presidio entity names.

    ``None`` (or empty list) → ``None``, which the caller passes to
    Presidio to mean "every entity recognised by every loaded
    recognizer". A list of ids is fanned out into the union of
    Presidio entities across them; unknown ids are silently dropped
    (the caller emits its own warning so the policy author sees it).
    """
    if not type_ids_subset:
        return None
    entities: set[str] = set()
    for tid in type_ids_subset:
        spec = _BY_ID.get(tid)
        if spec is None:
            continue
        entities.update(spec.presidio_entities)
    return frozenset(entities) if entities else frozenset()


def type_for_entity(entity: str) -> str | None:
    """Inverse: which operator type does this Presidio entity belong to?"""
    return _ENTITY_TO_TYPE.get(entity)


def unknown_types(type_ids_subset: list[str]) -> list[str]:
    """Return the subset of ``type_ids_subset`` we don't know how to detect.

    The SDK and the backend both call this to emit a friendly warning
    when an operator policy references a type the engine can't fulfil
    — that's the bug that used to silently no-op when ``"passport"``
    was typed into the legacy free-text ``kinds`` field.
    """
    return [tid for tid in type_ids_subset if tid not in _BY_ID]
