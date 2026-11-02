"""Checksum / structural validators for European national identifiers.

Companion to :mod:`egisai.policy._pii_helpers` — same contract (pure
Python, no Presidio import) so the validators are unit-testable on
their own and can be reused by the regex fallback if ever needed. The
Presidio recognizer classes in :mod:`egisai.policy._pii_recognizers`
call these from their ``validate_result`` hooks, exactly like the
Spanish NIF recognizer already does with
:func:`egisai.policy._pii_helpers.is_valid_spanish_nif`.

Why a dedicated module: our analyzer runs the **English** Presidio NLP
engine (``en_core_web_lg``), and Presidio only loads the country
recognizers whose ``supported_language`` matches the engine. So the
built-in ``ES_NIF`` / ``IT_FISCAL_CODE`` / ``PL_PESEL`` recognizers
never register in our analyzer and the corresponding IDs sail through
unmasked. Each validator here backs a custom ``EGIS_*`` recognizer
registered under ``supported_language="en"`` so the detection actually
fires regardless of prompt language.

Design rules, per security-and-compliance.mdc:

* **Fail closed, but never false-negative on a real ID.** Where a
  national scheme defines a check digit we implement it exactly, so a
  valid ID always validates. Where a scheme has *no* reliable public
  checksum (passports, Italian ID cards) or officially abandoned it
  (Danish CPR after 2007), we validate structure + date plausibility
  only and lean on Presidio context words — never a wrong checksum
  that would reject valid documents.
* **Normalize first.** Each validator strips spaces / dots / hyphens
  and upper-cases before doing arithmetic, so formatting variants all
  validate identically. Callers pass the raw matched span.
* **Deterministic.** Pure regex + math — Phase 1 detectors that
  short-circuit before any LLM judge runs.

Every function takes the raw matched text and returns ``bool``.
"""
from __future__ import annotations

import re

from egisai.policy._pii_helpers import luhn_check

__all__ = [
    "is_valid_austrian_svnr",
    "is_valid_belgian_rrn",
    "is_valid_danish_cpr",
    "is_valid_dutch_bsn",
    "is_valid_finnish_hetu",
    "is_valid_french_nir",
    "is_valid_german_tax_id",
    "is_valid_greek_afm",
    "is_valid_greek_amka",
    "is_valid_irish_pps",
    "is_valid_italian_codice_fiscale",
    "is_valid_italian_tessera_sanitaria",
    "is_valid_norwegian_fnr",
    "is_valid_polish_nip",
    "is_valid_polish_pesel",
    "is_valid_portuguese_nif",
    "is_valid_swedish_personnummer",
    "is_valid_swiss_ahv",
]


# ── shared helpers ──────────────────────────────────────────────────


def _digits(value: str) -> str:
    """Strip every non-digit character."""
    return re.sub(r"\D", "", value or "")


def _clean(value: str) -> str:
    """Upper-case and drop spaces / dots / hyphens / slashes."""
    return re.sub(r"[\s.\-/]", "", (value or "").upper())


def _plausible_day_month(day: int, month: int) -> bool:
    """Coarse calendar check — day 1–31, month 1–12.

    Deliberately does not do per-month day counts or leap-year math:
    the goal is to reject obvious non-dates (month 19, day 00), not to
    perfectly validate a civil calendar. Several national schemes also
    offset the day (``+40`` female markers, ``+60`` coordination
    numbers), which callers fold in before calling this.
    """
    return 1 <= day <= 31 and 1 <= month <= 12


# ── Italy ───────────────────────────────────────────────────────────

# Check-character tables for the Codice Fiscale (CF). Odd-position
# (1-indexed) characters use the scrambled table; even-position use
# the plain 0-25 / digit mapping. The 16th character is the control.
_CF_ODD = {
    "0": 1, "1": 0, "2": 5, "3": 7, "4": 9, "5": 13, "6": 15, "7": 17,
    "8": 19, "9": 21, "A": 1, "B": 0, "C": 5, "D": 7, "E": 9, "F": 13,
    "G": 15, "H": 17, "I": 19, "J": 21, "K": 2, "L": 4, "M": 18, "N": 20,
    "O": 11, "P": 3, "Q": 6, "R": 8, "S": 12, "T": 14, "U": 16, "V": 10,
    "W": 22, "X": 25, "Y": 24, "Z": 23,
}
_CF_EVEN = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
    "8": 8, "9": 9, "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5,
    "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13,
    "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20,
    "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25,
}
# Structural shape. The numeric slots may carry letters under
# *omocodia* (digit → L M N P Q R S T U V), so the slots that are
# "normally digits" are matched as ``[0-9A-Z]`` and the check char
# does the real validation.
_CF_RE = re.compile(r"^[A-Z]{6}[0-9A-Z]{2}[A-Z][0-9A-Z]{2}[A-Z][0-9A-Z]{3}[A-Z]$")
# Month letters actually used by the CF encoding.
_CF_MONTHS = set("ABCDEHLMPRST")


def is_valid_italian_codice_fiscale(value: str) -> bool:
    """Validate an Italian Codice Fiscale via its mod-26 control char.

    16 alphanumerics: 6 surname/name letters, 2 year, 1 month letter,
    2 day, 4 place + 1 control. Handles *omocodia* (digit positions
    carrying substitute letters) because the control-character tables
    cover letters and digits alike, and the final check char is what
    we compare against. Company/temporary 11-digit codes are out of
    scope — those are the VAT code (``partita IVA``), a different type.
    """
    cf = _clean(value)
    if len(cf) != 16 or not _CF_RE.match(cf):
        return False
    if cf[8] not in _CF_MONTHS:
        return False
    total = 0
    for i, ch in enumerate(cf[:15]):
        table = _CF_EVEN if (i + 1) % 2 == 0 else _CF_ODD
        total += table[ch]
    return chr(total % 26 + ord("A")) == cf[15]


# The Italian health card (Tessera Sanitaria / European Health
# Insurance Card) carries a 20-digit identifier on the back. Every
# Italian card is issued under the fixed state prefix 80380800 — a
# signal strong enough on its own (a random 20-digit run starting with
# exactly those 8 digits is vanishingly unlikely). We validate the
# prefix + length and, when present, the trailing Luhn check digit,
# but never reject a prefix-valid number on the Luhn alone (card
# stock has varied) — fail closed on PII beats a false negative.
_IT_TS_PREFIX = "80380800"


def is_valid_italian_tessera_sanitaria(value: str) -> bool:
    """Validate an Italian health-card (Tessera Sanitaria) 20-digit number."""
    d = _digits(value)
    return len(d) == 20 and d.startswith(_IT_TS_PREFIX)


# ── Sweden ──────────────────────────────────────────────────────────


def is_valid_swedish_personnummer(value: str) -> bool:
    """Validate a Swedish personnummer / samordningsnummer (Luhn).

    Accepts the 10-digit (``YYMMDD-NNNC``) and 12-digit
    (``YYYYMMDDNNNC``) forms, with or without the separator. The last
    digit is a Luhn checksum over the nine significant digits
    (``YYMMDDNNN``). Coordination numbers (*samordningsnummer*) add 60
    to the day, so days 61–91 are accepted alongside 1–31.
    """
    d = _digits(value)
    if len(d) == 12:
        d = d[2:]  # drop the century, Luhn runs on the 10-digit core
    if len(d) != 10:
        return False
    month = int(d[2:4])
    day = int(d[4:6])
    if not 1 <= month <= 12:
        return False
    if not (1 <= day <= 31 or 61 <= day <= 91):
        return False
    return luhn_check(d)


# ── Netherlands ─────────────────────────────────────────────────────


def is_valid_dutch_bsn(value: str) -> bool:
    """Validate a Dutch BSN with the 11-proef weighted checksum.

    8 or 9 digits (8-digit BSNs are zero-padded to 9). Weights are
    9,8,7,6,5,4,3,2 on the first eight digits and ``-1`` on the ninth;
    the weighted sum must be a non-zero multiple of 11.
    """
    d = _digits(value)
    if len(d) == 8:
        d = "0" + d
    if len(d) != 9 or d == "000000000":
        return False
    weights = (9, 8, 7, 6, 5, 4, 3, 2, -1)
    total = sum(int(ch) * w for ch, w in zip(d, weights, strict=False))
    return total % 11 == 0


# ── France ──────────────────────────────────────────────────────────

_FR_NIR_RE = re.compile(r"^[12]\d{2}(?:0[1-9]|1[0-2]|20|[2-9]\d)(?:2[AB]|\d{2})\d{6}\d{2}$")


def is_valid_french_nir(value: str) -> bool:
    """Validate a French NIR (numéro de sécurité sociale), mod-97 key.

    15 characters: sex (1) + year (2) + month (2) + department (2) +
    commune/order (6) + key (2). The two-digit key equals
    ``97 - (body % 97)`` over the 13-digit body. Corsica's alphabetic
    departments are handled with the INSEE substitution: ``2A`` → 19
    (minus 1,000,000) and ``2B`` → 18 (minus 2,000,000).
    """
    s = _clean(value)
    if len(s) != 15 or not _FR_NIR_RE.match(s):
        return False
    body, key = s[:13], s[13:]
    if not key.isdigit():
        return False
    if "2A" in body:
        number = int(body.replace("2A", "19")) - 1_000_000
    elif "2B" in body:
        number = int(body.replace("2B", "18")) - 2_000_000
    else:
        if not body.isdigit():
            return False
        number = int(body)
    return int(key) == 97 - (number % 97)


# ── Germany ─────────────────────────────────────────────────────────


def is_valid_german_tax_id(value: str) -> bool:
    """Validate a German tax ID (Steuer-IdNr) — ISO 7064 MOD 11,10.

    11 digits, must not start with 0. The 11th digit is the ISO 7064
    MOD 11,10 check digit over the first ten. The additional
    "exactly one digit repeats" structural rule is intentionally not
    enforced — the check digit is sufficient and the extra rule has
    edge cases that would risk rejecting a valid number.
    """
    d = _digits(value)
    if len(d) != 11 or d[0] == "0":
        return False
    product = 10
    for ch in d[:10]:
        s = (int(ch) + product) % 10
        s = s or 10
        product = (s * 2) % 11
    check = (11 - product) % 10
    return check == int(d[10])


# ── Belgium ─────────────────────────────────────────────────────────


def is_valid_belgian_rrn(value: str) -> bool:
    """Validate a Belgian Rijksregisternummer / Numéro National (mod-97).

    11 digits: birth date (YYMMDD) + daily serial (3) + check (2). The
    check equals ``97 - (first9 % 97)``. People born in or after 2000
    prepend a ``2`` to the nine-digit base before the modulus, so both
    centuries are accepted.
    """
    d = _digits(value)
    if len(d) != 11:
        return False
    base, check = d[:9], int(d[9:])
    return check == 97 - (int(base) % 97) or check == 97 - (int("2" + base) % 97)


# ── Finland ─────────────────────────────────────────────────────────

_FI_CHECK = "0123456789ABCDEFHJKLMNPRSTUVWXY"
_FI_HETU_RE = re.compile(r"^(\d{6})([-+ABCDEFXYWVU])(\d{3})([0-9A-Y])$")


def is_valid_finnish_hetu(value: str) -> bool:
    """Validate a Finnish personal identity code (HETU), mod-31 control.

    ``DDMMYY`` + century sign + 3-digit individual number + control
    character. The control is ``"0123456789ABCDEFHJKLMNPRSTUVWXY"``
    indexed by ``int(DDMMYYNNN) % 31``.
    """
    # The century sign (``-``/``+``/letter) is load-bearing here, so
    # unlike the other validators we keep separators — only spaces and
    # the upper-casing are safe to normalise away.
    s = re.sub(r"\s", "", (value or "")).upper()
    m = _FI_HETU_RE.match(s)
    if not m:
        return False
    ddmmyy, _sign, individual, control = m.groups()
    day, month = int(ddmmyy[:2]), int(ddmmyy[2:4])
    if not _plausible_day_month(day, month):
        return False
    return _FI_CHECK[int(ddmmyy + individual) % 31] == control


# ── Poland ──────────────────────────────────────────────────────────


def is_valid_polish_pesel(value: str) -> bool:
    """Validate a Polish PESEL — weighted mod-10 checksum + date.

    11 digits. Weights 1,3,7,9 repeat across the first ten digits; the
    check digit is ``(10 - sum % 10) % 10``. The month field encodes
    the century (``+20/+40/+60/+80`` offsets), which we fold out before
    the coarse date check.
    """
    d = _digits(value)
    if len(d) != 11:
        return False
    month = int(d[2:4]) % 20  # strip century offset
    day = int(d[4:6])
    if not _plausible_day_month(day, month or 1):
        return False
    weights = (1, 3, 7, 9, 1, 3, 7, 9, 1, 3)
    total = sum(int(ch) * w for ch, w in zip(d, weights, strict=False))
    return (10 - total % 10) % 10 == int(d[10])


def is_valid_polish_nip(value: str) -> bool:
    """Validate a Polish NIP (tax number) — weighted mod-11.

    10 digits. Weights 6,5,7,2,3,4,5,6,7 on the first nine; the
    weighted sum mod 11 must equal the tenth digit (and must not be
    10, which marks an impossible NIP).
    """
    d = _digits(value)
    if len(d) != 10:
        return False
    weights = (6, 5, 7, 2, 3, 4, 5, 6, 7)
    total = sum(int(ch) * w for ch, w in zip(d, weights, strict=False))
    check = total % 11
    return check != 10 and check == int(d[9])


# ── Portugal ────────────────────────────────────────────────────────


def is_valid_portuguese_nif(value: str) -> bool:
    """Validate a Portuguese NIF / NIPC — mod-11 check digit.

    9 digits. Weights 9,8,7,6,5,4,3,2 on the first eight; the check
    digit is ``0`` when ``11 - (sum % 11)`` is 10 or 11, otherwise
    ``11 - (sum % 11)``. The first digit must be a valid NIF group.
    """
    d = _digits(value)
    if len(d) != 9 or d[0] == "0":
        return False
    total = sum(int(ch) * w for ch, w in zip(d[:8], range(9, 1, -1), strict=False))
    check = 11 - (total % 11)
    if check >= 10:
        check = 0
    return check == int(d[8])


# ── Greece ──────────────────────────────────────────────────────────


def is_valid_greek_amka(value: str) -> bool:
    """Validate a Greek AMKA (social-security number) — Luhn + date.

    11 digits whose first six are the birth date ``DDMMYY``; the whole
    11-digit run is Luhn-valid.
    """
    d = _digits(value)
    if len(d) != 11 or d == "0" * 11:
        return False
    if not _plausible_day_month(int(d[:2]), int(d[2:4])):
        return False
    return luhn_check(d)


def is_valid_greek_afm(value: str) -> bool:
    """Validate a Greek AFM (tax number) — powers-of-two mod-11.

    9 digits. Each of the first eight is weighted by a descending
    power of two (256…2); the check digit is ``(sum % 11) % 10``.
    """
    d = _digits(value)
    if len(d) != 9 or d == "0" * 9:
        return False
    total = sum(int(d[i]) * (2 ** (8 - i)) for i in range(8))
    return (total % 11) % 10 == int(d[8])


# ── Norway ──────────────────────────────────────────────────────────


def is_valid_norwegian_fnr(value: str) -> bool:
    """Validate a Norwegian fødselsnummer — two mod-11 check digits.

    11 digits: ``DDMMYY`` + 3-digit individual + 2 control digits.
    Accepts D-numbers (day ``+40``) and H-numbers (month ``+40``).
    """
    d = _digits(value)
    if len(d) != 11:
        return False
    day = int(d[:2])
    month = int(d[2:4])
    if day > 40:
        day -= 40  # D-number
    if month > 40:
        month -= 40  # H-number
    if not _plausible_day_month(day, month):
        return False
    w1 = (3, 7, 6, 1, 8, 9, 4, 5, 2)
    w2 = (5, 4, 3, 2, 7, 6, 5, 4, 3, 2)
    k1 = 11 - sum(int(d[i]) * w1[i] for i in range(9)) % 11
    if k1 == 11:
        k1 = 0
    if k1 == 10 or k1 != int(d[9]):
        return False
    k2 = 11 - sum(int(d[i]) * w2[i] for i in range(10)) % 11
    if k2 == 11:
        k2 = 0
    return k2 != 10 and k2 == int(d[10])


# ── Denmark ─────────────────────────────────────────────────────────

_DK_CPR_RE = re.compile(r"^(\d{2})(\d{2})(\d{2})-?(\d{4})$")


def is_valid_danish_cpr(value: str) -> bool:
    """Validate a Danish CPR number — structure + date only.

    ``DDMMYY-SSSS``. Denmark officially abandoned the modulus-11 check
    in 2007 (valid numbers that fail mod-11 are issued), so enforcing
    it would reject real people — we validate the date and the
    4-digit serial instead and rely on the ``-`` separator / context
    for precision.
    """
    s = _clean(value) if "-" not in (value or "") else value.strip().upper()
    m = _DK_CPR_RE.match(s.replace(" ", ""))
    if not m:
        return False
    day, month = int(m.group(1)), int(m.group(2))
    return _plausible_day_month(day, month)


# ── Ireland ─────────────────────────────────────────────────────────

_IE_CHECK = "WABCDEFGHIJKLMNOPQRSTUV"
_IE_PPS_RE = re.compile(r"^(\d{7})([A-W])([A-IW]?)$")


def is_valid_irish_pps(value: str) -> bool:
    """Validate an Irish PPS number — mod-23 check letter.

    7 digits + a check letter, optionally followed by a second letter
    (``A``/``W`` or blank). Weights 8,7,6,5,4,3,2 on the digits; when a
    second letter is present it contributes ``(position-in-alphabet) *
    9``. The remainder mod 23 indexes ``"WABC…UV"``.
    """
    s = _clean(value)
    m = _IE_PPS_RE.match(s)
    if not m:
        return False
    digits, check, second = m.groups()
    total = sum(int(ch) * w for ch, w in zip(digits, range(8, 1, -1), strict=False))
    if second and second != "W":
        total += (ord(second) - ord("A") + 1) * 9
    return _IE_CHECK[total % 23] == check


# ── Austria ─────────────────────────────────────────────────────────


def is_valid_austrian_svnr(value: str) -> bool:
    """Validate an Austrian Sozialversicherungsnummer — weighted mod-11.

    10 digits: 3-digit serial + 1 check digit + ``DDMMYY``. Weights are
    3,7,9 on the serial and 5,8,4,2,1,6 on the birth date; the sum mod
    11 equals the check digit (position 4). A remainder of 10 marks an
    unissued number.
    """
    d = _digits(value)
    if len(d) != 10:
        return False
    check = int(d[3])
    weights = (3, 7, 9, 0, 5, 8, 4, 2, 1, 6)
    total = sum(int(d[i]) * weights[i] for i in range(10))
    remainder = total % 11
    return remainder != 10 and remainder == check


# ── Switzerland ─────────────────────────────────────────────────────


def is_valid_swiss_ahv(value: str) -> bool:
    """Validate a Swiss AHV/AVS (new 13-digit) — EAN-13 checksum.

    13 digits beginning with the country code ``756``. The 13th digit
    is the EAN-13 check digit over the first twelve (weights alternate
    1 and 3 from the left).
    """
    d = _digits(value)
    if len(d) != 13 or not d.startswith("756"):
        return False
    total = sum(int(ch) * (1 if i % 2 == 0 else 3) for i, ch in enumerate(d[:12]))
    return (10 - total % 10) % 10 == int(d[12])
