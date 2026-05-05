"""Multi-signal PII detection.

Each detector validates mathematically rather than just pattern-matching:

- Credit cards: Luhn checksum
- SSNs: area-number range + context keywords
- API keys / secrets: Shannon entropy on base64/hex segments
- IBANs: ISO 7064 mod-97 checksum
- Emails: RFC-lite structural validation
- Phones: E.164-ish structure with context

The aggregate risk score combines finding count, per-finding
confidence, and the diversity of PII kinds — not a binary regex match.
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

# ── Context keywords that boost confidence when near a PII match ────
_CONTEXT_KEYWORDS: dict[str, list[str]] = {
    "credit_card": ["card", "visa", "mastercard", "amex", "payment", "credit", "debit", "cvv", "expir"],
    "ssn": ["ssn", "social security", "social-security", "tax id", "taxpayer"],
    "api_key": ["key", "secret", "token", "api_key", "apikey", "password", "credential", "bearer", "authorization"],
    "email": ["email", "e-mail", "mailto", "contact"],
    "phone": ["phone", "mobile", "cell", "tel", "call", "sms", "whatsapp"],
    "iban": ["iban", "bank", "account", "transfer", "swift", "bic"],
}

_CONTEXT_WINDOW = 60  # chars to look around a match for context keywords


# ── Reserved / placeholder email domains ────────────────────────────
#
# Email addresses landing on these domains are treated as documentation
# and test fixtures rather than real PII. Two sets:
#
# * ``_RFC_RESERVED_EMAIL_TLDS`` and ``_RFC_RESERVED_EMAIL_SLDS`` come
#   straight from RFC 2606 §3 / RFC 6761 §6 and are guaranteed never to
#   resolve. Skipping them avoids false positives on every code sample
#   that uses ``user@example.com`` style addresses.
#
# * ``_PLACEHOLDER_EMAIL_DOMAINS`` is a small list of widely-used
#   placeholder domains that aren't covered by the RFCs but show up in
#   docs, tutorials, and form auto-fill data so often that flagging
#   them is more noise than signal.
_RFC_RESERVED_EMAIL_TLDS: frozenset[str] = frozenset(
    {".test", ".example", ".invalid", ".localhost"}
)
_RFC_RESERVED_EMAIL_SLDS: frozenset[str] = frozenset(
    {"example.com", "example.net", "example.org"}
)
_PLACEHOLDER_EMAIL_DOMAINS: frozenset[str] = frozenset({"test.com"})


def _is_reserved_email_domain(domain: str) -> bool:
    """Return True if ``domain`` is RFC-reserved or a known placeholder.

    Case-insensitive. ``endswith`` covers any subdomain beneath a
    reserved TLD (e.g. ``api.example.invalid``).
    """
    if not domain:
        return False
    d = domain.lower().strip()
    if d == "localhost":
        return True
    if d in _RFC_RESERVED_EMAIL_SLDS or d in _PLACEHOLDER_EMAIL_DOMAINS:
        return True
    if any(d.endswith(tld) for tld in _RFC_RESERVED_EMAIL_TLDS):
        return True
    return False


@dataclass(frozen=True)
class PIIFinding:
    kind: str            # "credit_card", "ssn", "api_key", etc.
    value_redacted: str  # "4111-****-****-1111"
    confidence: float    # 0.0 – 1.0
    method: str          # "luhn", "entropy", "mod97", "pattern+context"


@dataclass(frozen=True)
class Sanitization:
    """Audit record for one redaction applied to a prompt.

    Carries the count and mask shape only — never the original
    value, by design.
    """
    kind: str        # 'ssn' | 'credit_card' | 'email' | 'phone' | 'iban' | 'api_key'
    count: int
    pattern: str     # mask shape, e.g. '###-##-####' for SSN


def _normalize_for_pii(text: str) -> str:
    """NFKC-normalize so non-ASCII digit scripts collapse to ASCII.

    Without this, ``\\d`` matches but downstream ``int()`` validation
    raises on fullwidth / Arabic-Indic / Devanagari codepoints and
    silently drops the finding.
    """
    if not text:
        return text
    try:
        return unicodedata.normalize("NFKC", text)
    except Exception:  # pragma: no cover
        return text


def scan(text: str) -> list[PIIFinding]:
    """Run all detectors and return validated findings."""
    text = _normalize_for_pii(text)
    findings: list[PIIFinding] = []
    findings.extend(_scan_credit_cards(text))
    findings.extend(_scan_ssns(text))
    findings.extend(_scan_api_keys(text))
    findings.extend(_scan_emails(text))
    findings.extend(_scan_phones(text))
    findings.extend(_scan_ibans(text))
    findings.extend(_scan_word_form_digits(text))
    return findings


# ── Word-form digit detection ───────────────────────────────────────
#
# Catches SSN- or credit-card-shape sequences spelled in English words
# ("one two three four five six seven eight nine"). English only —
# multi-language word-form detection is a future enhancement.

_WORD_DIGIT_TOKEN_MAP: dict[str, str] = {
    "zero": "0", "oh": "0", "naught": "0", "nought": "0",
    "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
}

_WORD_DIGIT_TOKEN = (
    r"(?:zero|oh|naught|nought|one|two|three|four|"
    r"five|six|seven|eight|nine)"
)

# Run of 9–19 digit-words (longest valid credit card is 19).
_WORD_DIGIT_RUN_RE = re.compile(
    rf"(?:\b{_WORD_DIGIT_TOKEN}\b[\s,.\-]*(?:and[\s,.\-]+)?){{9,19}}",
    re.IGNORECASE,
)
_WORD_DIGIT_TOKEN_RE = re.compile(_WORD_DIGIT_TOKEN, re.IGNORECASE)


def _word_run_to_digits(run: str) -> str:
    """Turn the matched word run into the digits it spells out."""
    return "".join(
        _WORD_DIGIT_TOKEN_MAP[w.lower()]
        for w in _WORD_DIGIT_TOKEN_RE.findall(run)
    )


def _scan_word_form_digits(text: str) -> list[PIIFinding]:
    """Detect SSN- or credit-card-shape sequences spelled in English words."""
    findings: list[PIIFinding] = []
    for match in _WORD_DIGIT_RUN_RE.finditer(text):
        digits = _word_run_to_digits(match.group(0))
        if len(digits) == 9:
            findings.append(PIIFinding(
                kind="ssn",
                value_redacted=f"***-**-{digits[-4:]}",
                confidence=0.95,
                method="word-form",
            ))
        elif 13 <= len(digits) <= 19 and _luhn_check(digits):
            findings.append(PIIFinding(
                kind="credit_card",
                value_redacted=digits[:4] + "-****-****-" + digits[-4:],
                confidence=0.95,
                method="word-form+luhn",
            ))
    return findings


# ── Sanitization ───────────────────────────────────────────────────
#
# ``scan`` finds PII; ``sanitize`` masks it in place. Same regexes
# and mathematical validators as the scanners — only validated PII
# is masked. Each masker preserves the shape of the original so
# downstream reasoning about format still works.

def _digit_template_for(kind: str, raw: str, mask_char: str) -> str:
    """Replace every digit in ``raw`` with ``mask_char``."""
    return re.sub(r"\d", mask_char, raw)


# Static placeholders for non-digit kinds.
_WORD_PLACEHOLDERS: dict[str, str] = {
    "email": "[email-redacted]",
    "iban": "[iban-redacted]",
    "api_key": "[secret-redacted]",
}

# Typed-label redaction. Replaces each validated PII match with a
# label like ``<SSN>`` so the original value never lands in a log.

_LABEL_FOR_KIND: dict[str, str] = {
    "ssn": "<SSN>",
    "credit_card": "<CREDIT_CARD>",
    "email": "<EMAIL>",
    "phone": "<PHONE>",
    "iban": "<IBAN>",
    "api_key": "<API_KEY>",
}


def label_redact(text: str, kinds: list[str] | None = None) -> str:
    """Replace each validated PII match in ``text`` with a typed label.

    ``"my SSN is 123-45-6789"`` → ``"my SSN is <SSN>"``
    ``"call jane@acme.com or 555-123-4567"`` → ``"call <EMAIL> or <PHONE>"``

    Uses the same validators as ``sanitize``, so Luhn-invalid 16-digit
    numbers are not redacted. ``kinds`` filters detectors; ``None``
    runs every one. The output is safe to persist in audit logs.
    """
    if not text:
        return text

    text = _normalize_for_pii(text)
    kind_filter = set(kinds) if kinds else None

    if kind_filter is None or "ssn" in kind_filter or "credit_card" in kind_filter:
        def _word_run_repl(m: re.Match[str]) -> str:
            digits = _word_run_to_digits(m.group(0))
            if len(digits) == 9 and (kind_filter is None or "ssn" in kind_filter):
                return _LABEL_FOR_KIND["ssn"]
            if (
                13 <= len(digits) <= 19
                and _luhn_check(digits)
                and (kind_filter is None or "credit_card" in kind_filter)
            ):
                return _LABEL_FOR_KIND["credit_card"]
            return m.group(0)
        text = _WORD_DIGIT_RUN_RE.sub(_word_run_repl, text)

    if kind_filter is None or "ssn" in kind_filter:
        def _ssn_repl(m: re.Match[str]) -> str:
            area, group, serial = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if area in _INVALID_SSN_AREAS or group == 0 or serial == 0:
                return m.group(0)
            return _LABEL_FOR_KIND["ssn"]
        text = _SSN_RE.sub(_ssn_repl, text)

    if kind_filter is None or "credit_card" in kind_filter:
        def _cc_repl(m: re.Match[str]) -> str:
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) < 13 or len(digits) > 19 or not _luhn_check(digits):
                return m.group(0)
            return _LABEL_FOR_KIND["credit_card"]
        text = _CC_RE.sub(_cc_repl, text)

    if kind_filter is None or "iban" in kind_filter:
        def _iban_repl(m: re.Match[str]) -> str:
            if not _iban_mod97(m.group(0)):
                return m.group(0)
            return _LABEL_FOR_KIND["iban"]
        text = _IBAN_RE.sub(_iban_repl, text)

    if kind_filter is None or "email" in kind_filter:
        def _email_repl(m: re.Match[str]) -> str:
            domain = m.group(0).rsplit("@", 1)[1]
            if _is_reserved_email_domain(domain):
                return m.group(0)
            return _LABEL_FOR_KIND["email"]
        text = _EMAIL_RE.sub(_email_repl, text)

    if kind_filter is None or "phone" in kind_filter:
        def _phone_repl(m: re.Match[str]) -> str:
            raw = m.group(0)
            digits = re.sub(r"\D", "", raw)
            if len(digits) < 10 or len(digits) > 11:
                return raw
            confidence = 0.55 + _context_boost(text, m.start(), m.end(), "phone")
            if confidence < 0.65:
                return raw
            return _LABEL_FOR_KIND["phone"]
        text = _PHONE_RE.sub(_phone_repl, text)

    if kind_filter is None or "api_key" in kind_filter:
        def _key_repl(m: re.Match[str]) -> str:
            raw = m.group(0)
            if _shannon_entropy(raw) < 3.5:
                return raw
            return _LABEL_FOR_KIND["api_key"]
        text = _KEY_RE.sub(_key_repl, text)

    return text


def sanitize(
    text: str,
    kinds: list[str] | None = None,
    mask_char: str = "#",
) -> tuple[str, list[Sanitization]]:
    """Mask validated PII in ``text``.

    Returns ``(redacted_text, [Sanitization, ...])`` with at most one
    record per kind (counts are rolled up).

    Parameters
    ----------
    text
        The user-supplied text to sanitize.
    kinds
        Restrict the active detectors. ``None`` runs every detector.
    mask_char
        Character substituted for each digit in digit-shaped PII
        (SSN, credit card, phone). Word-shaped PII (email, IBAN,
        api_key) always uses ``_WORD_PLACEHOLDERS``.
    """
    if not text:
        return text, []

    if not mask_char:
        mask_char = "#"

    text = _normalize_for_pii(text)

    kind_filter = set(kinds) if kinds else None
    counts: dict[str, int] = {}
    sample_pattern: dict[str, str] = {}

    def _bump(kind: str, rendered: str) -> None:
        counts[kind] = counts.get(kind, 0) + 1
        sample_pattern.setdefault(kind, rendered)

    if kind_filter is None or "ssn" in kind_filter or "credit_card" in kind_filter:
        def _word_run_repl(m: re.Match[str]) -> str:
            digits = _word_run_to_digits(m.group(0))
            if len(digits) == 9 and (kind_filter is None or "ssn" in kind_filter):
                rendered = (
                    mask_char * 3 + "-" + mask_char * 2 + "-" + mask_char * 4
                )
                _bump("ssn", rendered)
                return rendered
            if (
                13 <= len(digits) <= 19
                and _luhn_check(digits)
                and (kind_filter is None or "credit_card" in kind_filter)
            ):
                groups = [digits[i:i + 4] for i in range(0, len(digits), 4)]
                rendered = "-".join(mask_char * len(g) for g in groups)
                _bump("credit_card", rendered)
                return rendered
            return m.group(0)
        text = _WORD_DIGIT_RUN_RE.sub(_word_run_repl, text)

    if kind_filter is None or "ssn" in kind_filter:
        def _ssn_repl(m: re.Match[str]) -> str:
            area, group, serial = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if area in _INVALID_SSN_AREAS or group == 0 or serial == 0:
                return m.group(0)
            rendered = _digit_template_for("ssn", m.group(0), mask_char)
            _bump("ssn", rendered)
            return rendered
        text = _SSN_RE.sub(_ssn_repl, text)

    if kind_filter is None or "credit_card" in kind_filter:
        def _cc_repl(m: re.Match[str]) -> str:
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) < 13 or len(digits) > 19 or not _luhn_check(digits):
                return m.group(0)
            rendered = _digit_template_for("credit_card", m.group(0), mask_char)
            _bump("credit_card", rendered)
            return rendered
        text = _CC_RE.sub(_cc_repl, text)

    if kind_filter is None or "iban" in kind_filter:
        def _iban_repl(m: re.Match[str]) -> str:
            if not _iban_mod97(m.group(0)):
                return m.group(0)
            rendered = _WORD_PLACEHOLDERS["iban"]
            _bump("iban", rendered)
            return rendered
        text = _IBAN_RE.sub(_iban_repl, text)

    if kind_filter is None or "email" in kind_filter:
        def _email_repl(m: re.Match[str]) -> str:
            domain = m.group(0).rsplit("@", 1)[1]
            if _is_reserved_email_domain(domain):
                return m.group(0)
            rendered = _WORD_PLACEHOLDERS["email"]
            _bump("email", rendered)
            return rendered
        text = _EMAIL_RE.sub(_email_repl, text)

    if kind_filter is None or "phone" in kind_filter:
        def _phone_repl(m: re.Match[str]) -> str:
            raw = m.group(0)
            digits = re.sub(r"\D", "", raw)
            if len(digits) < 10 or len(digits) > 11:
                return raw
            confidence = 0.55 + _context_boost(text, m.start(), m.end(), "phone")
            if confidence < 0.65:
                return raw
            rendered = _digit_template_for("phone", raw, mask_char)
            _bump("phone", rendered)
            return rendered
        text = _PHONE_RE.sub(_phone_repl, text)

    if kind_filter is None or "api_key" in kind_filter:
        def _key_repl(m: re.Match[str]) -> str:
            raw = m.group(0)
            if _shannon_entropy(raw) < 3.5:
                return raw
            rendered = _WORD_PLACEHOLDERS["api_key"]
            _bump("api_key", rendered)
            return rendered
        text = _KEY_RE.sub(_key_repl, text)

    records = [
        Sanitization(kind=k, count=c, pattern=sample_pattern.get(k, "[redacted]"))
        for k, c in counts.items()
    ]
    return text, records


def compute_risk_score(findings: list[PIIFinding]) -> float:
    """Aggregate risk score from 0.0 (clean) to 1.0 (severe).

    Combines finding count, per-finding confidence, and diversity of
    PII kinds.
    """
    if not findings:
        return 0.0

    weighted_sum = sum(f.confidence for f in findings)
    type_diversity = len({f.kind for f in findings})

    raw = (weighted_sum * 0.3) + (type_diversity * 0.15)
    return min(1.0, round(raw / (1.0 + raw) * 2, 4))


# ── Credit card detection (Luhn validated) ──────────────────────────

_CC_RE = re.compile(
    r"\b"
    r"(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"  # issuer prefix
    r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{1,4}"
    r"\b"
)


def _luhn_check(digits: str) -> bool:
    """ISO/IEC 7812 Luhn algorithm."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _scan_credit_cards(text: str) -> list[PIIFinding]:
    findings: list[PIIFinding] = []
    for match in _CC_RE.finditer(text):
        raw = match.group()
        digits = re.sub(r"\D", "", raw)
        if len(digits) < 13 or len(digits) > 19:
            continue
        if not _luhn_check(digits):
            continue
        confidence = 0.85
        confidence += _context_boost(text, match.start(), match.end(), "credit_card")
        findings.append(PIIFinding(
            kind="credit_card",
            value_redacted=digits[:4] + "-****-****-" + digits[-4:],
            confidence=min(1.0, confidence),
            method="luhn",
        ))
    return findings


# ── SSN detection (area-number validated) ───────────────────────────

_SSN_RE = re.compile(r"\b(\d{3})[\s\-](\d{2})[\s\-](\d{4})\b")

# Invalid SSN area numbers (SSA rules)
_INVALID_SSN_AREAS = {0, 666} | set(range(900, 1000))


def _scan_ssns(text: str) -> list[PIIFinding]:
    findings: list[PIIFinding] = []
    for match in _SSN_RE.finditer(text):
        area, group, serial = int(match.group(1)), int(match.group(2)), int(match.group(3))
        if area in _INVALID_SSN_AREAS or group == 0 or serial == 0:
            continue
        confidence = 0.70
        confidence += _context_boost(text, match.start(), match.end(), "ssn")
        findings.append(PIIFinding(
            kind="ssn",
            value_redacted=f"***-**-{serial:04d}",
            confidence=min(1.0, confidence),
            method="area_validation+context",
        ))
    return findings


# ── API key / secret detection (entropy-based) ─────────────────────

_KEY_RE = re.compile(
    r"(?:"
    r"(?:sk|pk|api|key|token|secret|bearer|ghp|gho|ghu|ghs|ghr|glpat|"
    r"AKIA|ASIA|xox[bpsar]|eyJ|sl\.)"  # known prefixes
    r"[\-_]?"
    r"[A-Za-z0-9\-_\.]{20,80}"
    r")",
)

# Fallback: any long high-entropy hex/base64 blob
_BLOB_RE = re.compile(r"\b[A-Za-z0-9+/\-_]{32,128}={0,2}\b")


def _shannon_entropy(s: str) -> float:
    """Shannon entropy in bits per character."""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for ch in s:
        freq[ch] = freq.get(ch, 0) + 1
    length = len(s)
    return -sum((c / length) * math.log2(c / length) for c in freq.values())


def _scan_api_keys(text: str) -> list[PIIFinding]:
    findings: list[PIIFinding] = []
    seen_spans: set[tuple[int, int]] = set()

    for match in _KEY_RE.finditer(text):
        raw = match.group()
        entropy = _shannon_entropy(raw)
        if entropy < 3.5:
            continue
        span = (match.start(), match.end())
        seen_spans.add(span)
        confidence = min(1.0, 0.75 + (entropy - 3.5) * 0.1)
        confidence += _context_boost(text, match.start(), match.end(), "api_key")
        findings.append(PIIFinding(
            kind="api_key",
            value_redacted=raw[:6] + "..." + raw[-4:],
            confidence=min(1.0, confidence),
            method="prefix+entropy",
        ))

    for match in _BLOB_RE.finditer(text):
        span = (match.start(), match.end())
        if any(s[0] <= span[0] and s[1] >= span[1] for s in seen_spans):
            continue
        raw = match.group()
        entropy = _shannon_entropy(raw)
        if entropy < 4.5 or len(raw) < 32:
            continue
        context_bonus = _context_boost(text, match.start(), match.end(), "api_key")
        if context_bonus == 0 and entropy < 5.0:
            continue
        confidence = min(1.0, 0.55 + (entropy - 4.5) * 0.15 + context_bonus)
        findings.append(PIIFinding(
            kind="api_key",
            value_redacted=raw[:6] + "..." + raw[-4:],
            confidence=min(1.0, confidence),
            method="entropy",
        ))

    return findings


# ── Email detection ─────────────────────────────────────────────────

_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
)


def _scan_emails(text: str) -> list[PIIFinding]:
    findings: list[PIIFinding] = []
    for match in _EMAIL_RE.finditer(text):
        raw = match.group()
        local, domain = raw.rsplit("@", 1)
        if _is_reserved_email_domain(domain):
            continue
        confidence = 0.80
        confidence += _context_boost(text, match.start(), match.end(), "email")
        findings.append(PIIFinding(
            kind="email",
            value_redacted=local[0] + "***@" + domain,
            confidence=min(1.0, confidence),
            method="structural",
        ))
    return findings


# ── Phone number detection ──────────────────────────────────────────

_PHONE_RE = re.compile(
    r"(?<!\d)"
    r"(?:\+1[\s\-]?)?"
    r"\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}"
    r"(?!\d)"
)


def _scan_phones(text: str) -> list[PIIFinding]:
    findings: list[PIIFinding] = []
    for match in _PHONE_RE.finditer(text):
        raw = match.group()
        digits = re.sub(r"\D", "", raw)
        if len(digits) < 10 or len(digits) > 11:
            continue
        confidence = 0.55
        confidence += _context_boost(text, match.start(), match.end(), "phone")
        if confidence < 0.65:
            continue
        findings.append(PIIFinding(
            kind="phone",
            value_redacted="(***) ***-" + digits[-4:],
            confidence=min(1.0, confidence),
            method="pattern+context",
        ))
    return findings


# ── IBAN detection (mod-97 checksum) ────────────────────────────────

_IBAN_RE = re.compile(r"\b([A-Z]{2}\d{2}[\s]?[\dA-Z]{4}[\s]?(?:[\dA-Z]{4}[\s]?){1,7}[\dA-Z]{1,4})\b")


def _iban_mod97(iban: str) -> bool:
    """ISO 7064 mod-97 IBAN validation."""
    clean = re.sub(r"\s", "", iban)
    if len(clean) < 15 or len(clean) > 34:
        return False
    rearranged = clean[4:] + clean[:4]
    numeric = ""
    for ch in rearranged:
        if ch.isdigit():
            numeric += ch
        elif ch.isalpha():
            numeric += str(ord(ch.upper()) - 55)
        else:
            return False
    return int(numeric) % 97 == 1


def _scan_ibans(text: str) -> list[PIIFinding]:
    findings: list[PIIFinding] = []
    for match in _IBAN_RE.finditer(text):
        raw = match.group()
        if not _iban_mod97(raw):
            continue
        clean = re.sub(r"\s", "", raw)
        confidence = 0.95
        confidence += _context_boost(text, match.start(), match.end(), "iban")
        findings.append(PIIFinding(
            kind="iban",
            value_redacted=clean[:4] + "****" + clean[-4:],
            confidence=min(1.0, confidence),
            method="mod97",
        ))
    return findings


# ── Context boosting ────────────────────────────────────────────────

def _context_boost(text: str, start: int, end: int, kind: str) -> float:
    """Up to +0.15 confidence based on nearby context keywords."""
    window_start = max(0, start - _CONTEXT_WINDOW)
    window_end = min(len(text), end + _CONTEXT_WINDOW)
    window = text[window_start:window_end].lower()

    keywords = _CONTEXT_KEYWORDS.get(kind, [])
    hits = sum(1 for kw in keywords if kw in window)
    return min(0.15, hits * 0.05)
