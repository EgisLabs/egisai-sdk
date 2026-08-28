"""Prompt-injection detection — deterministic, local, no network.

The problem this solves is not "somebody typed a rude prompt". It is
that an agent reads text it did not author — a web page, a PDF, a
Jira ticket, a tool result — and that text contains instructions
addressed to the model rather than to the reader. The model has no way
to tell the difference, so the instruction runs with the agent's
credentials. Every published agent compromise of the last two years is
some version of that sentence.

This module is the local *pre-filter* tier
-------------------------------------------
Detection is two-tier by design. This module is the first tier: a
fast, standard, fully-local regex pass that runs on every governed
call, sub-millisecond, offline, and leaks nothing. Everything in it
is public knowledge — the shapes below are documented in the OWASP
LLM Top 10 and every prompt-injection write-up of the last two years,
so keeping them local costs no proprietary advantage and buys instant,
network-free blocking of the obvious attacks.

The second tier is the platform's *smart* tier — a fine-tuned
classifier plus EgisAI's calibrated pattern/scoring extensions —
reached over HTTP by :class:`egisai.policy.injection_client.InjectionBlocker`.
That tier holds the proprietary IP and never ships in a public PyPI
artefact. The engine escalates to it only when this local pre-filter
did *not* already block, and only after Phase 1 has masked PII, so an
obvious attack is still refused instantly with no network and no data
egress. Offline / air-gapped deployments run on this local tier alone
(documented best-effort); online deployments additionally get the
classifier's recall on paraphrased, multilingual, and novel attacks.

Why patterns rather than a model *here*
---------------------------------------
A classifier would score better on a benchmark. It would also mean a
network call or a 400 MB ONNX bundle inside the hot path of every
governed request, which breaks two rules this local tier does not
break: it stays local, and it stays under a millisecond. So this tier
is regex, scoped to the *shapes* injections have to take, with the
model-grade recall provided by the smart tier above.

That scoping is what makes it work. An injection has a job — override
the model's instructions, extract its configuration, or get it to
exfiltrate what it can reach — and each job has a small set of
surface forms in any language that reads left to right. We match those
forms, score them, and let the operator set the bar.

The six classes
---------------
``instruction_override``
    "Ignore the above", "disregard your previous instructions",
    "forget everything you were told". The classic.
``role_hijack``
    Fabricated conversation structure — ``[system]``, ``###
    system:``, ``<|im_start|>system`` — trying to make the model read
    attacker text as a higher-privilege turn.
``exfiltration``
    "Send the contents to…", "POST this to…", a markdown image whose
    URL carries the data. The instruction that turns a read into a
    breach.
``prompt_extraction``
    "Repeat the words above", "what is your system prompt", "output
    your instructions verbatim". Recon, and usually the first step.
``guardrail_bypass``
    "Developer mode", "DAN", "you have no restrictions", "this is a
    hypothetical so the rules don't apply".
``encoded_payload``
    Long base64 runs, dense zero-width characters, or tag-block
    Unicode — an instruction hidden from the human reviewing the page
    but perfectly legible to the tokenizer.

Scoring
-------
Every hit carries a weight. The score is the highest single weight
plus a small bonus for corroboration, capped at 1.0 — deliberately
*not* a sum, because three weak lexical hits in a long document is
what normal prose looks like, while one unambiguous hit is an attack
whatever else is on the page.

The operator sets ``threshold``. Below it, nothing happens. At or
above it, the rule's ``action`` applies: ``flag`` records the finding
and lets the call through, ``block`` refuses.

What this deliberately does not do
----------------------------------
It does not attempt to be complete, and the docstring says so where an
operator will read it. Paraphrased instructions in Spanish, an
injection written as a poem, or a novel encoding will pass. The honest
framing is a cheap, high-precision first filter that runs on every
call — with ``semantic_guard`` behind it for the cases only a language
model can judge.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = [
    "CLASSES",
    "InjectionFinding",
    "InjectionResult",
    "scan",
]


#: The classes a finding can carry. A tuple rather than an enum so the
#: wire vocabulary can grow without a migration on either side.
CLASSES: tuple[str, ...] = (
    "instruction_override",
    "role_hijack",
    "exfiltration",
    "prompt_extraction",
    "guardrail_bypass",
    "encoded_payload",
)


@dataclass(frozen=True, slots=True)
class InjectionFinding:
    """One pattern that fired.

    ``excerpt`` is the matched span, truncated. It is included because
    an operator triaging a block needs to see *what* fired, and the
    matched text is by construction the attacker's words rather than
    the customer's data — but it is truncated hard because a greedy
    pattern on a long document could otherwise pull a paragraph of
    surrounding content into an audit row.
    """

    cls: str
    weight: float
    excerpt: str


@dataclass(frozen=True, slots=True)
class InjectionResult:
    score: float
    findings: tuple[InjectionFinding, ...]

    @property
    def classes(self) -> tuple[str, ...]:
        """Distinct classes that fired, strongest first."""
        seen: list[str] = []
        for f in sorted(self.findings, key=lambda x: -x.weight):
            if f.cls not in seen:
                seen.append(f.cls)
        return tuple(seen)

    @property
    def primary(self) -> InjectionFinding | None:
        return max(self.findings, key=lambda f: f.weight, default=None)


# ── Patterns ────────────────────────────────────────────────────────
#
# Weights are calibrated so a single 0.9 hit clears any sane threshold
# on its own, a 0.6 needs corroboration, and a 0.35 is a hint that only
# matters in company. Every pattern is anchored on a verb or a literal
# delimiter — nothing here matches on a bare noun, because "system" and
# "instructions" are words that appear in ordinary technical writing
# and a detector that fires on them is a detector operators turn off.
#
# Written to be readable rather than clever: no nested quantifiers, no
# backreferences, bounded ``{0,n}`` gaps instead of ``.*``. That keeps
# them outside the shapes ``_regex_safe`` rejects, and keeps the whole
# pass linear in the length of the input.

_PATTERNS: tuple[tuple[str, str, float], ...] = (
    # ── instruction_override ────────────────────────────────────────
    (
        "instruction_override",
        r"\b(?:ignore|disregard|forget|discard|override)\b[^.\n]{0,40}?"
        r"\b(?:above|previous|prior|earlier|preceding|foregoing|all)\b"
        r"[^.\n]{0,30}?\b(?:instruction|instructions|prompt|prompts|"
        r"direction|directions|rule|rules|guideline|guidelines|context)\b",
        0.9,
    ),
    (
        "instruction_override",
        r"\b(?:ignore|disregard|forget)\b[^.\n]{0,20}?"
        r"\b(?:everything|anything)\b[^.\n]{0,30}?"
        r"\b(?:above|before|previously|told|said)\b",
        0.85,
    ),
    (
        "instruction_override",
        r"\byour\s+(?:new|real|actual|true|updated)\s+"
        r"(?:instruction|instructions|task|role|job|objective|mission)\b",
        0.8,
    ),
    (
        "instruction_override",
        r"\b(?:from\s+now\s+on|starting\s+now|going\s+forward)\b"
        r"[^.\n]{0,30}?\byou\s+(?:will|must|are|shall)\b",
        0.55,
    ),
    (
        "instruction_override",
        r"\b(?:stop|cease)\s+(?:following|obeying|applying)\b",
        0.6,
    ),
    # ── role_hijack ─────────────────────────────────────────────────
    #
    # Chat-template delimiters appearing inside content are the single
    # highest-precision signal in this whole module. A legitimate
    # document does not contain ``<|im_start|>``.
    (
        "role_hijack",
        r"<\|(?:im_start|im_end|system|endoftext|start_header_id)\|?>?",
        0.95,
    ),
    (
        "role_hijack",
        r"(?:^|\n)\s*(?:#{1,4}\s*)?\[?\s*"
        r"(?:system|assistant|developer)\s*\]?\s*(?::|>)\s",
        0.75,
    ),
    (
        "role_hijack",
        r"\[\s*(?:INST|/INST|SYSTEM|SYS)\s*\]",
        0.9,
    ),
    (
        "role_hijack",
        r"\byou\s+are\s+now\s+(?:a|an|the)\b[^.\n]{0,40}?"
        r"\b(?:assistant|model|ai|bot|agent|system)\b",
        0.65,
    ),
    (
        "role_hijack",
        r"\b(?:act|behave|respond)\s+as\s+(?:if\s+you\s+are\s+)?"
        r"(?:a|an|the)\s+(?:system|admin|administrator|root|developer)\b",
        0.7,
    ),
    # ── exfiltration ────────────────────────────────────────────────
    (
        "exfiltration",
        r"\b(?:send|post|upload|transmit|forward|exfiltrate|leak)\b"
        r"[^.\n]{0,40}?\bto\b\s*(?:https?://|www\.|[\w.-]+@)",
        0.9,
    ),
    (
        "exfiltration",
        r"!\[[^\]]{0,40}\]\(\s*https?://[^)\s]{0,200}"
        r"(?:\?|&)[^)\s]{0,200}\)",
        0.6,
    ),
    (
        "exfiltration",
        r"\b(?:curl|wget|fetch|requests\.(?:get|post))\b[^\n]{0,40}?"
        r"https?://",
        0.55,
    ),
    (
        "exfiltration",
        r"\b(?:include|append|attach|embed)\b[^.\n]{0,30}?"
        r"\b(?:api[_\s-]?key|token|password|secret|credential"
        r"|conversation|chat\s+history)\b[^.\n]{0,40}?"
        r"\b(?:url|link|image|request|query)\b",
        0.85,
    ),
    # ── prompt_extraction ───────────────────────────────────────────
    (
        "prompt_extraction",
        r"\b(?:repeat|print|output|reveal|show|display|echo|dump|recite)\b"
        r"[^.\n]{0,40}?\b(?:system\s+prompt|initial\s+prompt|"
        r"your\s+instructions|the\s+instructions\s+above|"
        r"everything\s+above|the\s+text\s+above|words\s+above|"
        r"prompt\s+verbatim)\b",
        0.9,
    ),
    (
        "prompt_extraction",
        r"\bwhat\s+(?:is|are|were)\b[^?\n]{0,30}?"
        r"\b(?:your\s+system\s+prompt|your\s+original\s+instructions|"
        r"your\s+initial\s+instructions)\b",
        0.85,
    ),
    (
        "prompt_extraction",
        r"\bverbatim\b[^.\n]{0,30}?\b(?:above|prompt|instructions)\b",
        0.6,
    ),
    # ── guardrail_bypass ────────────────────────────────────────────
    (
        "guardrail_bypass",
        r"\b(?:developer|debug|god|jailbreak|unrestricted|unfiltered)\s+mode\b",
        0.85,
    ),
    (
        "guardrail_bypass",
        r"\b(?:DAN|AIM|STAN)\s+mode\b|\bdo\s+anything\s+now\b",
        0.85,
    ),
    (
        "guardrail_bypass",
        r"\byou\s+(?:have\s+no|are\s+not\s+bound\s+by|are\s+free\s+from)\b"
        r"[^.\n]{0,30}?\b(?:restriction|restrictions|limit|limits|"
        r"rule|rules|filter|filters|guideline|guidelines|policy|policies)\b",
        0.8,
    ),
    (
        "guardrail_bypass",
        r"\b(?:this\s+is|it'?s)\s+(?:just|only|purely)?\s*"
        r"(?:a\s+)?(?:hypothetical|fiction|fictional|roleplay|"
        r"simulation|test|game)\b[^.\n]{0,40}?"
        r"\b(?:so|therefore|thus|hence)\b",
        0.5,
    ),
    (
        "guardrail_bypass",
        r"\bwithout\s+(?:any\s+)?(?:warnings?|disclaimers?|refusals?|"
        r"moral\s+judgement)\b",
        0.45,
    ),
)


_COMPILED: tuple[tuple[str, re.Pattern[str], float], ...] = tuple(
    (cls, re.compile(src, re.IGNORECASE | re.MULTILINE), weight)
    for cls, src, weight in _PATTERNS
)


# ── Encoded payloads ────────────────────────────────────────────────
#
# Handled outside the pattern table because they are measurements, not
# matches: "how much of this looks like base64" and "how many invisible
# characters are there" are thresholds, and expressing a threshold as a
# regex means picking an arbitrary run length and pretending it's a
# rule.

#: A base64 run this long inside prose is not prose. 120 characters is
#: about 90 bytes decoded — long enough to carry an instruction, short
#: enough that a legitimate inline hash or short token (a JWT header,
#: a git SHA, a data URI's first line) doesn't trip it.
_B64_RUN = re.compile(r"[A-Za-z0-9+/]{120,}={0,2}")

#: Zero-width and directional-override characters. Individually these
#: are legitimate (a ZWJ inside an emoji sequence, a soft hyphen in
#: typeset text); in quantity inside a document they are a payload
#: hidden from every human who reviews the page.
_INVISIBLE = re.compile(
    "[\u200b\u200c\u200d\u2060\ufeff\u00ad\u202a-\u202e\u2066-\u2069]"
)
_INVISIBLE_MIN = 12

#: Unicode "tag" characters (U+E0000 block). They render as nothing
#: anywhere, and they map one-to-one onto ASCII — which makes them a
#: perfect, and increasingly popular, carrier for a full instruction.
#: Any occurrence at all is reportable; there is no benign use in
#: content an agent reads.
_TAG_CHARS = re.compile("[\U000e0000-\U000e007f]")


def _excerpt(text: str, start: int, end: int, *, limit: int = 80) -> str:
    """The matched span, bounded and flattened onto one line."""
    span = text[start:end]
    span = " ".join(span.split())
    return span if len(span) <= limit else span[: limit - 1] + "…"


def scan(text: str, *, classes: tuple[str, ...] | None = None) -> InjectionResult:
    """Score ``text`` for prompt-injection shapes.

    ``classes`` narrows the scan to a subset — an operator who only
    cares about exfiltration attempts in tool results shouldn't pay for
    the jailbreak patterns, and shouldn't get their false positives
    either. ``None`` runs everything.

    Normalizes to NFKC first, for the same reason the PII detectors do:
    an instruction written in fullwidth or mathematical-alphanumeric
    characters is the same instruction to a tokenizer, and a detector
    that only reads ASCII is a detector with a documented bypass.

    Never raises. A detector that can throw is a detector that can take
    down the customer's call path, and this one runs on every request.
    """
    if not text:
        return InjectionResult(score=0.0, findings=())

    try:
        return _scan(text, classes)
    except Exception:  # noqa: BLE001 — see the docstring
        return InjectionResult(score=0.0, findings=())


def _scan(text: str, classes: tuple[str, ...] | None) -> InjectionResult:
    wanted = set(classes) if classes else None

    # Measure the invisible characters *before* normalizing — NFKC
    # leaves most of them alone but the count is about the raw bytes
    # the model will actually receive.
    findings: list[InjectionFinding] = []
    if wanted is None or "encoded_payload" in wanted:
        findings.extend(_encoded_findings(text))

    normalized = unicodedata.normalize("NFKC", text)
    for cls, pattern, weight in _COMPILED:
        if wanted is not None and cls not in wanted:
            continue
        match = pattern.search(normalized)
        if match is None:
            continue
        findings.append(
            InjectionFinding(
                cls=cls,
                weight=weight,
                excerpt=_excerpt(normalized, match.start(), match.end()),
            )
        )

    return InjectionResult(score=_score(findings), findings=tuple(findings))


def _encoded_findings(text: str) -> list[InjectionFinding]:
    out: list[InjectionFinding] = []

    tag = _TAG_CHARS.search(text)
    if tag is not None:
        n = len(_TAG_CHARS.findall(text))
        out.append(
            InjectionFinding(
                cls="encoded_payload",
                weight=0.95,
                excerpt=f"{n} invisible Unicode tag character(s)",
            )
        )

    invisible = len(_INVISIBLE.findall(text))
    if invisible >= _INVISIBLE_MIN:
        out.append(
            InjectionFinding(
                cls="encoded_payload",
                weight=0.7,
                excerpt=f"{invisible} zero-width or directional characters",
            )
        )

    b64 = _B64_RUN.search(text)
    if b64 is not None:
        out.append(
            InjectionFinding(
                cls="encoded_payload",
                weight=0.5,
                excerpt=f"{len(b64.group(0))}-character base64 run",
            )
        )

    return out


def _score(findings: list[InjectionFinding]) -> float:
    """Highest weight, plus a bounded bonus for corroboration.

    Not a sum. Summing rewards length: paste a long enough document and
    enough weak lexical patterns eventually clear any threshold, which
    turns the detector into a document-size alarm. Taking the maximum
    keeps one unambiguous hit decisive on its own, and the bonus — a
    tenth per additional *distinct class*, capped at two — recognizes
    that "override the instructions AND send the result somewhere" is
    more than either half.
    """
    if not findings:
        return 0.0
    best = max(f.weight for f in findings)
    extra_classes = len({f.cls for f in findings}) - 1
    bonus = 0.1 * min(extra_classes, 2)
    return round(min(1.0, best + bonus), 3)
