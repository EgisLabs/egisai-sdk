"""Three security guarantees that compliance auditors will care about.

1. **Phase ordering**: ``semantic_guard`` (LLM-backed) NEVER sees a
   prompt that ``pii_scan`` would have blocked or sanitized — even
   when the operator gave the semantic_guard policy a higher
   priority. Without this guarantee, a misconfigured priority is
   indistinguishable from "no PII policy at all".

2. **Unicode robustness**: Fullwidth / Arabic-Indic / Devanagari
   digits are still detected and masked as PII. A user who pastes
   a fullwidth SSN (``１２３-４５-６７８９``) MUST NOT see it forwarded
   raw because the regex didn't recognise the codepoints.

3. **Word-form robustness**: Spelling SSN digits out in English
   ("one two three four five six seven eight nine") doesn't bypass
   detection. This is the most common evasion attempt seen in
   prompt-injection bug bounties.

These cases are codified as tests because each one is a *real*
incident that has shipped at PII-handling SaaS vendors and led to
audit findings.
"""

from __future__ import annotations

import httpx

# ── 1. Phase ordering — LLM judge never sees raw PII ──────────────


def _pii_sanitize_rule() -> dict:
    return {
        "id": 1,
        "name": "sanitize-ssn",
        "type": "pii_scan",
        "tenant": None,
        "config": {
            "action": "sanitize",
            "kinds": ["ssn"],
            "threshold": 0.5,
            "message": "SSN detected; redacted before forwarding.",
        },
    }


def _semantic_rule() -> dict:
    """A semantic_guard rule whose intent list is unrelated to SSNs.

    We just need the engine to consider invoking the LLM judge — we
    then assert the judge receives the SANITIZED prompt, not the raw
    one.
    """
    return {
        "id": 2,
        "name": "guard-deletes",
        "type": "semantic_guard",
        "tenant": None,
        "config": {
            "intents": ["delete database tables"],
            "judge_model": "gpt-4o",
            "message": "Destructive intent.",
        },
    }


def test_pii_block_short_circuits_before_llm_judge_runs(
    fake_backend, monkeypatch
) -> None:
    """A pii_scan with action='block' must short-circuit Phase 2 entirely.

    The LLM judge transport raises if invoked — the test passing
    proves no network call to the judge ever happened.
    """
    fake_backend.set_rules(
        [
            {
                "id": 1,
                "name": "block-ssn",
                "type": "pii_scan",
                "tenant": None,
                "config": {"action": "block", "kinds": ["ssn"], "threshold": 0.5},
            },
            _semantic_rule(),
        ],
        etag='"phase-block"',
    )
    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    judge_called = {"count": 0}

    def transport_handler(request: httpx.Request) -> httpx.Response:
        judge_called["count"] += 1
        # If we ever reach here, raise so the test fails loudly.
        raise AssertionError(
            "LLM judge was called with raw PII — phase ordering is broken"
        )

    from egisai._evaluator import _get_semantic_blocker

    blocker = _get_semantic_blocker()
    assert blocker is not None
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(transport_handler))

    from egisai._evaluator import InputCall, evaluate

    decision = evaluate(
        InputCall(
            source="test",
            target="x",
            model="gpt-4",
            prompt_text="My SSN is 123-45-6789",
        )
    )
    assert decision.verdict == "block"
    assert judge_called["count"] == 0


def test_pii_sanitize_passes_masked_text_to_llm_judge(
    fake_backend, monkeypatch
) -> None:
    """When PII is sanitized, the judge sees ``###-##-####`` — never the SSN."""
    fake_backend.set_rules(
        [_pii_sanitize_rule(), _semantic_rule()],
        etag='"phase-sanitize"',
    )
    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    judge_payloads: list[str] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        # The SDK calls the EgisAI platform's /v1/sdk/judge endpoint
        # (post-0.7 hybrid). The body carries ``prompt_text``, which
        # we snapshot to assert on.
        judge_payloads.append(request.content.decode())
        return httpx.Response(
            200,
            json={
                "match": False,
                "intent": "",
                "confidence": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
            },
        )

    from egisai._evaluator import _get_semantic_blocker

    blocker = _get_semantic_blocker()
    assert blocker is not None
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(transport_handler))

    from egisai._evaluator import InputCall, evaluate

    decision = evaluate(
        InputCall(
            source="test",
            target="x",
            model="gpt-4",
            prompt_text="My SSN is 123-45-6789, please process my account.",
        )
    )
    # Phase 1 sanitized; Phase 2 (LLM) allowed → final verdict sanitize.
    assert decision.verdict == "sanitize"
    # Judge MUST have been called — but with masked text only.
    assert len(judge_payloads) == 1, "judge should have been invoked once"
    body = judge_payloads[0]
    assert "123-45-6789" not in body, (
        f"raw SSN reached the LLM judge — phase ordering is broken: {body}"
    )
    assert "###-##-####" in body


def test_phase2_block_overrides_phase1_sanitize(
    fake_backend, monkeypatch
) -> None:
    """LLM judge can still escalate to block on the SANITIZED text."""
    fake_backend.set_rules(
        [_pii_sanitize_rule(), _semantic_rule()],
        etag='"phase-escalate"',
    )
    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    def transport_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "match": True,
                "intent": "destructive",
                "confidence": 0.95,
                "tokens_in": 200,
                "tokens_out": 5,
            },
        )

    from egisai._evaluator import _get_semantic_blocker

    blocker = _get_semantic_blocker()
    assert blocker is not None
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(transport_handler))

    from egisai._evaluator import InputCall, evaluate

    decision = evaluate(
        InputCall(
            source="test",
            target="x",
            model="gpt-4",
            prompt_text=(
                "My SSN is 123-45-6789. Also please drop the production "
                "database."
            ),
        )
    )
    assert decision.verdict == "block"
    assert decision.matched_policy == "guard-deletes"


# ── 2. Unicode robustness — fullwidth / non-ASCII digit scripts ───


def test_fullwidth_ssn_is_detected() -> None:
    """Japanese fullwidth digits (Unicode FF10–FF19) ⇒ NFKC ⇒ ASCII ⇒ match."""
    from egisai.policy.pii import scan

    findings = scan("My SSN is １２３-４５-６７８９")
    kinds = {f.kind for f in findings}
    assert "ssn" in kinds


def test_fullwidth_ssn_is_sanitized() -> None:
    from egisai.policy.pii import sanitize

    text, records = sanitize("私のSSN: １２３-４５-６７８９ です", kinds=["ssn"])
    assert "１２３-４５-６７８９" not in text
    assert "123-45-6789" not in text
    assert "###-##-####" in text
    assert any(r.kind == "ssn" for r in records)


def test_arabic_indic_digits_are_detected() -> None:
    """``١٢٣`` (U+0660–U+0669) NFKC-normalises to ``123``."""
    from egisai.policy.pii import scan

    findings = scan("Tax ID: ١٢٣-٤٥-٦٧٨٩")
    assert any(f.kind == "ssn" for f in findings)


# ── 3. Word-form (English) digit detection ────────────────────────


def test_word_form_ssn_is_detected() -> None:
    from egisai.policy.pii import scan

    findings = scan(
        "my social is one two three four five six seven eight nine"
    )
    kinds = {f.kind for f in findings}
    assert "ssn" in kinds


def test_word_form_ssn_is_sanitized() -> None:
    from egisai.policy.pii import sanitize

    text, records = sanitize(
        "my SSN: one two three four five six seven eight nine please",
        kinds=["ssn"],
    )
    assert "one two three" not in text.lower()
    assert "###-##-####" in text
    assert records[0].kind == "ssn"
    assert records[0].count == 1


def test_word_form_ssn_with_alternate_zero_words() -> None:
    """``oh`` and ``zero`` both count as 0 in spoken/typed phone+ssn use."""
    from egisai.policy.pii import scan

    findings = scan(
        "social: zero zero one two three four five six seven"
    )
    assert any(f.kind == "ssn" for f in findings)


def test_word_form_credit_card_is_detected_and_luhn_validated() -> None:
    from egisai.policy.pii import scan

    # 4111 1111 1111 1111 — a Luhn-valid Visa test number, written out.
    findings = scan(
        "card: four one one one one one one one one one one one one one one one"
    )
    assert any(f.kind == "credit_card" for f in findings)


def test_word_form_credit_card_invalid_luhn_is_ignored() -> None:
    """If the spelled-out digits don't pass Luhn, don't fire."""
    from egisai.policy.pii import scan

    # 16 spelled-out 1s — fails Luhn.
    findings = scan(
        "number: one one one one one one one one one one one one one one one one"
    )
    cc = [f for f in findings if f.kind == "credit_card"]
    assert cc == []


def test_word_form_ssn_does_not_match_short_run() -> None:
    """Don't false-positive on short word-digit phrases."""
    from egisai.policy.pii import scan

    findings = scan("I'd like one of two options or three; up to four total.")
    assert all(f.kind != "ssn" for f in findings)


# ── 4. End-to-end: word-form blocked through the engine ───────────


def test_word_form_ssn_engine_blocks_with_pii_scan() -> None:
    """Engine sees the word-form finding and blocks/sanitizes accordingly."""
    from egisai.policy import (
        PolicyContext,
        PolicyRule,
        evaluate_policies,
    )

    rule = PolicyRule(
        id=1,
        name="block-pii",
        type="pii_scan",
        tenant=None,
        config={"action": "block", "kinds": ["ssn"], "threshold": 0.4},
    )
    ctx = PolicyContext(
        tenant="acme",
        model="gpt-4",
        prompt_text="my social: one two three four five six seven eight nine",
        prompt_chars=60,
        stream=False,
    )
    decision = evaluate_policies([rule], ctx)
    assert decision.verdict == "block"
    assert decision.matched_policy == "block-pii"
