"""End-to-end: pii_scan policy → SDK blocks the call."""

from __future__ import annotations


def _pii_rule() -> dict:
    return {
        "id": 1,
        "name": "block-pii",
        "type": "pii_scan",
        "tenant": None,
        "config": {
            "threshold": 0.4,
            "kinds": ["credit_card", "ssn"],
            "message": "PII detected — blocked.",
        },
    }


def test_evaluate_blocks_ssn(fake_backend) -> None:
    fake_backend.set_rules([_pii_rule()], etag='"pii"')

    import egisai

    egisai.init(api_key="egis_live_x", app="a", env="t",
                base_url="http://fake", enable_sse=False)

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
    assert decision.matched_policy == "block-pii"


def test_evaluate_allows_clean_prompt(fake_backend) -> None:
    fake_backend.set_rules([_pii_rule()], etag='"pii"')

    import egisai

    egisai.init(api_key="egis_live_x", app="a", env="t",
                base_url="http://fake", enable_sse=False)

    from egisai._evaluator import InputCall, evaluate

    decision = evaluate(
        InputCall(
            source="test",
            target="x",
            model="gpt-4",
            prompt_text="Say hello in French",
        )
    )
    assert decision.verdict == "allow"
