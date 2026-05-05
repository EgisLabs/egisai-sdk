"""label_redact correctness + the gate's compliance contract on previews.

Two distinct things under test:

1. ``policy.pii.label_redact`` — replaces validated PII in text with
   typed labels (``<SSN>``, ``<EMAIL>``, …), preserving everything
   else. This is the function the SDK gate calls at audit-log time
   to make sure the persisted ``request_text`` field never carries
   raw PII regardless of verdict.

2. The gate's *prompt_preview* / *prompt_preview_before* event
   fields. Compliance contract:

   - ``prompt_preview`` is set on EVERY event, label-redacted.
     Even a block on a PII regex MUST not leave the raw digits in
     the persisted preview (this was a real leak — see the
     historical comment in ``_apply_sanitization``).
   - ``prompt_preview_before`` is set ONLY for ``verdict='sanitize'``.
     Same compliance guarantee: typed labels, never raw digits.
   - The SDK-internal ``_prompt_text_original`` stash key is
     stripped at enqueue time and MUST NOT reach the wire.
"""

from __future__ import annotations

# ── 1. label_redact unit tests ─────────────────────────────────


def test_label_redact_replaces_ssn() -> None:
    from egisai.policy.pii import label_redact

    assert label_redact("My SSN is 123-45-6789.") == "My SSN is <SSN>."


def test_label_redact_replaces_email() -> None:
    from egisai.policy.pii import label_redact

    assert label_redact("contact jane@acme.com please") == "contact <EMAIL> please"


def test_label_redact_replaces_credit_card_luhn_validated_only() -> None:
    """Only Luhn-valid sequences should be redacted."""
    from egisai.policy.pii import label_redact

    # Luhn-valid Visa test number → label.
    assert "<CREDIT_CARD>" in label_redact("card 4111-1111-1111-1111")
    # Luhn-INVALID 16-digit number → left alone (would be a false positive).
    out = label_redact("phantom 4111-1111-1111-1234")
    assert "<CREDIT_CARD>" not in out
    assert "4111-1111-1111-1234" in out


def test_label_redact_clean_text_is_unchanged() -> None:
    from egisai.policy.pii import label_redact

    assert label_redact("Say hello in French.") == "Say hello in French."


def test_label_redact_empty_text_passes_through() -> None:
    from egisai.policy.pii import label_redact

    assert label_redact("") == ""
    assert label_redact(None) is None  # type: ignore[arg-type]


def test_label_redact_handles_word_form_ssn() -> None:
    """Spelled-out SSN → single ``<SSN>`` label, not a partial collapse."""
    from egisai.policy.pii import label_redact

    out = label_redact("social: one two three four five six seven eight nine")
    assert out.count("<SSN>") == 1
    assert "one two three" not in out.lower()


def test_label_redact_handles_fullwidth_digits() -> None:
    """NFKC normalisation is in scan + label_redact."""
    from egisai.policy.pii import label_redact

    out = label_redact("My SSN is １２３-４５-６７８９")
    assert "<SSN>" in out


def test_label_redact_kinds_filter() -> None:
    """When ``kinds`` is given, only those detectors run."""
    from egisai.policy.pii import label_redact

    raw = "ssn 123-45-6789 and email jane@acme.com"
    # Only SSN — email left alone.
    out = label_redact(raw, kinds=["ssn"])
    assert "<SSN>" in out
    assert "jane@acme.com" in out
    assert "<EMAIL>" not in out


# ── 2. Gate stamps prompt_preview correctly ────────────────────


def _block_pii_rule() -> dict:
    return {
        "id": 1,
        "name": "block-ssn",
        "type": "pii_scan",
        "tenant": None,
        "config": {"action": "block", "kinds": ["ssn"], "threshold": 0.5},
    }


def _sanitize_pii_rule() -> dict:
    return {
        "id": 1,
        "name": "redact-ssn",
        "type": "pii_scan",
        "tenant": None,
        "config": {
            "action": "sanitize",
            "kinds": ["ssn"],
            "threshold": 0.5,
        },
    }


def _allow_only_rule() -> dict:
    """No PII rule at all — every call goes through unchanged."""
    return {
        "id": 1,
        "name": "allow-everything",
        "type": "allow_model",
        "tenant": None,
        "config": {"models": ["gpt-4"]},
    }


def _drain_events() -> list[dict]:
    from egisai import _logger

    out: list[dict] = []
    while not _logger._q.empty():
        out.append(_logger._q.get_nowait())
    return out


def test_block_on_pii_does_not_leak_raw_value_in_prompt_preview(
    fake_backend,
) -> None:
    """Compliance regression: a block-on-PII MUST label-redact the preview.

    Historically the gate built ``payload_preview`` from the raw payload
    BEFORE any redaction ran, then enqueued the event for blocked calls
    immediately. That meant blocked PII calls carried the raw digits to
    the audit log — silent leak. The test asserts the leak is closed.
    """
    import egisai
    from egisai._patches._common import gate_call

    fake_backend.set_rules([_block_pii_rule()], etag='"block-pii"')
    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="stub",
    )

    payload = {
        "messages": [{"role": "user", "content": "My SSN is 123-45-6789, save it."}],
        "tools": None,
    }

    def fake_forward():
        raise AssertionError("forward() should not run on a block verdict")

    def stub(decision, trace_id, model):
        # Patch wires this for blocked calls.
        return {"blocked": True}

    gate_call(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4",
        prompt_text="My SSN is 123-45-6789, save it.",
        stream=False,
        payload=payload,
        stub_factory=stub,
        forward=fake_forward,
    )

    drained = _drain_events()
    assert len(drained) == 1
    ev = drained[0]
    assert ev["verdict"] == "block"
    # The persisted preview MUST have the SSN replaced by ``<SSN>``.
    assert "123-45-6789" not in ev["prompt_preview"]
    assert "<SSN>" in ev["prompt_preview"]
    # The internal stash key MUST NOT reach the wire.
    assert "_prompt_text_original" not in ev


def test_sanitize_emits_both_before_and_after_previews(fake_backend) -> None:
    """Sanitize sets prompt_preview_before (label form) and prompt_preview
    (post-mask form). Neither carries raw digits."""
    import egisai
    from egisai._patches._common import gate_call

    fake_backend.set_rules([_sanitize_pii_rule()], etag='"san-pii"')
    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    payload = {
        "messages": [{"role": "user", "content": "My SSN is 123-45-6789."}],
        "tools": None,
    }

    def fake_forward():
        class _R:
            usage = type("U", (), {"prompt_tokens": 5, "completion_tokens": 2})()
        return _R()

    gate_call(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4",
        prompt_text="My SSN is 123-45-6789.",
        stream=False,
        payload=payload,
        forward=fake_forward,
    )

    drained = _drain_events()
    assert len(drained) == 1
    ev = drained[0]

    assert ev["verdict"] == "sanitize"
    # Before — structural label form.
    before = ev["prompt_preview_before"]
    assert before is not None
    assert "<SSN>" in before
    assert "123-45-6789" not in before
    # After — masked form (digits replaced with #).
    after = ev["prompt_preview"]
    assert after is not None
    assert "###-##-####" in after
    assert "123-45-6789" not in after


def test_allow_emits_prompt_preview_no_before(fake_backend) -> None:
    """Allow verdict: ``prompt_preview`` set, ``prompt_preview_before`` absent."""
    import egisai
    from egisai._patches._common import gate_call

    fake_backend.set_rules([_allow_only_rule()], etag='"allow"')
    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    payload = {"messages": [{"role": "user", "content": "Say hello"}], "tools": None}

    def fake_forward():
        class _R:
            usage = type("U", (), {"prompt_tokens": 3, "completion_tokens": 1})()
        return _R()

    gate_call(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4",
        prompt_text="Say hello",
        stream=False,
        payload=payload,
        forward=fake_forward,
    )

    drained = _drain_events()
    ev = drained[-1]
    assert ev["verdict"] == "allow"
    assert ev["prompt_preview"] == "Say hello"
    # Before is only set for sanitize. Either absent or None.
    assert not ev.get("prompt_preview_before")


def test_prompt_preview_truncated_to_2kb(fake_backend) -> None:
    """A 5 KB prompt persists as ~2 KB + ``...`` suffix — bound the audit row."""
    import egisai
    from egisai._patches._common import gate_call

    fake_backend.set_rules([_allow_only_rule()], etag='"allow"')
    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    big = "abcdef" * 1000  # 6 KB of plain ASCII, no PII
    payload = {"messages": [{"role": "user", "content": big}], "tools": None}

    def fake_forward():
        class _R:
            usage = type("U", (), {"prompt_tokens": 1000, "completion_tokens": 1})()
        return _R()

    gate_call(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4",
        prompt_text=big,
        stream=False,
        payload=payload,
        forward=fake_forward,
    )

    drained = _drain_events()
    ev = drained[-1]
    assert ev["prompt_preview"].endswith("...")
    # 2048 cap is the contract; allow a small slack for ``...`` suffix.
    assert len(ev["prompt_preview"]) <= 2048
