"""pii_scan with action='sanitize' → mask PII in the prompt and forward.

Three layers under test:

  1. ``policy.pii.sanitize`` — the lowest-level masker. Validates that
     the regex-based detectors mask SSN/CC/email/phone/IBAN/api-keys
     while preserving shape, and that the audit record (Sanitization)
     reflects what was redacted *without* leaking the original value.

  2. ``policy.engine.evaluate_policies`` — confirm that pii_scan with
     ``action: "sanitize"`` returns a sanitize-verdict decision while
     the same policy with the legacy default still blocks.

  3. ``_evaluator.mutate_prompt_text`` — the framework-agnostic walker
     that the SDK gate calls on payloads before forward(). Covers
     OpenAI chat ``messages``, Responses-API ``input``, and Gemini
     ``contents`` shapes, and verifies that system messages are NOT
     mutated.

End-to-end (gate-level) coverage lives in test_patches.py via fake
clients; this module exercises the units that compose the feature.
"""

from __future__ import annotations

# ── Layer 1: pii.sanitize ─────────────────────────────────────────


def test_sanitize_masks_ssn_preserving_shape() -> None:
    from egisai.policy.pii import sanitize

    text, records = sanitize("My SSN is 123-45-6789, please keep it safe.")

    assert "123-45-6789" not in text
    assert "###-##-####" in text
    assert len(records) == 1
    assert records[0].type == "ssn"
    assert records[0].count == 1
    assert records[0].pattern == "###-##-####"


def test_sanitize_only_kinds_filter_skips_others() -> None:
    """Asking only for SSN must leave a credit card alone."""
    from egisai.policy.pii import sanitize

    raw = "SSN 123-45-6789 and card 4111-1111-1111-1111"
    text, records = sanitize(raw, types=["ssn"])

    assert "###-##-####" in text
    assert "4111-1111-1111-1111" in text  # CC untouched
    types = {r.type for r in records}
    assert types == {"ssn"}


def test_sanitize_clean_prompt_is_no_op() -> None:
    from egisai.policy.pii import sanitize

    raw = "Say hello in French."
    text, records = sanitize(raw)

    assert text == raw
    assert records == []


def test_sanitize_records_count_when_multiple_matches() -> None:
    from egisai.policy.pii import sanitize

    # Both SSNs have valid area numbers (≠ 0/666/900-999) so the
    # SSA-rule validator accepts them as real SSNs to redact.
    raw = "SSNs: 123-45-6789 and 222-33-4444"
    text, records = sanitize(raw, types=["ssn"])

    assert "123-45-6789" not in text
    assert "222-33-4444" not in text
    assert len(records) == 1
    # Single rolled-up record per kind, but with the full count.
    assert records[0].type == "ssn"
    assert records[0].count == 2


def test_sanitize_does_not_log_original_value() -> None:
    """SOC 2 / GDPR guarantee: the audit record never carries raw PII."""
    from egisai.policy.pii import sanitize

    raw = "John Doe SSN 123-45-6789"
    _, records = sanitize(raw, types=["ssn"])

    blob = repr(records)
    assert "123-45-6789" not in blob
    assert "John Doe" not in blob


# ── Layer 2: policy.engine ────────────────────────────────────────


def _make_ctx(text: str):
    from egisai.policy import PolicyContext

    return PolicyContext(
        tenant="acme",
        model="gpt-4",
        prompt_text=text,
        prompt_chars=len(text),
        stream=False,
    )


def _ssn_rule(action: str | None) -> list:
    from egisai.policy import PolicyRule

    config = {"threshold": 0.5, "kinds": ["ssn"]}
    if action is not None:
        config["action"] = action
    return [
        PolicyRule(
            id=1,
            name="ssn-handler",
            type="pii_scan",
            tenant=None,
            config=config,
        )
    ]


def test_pii_scan_default_still_blocks() -> None:
    """Backward compat: omitted action == legacy block behavior."""
    from egisai.policy import evaluate_policies

    decision = evaluate_policies(_ssn_rule(None), _make_ctx("SSN 123-45-6789"))
    assert decision.verdict == "block"


def test_pii_scan_action_block_is_explicit_block() -> None:
    from egisai.policy import evaluate_policies

    decision = evaluate_policies(_ssn_rule("block"), _make_ctx("SSN 123-45-6789"))
    assert decision.verdict == "block"


def test_pii_scan_action_sanitize_returns_sanitize_verdict() -> None:
    from egisai.policy import evaluate_policies

    decision = evaluate_policies(
        _ssn_rule("sanitize"), _make_ctx("SSN 123-45-6789")
    )
    assert decision.verdict == "sanitize"
    assert "ssn" in decision.sanitize_kinds
    assert decision.matched_policy == "ssn-handler"


def test_pii_scan_action_sanitize_clean_prompt_allows() -> None:
    """No PII detected ⇒ the rule doesn't fire at all."""
    from egisai.policy import evaluate_policies

    decision = evaluate_policies(
        _ssn_rule("sanitize"), _make_ctx("Hello, how are you?")
    )
    assert decision.verdict == "allow"


# ── Layer 3: _evaluator.mutate_prompt_text ────────────────────────


def test_mutate_walks_openai_chat_messages_shape() -> None:
    from egisai._evaluator import mutate_prompt_text

    payload = {
        "messages": [
            {"role": "system", "content": "Helper."},
            {"role": "user", "content": "My SSN is 123-45-6789"},
        ]
    }
    mutated = mutate_prompt_text(payload, lambda s: s.replace("123-45-6789", "###-##-####"))
    assert mutated is True
    # System untouched.
    assert payload["messages"][0]["content"] == "Helper."
    # User mutated in place.
    assert payload["messages"][1]["content"] == "My SSN is ###-##-####"


def test_mutate_walks_openai_responses_string_input() -> None:
    from egisai._evaluator import mutate_prompt_text

    payload = {"input": "leak: 123-45-6789"}
    mutate_prompt_text(payload, lambda s: s.replace("123-45-6789", "###-##-####"))
    assert payload["input"] == "leak: ###-##-####"


def test_mutate_walks_gemini_contents_shape() -> None:
    from egisai._evaluator import mutate_prompt_text

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "ssn 123-45-6789"}]},
        ]
    }
    mutate_prompt_text(payload, lambda s: s.replace("123-45-6789", "###-##-####"))
    assert payload["contents"][0]["parts"][0]["text"] == "ssn ###-##-####"


def test_mutate_walks_openai_vision_parts_shape() -> None:
    """Vision/multimodal messages have ``content`` as list of parts."""
    from egisai._evaluator import mutate_prompt_text

    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "card 4111 1111 1111 1111"},
                    {"type": "image_url", "image_url": {"url": "data:..."}},
                ],
            }
        ]
    }
    mutate_prompt_text(payload, lambda s: s.replace("4111 1111 1111 1111", "REDACTED"))
    assert "REDACTED" in payload["messages"][0]["content"][0]["text"]


def test_mutate_returns_false_when_nothing_changed() -> None:
    from egisai._evaluator import mutate_prompt_text

    payload = {"messages": [{"role": "user", "content": "clean prompt"}]}
    assert mutate_prompt_text(payload, lambda s: s) is False


# ── End-to-end via the gate (sync OpenAI adapter) ─────────────────


def _sanitize_rule_ssn() -> dict:
    return {
        "id": 1,
        "name": "redact-ssn",
        "type": "pii_scan",
        "tenant": None,
        "config": {
            "action": "sanitize",
            "threshold": 0.5,
            "kinds": ["ssn"],
            "message": "SSN redacted before forwarding to model.",
        },
    }


def test_gate_sanitizes_payload_before_forward(fake_backend) -> None:
    """The model SDK must see the masked text — never the raw SSN."""
    import egisai
    from egisai._patches._common import gate_call

    fake_backend.set_rules([_sanitize_rule_ssn()], etag='"sanitize"')
    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    seen_messages: list = []

    # The "framework SDK" — captures whatever payload the gate hands
    # over to forward(). In the real OpenAI patch this is what's sent
    # over the wire by httpx; here we just record it.
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "My SSN is 123-45-6789, save it."},
    ]
    payload = {"messages": messages, "tools": None}

    def fake_forward():
        # Snapshot what the framework SDK would serialize. We deep-
        # copy because dict identity vs value matters: the gate
        # mutated this same list in place, so a reference snapshot
        # would show the masked version even if we asserted later.
        import copy
        seen_messages.append(copy.deepcopy(messages))

        class _FakeResp:
            usage = type("U", (), {"prompt_tokens": 10, "completion_tokens": 4})()

        return _FakeResp()

    gate_call(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4",
        prompt_text="My SSN is 123-45-6789, save it.",
        stream=False,
        payload=payload,
        forward=fake_forward,
    )

    assert seen_messages, "fake_forward should have been invoked once"
    rendered = seen_messages[0]
    # System prompt untouched.
    assert rendered[0]["content"] == "You are a helpful assistant."
    # User prompt sanitized — the SSN never reaches the model.
    assert "123-45-6789" not in rendered[1]["content"]
    assert "###-##-####" in rendered[1]["content"]


def test_gate_emits_sanitize_event(fake_backend) -> None:
    """The audit log row records verdict='sanitize' + the redaction tally."""
    import egisai
    from egisai._patches._common import gate_call

    fake_backend.set_rules([_sanitize_rule_ssn()], etag='"sanitize"')
    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    payload = {
        "messages": [
            {"role": "user", "content": "SSN 123-45-6789 please."},
        ],
        "tools": None,
    }

    def fake_forward():
        class _FakeResp:
            usage = type("U", (), {"prompt_tokens": 5, "completion_tokens": 2})()
        return _FakeResp()

    gate_call(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4",
        prompt_text="SSN 123-45-6789 please.",
        stream=False,
        payload=payload,
        forward=fake_forward,
    )

    # Drain the queue and find the most recent emitted event.
    from egisai import _logger

    drained = []
    while not _logger._q.empty():
        drained.append(_logger._q.get_nowait())

    assert drained, "gate_call should enqueue exactly one event for an allowed/sanitized call"
    ev = drained[-1]
    assert ev["verdict"] == "sanitize"
    # Audit record carries the count + pattern, never the raw value.
    sanitizations = ev.get("sanitizations") or []
    assert len(sanitizations) == 1
    # Wire field is now ``type`` (renamed from ``kind`` in this release).
    assert sanitizations[0]["type"] == "ssn"
    assert sanitizations[0]["count"] == 1
    assert sanitizations[0]["pattern"] == "###-##-####"
    # Crucial compliance assertion: the original SSN is nowhere in
    # the event we'd ship to the backend.
    assert "123-45-6789" not in repr(ev)


# ── Configurable mask_char (Pass B) ───────────────────────────────


def test_sanitize_honors_custom_mask_char_for_ssn() -> None:
    """Operator picks 'X' → SSN becomes XXX-XX-XXXX, separators kept."""
    from egisai.policy.pii import sanitize

    text, records = sanitize("SSN 123-45-6789", mask_char="X")
    assert "XXX-XX-XXXX" in text
    assert "###-##-####" not in text
    assert records[0].pattern == "XXX-XX-XXXX"


def test_sanitize_honors_custom_mask_char_for_credit_card() -> None:
    """Mask char threads through to other digit-shaped kinds too."""
    from egisai.policy.pii import sanitize

    text, records = sanitize("card 4111-1111-1111-1111", mask_char="*")
    assert "****-****-****-****" in text
    assert records[0].pattern == "****-****-****-****"


def test_sanitize_email_ignores_mask_char() -> None:
    """Word-shaped PII keeps its labeled placeholder regardless of mask_char.

    A single character can't faithfully redact an arbitrary email; the
    placeholder is the safe choice. The audit record reflects that.
    """
    from egisai.policy.pii import sanitize

    text, records = sanitize("contact me: jane@acme.com", mask_char="*")
    assert "[email-redacted]" in text
    assert "*" not in records[0].pattern


def test_sanitize_empty_mask_char_falls_back_to_default() -> None:
    """Empty/None mask_char must NOT clobber the prompt to nothing.

    Operator misconfiguration should fail-safe to '#'. This is the
    fail-open compliance contract — when the input is malformed,
    we still mask the PII rather than letting it through.
    """
    from egisai.policy.pii import sanitize

    text, records = sanitize("SSN 123-45-6789", mask_char="")
    assert "###-##-####" in text
    assert records[0].pattern == "###-##-####"


def test_engine_passes_mask_char_from_policy_config() -> None:
    """End-to-end at the engine layer: config.mask_char reaches the decision."""
    from egisai.policy import PolicyContext, PolicyRule, evaluate_policies

    rule = PolicyRule(
        id=1,
        name="ssn-x",
        type="pii_scan",
        tenant=None,
        config={
            "action": "sanitize",
            "kinds": ["ssn"],
            "threshold": 0.5,
            "mask_char": "X",
        },
    )
    decision = evaluate_policies(
        [rule],
        PolicyContext(
            tenant="acme",
            model="gpt-4",
            prompt_text="SSN 123-45-6789",
            prompt_chars=15,
            stream=False,
        ),
    )
    assert decision.verdict == "sanitize"
    assert decision.sanitize_mask_char == "X"


def test_gate_applies_custom_mask_char_to_payload(fake_backend) -> None:
    """The model receives masked text using the operator-chosen char."""
    import egisai
    from egisai._patches._common import gate_call

    fake_backend.set_rules(
        [
            {
                "id": 1,
                "name": "ssn-redact-X",
                "type": "pii_scan",
                "tenant": None,
                "config": {
                    "action": "sanitize",
                    "mask_char": "X",
                    "kinds": ["ssn"],
                    "threshold": 0.5,
                },
            }
        ],
        etag='"x-mask"',
    )
    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    seen: list = []
    messages = [{"role": "user", "content": "My SSN is 123-45-6789."}]

    def fake_forward():
        import copy
        seen.append(copy.deepcopy(messages))

        class _R:
            usage = type("U", (), {"prompt_tokens": 5, "completion_tokens": 2})()
        return _R()

    gate_call(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4",
        prompt_text="My SSN is 123-45-6789.",
        stream=False,
        payload={"messages": messages, "tools": None},
        forward=fake_forward,
    )
    assert seen, "fake_forward should have been called once"
    body = seen[0][0]["content"]
    assert "XXX-XX-XXXX" in body
    assert "###-##-####" not in body
    assert "123-45-6789" not in body
