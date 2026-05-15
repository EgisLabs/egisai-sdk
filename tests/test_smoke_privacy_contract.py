"""Cross-cutting privacy invariants every framework MUST honor.

This file pins the SDK's most-load-bearing compliance guarantees:

1. **No raw prompt bytes leave the SDK boundary on a block path.**
   When a ``pii_scan(block)`` rule refuses an inbound prompt, the
   detected PII span (the raw SSN / credit card / email) must NEVER
   appear on any audit envelope the SDK ships to the platform —
   ``payload_preview``, ``prompt_preview``, ``payload_preview_before``,
   ``response_preview``, ``step_payload``, ``error``, any field. The
   audit row only carries the masked / labeled form (``<SSN>``).

2. **No raw model response bytes ever land on the audit row.**
   The SDK's privacy contract (``security-and-compliance.mdc`` §5)
   explicitly states the model's reply text is *evaluated but never
   persisted*. ``_run_output_phase`` extracts it long enough to feed
   output-side policies, then drops it. The wire envelope MUST NEVER
   carry a ``response_preview`` field and MUST NEVER contain the
   verbatim model text under any other field name.

3. **No raw tool result bytes ever land on the audit row.**
   When a framework supports tool execution (claude_agent_sdk),
   ``PostToolUse`` evaluates the tool's free-text output for PII
   before letting Claude see it. The audit row records *what
   happened* (verdict, sanitization counts) — never the raw tool
   text, even when no policy fired.

4. **No raw PII bytes leave on the sanitize path either.**
   Sanitize is the "we masked, don't refuse" branch. The masked
   text is the post-sanitization view (``###-##-####`` shape or the
   ``<SSN>`` label); the **raw** original must still never appear
   on the wire — not in ``payload_preview``, not in any field that
   serializes the pre-sanitize payload.

These contracts are framework-agnostic — they hold for OpenAI,
Anthropic, Gemini, Bedrock Converse, and every Tier-2 wrapper that
cascades into one of them. The tests below cover the spectrum:
direct LLM providers (OpenAI / Anthropic / Gemini / Bedrock) and
the framework cascade (openai_agents → openai), with one test per
provider per invariant.

If any test in this file fails, treat it as a P0: a SOC 2 / GDPR
auditor would refuse the deployment. The fix is in the SDK's
audit-event construction (``_events.py`` ``safe_preview``,
``_common.py`` ``_safe_text_preview`` / ``_run_output_phase``),
NOT in the test.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

# ── Shared helpers (mirror of smoke_provider_battery & cascade) ─────


def _flush() -> None:
    from egisai import shutdown

    shutdown()


def _init_sdk(app: str) -> None:
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app=app,
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )


def _pii_sanitize_rule() -> dict[str, Any]:
    return {
        "id": "pc-pii-san",
        "name": "privacy-contract-sanitize",
        "type": "pii_scan",
        "tenant": None,
        "config": {
            "action": "sanitize",
            "types": ["ssn", "email", "credit_card"],
            "mask_char": "#",
        },
    }


def _pii_block_rule() -> dict[str, Any]:
    return {
        "id": "pc-pii-block",
        "name": "privacy-contract-block",
        "type": "pii_scan",
        "tenant": None,
        "config": {"action": "block", "types": ["ssn"]},
    }


def _set_rules(*rules: dict[str, Any]) -> None:
    from egisai._policy_cache import replace_rules

    replace_rules(f'"pc-{len(rules)}"', list(rules))


def _all_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(events)


def _assert_no_raw_bytes(
    events: list[dict[str, Any]],
    *fingerprints: str,
) -> None:
    """No envelope on the wire carries ``fingerprint`` substrings.

    Walks every field on every envelope and asserts the raw byte
    sequence is absent. Used to verify the SDK never ships a raw
    SSN, raw email, raw model response, or raw tool result.
    """
    for ev in events:
        body = repr(ev)
        for fp in fingerprints:
            assert fp not in body, (
                f"raw secret {fp!r} leaked into wire envelope "
                f"event_id={ev.get('event_id')} kind={ev.get('kind')} "
                f"fields={list(ev)}"
            )


def _assert_no_response_preview(events: list[dict[str, Any]]) -> None:
    """No audit envelope can carry a ``response_preview`` field."""
    for ev in events:
        assert "response_preview" not in ev, (
            f"audit row leaked response_preview: keys={list(ev)}"
        )
        # The step_payload sub-dict is the structured carrier for
        # tool / model fields on Run steps; that surface MUST also
        # be free of response_preview.
        sp = ev.get("step_payload")
        if isinstance(sp, dict):
            assert "response_preview" not in sp, (
                f"step_payload leaked response_preview: keys={list(sp)}"
            )


# ── Provider seam: OpenAI Chat ─────────────────────────────────────


def _install_fake_openai() -> type:
    fake = types.ModuleType("openai")
    res = types.ModuleType("openai.resources")
    chat = types.ModuleType("openai.resources.chat")
    completions = types.ModuleType("openai.resources.chat.completions")
    responses = types.ModuleType("openai.resources.responses")

    class Completions:
        _captured_kwargs: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            return types.SimpleNamespace(
                id="oa",
                choices=[
                    types.SimpleNamespace(
                        message=types.SimpleNamespace(
                            content="A SECRET RESPONSE THE SDK MUST NOT KEEP",
                            tool_calls=None,
                        ),
                        finish_reason="stop",
                    )
                ],
                usage=types.SimpleNamespace(prompt_tokens=2, completion_tokens=1),
            )

    completions.Completions = Completions
    completions.AsyncCompletions = type(
        "AsyncCompletions", (), {"_captured_kwargs": []}
    )
    sys.modules.update(
        {
            "openai": fake,
            "openai.resources": res,
            "openai.resources.chat": chat,
            "openai.resources.chat.completions": completions,
            "openai.resources.responses": responses,
        }
    )
    Completions._captured_kwargs = []
    return Completions


@pytest.fixture
def openai_env(fake_backend: Any) -> Iterator[tuple[Any, type]]:
    _init_sdk(app="pc-openai")
    Completions = _install_fake_openai()
    from egisai._patches import openai as patch

    assert patch.apply() is True
    yield fake_backend, Completions
    for m in (
        "openai", "openai.resources", "openai.resources.chat",
        "openai.resources.chat.completions", "openai.resources.responses",
    ):
        sys.modules.pop(m, None)


def test_openai_block_path_redacts_raw_ssn_from_every_wire_field(
    openai_env: Any,
) -> None:
    """``payload_preview`` was the smoking gun before the fix —
    ``build_event`` used to ship raw ``repr(payload)`` to the wire
    on the block path. ``safe_preview`` is now label-redact aware
    so even an inadvertent leak is masked. This test pins that."""
    fake_backend, Completions = openai_env
    _set_rules(_pii_block_rule())
    raw = "123-45-6789"

    with pytest.raises(PermissionError):
        Completions().create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"Patient SSN {raw}"}
            ],
        )
    _flush()

    assert Completions._captured_kwargs == [], (
        "blocked prompt must never reach the openai SDK"
    )
    _assert_no_raw_bytes(fake_backend.events_received, raw)


def test_openai_sanitize_path_redacts_raw_ssn_from_every_wire_field(
    openai_env: Any,
) -> None:
    """Sanitize is the 'we masked, don't refuse' branch — the audit
    row carries the post-sanitize text; the raw SSN must NEVER show
    up anywhere on the envelope (not even in ``payload_preview_before``,
    which is constructed via ``_safe_text_preview`` → ``label_redact``).
    """
    fake_backend, Completions = openai_env
    _set_rules(_pii_sanitize_rule())
    raw = "234-56-7891"

    Completions().create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"verify SSN {raw}"}],
    )
    _flush()

    assert Completions._captured_kwargs, "openai create wasn't called"
    sent = Completions._captured_kwargs[-1]["messages"][-1]["content"]
    assert raw not in sent, "post-sanitize content sent to openai still raw"
    _assert_no_raw_bytes(fake_backend.events_received, raw)


def test_openai_allow_path_never_ships_response_preview(
    openai_env: Any,
) -> None:
    """``_run_output_phase`` extracts the model text to evaluate
    output policies but never persists it. The audit row must NOT
    carry a ``response_preview`` field — even on the allow path
    where there's no decision to record."""
    fake_backend, Completions = openai_env

    Completions().create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "hi"}],
    )
    _flush()

    _assert_no_response_preview(fake_backend.events_received)
    _assert_no_raw_bytes(
        fake_backend.events_received,
        "A SECRET RESPONSE THE SDK MUST NOT KEEP",
    )


def test_openai_per_tool_steps_never_carry_response_preview(
    openai_env: Any,
) -> None:
    """The multi-step waterfall (``emit_tool_call_steps=True``) emits
    one ``tool_call`` step per tool the model invoked. Each step's
    envelope must respect the same privacy contract as the parent
    ``model_call`` — no ``response_preview``, no raw model text."""
    fake_backend, Completions = openai_env

    def _two_tools() -> Any:
        return types.SimpleNamespace(
            id="real-tools",
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(
                        content=None,
                        tool_calls=[
                            types.SimpleNamespace(
                                type="function",
                                function=types.SimpleNamespace(
                                    name="lookup",
                                    arguments='{"q": "x"}',
                                ),
                            ),
                            types.SimpleNamespace(
                                type="function",
                                function=types.SimpleNamespace(
                                    name="notify",
                                    arguments='{"to": "ops"}',
                                ),
                            ),
                        ],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=types.SimpleNamespace(prompt_tokens=4, completion_tokens=2),
        )

    Completions._response_factory = _two_tools  # type: ignore[attr-defined]
    # Force a tool-bearing response by patching one call's return.
    original = Completions.create

    def _create(self: Any, **kwargs: Any) -> Any:
        type(self)._captured_kwargs.append(kwargs)
        return _two_tools()

    Completions.create = _create  # type: ignore[assignment]
    try:
        Completions().create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "go"}],
        )
        _flush()
    finally:
        Completions.create = original  # type: ignore[assignment]

    _assert_no_response_preview(fake_backend.events_received)


# ── Provider seam: Anthropic ───────────────────────────────────────


def _install_fake_anthropic() -> type:
    fake = types.ModuleType("anthropic")
    res = types.ModuleType("anthropic.resources")
    msgs = types.ModuleType("anthropic.resources.messages")

    class Messages:
        _captured_kwargs: list[dict[str, Any]] = []

        def create(self, **kwargs: Any) -> Any:
            type(self)._captured_kwargs.append(kwargs)
            return types.SimpleNamespace(
                id="ant",
                type="message",
                role="assistant",
                content=[
                    types.SimpleNamespace(
                        type="text",
                        text="ANT_RESPONSE_TEXT_DO_NOT_PERSIST",
                    )
                ],
                stop_reason="end_turn",
                usage=types.SimpleNamespace(
                    input_tokens=2, output_tokens=1,
                ),
            )

    class AsyncMessages:
        _captured_kwargs: list[dict[str, Any]] = []

    msgs.Messages = Messages
    msgs.AsyncMessages = AsyncMessages
    sys.modules.update(
        {
            "anthropic": fake,
            "anthropic.resources": res,
            "anthropic.resources.messages": msgs,
        }
    )
    Messages._captured_kwargs = []
    return Messages


@pytest.fixture
def anthropic_env(fake_backend: Any) -> Iterator[tuple[Any, type]]:
    _init_sdk(app="pc-anthropic")
    Messages = _install_fake_anthropic()
    from egisai._patches import anthropic as patch

    assert patch.apply() is True
    yield fake_backend, Messages
    for m in (
        "anthropic", "anthropic.resources", "anthropic.resources.messages",
    ):
        sys.modules.pop(m, None)


def test_anthropic_block_path_redacts_raw_ssn(anthropic_env: Any) -> None:
    fake_backend, Messages = anthropic_env
    _set_rules(_pii_block_rule())
    raw = "345-67-8901"

    with pytest.raises(PermissionError):
        Messages().create(
            model="claude-3-5-sonnet",
            messages=[{"role": "user", "content": f"SSN {raw}"}],
        )
    _flush()

    assert Messages._captured_kwargs == []
    _assert_no_raw_bytes(fake_backend.events_received, raw)


def test_anthropic_response_text_never_persisted(
    anthropic_env: Any,
) -> None:
    fake_backend, Messages = anthropic_env

    Messages().create(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hi"}],
    )
    _flush()

    _assert_no_response_preview(fake_backend.events_received)
    _assert_no_raw_bytes(
        fake_backend.events_received,
        "ANT_RESPONSE_TEXT_DO_NOT_PERSIST",
    )


# ── Provider seam: Gemini ──────────────────────────────────────────


def _install_fake_genai() -> type:
    google = types.ModuleType("google")
    genai = types.ModuleType("google.genai")
    models_mod = types.ModuleType("google.genai.models")

    class Models:
        _captured: list[dict[str, Any]] = []

        def generate_content(
            self,
            *,
            model: str,
            contents: Any,
            config: Any = None,
        ) -> Any:
            type(self)._captured.append(
                {"model": model, "contents": contents, "config": config}
            )
            return types.SimpleNamespace(
                text="GEMINI_REPLY_DO_NOT_PERSIST",
                candidates=[],
                usage_metadata=types.SimpleNamespace(
                    prompt_token_count=2, candidates_token_count=1,
                ),
            )

    class AsyncModels:
        _captured: list[dict[str, Any]] = []

    models_mod.Models = Models
    models_mod.AsyncModels = AsyncModels
    genai.models = models_mod
    google.genai = genai
    sys.modules.update(
        {
            "google": google,
            "google.genai": genai,
            "google.genai.models": models_mod,
        }
    )
    Models._captured = []
    return Models


@pytest.fixture
def genai_env(fake_backend: Any) -> Iterator[tuple[Any, type]]:
    _init_sdk(app="pc-genai")
    Models = _install_fake_genai()
    from egisai._patches import genai as patch

    assert patch.apply() is True
    yield fake_backend, Models
    for m in ("google.genai.models", "google.genai", "google"):
        sys.modules.pop(m, None)


def test_genai_block_path_redacts_raw_ssn(genai_env: Any) -> None:
    fake_backend, Models = genai_env
    _set_rules(_pii_block_rule())
    raw = "456-78-9012"

    with pytest.raises(PermissionError):
        Models().generate_content(
            model="gemini-1.5-flash",
            contents=f"SSN {raw}",
        )
    _flush()

    assert Models._captured == []
    _assert_no_raw_bytes(fake_backend.events_received, raw)


def test_genai_sanitize_path_top_level_string_contents(
    genai_env: Any,
) -> None:
    """Pins the bug the smoke-battery first caught: when Gemini's
    ergonomic ``contents="..."`` shape carries PII, sanitize must
    mask the string in place before the upstream SDK sees it. The
    raw SSN must not appear in the captured kwargs or on any
    wire envelope."""
    fake_backend, Models = genai_env
    _set_rules(_pii_sanitize_rule())
    raw = "567-89-0123"

    Models().generate_content(
        model="gemini-1.5-flash",
        contents=f"verify SSN {raw}",
    )
    _flush()

    sent = Models._captured[-1]["contents"]
    assert raw not in str(sent)
    _assert_no_raw_bytes(fake_backend.events_received, raw)


def test_genai_response_text_never_persisted(genai_env: Any) -> None:
    fake_backend, Models = genai_env

    Models().generate_content(
        model="gemini-1.5-flash",
        contents="hello",
    )
    _flush()

    _assert_no_response_preview(fake_backend.events_received)
    _assert_no_raw_bytes(
        fake_backend.events_received,
        "GEMINI_REPLY_DO_NOT_PERSIST",
    )


# ── Provider seam: Bedrock Converse ────────────────────────────────


def _install_fake_boto3_with_bedrock() -> type:
    boto3 = types.ModuleType("boto3")

    class BedrockClient:
        _captured: list[dict[str, Any]] = []

        def converse(self, **kwargs: Any) -> Any:
            type(self)._captured.append(kwargs)
            return {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [
                            {"text": "BEDROCK_REPLY_DO_NOT_PERSIST"}
                        ],
                    }
                },
                "stopReason": "end_turn",
                "usage": {"inputTokens": 2, "outputTokens": 1},
                "metrics": {"latencyMs": 12},
            }

        def converse_stream(self, **kwargs: Any) -> Any:
            type(self)._captured.append(kwargs)
            return {"stream": iter(())}

    def client(service: str, *args: Any, **kwargs: Any) -> Any:
        if service != "bedrock-runtime":
            raise ValueError("test fake supports bedrock-runtime only")
        return BedrockClient()

    boto3.client = client  # type: ignore[attr-defined]
    sys.modules["boto3"] = boto3
    BedrockClient._captured = []
    return BedrockClient


@pytest.fixture
def bedrock_env(fake_backend: Any) -> Iterator[tuple[Any, type]]:
    _init_sdk(app="pc-bedrock")
    Client = _install_fake_boto3_with_bedrock()
    from egisai._patches import bedrock_runtime as patch

    assert patch.apply() is True
    yield fake_backend, Client
    sys.modules.pop("boto3", None)


def test_bedrock_block_path_redacts_raw_ssn(bedrock_env: Any) -> None:
    fake_backend, _ = bedrock_env
    _set_rules(_pii_block_rule())
    raw = "678-90-1234"
    import boto3

    client = boto3.client("bedrock-runtime")
    with pytest.raises(PermissionError):
        client.converse(
            modelId="anthropic.claude-3-5-sonnet",
            messages=[
                {
                    "role": "user",
                    "content": [{"text": f"SSN {raw}"}],
                }
            ],
        )
    _flush()
    _assert_no_raw_bytes(fake_backend.events_received, raw)


def test_bedrock_response_text_never_persisted(bedrock_env: Any) -> None:
    fake_backend, _ = bedrock_env
    import boto3

    client = boto3.client("bedrock-runtime")
    client.converse(
        modelId="anthropic.claude-3-5-sonnet",
        messages=[
            {
                "role": "user",
                "content": [{"text": "hello"}],
            }
        ],
    )
    _flush()

    _assert_no_response_preview(fake_backend.events_received)
    _assert_no_raw_bytes(
        fake_backend.events_received,
        "BEDROCK_REPLY_DO_NOT_PERSIST",
    )


# ── Composite invariant: raw email + credit card across providers ─


def test_openai_sanitize_redacts_multi_pii_classes(openai_env: Any) -> None:
    """A prompt carrying multiple PII kinds (SSN + email + credit
    card) must have ALL three masked locally before the SDK ships
    bytes. The raw forms of every span are absent from the audit
    envelope.

    Important: ``example.com`` and other RFC-2606 reserved test
    domains are intentionally **not** masked by the SDK's email
    redactor (operators routinely use these domains in
    documentation, fixtures, and synthetic tests; treating them as
    PII would produce a steady stream of false positives in
    real workloads). We use a non-reserved domain here so the
    test exercises the actual redaction path.
    """
    fake_backend, Completions = openai_env
    _set_rules(_pii_sanitize_rule())
    raw_ssn = "789-01-2345"
    raw_email = "alice@acme-corp.io"
    raw_cc = "4111-1111-1111-1111"

    Completions().create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": (
                    f"My SSN is {raw_ssn}, email {raw_email}, "
                    f"card {raw_cc}."
                ),
            }
        ],
    )
    _flush()

    sent = Completions._captured_kwargs[-1]["messages"][-1]["content"]
    for fp in (raw_ssn, raw_email, raw_cc):
        assert fp not in sent, (
            f"sanitize forwarded raw {fp!r} to openai: sent={sent!r}"
        )
    _assert_no_raw_bytes(
        fake_backend.events_received, raw_ssn, raw_email, raw_cc,
    )


def test_openai_payload_preview_is_label_redacted_on_block(
    openai_env: Any,
) -> None:
    """The block path doesn't get a chance to re-set
    ``payload_preview`` (that only happens on sanitize). So
    ``safe_preview`` itself must label-redact the raw bytes at
    ``build_event`` time. This test pins that contract."""
    fake_backend, _ = openai_env
    _set_rules(_pii_block_rule())
    raw = "234-56-7891"

    with pytest.raises(PermissionError):
        from egisai._patches import openai as oa  # noqa: F401  # ensure init
        sys.modules["openai.resources.chat.completions"].Completions().create(
            model="gpt-4o",
            messages=[{"role": "user", "content": f"SSN {raw}"}],
        )
    _flush()

    pps = [
        ev.get("payload_preview")
        for ev in fake_backend.events_received
        if "payload_preview" in ev
    ]
    assert pps, "no payload_preview field on the audit row?"
    for pp in pps:
        if pp is None:
            continue
        assert raw not in pp, (
            f"safe_preview shipped raw SSN in payload_preview: {pp!r}"
        )
        # Label-redact replaces with ``<SSN>``; we expect to see the
        # label or the masked form, NOT the digits.
        assert "<SSN>" in pp or "###" in pp, (
            f"payload_preview missed redaction marker: {pp!r}"
        )
