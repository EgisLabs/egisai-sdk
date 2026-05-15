"""Advisory-but-honest contract for the Bedrock Agent patch.

``bedrock_agent`` is the **only** integration in egisai's stable
matrix where the SDK can neither pre-block tool execution nor
sanitize tool results — the Action Groups run on AWS-managed
infrastructure, outside this SDK's Python process. By the time the
streamed ``invoke_agent`` response reaches the patch, AWS has
already dispatched the Action Group **and** fed the Action Group's
result back to the model.

The honest posture documented in ``SECURITY.md`` is:

* Input-side policies (``pii_scan`` on ``inputText``,
  ``deny_regex`` on the user's prompt, ``max_prompt_chars``,
  ``allow_model``) fire BEFORE ``boto3``'s ``InvokeAgent`` is
  called — these are real enforcement.
* Output-side policies (``deny_tool_call`` / ``deny_mcp_call`` /
  ``pii_scan`` over the trace events) are **advisory only** —
  the SDK can observe and stamp the audit row, but cannot
  un-execute the Action Group.

This file pins that contract so a future refactor doesn't
silently flip ``enforcement_status`` from ``"advisory"`` to
``"enforced"`` for bedrock_agent. A SOC 2 / GDPR auditor reading
"enforced" on a Bedrock Agent audit row would conclude egisai
prevented the leak; ``advisory`` is the truth. Stamping the wrong
value is a compliance regression.

The tests deliberately don't try to instantiate the real
``bedrock-agent-runtime`` client (depending on AWS in CI is a
brittle path); we stand up a minimal in-process double of
``boto3.client("bedrock-agent-runtime").invoke_agent`` whose
``EventStream`` mirrors the documented shape AWS returns. The
patches duck-type on ``client.invoke_agent``, so the stub is
enough to exercise the gate seam end-to-end.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from typing import Any

import pytest

# ── Shared helpers ──────────────────────────────────────────────────


def _flush() -> None:
    from egisai import shutdown

    shutdown()


def _init_sdk() -> None:
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="bedrock-agent-smoke",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )


def _pii_block_rule() -> dict[str, Any]:
    return {
        "id": "ba-pii-block",
        "name": "smoke-block-pii",
        "type": "pii_scan",
        "tenant": None,
        "config": {"action": "block", "types": ["ssn"]},
    }


def _pii_sanitize_rule() -> dict[str, Any]:
    return {
        "id": "ba-pii-san",
        "name": "smoke-sanitize-pii",
        "type": "pii_scan",
        "tenant": None,
        "config": {
            "action": "sanitize",
            "types": ["ssn"],
            "mask_char": "#",
        },
    }


def _set_rules(*rules: dict[str, Any]) -> None:
    from egisai._policy_cache import replace_rules

    replace_rules(f'"ba-{len(rules)}"', list(rules))


def _all_audit_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(events)


def _install_fake_boto3_bedrock_agent() -> tuple[type, list[dict[str, Any]]]:
    """Plant a fake ``boto3.client("bedrock-agent-runtime")``.

    Captures every ``invoke_agent`` invocation and returns a fake
    EventStream-shaped iterator so the patch's stream handling can
    walk it without crashing. Also stubs the ``bedrock-agent``
    service (``get_agent``) so ``_resolve_friendly_name`` returns
    a known display name.
    """
    boto3 = types.ModuleType("boto3")
    captured: list[dict[str, Any]] = []

    class BedrockAgentClient:
        def get_agent(self, *, agentId: str) -> Any:
            return {"agent": {"agentName": f"BA-{agentId[:6]}"}}

    class BedrockAgentRuntimeClient:
        def invoke_agent(self, **kwargs: Any) -> Any:
            captured.append(dict(kwargs))
            # Mirror AWS's documented streamed return shape: a
            # ``completion`` field whose value is an iterable of
            # event dicts. Our fake yields nothing so the test
            # doesn't have to consume the stream.
            return {"completion": iter(()), "contentType": "text/event-stream"}

    def client(service: str, *args: Any, **kwargs: Any) -> Any:
        if service == "bedrock-agent":
            return BedrockAgentClient()
        if service == "bedrock-agent-runtime":
            return BedrockAgentRuntimeClient()
        raise ValueError(f"test fake doesn't support service={service!r}")

    boto3.client = client  # type: ignore[attr-defined]
    sys.modules["boto3"] = boto3
    return BedrockAgentRuntimeClient, captured


@pytest.fixture
def bedrock_agent_smoke(
    fake_backend: Any,
) -> Iterator[tuple[Any, type, list[dict[str, Any]]]]:
    _init_sdk()
    Client, captured = _install_fake_boto3_bedrock_agent()
    from egisai._patches import bedrock_agent as patch

    assert patch.apply() is True
    yield fake_backend, Client, captured
    sys.modules.pop("boto3", None)
    # The patch keeps an internal name-cache; reset so the next test
    # gets a clean slate. Note: 0.25.15 removed the
    # ``_PATCHED_CLIENT_IDS`` set that this teardown used to clear —
    # CPython's id() reuse made it a stale-state bug rather than an
    # idempotency aid. The ``__egisai_wrapped__`` sentinel attribute
    # on the wrapped method covers the same intent without
    # process-wide bookkeeping.
    patch._NAME_CACHE.clear()


# ── 1. Allow path: identity registered, audit row produced ─────────


def test_invoke_agent_allow_path_records_audit_row(
    bedrock_agent_smoke: Any,
) -> None:
    """No policies, clean prompt: ``invoke_agent`` is forwarded; the
    audit row carries the bedrock_agent identity and ``verdict=allow``."""
    fake_backend, _, captured = bedrock_agent_smoke
    import boto3

    client = boto3.client("bedrock-agent-runtime")
    client.invoke_agent(
        agentId="11111111-2222-3333-4444-555555555555",
        agentAliasId="ALIASA",
        sessionId="session-1",
        inputText="hello agent",
    )
    _flush()

    assert captured, "boto3.invoke_agent was never called"
    assert captured[-1]["inputText"] == "hello agent"
    events = _all_audit_events(fake_backend.events_received)
    allow = [e for e in events if e.get("verdict") == "allow"]
    assert allow, f"no allow verdict on the audit row: {events!r}"
    assert any(
        e.get("source") == "bedrock_agent" for e in events
    ), "audit row missing bedrock_agent source"


# ── 2. Input-side block actually prevents the AWS call ─────────────


def test_invoke_agent_input_block_short_circuits_boto3(
    bedrock_agent_smoke: Any,
) -> None:
    """When ``pii_scan(block)`` matches the ``inputText`` prompt,
    the patch MUST raise ``PermissionError`` BEFORE ``boto3.invoke_agent``
    is called. The input-side gate is the only real enforcement
    point on bedrock_agent — pin it.
    """
    fake_backend, _, captured = bedrock_agent_smoke
    _set_rules(_pii_block_rule())
    raw = "123-45-6789"
    import boto3

    client = boto3.client("bedrock-agent-runtime")
    with pytest.raises(PermissionError):
        client.invoke_agent(
            agentId="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            agentAliasId="LIVE",
            sessionId="s",
            inputText=f"Patient SSN {raw}",
        )
    _flush()

    assert captured == [], (
        "blocked invoke_agent still reached boto3 — input gate "
        "is broken on bedrock_agent"
    )
    # Wire envelope never carries the raw SSN, regardless of
    # which side blocked.
    for ev in fake_backend.events_received:
        assert raw not in repr(ev), (
            f"raw SSN leaked into audit envelope on bedrock_agent "
            f"block path: keys={list(ev)}"
        )


# ── 3. Input-side sanitize masks the inputText before AWS ──────────


def test_invoke_agent_input_sanitize_masks_inputText_before_boto3(
    bedrock_agent_smoke: Any,
) -> None:
    """``pii_scan(sanitize)`` on the prompt MUST mutate the
    ``inputText`` kwarg before ``boto3.invoke_agent`` is called.
    AWS never sees the raw SSN; the audit row records the
    sanitization.
    """
    fake_backend, _, captured = bedrock_agent_smoke
    _set_rules(_pii_sanitize_rule())
    raw = "234-56-7891"
    import boto3

    client = boto3.client("bedrock-agent-runtime")
    client.invoke_agent(
        agentId="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        agentAliasId="LIVE",
        sessionId="s",
        inputText=f"verify SSN {raw} please",
    )
    _flush()

    assert captured, "boto3.invoke_agent was not called"
    assert raw not in captured[-1]["inputText"], (
        f"sanitize did not mask SSN before AWS call: "
        f"inputText={captured[-1]['inputText']!r}"
    )
    for ev in fake_backend.events_received:
        assert raw not in repr(ev), (
            f"raw SSN leaked into audit envelope on bedrock_agent "
            f"sanitize path: keys={list(ev)}"
        )


# ── 4. Identity registration through the ensure pipe ───────────────


def test_invoke_agent_registers_distinct_identities_per_alias(
    bedrock_agent_smoke: Any,
) -> None:
    """Two ``invoke_agent`` calls with the SAME ``agentId`` but
    DIFFERENT ``agentAliasId`` MUST issue two distinct ensure
    requests with different identity bundles. The fingerprint is
    ``("bedrock_agent", agent_id, alias_id)``; a regression that
    dropped ``alias_id`` from the bundle would collapse DRAFT and
    LIVE into one dashboard row, which would confuse operators
    running canary deployments where one alias points at the new
    Action Group revision and the other still points at the old
    one.

    NB: the real platform dedupes by ``(org_id, identity_hash)``
    server-side; our ``FakeBackend`` dedupes by display name (and
    AWS's ``get_agent`` returns ONE friendly name per agentId, so
    both aliases share the display string). The right invariant
    to assert in the fake is therefore on ``ensure_requests``
    (the raw bodies the SDK sent), where the ``identity_hash``
    field of the two requests must differ.
    """
    fake_backend, _, _ = bedrock_agent_smoke
    import boto3

    client = boto3.client("bedrock-agent-runtime")
    for alias in ("DRAFT", "LIVE"):
        client.invoke_agent(
            agentId="same-agent-id-shared",
            agentAliasId=alias,
            sessionId="s",
            inputText="hi",
        )
    _flush()

    # Bedrock-agent's ensure requests should have been issued
    # twice — once per alias — with distinct identity bundles.
    # The wire schema for ``/v1/sdk/agents/ensure`` uses
    # ``identity_source`` (see ``egisai._backend.ensure_agent``)
    # so the filter is on that field, not ``source``.
    bedrock_ensures = [
        r for r in fake_backend.ensure_requests
        if r.get("identity_source") == "framework:bedrock_agent"
    ]
    assert len(bedrock_ensures) == 2, (
        f"expected 2 ensure requests (one per alias), got "
        f"{len(bedrock_ensures)}: {bedrock_ensures!r}"
    )
    hashes = {r.get("identity_hash") for r in bedrock_ensures}
    assert len(hashes) == 2, (
        f"DRAFT and LIVE collapsed into one identity_hash: {hashes!r}"
    )


# ── 5. Privacy contract: no raw inputText on the audit row ─────────


def test_invoke_agent_audit_row_carries_no_raw_inputText(
    bedrock_agent_smoke: Any,
) -> None:
    """Even on the allow path (where no policy ran), the audit
    envelope must NOT carry the raw ``inputText`` verbatim. The
    SDK's privacy contract redacts (``label_redact``) every
    string surfaced through ``payload_preview``.

    Important: 'no raw inputText' applies to **embedded PII**,
    not to free-form natural language. A non-PII prompt
    (``"hello agent"``) will show up in ``payload_preview`` as-is
    because there's nothing to label_redact. The probe text below
    contains an SSN so we have a sharp invariant to assert on.
    """
    fake_backend, _, _ = bedrock_agent_smoke
    raw = "345-67-8901"
    import boto3

    client = boto3.client("bedrock-agent-runtime")
    client.invoke_agent(
        agentId="agent-id",
        agentAliasId="LIVE",
        sessionId="s",
        inputText=f"Auditor request: SSN {raw}",
    )
    _flush()

    for ev in fake_backend.events_received:
        assert raw not in repr(ev), (
            f"raw SSN leaked into bedrock_agent audit envelope: "
            f"keys={list(ev)}"
        )


# ── 6. Documented advisory limitation — tool/output side ───────────


def test_invoke_agent_does_not_register_output_signal_extractor(
    bedrock_agent_smoke: Any,
) -> None:
    """The patch deliberately does NOT pass an
    ``extract_output_signals`` callback to ``gate_call``. That's
    the contract documented in the patch's module docstring (the
    EventStream must be consumed exactly once by the caller; the
    SDK can't replay it).

    Consequence: ``deny_tool_call`` / ``deny_mcp_call`` /
    output-side ``pii_scan`` rules can stamp an advisory audit row
    if AWS happens to surface that information through the trace
    events the caller iterates, but the SDK itself does NOT
    enforce them.

    This test is the "negative" invariant — it pins that we did
    NOT silently wire output-side enforcement without updating
    SECURITY.md.
    """
    fake_backend, _, captured = bedrock_agent_smoke
    import boto3

    client = boto3.client("bedrock-agent-runtime")
    # The returned dict has a ``completion`` field that's an
    # iterator. The fact that the patch returns it untouched
    # (rather than wrapping with a replay proxy) is the visible
    # signal that no output extractor is installed.
    result = client.invoke_agent(
        agentId="a",
        agentAliasId="L",
        sessionId="s",
        inputText="hello",
    )
    _flush()

    assert "completion" in result, (
        "patch must NOT swallow the EventStream — caller iterates it"
    )
    # And the audit row stamped at this seam doesn't carry a
    # ``response_decision`` block (output phase didn't run).
    for ev in fake_backend.events_received:
        if ev.get("source") == "bedrock_agent":
            assert "response_decision" not in ev, (
                "bedrock_agent stamped a response_decision — "
                "would mean output-side enforcement was wired without "
                "updating the advisory contract."
            )


# ── 7. No false "enforced" stamps for tool/output decisions ────────


def test_invoke_agent_audit_row_does_not_stamp_enforced_on_block(
    bedrock_agent_smoke: Any,
) -> None:
    """Input-side blocks on bedrock_agent stamp ``enforcement_status =
    "enforced"`` (we really did prevent the AWS call); output-side
    decisions, when they exist, must be ``advisory``. This test
    pins the input-side honesty: a real input block produces an
    ``enforced`` stamp.

    The complementary "output decisions don't ship as enforced"
    invariant is enforced upstream by the fact that the patch
    doesn't pass ``extract_output_signals`` (covered by the
    previous test). When that field is None,
    ``_dispatch_per_tool_steps`` never runs and the only audit row
    on the wire is the input-side one.
    """
    fake_backend, _, _ = bedrock_agent_smoke
    _set_rules(_pii_block_rule())
    import boto3

    client = boto3.client("bedrock-agent-runtime")
    with pytest.raises(PermissionError):
        client.invoke_agent(
            agentId="a",
            agentAliasId="L",
            sessionId="s",
            inputText="block me SSN 456-78-9012",
        )
    _flush()

    # The ``run.end`` aggregate also carries ``verdict='block'``
    # (rolled up from the only step) but doesn't itself stamp an
    # ``enforcement_status`` field — that lives on each step. Filter
    # to step / legacy single-row events so the assertion targets the
    # row whose contract this test is actually pinning.
    blocks = [
        ev for ev in fake_backend.events_received
        if ev.get("verdict") == "block"
        and ev.get("kind") != "run.end"
    ]
    assert blocks, "input block produced no audit row"
    for ev in blocks:
        # Input-side blocks are real prevention — the call to
        # boto3 never fires, so the stamp is honest.
        assert ev.get("enforcement_status") == "enforced", (
            f"input-side block stamped non-enforced: {ev!r}"
        )
