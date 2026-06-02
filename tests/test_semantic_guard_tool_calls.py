"""``semantic_guard`` with ``targets: ["tool_calls"]`` — intent
classification of agent ACTIONS, not just prose.

Pre-0.24, ``semantic_guard`` only judged text — a user prompt on
the input side, the model's accumulated assistant text on the
output side. That covered "the user asked the agent to do
something destructive" but left the most important agentic
failure mode untouched: **the agent itself decides to do
something destructive**. A model that calls
``db_execute(query="DELETE FROM users")`` would slip past every
``semantic_guard`` rule whose intent list described destructive
behavior in plain English — the judge simply was never asked
about the tool call.

0.24 adds an opt-in ``targets`` config field. When operators set
``targets: ["tool_calls"]`` (or ``["text", "tool_calls"]``), the
matcher synthesizes one sentence per pending tool call —

    "The agent is requesting to invoke tool 'X' with arguments {...}"

— PII-label-redacts the arguments, and asks the judge whether
that intent matches anything on the operator's block list.

These tests pin every contract that matters:

* the matcher does call the judge when ``targets`` includes
  ``"tool_calls"`` and the context carries tool calls;
* it does NOT call the judge when ``targets`` omits
  ``"tool_calls"`` (backwards-compat for every rule shipped
  before this version);
* tool arguments are PII-label-redacted BEFORE leaving the SDK
  (security-and-compliance.mdc §1);
* multi-tool blame is correctly attributed to the offending
  tool's name in ``matched_policy.message``;
* the same rule still fires on text alone when
  ``targets=["text"]`` — no regression on the legacy path.

The matcher is exercised in isolation here. End-to-end coverage
through the Claude Agent SDK PreToolUse hook lives in
``test_claude_agent_sdk_pretooluse.py``.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

# ── Helpers ──────────────────────────────────────────────────────────


def _stub_judge_match(intent: str, confidence: float = 0.92) -> dict[str, Any]:
    return {
        "match": True,
        "intent": intent,
        "confidence": confidence,
        "tokens_in": 220,
        "tokens_out": 10,
    }


def _stub_judge_no_match() -> dict[str, Any]:
    return {
        "match": False,
        "intent": "",
        "confidence": 0.0,
        "tokens_in": 200,
        "tokens_out": 5,
    }


def _make_blocker(
    handler: Any,
) -> Any:
    """Build a ``SemanticBlocker`` wired to an in-memory transport."""
    from egisai.policy.semantic import SemanticBlocker

    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
    )
    blocker._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return blocker


def _rule(
    *,
    intents: list[str],
    targets: list[str] | None = None,
    message: str | None = None,
) -> Any:
    """Build a ``PolicyRule`` for the engine matcher."""
    from egisai.policy.engine import PolicyRule

    config: dict[str, Any] = {"intents": intents}
    if targets is not None:
        config["targets"] = targets
    if message is not None:
        config["message"] = message
    return PolicyRule(
        id="r1",
        name="forbid-destructive-actions",
        type="semantic_guard",
        tenant=None,
        config=config,
    )


# ── 1. Matcher fires on tool_calls when targets includes "tool_calls" ─


def test_matcher_judges_tool_call_when_targets_include_tool_calls() -> None:
    """Operator policy: ``targets: ["tool_calls"]`` + intent
    "delete rows from a database table". Model calls
    ``db_execute(query="DELETE FROM users")``. Matcher synthesizes
    a sentence, sends it to the judge, judge says match → block."""
    from egisai.policy.engine import _semantic_guard_match

    captured_bodies: list[dict[str, Any]] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode()))
        return httpx.Response(
            200, json=_stub_judge_match("delete rows from a database table"),
        )

    blocker = _make_blocker(transport_handler)

    record = _semantic_guard_match(
        policy=_rule(
            intents=["delete rows from a database table"],
            targets=["tool_calls"],
        ),
        text="",  # no assistant text yet — PreToolUse fires with text=""
        tool_calls=[
            {"name": "db_execute", "input": {"query": "DELETE FROM users"}}
        ],
        semantic_blocker=blocker,
        side="output",
    )

    assert record is not None, "matcher must block when judge matches the tool call"
    assert record.verdict == "block"
    assert record.reason_code == "semantic_blocked_tool"
    # Blame attribution: the message names the offending tool so the
    # audit row tells operators WHICH tool tripped the rule.
    assert "'db_execute'" in record.message
    # The judge was called once.
    assert len(captured_bodies) == 1
    body = captured_bodies[0]
    # The synthesized sentence is what reached the platform.
    assert body["prompt_text"].startswith(
        "The agent is requesting to invoke tool 'db_execute' with arguments:"
    )
    # The intents list is forwarded unchanged.
    assert body["intents"] == ["delete rows from a database table"]


def test_matcher_blames_first_matching_tool_in_input_order() -> None:
    """When multiple tool calls in a turn match, the matcher's
    audit row blames the FIRST matching tool by input order — that
    keeps the dashboard's per-turn narrative deterministic ("the
    agent tried to do X first") regardless of which judge call
    happened to return first under parallelism.

    Cost note (BUG 3 fix, 0.30+): tool_calls evaluation now runs
    every per-tool judge round-trip in parallel inside a bounded
    thread pool, so the matcher pays for ALL N tools instead of
    short-circuiting on the first match. Trade-off: we burn 3x
    judge cost in the rare "first tool matches" case to collapse
    wall-clock latency from ``sum(t_i)`` to ``max(t_i)`` in the
    common "no match across N tools" case. Pre-fix a 6-tool turn
    paid ~3.6 s sequentially when nothing matched; post-fix it
    pays ~0.6 s. The match-blame contract below stays unchanged.
    """
    from egisai.policy.engine import _semantic_guard_match

    calls: list[str] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        calls.append(body["prompt_text"])
        if "db_execute" in body["prompt_text"]:
            return httpx.Response(
                200, json=_stub_judge_match("delete rows from a database table"),
            )
        return httpx.Response(200, json=_stub_judge_no_match())

    blocker = _make_blocker(transport_handler)

    record = _semantic_guard_match(
        policy=_rule(
            intents=["delete rows from a database table"],
            targets=["tool_calls"],
        ),
        text="",
        tool_calls=[
            {"name": "db_execute", "input": {"query": "DELETE FROM users"}},
            {"name": "send_email", "input": {"to": "ops@example.com"}},
            {"name": "post_to_slack", "input": {"channel": "#alerts"}},
        ],
        semantic_blocker=blocker,
        side="output",
    )

    assert record is not None
    assert "'db_execute'" in record.message, (
        "the audit message must name the FIRST matching tool by "
        "input order even when judge calls run in parallel"
    )


def test_matcher_walks_every_tool_until_a_match() -> None:
    """If the first N tool calls don't match, the matcher's
    verdict still attributes the block to the tool that DID match.
    This is the contract that makes the rule useful: a destructive
    tool buried among benign ones still gets caught and named.

    Pre-0.30 the matcher short-circuited and only paid for judge
    calls up to the first match. Post-fix it parallelizes every
    tool's judge call inside a bounded thread pool, so the wall-
    clock for an N-tool turn collapses toward the slowest single
    call instead of the sum. The naming contract is unchanged.
    """
    from egisai.policy.engine import _semantic_guard_match

    calls: list[str] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        calls.append(body["prompt_text"])
        if "destroy_production" in body["prompt_text"]:
            return httpx.Response(
                200, json=_stub_judge_match("wipe the production database"),
            )
        return httpx.Response(200, json=_stub_judge_no_match())

    blocker = _make_blocker(transport_handler)

    record = _semantic_guard_match(
        policy=_rule(
            intents=["wipe the production database"],
            targets=["tool_calls"],
        ),
        text="",
        tool_calls=[
            {"name": "list_customers", "input": {}},
            {"name": "send_email", "input": {"to": "ops@example.com"}},
            {"name": "destroy_production", "input": {"confirm": True}},
            {"name": "log_event", "input": {"event": "done"}},
        ],
        semantic_blocker=blocker,
        side="output",
    )

    assert record is not None
    assert "'destroy_production'" in record.message
    # Parallel evaluation: the matcher fires all 4 judge calls
    # concurrently regardless of which one matches. Asserting
    # ``len(calls) == 4`` pins the new contract; the old "walked
    # exactly 3" assertion was a sequential-loop artifact.
    assert len(calls) == 4, (
        "every tool gets a parallel judge call (post-BUG-3 fix); "
        "no early termination in the input order"
    )


def test_matcher_parallelizes_judge_calls_for_multi_tool_turns() -> None:
    """Regression — BUG 3: when ``targets=["tool_calls"]`` and the
    turn has N tools, the matcher MUST NOT issue N sequential
    judge round-trips. Pre-fix this loop was strictly serial — a
    6-tool turn paid ~6 × P50 judge latency in policy_latency_ms
    on its own. Post-fix it parks each tool's blocking
    ``semantic_blocker.check`` on a worker thread so wall-clock
    collapses to ``max(t_i) + thread overhead`` rather than
    ``sum(t_i)``.

    The test stubs the judge with a per-call sleep so a *strictly
    sequential* implementation would take ``N × delay`` and a
    *parallel* implementation takes ~``delay``. Asserting on the
    elapsed wall-clock makes the regression alarm fire if some
    future refactor accidentally re-serialises the loop.
    """
    import time as _time

    from egisai.policy.engine import _semantic_guard_match

    delay_per_call = 0.15  # 150 ms each — exaggerates the difference
    n_tools = 6

    def transport_handler(request: httpx.Request) -> httpx.Response:
        _time.sleep(delay_per_call)
        return httpx.Response(200, json=_stub_judge_no_match())

    blocker = _make_blocker(transport_handler)

    started = _time.monotonic()
    record = _semantic_guard_match(
        policy=_rule(
            intents=["wipe the production database"],
            targets=["tool_calls"],
        ),
        text="",
        tool_calls=[
            {"name": f"safe_tool_{i}", "input": {"i": i}}
            for i in range(n_tools)
        ],
        semantic_blocker=blocker,
        side="output",
    )
    elapsed = _time.monotonic() - started

    assert record is None, (
        "no match expected — every tool returns no_match"
    )
    # Sequential lower bound: ``n_tools * delay_per_call`` =
    # 6 × 150 ms = 900 ms. Parallel upper bound (with thread pool
    # overhead): generous 600 ms ceiling. We anchor the assertion
    # well below the sequential lower bound so a flake-free CI
    # signal still proves parallelism.
    sequential_floor = n_tools * delay_per_call
    assert elapsed < sequential_floor * 0.7, (
        f"expected parallel evaluation; took {elapsed:.2f}s for "
        f"{n_tools} tools at {delay_per_call:.2f}s each. "
        f"Sequential lower bound is {sequential_floor:.2f}s; the "
        f"matcher must collapse to ~max(t_i) + pool overhead."
    )


# ── 2. Backwards-compat: no ``targets`` field = text-only ────────────


def test_no_targets_field_keeps_legacy_text_only_behavior() -> None:
    """Every ``semantic_guard`` policy shipped before 0.24 has no
    ``targets`` field. The matcher MUST treat them as
    ``targets=["text"]`` — i.e. judge the text, ignore the tool
    calls. Any other default would silently change the meaning of
    every customer's existing rules on upgrade."""
    from egisai.policy.engine import _semantic_guard_match

    calls: list[str] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        calls.append(body["prompt_text"])
        return httpx.Response(200, json=_stub_judge_no_match())

    blocker = _make_blocker(transport_handler)

    record = _semantic_guard_match(
        policy=_rule(
            intents=["wipe the production database"],
            targets=None,  # legacy rule — no targets field
        ),
        text="",  # text is empty (PreToolUse path)
        tool_calls=[
            {"name": "destroy_production", "input": {"confirm": True}},
        ],
        semantic_blocker=blocker,
        side="output",
    )

    assert record is None, "no targets ⇒ tool_calls ignored ⇒ no match on empty text"
    assert calls == [], (
        "legacy rule with no targets MUST NOT round-trip the judge "
        "for a tool call (would be a silent behavior change on upgrade)"
    )


def test_targets_explicit_text_still_works_on_text() -> None:
    """An explicit ``targets: ["text"]`` rule is identical in
    behavior to a no-targets rule — the judge sees the text."""
    from egisai.policy.engine import _semantic_guard_match

    captured: list[dict[str, Any]] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content.decode()))
        return httpx.Response(
            200, json=_stub_judge_match("delete all users"),
        )

    blocker = _make_blocker(transport_handler)

    record = _semantic_guard_match(
        policy=_rule(
            intents=["delete all users"],
            targets=["text"],
        ),
        text="Please remove every user from the database.",
        tool_calls=[
            # Even with tool calls in scope, ``targets=["text"]``
            # means the matcher must NOT touch them.
            {"name": "db_execute", "input": {"query": "DROP TABLE users"}}
        ],
        semantic_blocker=blocker,
        side="prompt",
    )

    assert record is not None
    assert record.reason_code == "semantic_blocked"  # text path, not tool
    assert len(captured) == 1
    assert captured[0]["prompt_text"] == (
        "Please remove every user from the database."
    )


def test_targets_both_judges_text_first_then_tool_calls() -> None:
    """``targets: ["text", "tool_calls"]`` — text is judged first.
    On match, tool_calls are skipped (one judge call total, lower
    cost). On no-match for text, the matcher continues to the
    tools."""
    from egisai.policy.engine import _semantic_guard_match

    captured: list[dict[str, Any]] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        captured.append(body)
        # Text path always reports no_match here; tool path matches.
        if "invoke tool" in body["prompt_text"]:
            return httpx.Response(
                200, json=_stub_judge_match("wipe the production database"),
            )
        return httpx.Response(200, json=_stub_judge_no_match())

    blocker = _make_blocker(transport_handler)

    record = _semantic_guard_match(
        policy=_rule(
            intents=["wipe the production database"],
            targets=["text", "tool_calls"],
        ),
        text="Here is some benign assistant chatter.",
        tool_calls=[
            {"name": "destroy_production", "input": {"confirm": True}},
        ],
        semantic_blocker=blocker,
        side="output",
    )

    assert record is not None
    assert record.reason_code == "semantic_blocked_tool"
    assert "'destroy_production'" in record.message
    # Both probes ran: first the text (no_match), then the tool
    # (match). Order is text → tools so blame on prose stays
    # primary when both could fire.
    assert len(captured) == 2
    assert "benign assistant chatter" in captured[0]["prompt_text"]
    assert "invoke tool 'destroy_production'" in captured[1]["prompt_text"]


# ── 3. Privacy contract: PII in tool args is label-redacted ──────────


def test_pii_in_tool_args_is_label_redacted_before_judge() -> None:
    """security-and-compliance.mdc §1 — raw PII MUST be masked
    before reaching ANY third party, including our own LLM-based
    policy judges. Tool arguments are user-controlled (a model
    wrote them); they routinely carry email / SSN / credit-card.
    The matcher MUST label_redact before serializing for the judge.

    Intent classification accuracy is preserved because the judge
    cares about verb/noun shape ("delete <NAME>"), not the exact
    identifier values."""
    from egisai.policy.engine import _semantic_guard_match

    captured_bodies: list[dict[str, Any]] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode()))
        return httpx.Response(200, json=_stub_judge_no_match())

    blocker = _make_blocker(transport_handler)

    _ = _semantic_guard_match(
        policy=_rule(
            intents=["delete this customer's record"],
            targets=["tool_calls"],
        ),
        text="",
        tool_calls=[
            {
                "name": "delete_customer",
                "input": {
                    # All three of these are first-party detectable PII
                    # (regex+checksum on the fallback path; Presidio
                    # on the warm path). Any one being present means
                    # the rendered JSON for the judge must mask it.
                    # NB: ``acme.com`` is used deliberately —
                    # ``example.com`` is RFC-2606 reserved and the
                    # SDK explicitly skips redacting it so docs /
                    # tests aren't full of ``<EMAIL>`` noise (see
                    # ``_pii_helpers.is_reserved_email_domain``).
                    "email": "victim@acme.com",
                    "ssn": "123-45-6789",
                    "credit_card": "4111-1111-1111-1111",
                },
            }
        ],
        semantic_blocker=blocker,
        side="output",
    )

    assert len(captured_bodies) == 1
    sent_text = captured_bodies[0]["prompt_text"]
    # The tool name still reaches the judge (intent classification
    # needs the action verb) — the args are scrubbed.
    assert "delete_customer" in sent_text
    # None of the raw PII values appear on the wire.
    assert "victim@acme.com" not in sent_text
    assert "123-45-6789" not in sent_text
    assert "4111-1111-1111-1111" not in sent_text


# ── 4. Edge cases ────────────────────────────────────────────────────


def test_no_tool_calls_no_text_returns_none() -> None:
    """A rule with ``targets=["tool_calls"]`` and an empty
    tool_calls list is a no-op — the matcher must short-circuit
    without calling the judge."""
    from egisai.policy.engine import _semantic_guard_match

    calls: list[str] = []

    def transport_handler(_request: httpx.Request) -> httpx.Response:
        calls.append("called")
        return httpx.Response(200, json=_stub_judge_no_match())

    blocker = _make_blocker(transport_handler)

    record = _semantic_guard_match(
        policy=_rule(
            intents=["x"],
            targets=["tool_calls"],
        ),
        text="ignored",
        tool_calls=[],
        semantic_blocker=blocker,
        side="output",
    )

    assert record is None
    assert calls == []


def test_no_semantic_blocker_is_silent_no_op() -> None:
    """An SDK installed without ``init()`` has no live blocker.
    The matcher must return None and never raise. Same shape as
    the pre-0.24 contract."""
    from egisai.policy.engine import _semantic_guard_match

    record = _semantic_guard_match(
        policy=_rule(
            intents=["delete all users"],
            targets=["text", "tool_calls"],
        ),
        text="Delete all users",
        tool_calls=[{"name": "db_execute", "input": {}}],
        semantic_blocker=None,
        side="output",
    )
    assert record is None


def test_tool_call_with_no_name_is_skipped() -> None:
    """Defensive: a malformed tool_call dict without ``name`` must
    not crash the matcher. The matcher silently moves on to the
    next call."""
    from egisai.policy.engine import _semantic_guard_match

    captured_bodies: list[dict[str, Any]] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode()))
        return httpx.Response(
            200, json=_stub_judge_match("wipe the production database"),
        )

    blocker = _make_blocker(transport_handler)

    record = _semantic_guard_match(
        policy=_rule(
            intents=["wipe the production database"],
            targets=["tool_calls"],
        ),
        text="",
        tool_calls=[
            {"input": {"query": "DELETE FROM users"}},  # NO NAME
            None,  # type: ignore[list-item]  # NOT A DICT
            {"name": "destroy_production", "input": {"confirm": True}},
        ],
        semantic_blocker=blocker,
        side="output",
    )

    assert record is not None
    assert "'destroy_production'" in record.message
    # The first two malformed entries were skipped — only ONE judge
    # call landed on the valid third entry.
    assert len(captured_bodies) == 1


def test_alternative_arguments_field_is_recognized() -> None:
    """Different framework extractors normalize tool args under
    different keys: ``input`` (Anthropic / Claude Agent SDK /
    OpenAI Responses) vs ``arguments`` (OpenAI Chat Completions).
    The matcher reads both so the rule works across the SDK's
    integration matrix."""
    from egisai.policy.engine import _semantic_guard_match

    captured_bodies: list[dict[str, Any]] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        captured_bodies.append(json.loads(request.content.decode()))
        return httpx.Response(
            200, json=_stub_judge_match("delete all users"),
        )

    blocker = _make_blocker(transport_handler)

    record = _semantic_guard_match(
        policy=_rule(
            intents=["delete all users"],
            targets=["tool_calls"],
        ),
        text="",
        # OpenAI Chat Completions normalization: args under
        # ``arguments`` not ``input``.
        tool_calls=[
            {"name": "db_execute", "arguments": '{"query": "DELETE FROM users"}'},
        ],
        semantic_blocker=blocker,
        side="output",
    )

    assert record is not None
    assert "'db_execute'" in record.message
    assert "db_execute" in captured_bodies[0]["prompt_text"]


# ── 5. Output-side eval routes tool_calls through the matcher ────────


def test_evaluate_output_policies_routes_tool_calls_to_matcher() -> None:
    """End-to-end through the public output-policy entry point: an
    operator with a ``semantic_guard`` rule with
    ``targets=["tool_calls"]`` sees the call refused when the
    judge matches the synthesized tool description.

    This is the contract that ``claude_agent_sdk``'s PreToolUse
    hook, the OpenAI per-turn output policy, and every other
    framework's output evaluator rely on — no patch-specific code,
    just a policy config flag."""
    from egisai.policy.engine import OutputPolicyContext, evaluate_output_policies

    def transport_handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json=_stub_judge_match("wipe the production database"),
        )

    blocker = _make_blocker(transport_handler)

    decision = evaluate_output_policies(
        [_rule(
            intents=["wipe the production database"],
            targets=["tool_calls"],
            message="Refused: agent attempted a destructive action.",
        )],
        OutputPolicyContext(
            tenant="t",
            model="claude-sonnet-4",
            text="",  # PreToolUse path — no model text yet
            tool_names=["destroy_production"],
            tool_calls=[
                {"name": "destroy_production", "input": {"confirm": True}}
            ],
            mcp_targets=[],
            stream=True,
        ),
        semantic_blocker=blocker,
    )

    assert decision.verdict == "block"
    assert decision.matched_policy == "forbid-destructive-actions"
    assert decision.message == "Refused: agent attempted a destructive action."


def test_evaluate_output_policies_allows_when_targets_excludes_tool_calls() -> None:
    """Regression: a legacy rule (no ``targets``) MUST NOT block on
    tool calls — only on text. Same call as the test above, but
    with a no-targets rule. Judge would match if asked, but the
    matcher must never ask (privacy + cost + backwards-compat)."""
    from egisai.policy.engine import OutputPolicyContext, evaluate_output_policies

    judge_calls: list[str] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        judge_calls.append("called")
        return httpx.Response(
            200, json=_stub_judge_match("wipe the production database"),
        )

    blocker = _make_blocker(transport_handler)

    decision = evaluate_output_policies(
        [_rule(intents=["wipe the production database"], targets=None)],
        OutputPolicyContext(
            tenant="t",
            model="claude-sonnet-4",
            text="",  # no text ⇒ legacy text-only matcher short-circuits
            tool_names=["destroy_production"],
            tool_calls=[
                {"name": "destroy_production", "input": {"confirm": True}}
            ],
            mcp_targets=[],
            stream=True,
        ),
        semantic_blocker=blocker,
    )

    assert decision.verdict == "allow"
    assert judge_calls == [], (
        "no targets ⇒ tool_calls invisible ⇒ judge MUST NOT be called"
    )


# ── 6. Synthesis helper unit test ───────────────────────────────────


def test_synthesize_tool_call_text_shape() -> None:
    """The synthesizer renders a sentence that:
    1. is grammatical English (the judge model expects free text)
    2. includes the tool name verbatim (intent classification needs
       the action verb)
    3. JSON-serializes the args so structured fields are visible
       to the judge
    4. PII-label-redacts the rendered args BEFORE returning"""
    from egisai.policy.engine import _synthesize_tool_call_text

    out = _synthesize_tool_call_text(
        "send_email",
        # ``acme.com`` (not ``example.com``) — RFC-2606 reserved
        # domains are deliberately not redacted so docs / tests
        # don't surface as ``<EMAIL>``. See ``is_reserved_email_domain``.
        {"to": "alice@acme.com", "subject": "hi"},
    )
    assert out.startswith(
        "The agent is requesting to invoke tool 'send_email' with arguments:"
    )
    # Email gets masked.
    assert "alice@acme.com" not in out
    # Subject (benign string) is preserved so the judge can see
    # what the agent intends to do.
    assert "hi" in out


def test_synthesize_tool_call_text_none_args() -> None:
    """A tool call with no arguments still renders as a sentence —
    the judge can decide on the name alone ("the agent is invoking
    'drop_all_tables'"  is enough signal even without args)."""
    from egisai.policy.engine import _synthesize_tool_call_text

    out = _synthesize_tool_call_text("drop_all_tables", None)
    assert "'drop_all_tables'" in out
    # Must not crash on None args.
    assert isinstance(out, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
