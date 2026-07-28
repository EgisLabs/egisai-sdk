"""Judge verdict cache + Phase-2 parallel fan-out.

Both changes exist to stop ``policy_latency_ms`` growing with the
number of ``semantic_guard`` policies an operator configures. Both are
meant to be *accuracy-neutral*, so most of what follows pins the
invariants that make that claim true rather than the speedup itself:

* every judge call carries the byte-identical payload it carried when
  Phase 2 was serial;
* records come back in policy order, so ``matched_policy`` can't race;
* token spend still lands on the audit row even though the calls now
  happen on worker threads;
* a cache hit reports zero token spend, because it spent none;
* outages are never memoized.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import httpx
import pytest

from egisai._context import (
    add_policy_usage,
    get_policy_usage,
    reset_policy_usage,
)
from egisai.policy.engine import (
    PolicyContext,
    PolicyRule,
    evaluate_policies,
)
from egisai.policy.semantic import SemanticBlocker

# ── Helpers ──────────────────────────────────────────────────────────


def _blocker(
    handler: Any, *, cache_ttl: float = 60.0, on_outage: str = "allow"
) -> SemanticBlocker:
    b = SemanticBlocker(
        platform_api_key="egis_live_x",
        platform_base_url="http://fake",
        on_outage=on_outage,
        judge_cache_ttl_secs=cache_ttl,
    )
    b._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def _guard(name: str, intent: str) -> PolicyRule:
    return PolicyRule(
        id=name,
        name=name,
        type="semantic_guard",
        tenant=None,
        config={"intents": [intent]},
    )


def _ctx(prompt_text: str = "hello") -> PolicyContext:
    return PolicyContext(
        tenant="t",
        model="gpt-4",
        prompt_text=prompt_text,
        prompt_chars=len(prompt_text),
        stream=False,
    )


def _allow_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "match": False,
            "intent": "",
            "confidence": 0.0,
            "tokens_in": 10,
            "tokens_out": 2,
        },
    )


def _block_response(intent: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "match": True,
            "intent": intent,
            "confidence": 0.99,
            "tokens_in": 10,
            "tokens_out": 2,
        },
    )


# ── Verdict cache ────────────────────────────────────────────────────


def test_identical_question_is_answered_from_cache() -> None:
    """Same prompt + same intents ⇒ one round-trip, not two."""
    calls: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append({})
        return _allow_response()

    b = _blocker(handler)
    cfg = {"intents": ["delete the production database"]}

    assert b.check("drop all tables", cfg) is None
    assert b.check("drop all tables", cfg) is None

    assert len(calls) == 1, "second identical question must not hit the network"


def test_cache_hit_preserves_a_block_verdict() -> None:
    """Caching must not soften enforcement — a blocked prompt stays
    blocked on the hit, with the same intent and confidence."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return _block_response("delete the production database")

    b = _blocker(handler)
    cfg = {"intents": ["delete the production database"]}

    first = b.check("drop all tables", cfg)
    second = b.check("drop all tables", cfg)

    assert first is not None and second is not None
    assert len(calls) == 1
    assert second.intent == first.intent
    assert second.similarity == first.similarity


def test_different_prompt_is_a_separate_key() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append("x")
        return _allow_response()

    b = _blocker(handler)
    cfg = {"intents": ["delete the production database"]}

    b.check("drop all tables", cfg)
    b.check("drop the users table", cfg)

    assert len(calls) == 2


def test_editing_the_policy_invalidates_structurally() -> None:
    """A changed intent list is a different key, so a policy edit can
    never be served a verdict scored against the old rule. This is the
    property that makes the cache safe for governance."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append("x")
        return _allow_response()

    b = _blocker(handler)

    b.check("drop all tables", {"intents": ["delete the database"]})
    b.check("drop all tables", {"intents": ["delete the database", "wipe logs"]})

    assert len(calls) == 2


def test_changed_threshold_is_a_separate_key() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append("x")
        return _allow_response()

    b = _blocker(handler)

    b.check("drop all tables", {"intents": ["delete"], "threshold": 0.7})
    b.check("drop all tables", {"intents": ["delete"], "threshold": 0.9})

    assert len(calls) == 2


def test_cache_hit_books_zero_tokens() -> None:
    """A hit spent no tokens, so it must not inflate the cost columns."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _allow_response()

    b = _blocker(handler)
    cfg = {"intents": ["delete the production database"]}

    reset_policy_usage()
    b.check("drop all tables", cfg)
    after_miss = get_policy_usage()

    b.check("drop all tables", cfg)
    after_hit = get_policy_usage()

    assert after_miss == (10, 2)
    assert after_hit == after_miss, "cache hit must add no token spend"


def test_ttl_zero_disables_the_cache() -> None:
    """Escape hatch for auditors who require every call to be judged."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append("x")
        return _allow_response()

    b = _blocker(handler, cache_ttl=0.0)
    cfg = {"intents": ["delete"]}

    b.check("drop all tables", cfg)
    b.check("drop all tables", cfg)

    assert len(calls) == 2


def test_expired_entry_is_refetched() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append("x")
        return _allow_response()

    b = _blocker(handler, cache_ttl=0.05)
    cfg = {"intents": ["delete"]}

    b.check("drop all tables", cfg)
    time.sleep(0.08)
    b.check("drop all tables", cfg)

    assert len(calls) == 2


def test_outage_is_never_cached_fail_open() -> None:
    """An unreachable judge is a fact about the network at one instant,
    not a verdict. Memoizing it would let a blip govern the whole TTL
    window."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append("x")
        raise httpx.ConnectError("down")

    b = _blocker(handler, on_outage="allow")
    cfg = {"intents": ["delete"]}

    assert b.check("drop all tables", cfg) is None
    assert b.check("drop all tables", cfg) is None

    assert len(calls) == 2, "outage must be retried, not served from cache"


def test_outage_is_never_cached_fail_closed() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append("x")
        raise httpx.ConnectError("down")

    b = _blocker(handler, on_outage="block")
    cfg = {"intents": ["delete"]}

    first = b.check("drop all tables", cfg)
    second = b.check("drop all tables", cfg)

    assert first is not None and second is not None
    assert len(calls) == 2


def test_recovery_after_outage_is_not_masked_by_a_cached_failure() -> None:
    """The judge going down and coming back must produce a real verdict
    on the next call."""
    state = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["n"] += 1
        if state["n"] == 1:
            raise httpx.ConnectError("down")
        return _block_response("delete the production database")

    b = _blocker(handler, on_outage="allow")
    cfg = {"intents": ["delete the production database"]}

    assert b.check("drop all tables", cfg) is None
    recovered = b.check("drop all tables", cfg)
    assert recovered is not None
    assert recovered.intent == "delete the production database"


def test_cache_is_bounded() -> None:
    """An unbounded dict in a long-lived server process is a leak."""
    from egisai.policy import semantic as semantic_mod

    def handler(request: httpx.Request) -> httpx.Response:
        return _allow_response()

    b = _blocker(handler)
    for i in range(semantic_mod._JUDGE_CACHE_MAX + 10):
        b.check(f"prompt {i}", {"intents": ["delete"]})

    assert len(b._cache) <= semantic_mod._JUDGE_CACHE_MAX


def test_cache_is_per_blocker_instance() -> None:
    """Two independently configured blockers must not share verdicts."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append("x")
        return _allow_response()

    a = _blocker(handler)
    b = _blocker(handler)
    cfg = {"intents": ["delete"]}

    a.check("drop all tables", cfg)
    b.check("drop all tables", cfg)

    assert len(calls) == 2


# ── Phase 2 fan-out ──────────────────────────────────────────────────


def test_every_guard_is_judged_and_payloads_are_unchanged() -> None:
    """Parallelism must not change *what* is asked, only when.

    Each policy still gets its own round-trip carrying its own intent
    list against the same prompt — byte-identical to the serial path.
    """
    seen: list[dict[str, Any]] = []
    lock = threading.Lock()

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        with lock:
            seen.append(json.loads(request.content))
        return _allow_response()

    b = _blocker(handler)
    policies = [
        _guard("a", "delete the database"),
        _guard("b", "exfiltrate customer data"),
        _guard("c", "suppress the audit log"),
    ]
    ctx = _ctx()

    decision = evaluate_policies(policies, ctx, semantic_blocker=b)

    assert decision.verdict == "allow"
    assert len(seen) == 3
    assert {tuple(s["intents"]) for s in seen} == {
        ("delete the database",),
        ("exfiltrate customer data",),
        ("suppress the audit log",),
    }
    assert {s["prompt_text"] for s in seen} == {"hello"}


def test_fan_out_actually_overlaps() -> None:
    """The point of the change: N guards cost ~1 round-trip of wall
    clock, not N."""
    delay = 0.25
    lock = threading.Lock()
    concurrent = {"now": 0, "peak": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            concurrent["now"] += 1
            concurrent["peak"] = max(concurrent["peak"], concurrent["now"])
        time.sleep(delay)
        with lock:
            concurrent["now"] -= 1
        return _allow_response()

    b = _blocker(handler)
    policies = [_guard(f"g{i}", f"intent {i}") for i in range(4)]
    ctx = _ctx()

    started = time.monotonic()
    evaluate_policies(policies, ctx, semantic_blocker=b)
    elapsed = time.monotonic() - started

    assert concurrent["peak"] > 1, "guards were still evaluated serially"
    assert elapsed < delay * len(policies) * 0.8, (
        f"4 guards took {elapsed:.2f}s; serial would be ~{delay * 4:.2f}s"
    )


def test_matched_policy_follows_policy_order_not_completion_order() -> None:
    """``_synthesize_decision`` names the first block at the winning
    precedence, so a slow-but-earlier policy must still win over a
    fast-but-later one. Otherwise the dashboard's attributed policy
    becomes a race."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        intent = body["intents"][0]
        # The earlier policy answers LAST.
        if intent == "first intent":
            time.sleep(0.2)
        return _block_response(intent)

    b = _blocker(handler)
    policies = [
        _guard("first", "first intent"),
        _guard("second", "second intent"),
    ]
    ctx = _ctx()

    decision = evaluate_policies(policies, ctx, semantic_blocker=b)

    assert decision.verdict == "block"
    assert decision.matched_policy == "first"
    assert [r.name for r in decision.matched_policies] == ["first", "second"]


def test_token_spend_survives_the_worker_threads() -> None:
    """The regression this change had to fix first.

    Worker threads don't inherit context vars, so before ``_fan_out``
    copied the gate's context every judge call made off the main thread
    silently dropped its token spend off the audit row.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return _allow_response()

    b = _blocker(handler)
    policies = [_guard(f"g{i}", f"intent {i}") for i in range(3)]
    ctx = _ctx()

    reset_policy_usage()
    evaluate_policies(policies, ctx, semantic_blocker=b)

    # 3 guards × (10 in, 2 out), all charged from worker threads.
    assert get_policy_usage() == (30, 6)


def test_one_failing_guard_does_not_poison_the_batch() -> None:
    """Fail-open per task: a dead judge call for one policy must not
    stop the others from being enforced."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content)
        if body["intents"][0] == "boom":
            raise httpx.ConnectError("down")
        return _block_response(body["intents"][0])

    b = _blocker(handler, on_outage="allow")
    policies = [_guard("bad", "boom"), _guard("good", "delete the database")]
    ctx = _ctx()

    decision = evaluate_policies(policies, ctx, semantic_blocker=b)

    assert decision.verdict == "block"
    assert decision.matched_policy == "good"


def test_single_guard_skips_threading_entirely() -> None:
    """The common one-policy turn must pay no executor overhead."""
    threads: set[str] = set()

    def handler(request: httpx.Request) -> httpx.Response:
        threads.add(threading.current_thread().name)
        return _allow_response()

    b = _blocker(handler)
    ctx = _ctx()

    evaluate_policies([_guard("only", "delete")], ctx, semantic_blocker=b)

    assert threads == {threading.current_thread().name}


def test_deterministic_phase_never_threads() -> None:
    """Phase 1 is pure-Python regex/math; a thread hand-off would cost
    more than the work, and the judge must never be reachable from it."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append("x")
        return _allow_response()

    b = _blocker(handler)
    policies = [
        PolicyRule(
            id="1",
            name="len",
            type="max_prompt_chars",
            tenant=None,
            config={"max_chars": 3},
        ),
        PolicyRule(
            id="2",
            name="re",
            type="deny_regex",
            tenant=None,
            config={"patterns": ["secret"]},
        ),
    ]
    ctx = _ctx("a much longer prompt")

    decision = evaluate_policies(policies, ctx, semantic_blocker=b)

    assert decision.verdict == "block"
    assert calls == [], "Phase 1 must not reach the judge"


def test_nested_fan_out_stays_within_the_global_budget() -> None:
    """Phase 2 fans out across policies and each guard fans out across
    tool calls. Uncoordinated, that product would rate-limit us into
    ``Retry-After`` storms."""
    from egisai.policy import engine as engine_mod
    from egisai.policy.engine import OutputPolicyContext, evaluate_output_policies

    lock = threading.Lock()
    concurrent = {"now": 0, "peak": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        with lock:
            concurrent["now"] += 1
            concurrent["peak"] = max(concurrent["peak"], concurrent["now"])
        time.sleep(0.05)
        with lock:
            concurrent["now"] -= 1
        return _allow_response()

    b = _blocker(handler)
    policies = [
        PolicyRule(
            id=f"g{i}",
            name=f"g{i}",
            type="semantic_guard",
            tenant=None,
            phase="response",
            config={"intents": [f"intent {i}"], "targets": ["tool_calls"]},
        )
        for i in range(4)
    ]
    ctx = OutputPolicyContext(
        tenant="t",
        model="gpt-4",
        text="",
        tool_names=[f"tool_{j}" for j in range(6)],
        tool_calls=[{"name": f"tool_{j}", "input": {"a": j}} for j in range(6)],
        mcp_targets=[],
        stream=False,
    )

    evaluate_output_policies(policies, ctx, semantic_blocker=b)

    assert concurrent["peak"] > 1, "no parallelism happened at all"
    assert concurrent["peak"] <= engine_mod._TOOL_JUDGE_MAX_WORKERS, (
        f"peak concurrency {concurrent['peak']} exceeded the "
        f"{engine_mod._TOOL_JUDGE_MAX_WORKERS} budget"
    )


def test_concurrent_token_accounting_is_not_lost_to_races() -> None:
    """The accumulator is mutated from many threads at once."""
    reset_policy_usage()
    barrier = threading.Barrier(8)

    def worker() -> None:
        barrier.wait()
        for _ in range(50):
            add_policy_usage(1, 1)

    import contextvars

    threads = [
        threading.Thread(target=contextvars.copy_context().run, args=(worker,))
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert get_policy_usage() == (400, 400)


@pytest.mark.parametrize("count", [2, 3, 8, 12])
def test_all_guards_are_judged_regardless_of_batch_size(count: int) -> None:
    """More policies than the worker cap must still all be evaluated —
    they degrade into serialized batches, never get dropped."""
    lock = threading.Lock()
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        with lock:
            seen.append(json.loads(request.content)["intents"][0])
        return _allow_response()

    b = _blocker(handler)
    policies = [_guard(f"g{i}", f"intent {i}") for i in range(count)]
    ctx = _ctx()

    evaluate_policies(policies, ctx, semantic_blocker=b)

    assert sorted(seen) == sorted(f"intent {i}" for i in range(count))
