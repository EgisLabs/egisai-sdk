"""Policy is processing time, not wait.

Composes local CPU + judge prompt+completion, counts a merged
``judge_group`` once, and never falls back to the HTTP wait when
clocks are missing.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from egisai.policy import _processing
from egisai.policy.engine import (
    PolicyContext,
    PolicyRule,
    evaluate_output_policies,
    evaluate_policies,
)
from egisai.policy.semantic import SemanticBlocker


def _ctx(prompt: str = "hello world") -> PolicyContext:
    return PolicyContext(
        tenant="t",
        model="gpt-4",
        prompt_text=prompt,
        prompt_chars=len(prompt),
        stream=False,
        hook="model",
    )


def _pii() -> PolicyRule:
    return PolicyRule(
        id=None,
        name="pii",
        type="pii_scan",
        tenant=None,
        config={"action": "block", "types": ["ssn"]},
    )


def _guard(name: str, intent: str, *, threshold: float = 0.75) -> PolicyRule:
    return PolicyRule(
        id=name,
        name=name,
        type="semantic_guard",
        tenant=None,
        config={"intents": [intent], "threshold": threshold},
    )


def _blocker(
    handler: Any, *, cache_ttl: float = 0.0
) -> SemanticBlocker:
    b = SemanticBlocker(
        platform_api_key="egis_live_x",
        platform_base_url="http://fake",
        on_outage="allow",
        judge_cache_ttl_secs=cache_ttl,
    )
    b._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def _allow_json(**extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {
        "match": False,
        "intent": "",
        "confidence": 0.0,
        "tokens_in": 10,
        "tokens_out": 2,
    }
    body.update(extra)
    return body


def _enforcement(decision: Any) -> dict[str, Any]:
    return {
        "verdict": decision.verdict,
        "reason_code": decision.reason_code,
        "message": decision.message,
        "matched_policy": decision.matched_policy,
        "matched": [(r.name, r.type, r.verdict) for r in decision.matched_policies],
        "sanitize_types": list(decision.sanitize_types),
        "semantic_in_scope": decision.semantic_in_scope,
        "timing_keys": [
            (t.type, t.name, t.hook) for t in decision.policy_timings
        ],
    }


def test_uses_processing_not_wait() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_allow_json(processing_ms=41.5))

    decision = evaluate_policies(
        [_guard("g1", "exfiltrate secrets")],
        _ctx("hello"),
        semantic_blocker=_blocker(handler),
    )
    assert decision.processing_ms == pytest.approx(41.5)
    assert decision.policy_timings[0].ms == pytest.approx(41.5)
    assert _processing.latency_ms(decision, 800) == 42


def test_missing_clocks_omit_not_wait() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_allow_json())

    decision = evaluate_policies(
        [_guard("g1", "exfiltrate secrets")],
        _ctx("hello"),
        semantic_blocker=_blocker(handler),
    )
    assert decision.processing_ms == pytest.approx(0.0)
    assert decision.policy_timings[0].ms == pytest.approx(0.0)


def test_cache_hit_is_zero() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_allow_json(processing_ms=40.0))

    blocker = _blocker(handler, cache_ttl=60.0)
    first = evaluate_policies(
        [_guard("g1", "exfiltrate secrets")],
        _ctx("same text"),
        semantic_blocker=blocker,
    )
    second = evaluate_policies(
        [_guard("g1", "exfiltrate secrets")],
        _ctx("same text"),
        semantic_blocker=blocker,
    )
    assert calls["n"] == 1
    assert first.processing_ms == pytest.approx(40.0)
    assert second.processing_ms == pytest.approx(0.0)


def test_merge_counts_judge_group_once(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGISAI_FAST_GOVERNANCE", "on")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_allow_json(processing_ms=50.0))

    decision = evaluate_policies(
        [
            _guard("g1", "exfiltrate secrets"),
            _guard("g2", "delete production"),
            _guard("g3", "disable audit"),
        ],
        _ctx("hello"),
        semantic_blocker=_blocker(handler),
    )
    assert calls["n"] == 1
    assert decision.processing_ms == pytest.approx(50.0, abs=2.0)
    assert len(decision.policy_timings) == 3
    groups = {t.judge_group for t in decision.policy_timings}
    assert len(groups) == 1


def test_parallel_wave_is_max_not_sum() -> None:
    n = {"i": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        n["i"] += 1
        ms = 10.0 if n["i"] == 1 else 40.0
        return httpx.Response(200, json=_allow_json(processing_ms=ms))

    decision = evaluate_policies(
        [
            _guard("g1", "exfiltrate secrets", threshold=0.7),
            _guard("g2", "delete production", threshold=0.9),
        ],
        _ctx("hello"),
        semantic_blocker=_blocker(handler),
    )
    assert n["i"] == 2
    assert decision.processing_ms == pytest.approx(40.0)


def test_sequential_phases_sum() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_allow_json(processing_ms=20.0))

    from egisai.policy.engine import OutputPolicyContext

    blocker = _blocker(handler)
    inp = evaluate_policies(
        [_guard("g1", "exfiltrate secrets")],
        _ctx("hello"),
        semantic_blocker=blocker,
    )
    out = evaluate_output_policies(
        [_guard("g2", "leak secrets")],
        OutputPolicyContext(
            tenant="t",
            model="gpt-4",
            text="hello back",
            tool_names=[],
            tool_calls=[],
            mcp_targets=[],
            stream=False,
            hook="response",
        ),
        semantic_blocker=blocker,
    )
    assert inp.processing_ms == pytest.approx(20.0)
    assert out.processing_ms == pytest.approx(20.0)
    assert _processing.latency_ms(inp, 999) + _processing.latency_ms(
        out, 999
    ) == 40


def test_enforcement_unchanged_when_json_has_clocks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_allow_json(processing_ms=40.0))

    policies = [_pii(), _guard("g1", "exfiltrate secrets")]
    a = evaluate_policies(policies, _ctx("hello"), semantic_blocker=_blocker(handler))
    b = evaluate_policies(policies, _ctx("hello"), semantic_blocker=_blocker(handler))
    assert _enforcement(a) == _enforcement(b)
    assert a.processing_ms is not None
    assert b.processing_ms is not None
    assert a.processing_ms >= 40.0
    assert b.processing_ms >= 40.0


def test_max_wave_counts_group_once() -> None:
    assert _processing.max_wave(
        [("g", 50.0), ("g", 50.0), ("g", 50.0)]
    ) == pytest.approx(50.0)
    assert _processing.max_wave(
        [("a", 10.0), ("b", 40.0)]
    ) == pytest.approx(40.0)
    assert _processing.max_wave([]) is None
    assert _processing.max_wave([("g", None)]) == pytest.approx(0.0)


def test_latency_ms_prefers_processing() -> None:
    class _D:
        processing_ms = 12.4

    assert _processing.latency_ms(_D(), 800) == 12
    assert _processing.latency_ms(None, 800) == 800
