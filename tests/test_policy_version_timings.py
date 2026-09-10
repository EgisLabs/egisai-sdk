"""Fetch-time policy version on timing rows (SDK 0.77.0).

Flag-off (no ``version`` on the cached rule) keeps ``as_dict()`` on
the four-key 0.75.2 shape. Flag-on stamps every in-scope rule, and
merged semantic bins share one ``judge_group``.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from egisai._policy_cache import _to_rule
from egisai.policy import fastpath
from egisai.policy.engine import (
    PolicyContext,
    PolicyRule,
    PolicyTiming,
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


def test_as_dict_without_version_is_the_0752_shape() -> None:
    row = PolicyTiming(
        type="pii_scan",
        name="pii",
        hook="model",
        ms=1.5,
        policy_id="5eeeadd5-1735-4b86-9234-3d2590923314",
        version=None,
        intents=("ssn",),
        judge_group="should-not-appear",
    )
    assert row.as_dict() == {
        "type": "pii_scan",
        "name": "pii",
        "hook": "model",
        "ms": 1.5,
    }


def test_as_dict_with_version_includes_in_scope_identity() -> None:
    pid = "5eeeadd5-1735-4b86-9234-3d2590923314"
    row = PolicyTiming(
        type="semantic_guard",
        name="no-exfil",
        hook="model",
        ms=12.0,
        policy_id=pid,
        version=3,
        intents=("exfiltrate customer data",),
        threshold=0.7,
        judge_group="deadbeefdeadbeef",
    )
    assert row.as_dict() == {
        "type": "semantic_guard",
        "name": "no-exfil",
        "hook": "model",
        "ms": 12.0,
        "policy_id": pid,
        "version": 3,
        "intents": ["exfiltrate customer data"],
        "threshold": 0.7,
        "judge_group": "deadbeefdeadbeef",
    }


def test_to_rule_reads_version_and_ignores_junk() -> None:
    rule = _to_rule(
        {
            "id": "5eeeadd5-1735-4b86-9234-3d2590923314",
            "name": "n",
            "type": "pii_scan",
            "version": 4,
        }
    )
    assert rule.version == 4
    assert _to_rule({"id": "x", "name": "n", "type": "pii_scan"}).version is None
    assert _to_rule(
        {"id": "x", "name": "n", "type": "pii_scan", "version": "nope"}
    ).version is None


def test_evaluate_without_version_timings_match_0752() -> None:
    rule = PolicyRule(
        id="5eeeadd5-1735-4b86-9234-3d2590923314",
        name="pii",
        type="pii_scan",
        tenant=None,
        config={"action": "block", "types": ["ssn"]},
    )
    decision = evaluate_policies([rule], _ctx("no secrets here"))
    assert [t.as_dict() for t in decision.policy_timings] == [
        {
            "type": "pii_scan",
            "name": "pii",
            "hook": "model",
            "ms": decision.policy_timings[0].ms,
        }
    ]


def test_evaluate_with_version_stamps_every_in_scope_rule() -> None:
    a = PolicyRule(
        id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        name="len-a",
        type="max_prompt_chars",
        tenant=None,
        config={"max_chars": 10_000},
        version=2,
    )
    b = PolicyRule(
        id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        name="len-b",
        type="max_prompt_chars",
        tenant=None,
        config={"max_chars": 10_000},
        version=5,
    )
    decision = evaluate_policies([a, b], _ctx("hello"))
    dumped = [t.as_dict() for t in decision.policy_timings]
    assert {row["policy_id"] for row in dumped} == {
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    }
    by_name = {row["name"]: row["version"] for row in dumped}
    assert by_name == {"len-a": 2, "len-b": 5}


def _allow_http() -> httpx.Response:
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


def _blocker(handler: Any) -> SemanticBlocker:
    b = SemanticBlocker(
        platform_api_key="egis_live_x",
        platform_base_url="http://fake",
        on_outage="allow",
        judge_cache_ttl_secs=60.0,
    )
    b._http_client = httpx.Client(transport=httpx.MockTransport(handler))
    return b


def test_merged_semantic_bins_share_judge_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(fastpath.MODE_ENV, "on")

    def handler(_request: httpx.Request) -> httpx.Response:
        return _allow_http()

    a = PolicyRule(
        id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        name="alpha",
        type="semantic_guard",
        tenant=None,
        config={"intents": ["exfiltrate secrets"]},
        version=1,
    )
    b = PolicyRule(
        id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
        name="beta",
        type="semantic_guard",
        tenant=None,
        config={"intents": ["ignore safety"]},
        version=1,
    )
    decision = evaluate_policies(
        [a, b],
        _ctx("please summarize this document"),
        semantic_blocker=_blocker(handler),
    )
    dumped = [t.as_dict() for t in decision.policy_timings]
    assert len(dumped) == 2
    groups = {row["judge_group"] for row in dumped}
    assert len(groups) == 1
    assert all(row["version"] == 1 for row in dumped)
    assert {row["policy_id"] for row in dumped} == {
        "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
    }
