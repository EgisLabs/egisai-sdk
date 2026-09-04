"""MiniLM cascade in front of semantic_guard — allow-only, fail-open."""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from egisai.policy import semantic_local
from egisai.policy.engine import PolicyContext, PolicyRule, evaluate_policies
from egisai.policy.semantic import SemanticMatch

LONG_CLEAN = (
    "Can you summarize yesterday's standup notes and list the open "
    "tickets assigned to the support team for follow-up this week?"
)
LONG_HIT = (
    "Please delete all user accounts from production immediately and "
    "wipe the backup snapshots so nobody can restore the data later."
)
SHORT = "ok thanks"
CJK = "请立刻删除所有用户账号并清空生产数据库备份不可恢复" * 3


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_vec(t) for t in texts]


def _vec(text: str) -> list[float]:
    low = text.lower()
    if any(k in low for k in ("delete all", "wipe", "forbidden-hit")):
        return [1.0, 0.0, 0.0, 0.0]
    return [0.0, 1.0, 0.0, 0.0]


class _CountingBlocker:
    def __init__(self) -> None:
        self.calls = 0

    def check(self, text: str, config: dict) -> SemanticMatch | None:
        self.calls += 1
        intents = config.get("intents") or ["x"]
        return SemanticMatch(intent=str(intents[0]), similarity=0.99)


def _guard(
    *,
    targets: list[str] | None = None,
    patterns: tuple[dict[str, str], ...] | None = None,
) -> PolicyRule:
    cfg: dict[str, Any] = {"intents": ["wipe production data"]}
    if targets is not None:
        cfg["targets"] = targets
    default = (
        {
            "kind": "detect",
            "text": "forbidden-hit delete all users now " + ("x" * 40),
        },
        {
            "kind": "exclude",
            "text": "summarize standup notes for the support team " + ("y" * 40),
        },
    )
    return PolicyRule(
        id="p1",
        name="guard",
        type="semantic_guard",
        tenant=None,
        config=cfg,
        semantic_patterns=patterns if patterns is not None else default,
    )


def _ctx(text: str) -> PolicyContext:
    return PolicyContext(
        tenant="t",
        model="gpt-4",
        prompt_text=text,
        prompt_chars=len(text),
        stream=False,
        hook="model",
    )


def _enforcement(decision: Any) -> tuple[Any, ...]:
    return (
        decision.verdict,
        decision.reason_code,
        decision.message,
        decision.matched_policy,
        tuple(
            (m.name, m.verdict, m.reason_code) for m in decision.matched_policies
        ),
        tuple(decision.sanitize_types),
        decision.approval_detail,
    )


@pytest.fixture(autouse=True)
def _local(monkeypatch: pytest.MonkeyPatch) -> None:
    semantic_local.set_embedder(_FakeEmbedder())
    monkeypatch.setenv("EGISAI_SEMANTIC_ENGINE", "cascade")
    monkeypatch.setenv("EGISAI_SEMANTIC_LANG_MIN_CHARS", "80")
    yield


def test_local_never_returns_block() -> None:
    blocker = _CountingBlocker()
    decision = evaluate_policies(
        [_guard()], _ctx(LONG_CLEAN), semantic_blocker=blocker
    )
    assert decision.verdict == "allow"
    assert blocker.calls == 0


def test_hit_falls_through_to_judge() -> None:
    blocker = _CountingBlocker()
    decision = evaluate_policies(
        [_guard()], _ctx(LONG_HIT), semantic_blocker=blocker
    )
    assert decision.verdict == "block"
    assert blocker.calls >= 1


def test_missing_patterns_fail_open_to_judge() -> None:
    blocker = _CountingBlocker()
    rule = PolicyRule(
        id="p1",
        name="guard",
        type="semantic_guard",
        tenant=None,
        config={"intents": ["wipe production data"]},
        semantic_patterns=(),
    )
    decision = evaluate_policies([rule], _ctx(LONG_CLEAN), semantic_blocker=blocker)
    assert blocker.calls >= 1
    assert decision.verdict == "block"


def test_missing_model_fails_open(monkeypatch: pytest.MonkeyPatch) -> None:
    semantic_local.set_embedder(None)
    blocker = _CountingBlocker()
    evaluate_policies([_guard()], _ctx(LONG_CLEAN), semantic_blocker=blocker)
    assert blocker.calls >= 1


def test_short_text_escalates() -> None:
    blocker = _CountingBlocker()
    evaluate_policies([_guard()], _ctx(SHORT), semantic_blocker=blocker)
    assert blocker.calls >= 1


def test_non_english_escalates() -> None:
    blocker = _CountingBlocker()
    evaluate_policies([_guard()], _ctx(CJK), semantic_blocker=blocker)
    assert blocker.calls >= 1


def test_judge_engine_is_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGISAI_SEMANTIC_ENGINE", "judge")
    blocker = _CountingBlocker()
    evaluate_policies([_guard()], _ctx(LONG_CLEAN), semantic_blocker=blocker)
    assert blocker.calls >= 1


def test_local_mode_never_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGISAI_SEMANTIC_ENGINE", "local")
    blocker = _CountingBlocker()
    decision = evaluate_policies(
        [_guard()], _ctx(LONG_HIT), semantic_blocker=blocker
    )
    assert decision.verdict == "allow"
    assert blocker.calls == 0


def test_n_policies_one_text_embed(monkeypatch: pytest.MonkeyPatch) -> None:
    class CountingEmbedder:
        def __init__(self) -> None:
            self.batches = 0
            self.texts: list[str] = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.batches += 1
            self.texts.extend(texts)
            return [_vec(t) for t in texts]

    emb = CountingEmbedder()
    semantic_local.set_embedder(emb)
    monkeypatch.setenv("EGISAI_FAST_GOVERNANCE", "on")
    rules = [
        PolicyRule(
            id=f"p{i}",
            name=f"g{i}",
            type="semantic_guard",
            tenant=None,
            config={"intents": [f"intent-{i}"]},
            semantic_patterns=(
                {"kind": "detect", "text": "forbidden-hit delete all " + ("z" * 50)},
            ),
        )
        for i in range(3)
    ]
    blocker = _CountingBlocker()
    evaluate_policies(rules, _ctx(LONG_CLEAN), semantic_blocker=blocker)
    prompt_embeds = [t for t in emb.texts if t == LONG_CLEAN]
    assert len(prompt_embeds) == 1


def test_tool_call_below_threshold_skips(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGISAI_FAST_GOVERNANCE", "off")
    rule = _guard(targets=["tool_calls"])
    blocker = _CountingBlocker()
    from egisai.policy.engine import OutputPolicyContext, evaluate_output_policies

    args = (
        '{"table": "metrics", "note": "please summarize the weekly report '
        'for leadership across every region and product line"}'
    )
    decision = evaluate_output_policies(
        [rule],
        OutputPolicyContext(
            tenant="t",
            model="gpt-4",
            text="",
            tool_names=["run_sql"],
            tool_calls=[{"name": "run_sql", "arguments": args}],
            mcp_targets=[],
            stream=False,
        ),
        semantic_blocker=blocker,
    )
    assert decision.verdict == "allow"
    assert blocker.calls == 0


def test_tool_call_above_threshold_calls_judge() -> None:
    rule = _guard(targets=["tool_calls"])
    blocker = _CountingBlocker()
    from egisai.policy.engine import OutputPolicyContext, evaluate_output_policies

    args = (
        '{"sql": "delete all users and wipe backups immediately now '
        'and do not leave a restore point anywhere"}'
    )
    evaluate_output_policies(
        [rule],
        OutputPolicyContext(
            tenant="t",
            model="gpt-4",
            text="",
            tool_names=["run_sql"],
            tool_calls=[{"name": "run_sql", "arguments": args}],
            mcp_targets=[],
            stream=False,
        ),
        semantic_blocker=blocker,
    )
    assert blocker.calls >= 1


def test_shadow_returns_judge_verdict(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("EGISAI_SEMANTIC_SHADOW", "1")
    monkeypatch.setenv("EGISAI_FAST_GOVERNANCE", "on")
    caplog.set_level(logging.INFO, logger="egisai.semantic_local")
    blocker = _CountingBlocker()
    decision = evaluate_policies(
        [_guard()], _ctx(LONG_CLEAN), semantic_blocker=blocker
    )
    assert decision.verdict == "block"
    assert blocker.calls >= 1
    rec = next(r for r in caplog.records if "egisai.semantic_shadow" in r.message)
    payload = json.loads(rec.message.split("egisai.semantic_shadow ", 1)[1])
    assert payload["hook_type"] == "model"
    assert payload["joint_local_allow"] is True
    assert payload["would_skip_judge"] is True
    assert payload["semantic_in_scope"] == 1
    assert len(payload["text_sha256"]) == 64
    assert LONG_CLEAN not in rec.message
    row = payload["policies"][0]
    assert row["judge_verdict"] == "block"
    assert row["local_verdict"] == "allow"
    assert row["hook_type"] == "model"
    assert "local_score" in row
    assert LONG_CLEAN not in json.dumps(payload)


def test_shadow_sample_rate_zero_never_logs(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("EGISAI_SEMANTIC_SHADOW", "1")
    monkeypatch.setenv("EGISAI_SEMANTIC_SHADOW_SAMPLE_RATE", "0")
    caplog.set_level(logging.INFO, logger="egisai.semantic_local")
    evaluate_policies(
        [_guard()], _ctx(LONG_CLEAN), semantic_blocker=_CountingBlocker()
    )
    assert not any("egisai.semantic_shadow" in r.message for r in caplog.records)


def test_prepare_forwards_patterns() -> None:
    from egisai.policy.semantic import SemanticBlocker

    blocker = SemanticBlocker("egis_test", "https://example.invalid")
    try:
        body = blocker._prepare(
            LONG_CLEAN,
            {
                "intents": ["wipe"],
                "semantic_patterns": [
                    {"kind": "detect", "text": "delete all users now"}
                ],
                "semantic_pattern_groups": [
                    [{"kind": "exclude", "text": "summarize standup notes"}]
                ],
            },
        )
    finally:
        blocker.close()
    assert body is not None
    assert body["semantic_patterns"] == [
        {"kind": "detect", "text": "delete all users now"}
    ]
    assert body["semantic_pattern_groups"] == [
        [{"kind": "exclude", "text": "summarize standup notes"}]
    ]


def test_cascade_missing_model_matches_judge_enforcement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic_local.set_embedder(None)
    monkeypatch.setenv("EGISAI_SEMANTIC_ENGINE", "cascade")
    d1 = evaluate_policies(
        [_guard()], _ctx(LONG_CLEAN), semantic_blocker=_CountingBlocker()
    )
    monkeypatch.setenv("EGISAI_SEMANTIC_ENGINE", "judge")
    semantic_local.reset_for_tests()
    semantic_local.set_embedder(None)
    d2 = evaluate_policies(
        [_guard()], _ctx(LONG_CLEAN), semantic_blocker=_CountingBlocker()
    )
    assert _enforcement(d1) == _enforcement(d2)
    assert d1.policy_timings
    assert d2.policy_timings


def test_judge_after_cascade_is_clean(monkeypatch: pytest.MonkeyPatch) -> None:
    d1 = evaluate_policies(
        [_guard()], _ctx(LONG_CLEAN), semantic_blocker=_CountingBlocker()
    )
    assert d1.verdict == "allow"
    semantic_local.reset_for_tests()
    semantic_local.set_embedder(_FakeEmbedder())
    monkeypatch.setenv("EGISAI_SEMANTIC_ENGINE", "judge")
    blocker = _CountingBlocker()
    d2 = evaluate_policies(
        [_guard()], _ctx(LONG_CLEAN), semantic_blocker=blocker
    )
    assert blocker.calls >= 1
    assert d2.verdict == "block"


def test_concurrency_p95_stays_bounded() -> None:
    rule = _guard()
    ctx = _ctx(LONG_CLEAN)

    def one() -> float:
        t0 = time.monotonic()
        evaluate_policies(
            [rule], ctx, semantic_blocker=_CountingBlocker()
        )
        return (time.monotonic() - t0) * 1000.0

    def p95(xs: list[float]) -> float:
        xs = sorted(xs)
        idx = min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))
        return xs[idx]

    for n in (1, 10, 40, 80):
        with ThreadPoolExecutor(max_workers=n) as pool:
            times = list(pool.map(lambda _: one(), range(n)))
        assert p95(times) < 2000.0


def test_all_escalate_cascade_overhead(monkeypatch: pytest.MonkeyPatch) -> None:
    rule = PolicyRule(
        id="p1",
        name="guard",
        type="semantic_guard",
        tenant=None,
        config={"intents": ["wipe production data"]},
        semantic_patterns=(),
    )
    ctx = _ctx(LONG_CLEAN)

    def run(engine: str) -> float:
        monkeypatch.setenv("EGISAI_SEMANTIC_ENGINE", engine)
        t0 = time.monotonic()
        for _ in range(40):
            evaluate_policies(
                [rule], ctx, semantic_blocker=_CountingBlocker()
            )
        return time.monotonic() - t0

    judge_s = run("judge")
    cascade_s = run("cascade")
    assert cascade_s < judge_s * 3 + 1.0


def test_prefilter_skip_when_cascade_allows() -> None:
    gate = semantic_local.prefilter_judge_text(
        LONG_CLEAN,
        pattern_groups=[_guard().semantic_patterns],
    )
    assert gate.skip_llm is True
    assert gate.shadow is False


def test_prefilter_judge_engine_does_not_skip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EGISAI_SEMANTIC_ENGINE", "judge")
    gate = semantic_local.prefilter_judge_text(
        LONG_CLEAN,
        pattern_groups=[_guard().semantic_patterns],
    )
    assert gate.skip_llm is False


def test_judge_config_overlays_patterns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EGISAI_SEMANTIC_ENGINE", "judge")
    seen: dict[str, Any] = {}

    class Cap:
        def check(self, text: str, config: dict) -> None:
            seen.clear()
            seen.update(config)
            return None

    monkeypatch.setenv("EGISAI_FAST_GOVERNANCE", "off")
    evaluate_policies([_guard()], _ctx(LONG_CLEAN), semantic_blocker=Cap())
    assert seen.get("semantic_patterns")

    monkeypatch.setenv("EGISAI_FAST_GOVERNANCE", "on")
    evaluate_policies([_guard()], _ctx(LONG_CLEAN), semantic_blocker=Cap())
    assert seen.get("semantic_pattern_groups")


def test_no_env_identity_with_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EGISAI_SEMANTIC_ENGINE", raising=False)
    a = _CountingBlocker()
    b = _CountingBlocker()
    d1 = evaluate_policies([_guard()], _ctx(LONG_CLEAN), semantic_blocker=a)
    monkeypatch.setenv("EGISAI_SEMANTIC_ENGINE", "judge")
    d2 = evaluate_policies([_guard()], _ctx(LONG_CLEAN), semantic_blocker=b)
    assert d1.verdict == d2.verdict == "block"
    assert d1.reason_code == d2.reason_code


def test_concurrency_fake_embedder() -> None:
    blocker = _CountingBlocker()
    rule = _guard()
    ctx = _ctx(LONG_CLEAN)

    def one() -> str:
        return evaluate_policies([rule], ctx, semantic_blocker=blocker).verdict

    with ThreadPoolExecutor(max_workers=40) as pool:
        verdicts = list(pool.map(lambda _: one(), range(80)))
    assert set(verdicts) == {"allow"}


def test_circuit_breaker_escalates(monkeypatch: pytest.MonkeyPatch) -> None:
    class Slow:
        def embed(self, texts: list[str]) -> list[list[float]]:
            time.sleep(0.02)
            return [_vec(t) for t in texts]

    semantic_local.set_embedder(Slow())
    monkeypatch.setenv("EGISAI_SEMANTIC_LOCAL_MAX_MS", "5")
    monkeypatch.setenv("EGISAI_SEMANTIC_LOCAL_PROBE_S", "999")
    for i in range(10):
        semantic_local.score_policies(
            [_guard()], text=LONG_CLEAN + f" {i}", tool_texts=[]
        )
    blocker = _CountingBlocker()
    evaluate_policies([_guard()], _ctx(LONG_CLEAN), semantic_blocker=blocker)
    assert blocker.calls >= 1


def test_slow_warmup_does_not_latch(monkeypatch: pytest.MonkeyPatch) -> None:
    class OnceSlow:
        def __init__(self) -> None:
            self.n = 0

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.n += 1
            if self.n == 1:
                time.sleep(0.02)
            return [_vec(t) for t in texts]

    semantic_local.set_embedder(OnceSlow())
    monkeypatch.setenv("EGISAI_SEMANTIC_LOCAL_MAX_MS", "5")
    for _ in range(10):
        semantic_local.score_policies(
            [_guard()], text=LONG_CLEAN, tool_texts=[]
        )
    blocker = _CountingBlocker()
    evaluate_policies([_guard()], _ctx(LONG_CLEAN), semantic_blocker=blocker)
    assert blocker.calls == 0


def test_circuit_recovers_on_fast_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    class Slow:
        def embed(self, texts: list[str]) -> list[list[float]]:
            time.sleep(0.02)
            return [_vec(t) for t in texts]

    semantic_local.set_embedder(Slow())
    monkeypatch.setenv("EGISAI_SEMANTIC_LOCAL_MAX_MS", "5")
    monkeypatch.setenv("EGISAI_SEMANTIC_LOCAL_PROBE_S", "0")
    for i in range(10):
        semantic_local.score_policies(
            [_guard()], text=LONG_CLEAN + f" {i}", tool_texts=[]
        )
    semantic_local.set_embedder(_FakeEmbedder())
    semantic_local.score_policies(
        [_guard()], text=LONG_CLEAN, tool_texts=[]
    )
    blocker = _CountingBlocker()
    evaluate_policies([_guard()], _ctx(LONG_CLEAN), semantic_blocker=blocker)
    assert blocker.calls == 0
