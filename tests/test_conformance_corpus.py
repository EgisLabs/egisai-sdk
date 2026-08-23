"""The Python engine's conformance run against the shared corpus.

The corpus in ``conformance/`` is language-neutral on purpose: it is
the contract every policy engine we ship must satisfy, and it lives
outside all of them so no single implementation can quietly become the
definition of correct.

This module is Python's side of that contract. It is deliberately thin
— it loads JSON, calls the engine, and compares. Any cleverness here
would be a place for the runner to agree with the engine for the wrong
reason.

If a case here fails, exactly one of two things is true: the engine
regressed, or someone changed intended behavior and did not update the
corpus. Both are worth stopping for. Do not "fix" a red case by editing
the expectation unless you also update its ``description`` to say what
changed and why.

See ``conformance/README.md`` for what is in scope (everything
deterministic) and what is not (the LLM judge's verdicts and the
ML-tier PII detectors, neither of which is reproducible).
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from conftest import CONFORMANCE_ROOT, skip_without_corpus

from egisai.policy.engine import (
    OutputPolicyContext,
    PolicyContext,
    PolicyRule,
    evaluate_output_policies,
    evaluate_policies,
)

CORPUS_ROOT = CONFORMANCE_ROOT
POLICY_CORPUS = CORPUS_ROOT / "policy-corpus"

# Off the monorepo — on the public mirror this package is published
# from — the corpus is not there to run against. See the note on
# ``IN_MONOREPO`` in ``conftest.py`` for why this is a skip rather
# than a silent pass.
pytestmark = skip_without_corpus


# ── Judge stubs ─────────────────────────────────────────────────────
#
# ``semantic_guard`` calls a model, so its *verdicts* cannot be pinned
# by a corpus. What the corpus does pin is everything around it: that
# it is deferred to phase 2, that it is skipped entirely after a
# deterministic block, and that it receives sanitized text. Those are
# properties of the engine, not of the model, and they are exactly the
# ones an independent implementation is likely to get wrong.


class _Match:
    def __init__(self, intent: str = "dangerous") -> None:
        self.intent = intent


class _RecordingBlocker:
    """A judge that answers predictably and remembers what it was asked.

    ``seen`` is what lets the corpus assert the sanitize-before-judge
    ordering: the case declares a substring that must NOT have reached
    the judge, and we check the actual argument rather than trusting
    the engine's own account of itself.
    """

    def __init__(self, *, verdict: str) -> None:
        self._verdict = verdict
        self.seen: list[str] = []

    def check(self, text: str, _config: dict[str, Any]) -> _Match | None:
        self.seen.append(text)
        return _Match() if self._verdict == "always" else None


def _blocker_for(case: dict[str, Any]) -> _RecordingBlocker | None:
    mode = case.get("semantic_blocker")
    if mode is None:
        return None
    return _RecordingBlocker(verdict=mode)


# ── Corpus loading ──────────────────────────────────────────────────


def _load_cases() -> list[tuple[str, dict[str, Any]]]:
    """Flatten every corpus file into ``(test_id, case)`` pairs."""
    found: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(POLICY_CORPUS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for case in payload["cases"]:
            found.append((f"{path.stem}::{case['id']}", case))
    return found


CASES = _load_cases()


def test_the_corpus_is_actually_present() -> None:
    """A missing corpus must fail loudly, not vacuously pass.

    Without this, a bad path or a renamed directory would leave the
    parametrized test with zero cases — the most dangerous kind of
    green, because it looks identical to a passing suite.
    """
    assert len(CASES) >= 30, (
        f"Only {len(CASES)} conformance cases loaded from {POLICY_CORPUS}. "
        "Either the corpus moved or the loader is not finding it."
    )


def _rule(spec: dict[str, Any]) -> PolicyRule:
    return PolicyRule(
        id=spec.get("id"),
        name=spec["name"],
        type=spec["type"],
        tenant=spec.get("tenant"),
        config=dict(spec.get("config") or {}),
        agent_ids=tuple(spec.get("agent_ids") or ()),
        phase=spec.get("phase", "both"),
        applies_to=tuple(spec.get("applies_to") or ()),
        mcp_server_ids=tuple(spec.get("mcp_server_ids") or ()),
    )


def _run(case: dict[str, Any], blocker: _RecordingBlocker | None) -> Any:
    ctx_spec = case["context"]
    if case["side"] == "input":
        text = ctx_spec.get("prompt_text", "")
        context = PolicyContext(
            tenant=ctx_spec.get("tenant", "t"),
            model=ctx_spec.get("model", "gpt-4o"),
            prompt_text=text,
            # Derived, never declared: a corpus that let a case state a
            # length different from its own text could pin behavior
            # that no real caller can reproduce.
            prompt_chars=len(text),
            stream=bool(ctx_spec.get("stream", False)),
            agent_id=ctx_spec.get("agent_id", ""),
        )
        return evaluate_policies(
            [_rule(r) for r in case["rules"]],
            context,
            blocker,
            surfaces=tuple(case.get("surfaces") or ("model",)),
        )

    context = OutputPolicyContext(
        tenant=ctx_spec.get("tenant", "t"),
        model=ctx_spec.get("model", "gpt-4o"),
        text=ctx_spec.get("text", ""),
        tool_names=list(ctx_spec.get("tool_names") or []),
        tool_calls=list(ctx_spec.get("tool_calls") or []),
        mcp_targets=list(ctx_spec.get("mcp_targets") or []),
        stream=bool(ctx_spec.get("stream", False)),
        allow_sanitize=bool(ctx_spec.get("allow_sanitize", False)),
    )
    return evaluate_output_policies(
        [_rule(r) for r in case["rules"]],
        context,
        blocker,
        surfaces=tuple(case.get("surfaces") or ("model", "tool", "mcp")),
    )


@pytest.mark.parametrize(
    "case", [c for _, c in CASES], ids=[i for i, _ in CASES]
)
def test_the_engine_agrees_with_the_corpus(case: dict[str, Any]) -> None:
    blocker = _blocker_for(case)
    decision = _run(case, blocker)
    expect = case["expect"]
    why = case["description"]

    assert decision.verdict == expect["verdict"], (
        f"{case['id']}: expected verdict {expect['verdict']!r}, got "
        f"{decision.verdict!r}.\n{why}"
    )

    if "matched_policy" in expect:
        assert decision.matched_policy == expect["matched_policy"], (
            f"{case['id']}: expected matched_policy "
            f"{expect['matched_policy']!r}, got "
            f"{decision.matched_policy!r}.\n{why}"
        )

    if "reason_code" in expect:
        assert decision.reason_code == expect["reason_code"], (
            f"{case['id']}: expected reason_code {expect['reason_code']!r}, "
            f"got {decision.reason_code!r}.\n{why}"
        )

    if "message" in expect:
        assert decision.message == expect["message"], (
            f"{case['id']}: expected message {expect['message']!r}, got "
            f"{decision.message!r}.\n{why}"
        )

    if "sanitize_types" in expect:
        assert list(decision.sanitize_types) == expect["sanitize_types"], (
            f"{case['id']}: expected sanitize_types "
            f"{expect['sanitize_types']}, got "
            f"{list(decision.sanitize_types)}.\n{why}"
        )

    if "matched_policy_count" in expect:
        assert len(decision.matched_policies) == expect[
            "matched_policy_count"
        ], (
            f"{case['id']}: expected "
            f"{expect['matched_policy_count']} recorded matches, got "
            f"{len(decision.matched_policies)} "
            f"({[m.name for m in decision.matched_policies]}).\n{why}"
        )

    if "matched_verdicts" in expect:
        # What each recorded rule said in isolation, in order. This is
        # how an advisory match (``injection_scan`` on ``flag``) gets
        # pinned: the call's verdict stays ``allow`` while the record
        # says ``flag``, and only checking both catches an engine that
        # silently promoted or dropped it.
        actual_verdicts = [m.verdict for m in decision.matched_policies]
        assert actual_verdicts == expect["matched_verdicts"], (
            f"{case['id']}: expected recorded verdicts "
            f"{expect['matched_verdicts']}, got {actual_verdicts}.\n{why}"
        )

    if "semantic_calls" in expect:
        actual = len(blocker.seen) if blocker else 0
        assert actual == expect["semantic_calls"], (
            f"{case['id']}: expected the judge to be consulted "
            f"{expect['semantic_calls']} time(s), got {actual}.\n{why}"
        )

    if "semantic_saw_text_without" in expect:
        assert blocker is not None and blocker.seen, (
            f"{case['id']}: the judge was never consulted, so the "
            f"sanitize-before-judge ordering was not exercised.\n{why}"
        )
        forbidden = expect["semantic_saw_text_without"]
        for seen in blocker.seen:
            assert forbidden not in seen, (
                f"{case['id']}: the judge received {forbidden!r}. Phase 1 "
                f"must mask before phase 2 sees the text.\n{why}"
            )
