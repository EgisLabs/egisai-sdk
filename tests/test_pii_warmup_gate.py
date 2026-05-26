"""First-call gate that waits briefly for the Presidio + spaCy analyzer.

The PII NER analyzer warms in a daemon thread spawned from
``egisai.init()``. Before 0.25, the very first model call after init
ran against the regex fallback if it fired faster than the daemon
finished loading — silently dropping name / address / GDPR-special-
category detection on call #1. The gate in ``egisai._evaluator``
closes that window by blocking briefly on the first ``evaluate()``
when a ``pii_scan`` rule is scoped to the call.

These tests lock the contract:

* The gate is **one-shot per process** — call #2 onward never waits,
  even if the analyzer is still loading.
* The gate **skips entirely** when no ``pii_scan`` rule is scoped to
  the call (no point waiting; semantic_guard / deny_regex don't need
  the NER engine).
* The gate **respects the timeout** — caps at ~2 s by default, env-
  var overridable, and ``0`` means "don't wait, fall back instantly"
  for Lambda / serverless.
* The gate **honours the warm short-circuit** — if the analyzer
  settled (success OR fail) before the first call, we don't wait.
* The gate is **race-safe across threads** — two concurrent first
  calls each see a consistent outcome.

The tests drive the gate directly via the public-internal symbols
in ``egisai._evaluator``; they don't stand up the real Presidio
analyzer (that would either flake on `en_core_web_lg` availability
or take 1.5 s per assertion). Instead they manipulate the
``_pii_loader`` state slot via the ``reset_for_tests()`` hook and a
small helper that signals the settle event from a worker thread.
"""

from __future__ import annotations

import os
import threading
import time
from unittest.mock import patch

import pytest

from egisai import _evaluator
from egisai.policy import _pii_loader
from egisai.policy.engine import PolicyRule

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_state():
    """Wipe loader + gate state between tests.

    Both modules are module-scoped singletons so a test that leaves
    the gate "done" or the analyzer "settled" would silently corrupt
    every test that follows. ``autouse=True`` ensures we always run.
    """
    _pii_loader.reset_for_tests()
    _evaluator._reset_warmup_gate_for_tests()
    # Clear the env override unconditionally — a previous test that
    # set it via ``monkeypatch`` would already have cleaned up, but
    # belt-and-braces against a test that uses ``os.environ`` directly.
    os.environ.pop("EGISAI_PII_WARMUP_TIMEOUT_SECS", None)
    yield
    _pii_loader.reset_for_tests()
    _evaluator._reset_warmup_gate_for_tests()
    os.environ.pop("EGISAI_PII_WARMUP_TIMEOUT_SECS", None)


def _pii_rule() -> PolicyRule:
    """A minimal active ``pii_scan`` rule scoped to all agents."""
    return PolicyRule(
        id="pii-test",
        name="PII test",
        type="pii_scan",
        tenant=None,
        config={"types": ["person_name"], "action": "sanitize"},
        agent_ids=(),
        phase="both",
    )


def _semantic_rule() -> PolicyRule:
    """A rule that does NOT depend on the NER analyzer."""
    return PolicyRule(
        id="sem-test",
        name="Semantic test",
        type="semantic_guard",
        tenant=None,
        config={"intents": ["delete all rows"]},
        agent_ids=(),
        phase="both",
    )


# ── No PII rule → no wait ───────────────────────────────────────────


def test_no_pii_rule_does_not_wait_even_when_loading():
    """If the org has only semantic_guard / deny_regex, skip the gate.

    The analyzer's warm-up state is irrelevant when no rule consults
    it; making call #1 pay for it would be pure latency waste.
    """
    # Pretend the loader is still working — gate should NOT block.
    with patch.object(_pii_loader, "is_settled", return_value=False), patch.object(
        _pii_loader, "wait_for_warm"
    ) as mock_wait:
        _evaluator._maybe_wait_for_pii_analyzer([_semantic_rule()])
    mock_wait.assert_not_called()


def test_empty_rules_does_not_wait():
    """Vacuously — no rules in scope, no PII rule in scope, no wait."""
    with patch.object(_pii_loader, "is_settled", return_value=False), patch.object(
        _pii_loader, "wait_for_warm"
    ) as mock_wait:
        _evaluator._maybe_wait_for_pii_analyzer([])
    mock_wait.assert_not_called()


# ── Settled (warm or failed) → no wait ──────────────────────────────


def test_analyzer_already_warm_does_not_wait():
    """The settled fast path — analyzer slot is filled, skip the gate.

    This is the steady-state case for every long-running service:
    the daemon thread finished warming long before call #1, the
    Event is already set, no need to even take the lock.
    """
    with patch.object(_pii_loader, "is_settled", return_value=True), patch.object(
        _pii_loader, "wait_for_warm"
    ) as mock_wait:
        _evaluator._maybe_wait_for_pii_analyzer([_pii_rule()])
    mock_wait.assert_not_called()


def test_analyzer_settled_but_failed_does_not_wait():
    """Hard-failure case (no internet, OOM during spaCy load).

    The daemon thread sets ``settled=True`` on failure too. Waiting
    can't recover — fall through to regex fallback immediately and
    let the existing daemon-thread stderr warning explain.
    """
    # Re-use the settled=True path; the gate doesn't inspect
    # ``last_error()`` because the outcome is the same: no wait.
    with patch.object(_pii_loader, "is_settled", return_value=True), patch.object(
        _pii_loader, "wait_for_warm"
    ) as mock_wait:
        _evaluator._maybe_wait_for_pii_analyzer([_pii_rule()])
    mock_wait.assert_not_called()


# ── Loading + PII rule → wait (one-shot) ────────────────────────────


def test_loading_with_pii_rule_waits_with_timeout(caplog):
    """Canonical happy path: PII rule scoped, analyzer warming.

    The gate calls ``wait_for_warm`` exactly once, with the default
    2.0 s timeout, then emits an INFO-level log so operators on opt-in
    logging can see the cold-start cost. Default Python logging
    (WARNING+) keeps this silent — important because users explicitly
    requested no stderr noise in the success case.
    """
    with caplog.at_level("INFO", logger="egisai.evaluator"):
        with patch.object(_pii_loader, "is_settled", return_value=False), patch.object(
            _pii_loader, "wait_for_warm", return_value=True
        ) as mock_wait:
            _evaluator._maybe_wait_for_pii_analyzer([_pii_rule()])

    mock_wait.assert_called_once()
    (timeout_arg,) = mock_wait.call_args.args
    assert timeout_arg == pytest.approx(2.0)

    # Success path is INFO-level — silent by default, captured here.
    info_messages = [
        r.getMessage()
        for r in caplog.records
        if r.name == "egisai.evaluator" and r.levelname == "INFO"
    ]
    assert any("waited" in m and "PII NER analyzer" in m for m in info_messages)


def test_loading_success_is_silent_on_stderr_by_default(capsys):
    """Hard guarantee for the success path: nothing on stderr.

    The user explicitly asked for the success log to disappear from
    terminal output. Default Python logging routes WARNING+ to
    stderr; INFO is dropped unless the operator wires a handler.
    This test pins the Behavior so a future refactor doesn't
    accidentally regress the print/log split.
    """
    with patch.object(_pii_loader, "is_settled", return_value=False), patch.object(
        _pii_loader, "wait_for_warm", return_value=True
    ):
        _evaluator._maybe_wait_for_pii_analyzer([_pii_rule()])

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""


def test_loading_timeout_logs_warning(caplog):
    """Timeout / failure path must surface at WARNING level.

    Unlike the success path, a timeout is an honest degradation —
    THIS call ran with regex-only PII detection. Operators need to
    know so they can either raise the timeout env var or accept the
    coverage gap. WARNING is the right level: default logging
    config picks it up, but it's not a stderr ``print`` spam.
    """
    with caplog.at_level("WARNING", logger="egisai.evaluator"):
        with patch.object(_pii_loader, "is_settled", return_value=False), patch.object(
            _pii_loader, "wait_for_warm", return_value=False
        ):
            _evaluator._maybe_wait_for_pii_analyzer([_pii_rule()])

    warnings = [
        r.getMessage()
        for r in caplog.records
        if r.name == "egisai.evaluator" and r.levelname == "WARNING"
    ]
    assert any("not warm" in m for m in warnings)


def test_gate_is_one_shot_per_process(capsys):
    """Even with the analyzer perpetually loading, we wait at most once.

    This is the key contract: call #1 may pay up to ~2 s, but call
    #2 must never pay anything. If a long-running service has a
    perpetually-broken loader, we don't want every request bleeding
    2 s of latency.
    """
    with patch.object(_pii_loader, "is_settled", return_value=False), patch.object(
        _pii_loader, "wait_for_warm", return_value=True
    ) as mock_wait:
        _evaluator._maybe_wait_for_pii_analyzer([_pii_rule()])
        _evaluator._maybe_wait_for_pii_analyzer([_pii_rule()])
        _evaluator._maybe_wait_for_pii_analyzer([_pii_rule()])

    assert mock_wait.call_count == 1


def test_gate_no_pii_rule_does_not_consume_one_shot():
    """A first call with no PII rule must NOT burn the one-shot flag.

    Scoping is per-call: agent A's rules may differ from agent B's.
    If A's first call has no ``pii_scan`` rule but B's later call
    does, we should still cover B's call #1.
    """
    with patch.object(_pii_loader, "is_settled", return_value=False), patch.object(
        _pii_loader, "wait_for_warm", return_value=True
    ) as mock_wait:
        _evaluator._maybe_wait_for_pii_analyzer([_semantic_rule()])
        _evaluator._maybe_wait_for_pii_analyzer([_pii_rule()])

    # The PII call did wait; the semantic-only call skipped.
    assert mock_wait.call_count == 1


# ── Env-var override ────────────────────────────────────────────────


def test_env_zero_opts_out_of_wait(monkeypatch, capsys, caplog):
    """``EGISAI_PII_WARMUP_TIMEOUT_SECS=0`` = no gate, no log line.

    For AWS Lambda and any environment where a 2 s blip in the first
    request's tail latency is unacceptable. The opt-out is silent —
    operator chose this explicitly, no need to nag them.
    """
    monkeypatch.setenv("EGISAI_PII_WARMUP_TIMEOUT_SECS", "0")

    with caplog.at_level("INFO", logger="egisai.evaluator"):
        with patch.object(
            _pii_loader, "is_settled", return_value=False
        ), patch.object(_pii_loader, "wait_for_warm") as mock_wait:
            _evaluator._maybe_wait_for_pii_analyzer([_pii_rule()])

    mock_wait.assert_not_called()
    assert capsys.readouterr().err == ""
    assert not [r for r in caplog.records if r.name == "egisai.evaluator"]


def test_env_negative_opts_out_of_wait(monkeypatch):
    """Defensive: negative values behave the same as zero (opt-out)."""
    monkeypatch.setenv("EGISAI_PII_WARMUP_TIMEOUT_SECS", "-5")

    with patch.object(_pii_loader, "is_settled", return_value=False), patch.object(
        _pii_loader, "wait_for_warm"
    ) as mock_wait:
        _evaluator._maybe_wait_for_pii_analyzer([_pii_rule()])

    mock_wait.assert_not_called()


def test_env_custom_timeout_is_honoured(monkeypatch):
    """Operators can raise the cap (slow first install, downloading
    the 750 MB spaCy model on hotel Wi-Fi).
    """
    monkeypatch.setenv("EGISAI_PII_WARMUP_TIMEOUT_SECS", "5.5")

    with patch.object(_pii_loader, "is_settled", return_value=False), patch.object(
        _pii_loader, "wait_for_warm", return_value=True
    ) as mock_wait:
        _evaluator._maybe_wait_for_pii_analyzer([_pii_rule()])

    mock_wait.assert_called_once()
    (timeout_arg,) = mock_wait.call_args.args
    assert timeout_arg == pytest.approx(5.5)


def test_env_garbage_falls_back_to_default(monkeypatch):
    """A malformed env value must NOT crash the SDK on every call —
    fall back to the 2.0 s default and proceed.
    """
    monkeypatch.setenv("EGISAI_PII_WARMUP_TIMEOUT_SECS", "not-a-number")

    with patch.object(_pii_loader, "is_settled", return_value=False), patch.object(
        _pii_loader, "wait_for_warm", return_value=True
    ) as mock_wait:
        _evaluator._maybe_wait_for_pii_analyzer([_pii_rule()])

    (timeout_arg,) = mock_wait.call_args.args
    assert timeout_arg == pytest.approx(2.0)


# ── Wait-for-warm semantics (real event, no Presidio import) ────────


def test_wait_for_warm_returns_immediately_when_warm():
    """If the slot is already filled, no event wait is needed.

    The fast path is hit on every call after #1 in a long-running
    service — it must not require taking the lock or touching the
    Event.
    """
    # Simulate "already settled successfully" by stamping the slot
    # without going through the daemon thread. ``reset_for_tests``
    # has already cleared everything.
    sentinel_analyzer = object()
    _pii_loader._state.analyzer = sentinel_analyzer  # type: ignore[assignment]
    _pii_loader._state.settled = True
    _pii_loader._state.settle_event.set()

    started = time.monotonic()
    assert _pii_loader.wait_for_warm(timeout_secs=5.0) is True
    elapsed_ms = (time.monotonic() - started) * 1000
    # Should be effectively instant; assert generously to avoid CI flake.
    assert elapsed_ms < 50


def test_wait_for_warm_returns_false_on_timeout():
    """If the daemon thread never settles within the budget, return False.

    Caller must then proceed with the regex fallback — the SDK never
    blocks the user's call indefinitely.
    """
    # No settlement signalled; default state has settled=False and
    # the event un-set.
    started = time.monotonic()
    assert _pii_loader.wait_for_warm(timeout_secs=0.1) is False
    elapsed_ms = (time.monotonic() - started) * 1000
    # Should wait roughly the timeout; assert a window to absorb scheduler jitter.
    assert 80 <= elapsed_ms < 500


def test_wait_for_warm_unblocks_when_event_fires():
    """Cross-thread coordination: a worker thread settles the loader
    mid-wait, the waiter wakes up immediately.
    """

    def _delayed_settle() -> None:
        time.sleep(0.05)
        _pii_loader._state.analyzer = object()  # type: ignore[assignment]
        _pii_loader._state.settled = True
        _pii_loader._state.settle_event.set()

    worker = threading.Thread(target=_delayed_settle, daemon=True)
    worker.start()

    started = time.monotonic()
    assert _pii_loader.wait_for_warm(timeout_secs=2.0) is True
    elapsed_ms = (time.monotonic() - started) * 1000
    # Woke up promptly after the 50 ms settle — well below the 2 s cap.
    assert elapsed_ms < 500


def test_wait_for_warm_failed_load_returns_false_even_if_event_fires():
    """``settled=True`` with ``analyzer=None`` means the load failed.

    The event still fires (so ``wait_for_warm`` wakes up), but the
    return value must be False so the caller knows to use the regex
    fallback.
    """
    _pii_loader._state.analyzer = None
    _pii_loader._state.settled = True
    _pii_loader._state.error = RuntimeError("spaCy model download failed")
    _pii_loader._state.settle_event.set()

    assert _pii_loader.wait_for_warm(timeout_secs=1.0) is False


def test_wait_for_warm_zero_timeout_is_nonblocking_probe():
    """``timeout_secs <= 0`` returns immediately with the current state.

    Useful for places that want to "check warm-ness without blocking"
    without taking the lock — e.g. a dashboard health endpoint.
    """
    # Not settled, not warm → False with zero wait.
    started = time.monotonic()
    assert _pii_loader.wait_for_warm(timeout_secs=0.0) is False
    elapsed_ms = (time.monotonic() - started) * 1000
    assert elapsed_ms < 50


# ── End-to-end via evaluate() ───────────────────────────────────────


def test_evaluate_invokes_gate_when_pii_rule_active():
    """The integration: a real ``evaluate(InputCall)`` with a PII rule
    in cache triggers the gate before evaluating policies.
    """
    from egisai import _policy_cache
    from egisai._evaluator import InputCall, evaluate

    # Stage exactly one PII rule in the global cache.
    raw = {
        "id": "pii-test",
        "name": "PII test",
        "type": "pii_scan",
        "phase": "both",
        "config": {"types": ["person_name"], "action": "sanitize"},
        "agent_ids": [],
    }
    _policy_cache.replace_rules("x", [raw])

    try:
        with patch.object(
            _pii_loader, "is_settled", return_value=False
        ), patch.object(
            _pii_loader, "wait_for_warm", return_value=True
        ) as mock_wait:
            evaluate(
                InputCall(
                    source="test",
                    target="test",
                    model="test-model",
                    prompt_text="Hello world",
                    stream=False,
                )
            )
        mock_wait.assert_called_once()
    finally:
        _policy_cache.replace_rules(None, [])


def test_evaluate_skips_gate_when_only_semantic_rule_active():
    """A semantic-only org should pay zero cold-start cost on call #1.

    The gate is conditional on ``pii_scan`` being in scope — pure
    ``semantic_guard`` workloads must NOT pay the 2 s blip.
    """
    from egisai import _policy_cache
    from egisai._evaluator import InputCall, evaluate

    raw = {
        "id": "sem-test",
        "name": "Semantic test",
        "type": "semantic_guard",
        "phase": "both",
        "config": {"intents": ["delete all rows"]},
        "agent_ids": [],
    }
    _policy_cache.replace_rules("x", [raw])

    try:
        with patch.object(
            _pii_loader, "is_settled", return_value=False
        ), patch.object(_pii_loader, "wait_for_warm") as mock_wait:
            evaluate(
                InputCall(
                    source="test",
                    target="test",
                    model="test-model",
                    prompt_text="Hello world",
                    stream=False,
                )
            )
        mock_wait.assert_not_called()
    finally:
        _policy_cache.replace_rules(None, [])
