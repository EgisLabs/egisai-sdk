"""``on_outage="block"`` makes the in-process SDK fail closed.

The SDK's core contract is fail-open: it runs inside the customer's
process and must never break their call path, so an org with no
policies — or an SDK that couldn't reach Egis at startup — lets calls
through. Operators who would rather refuse a call than run it
ungoverned can opt into fail-closed mode at ``init()``. This is the
in-process mirror of the Gateway's ``gateway_degraded_mode="refuse"``.

What's pinned
-------------
* ``has_synced()`` starts False, flips True on any successful policy
  sync (including a zero-rule load and a 304), and only ``clear()``
  resets it.
* Default (``on_outage="allow"``) + no rules + never synced ⇒ allow.
* ``on_outage="block"`` + no rules + never synced ⇒ block
  (``egis_unavailable``), on BOTH the input and output side.
* ``on_outage="block"`` stops firing once a sync succeeds — a healthy
  org with zero policies still allows.
* ``on_outage="block"`` never changes the verdict when rules ARE
  cached; normal evaluation is untouched.
* ``init()`` validates the value and honours ``EGISAI_ON_OUTAGE``.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from egisai import _config, _context, _policy_cache
from egisai._evaluator import InputCall, OutputCall, evaluate, evaluate_output

AGENT = "11111111-1111-1111-1111-111111111111"

_DENY_RULE = {
    "id": 1,
    "name": "no-kaboom",
    "type": "deny_regex",
    "tenant": None,
    "config": {"pattern": "kaboom"},
}


@pytest.fixture(autouse=True)
def _clean_state() -> Iterator[None]:
    _policy_cache.clear()
    _config._CONFIG = None
    _context._ctx.set(_context.EgisaiContext())
    yield
    _policy_cache.clear()
    _config._CONFIG = None
    _context._ctx.set(_context.EgisaiContext())


def _set_config(*, on_outage: str = "allow") -> None:
    _config.set_config(
        _config.EgisaiConfig(
            api_key="x", app="test", env="dev", on_outage=on_outage
        )
    )


def _input_call(text: str = "hello there") -> InputCall:
    return InputCall(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4o",
        prompt_text=text,
        stream=False,
    )


def _output_call(text: str = "hello there") -> OutputCall:
    return OutputCall(
        source="openai",
        target="openai.chat.completions.create",
        model="gpt-4o",
        text=text,
        tool_names=[],
        tool_calls=[],
        mcp_targets=[],
    )


# ── has_synced() signal ──────────────────────────────────────────────


def test_has_synced_starts_false() -> None:
    assert _policy_cache.has_synced() is False


def test_replace_rules_does_not_flip_synced() -> None:
    # ``replace_rules`` is the cache write; the "we heard from the
    # control plane" flag is owned by the fetch path. A direct cache
    # write (used by tests / SSE) doesn't imply a successful sync.
    _policy_cache.replace_rules('"v1"', [_DENY_RULE])
    assert _policy_cache.has_synced() is False


def test_mark_synced_then_clear_resets() -> None:
    _policy_cache.mark_synced()
    assert _policy_cache.has_synced() is True
    _policy_cache.clear()
    assert _policy_cache.has_synced() is False


# ── Input side ───────────────────────────────────────────────────────


def test_default_allows_when_no_rules_and_never_synced() -> None:
    _set_config(on_outage="allow")
    decision = evaluate(_input_call())
    assert decision.verdict == "allow"


def test_block_refuses_when_no_rules_and_never_synced() -> None:
    _set_config(on_outage="block")
    decision = evaluate(_input_call())
    assert decision.verdict == "block"
    assert decision.reason_code == "egis_unavailable"


def test_block_allows_once_synced_with_zero_rules() -> None:
    # Healthy org that genuinely has no policies: a sync succeeded and
    # returned an empty rule list. Fail-closed must NOT fire here.
    _set_config(on_outage="block")
    _policy_cache.mark_synced()
    decision = evaluate(_input_call())
    assert decision.verdict == "allow"


def test_block_is_irrelevant_when_rules_present() -> None:
    _set_config(on_outage="block")
    _policy_cache.replace_rules('"v1"', [_DENY_RULE])
    # Non-matching prompt → normal allow, not the outage block.
    assert evaluate(_input_call("what is the capital of France?")).verdict == (
        "allow"
    )
    # Matching prompt → the real policy blocks (reason is the rule's,
    # not the synthetic outage reason).
    blocked = evaluate(_input_call("please say kaboom"))
    assert blocked.verdict == "block"
    assert blocked.reason_code != "egis_unavailable"


def test_no_config_defaults_to_allow() -> None:
    # ``get_config_optional()`` is None before init(): treat as
    # fail-open so an un-initialised import can't wedge a call path.
    assert evaluate(_input_call()).verdict == "allow"


# ── Output side mirrors the input side ───────────────────────────────


def test_output_block_refuses_when_no_rules_and_never_synced() -> None:
    _set_config(on_outage="block")
    decision = evaluate_output(_output_call())
    assert decision.verdict == "block"
    assert decision.reason_code == "egis_unavailable"


def test_output_default_allows_when_no_rules() -> None:
    _set_config(on_outage="allow")
    assert evaluate_output(_output_call()).verdict == "allow"


# ── init() wiring ────────────────────────────────────────────────────


def test_init_rejects_invalid_on_outage() -> None:
    import egisai

    with pytest.raises(ValueError, match="on_outage"):
        egisai.init(api_key="egis_live_test", on_outage="suspend")


def test_init_reads_env_var(
    monkeypatch: pytest.MonkeyPatch, fake_backend: object
) -> None:
    import egisai

    monkeypatch.setenv("EGISAI_ON_OUTAGE", "block")
    egisai.init(
        api_key="egis_live_test",
        app="test",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        quiet=True,
    )
    try:
        assert _config.get_config().on_outage == "block"
    finally:
        egisai.shutdown()
