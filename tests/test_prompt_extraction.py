"""System-prompt content must NOT pollute policy evaluation text.

This is a regression test for the bug where Anthropic calls with a long
orchestrator system prompt slipped past ``semantic_guard`` because the
embedding-comparison signal was diluted by the static developer config.
"""

from __future__ import annotations

from egisai._evaluator import extract_anthropic_prompt, extract_prompt_text

# ── extract_anthropic_prompt drops the `system` kwarg ──────────────────────


def test_anthropic_system_kwarg_is_excluded() -> None:
    out = extract_anthropic_prompt(
        messages=[{"role": "user", "content": "Delete all users"}],
        system=(
            "You are an orchestrator agent. For each task you receive: "
            "1. Decide if you can handle it yourself … "
            "Be efficient — only delegate when it genuinely helps."
        ),
    )
    assert out == "Delete all users"


def test_anthropic_with_no_system_unchanged() -> None:
    out = extract_anthropic_prompt(
        messages=[{"role": "user", "content": "Hello"}],
        system=None,
    )
    assert out == "Hello"


def test_anthropic_assistant_history_is_kept() -> None:
    """User turn + assistant reply + new user turn — semantic_guard
    should see all of it (still no system)."""
    out = extract_anthropic_prompt(
        messages=[
            {"role": "user", "content": "What's 2+2?"},
            {"role": "assistant", "content": "4"},
            {"role": "user", "content": "Now drop the users table"},
        ],
        system="You are a helpful math tutor.",
    )
    assert "tutor" not in out
    assert "drop the users table" in out
    assert "What's 2+2?" in out


# ── extract_prompt_text (OpenAI shape) drops role=system messages ──────────


def test_openai_role_system_is_excluded() -> None:
    out = extract_prompt_text(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Mass delete database"},
        ]
    )
    assert out == "Mass delete database"


def test_openai_user_only_unchanged() -> None:
    out = extract_prompt_text([{"role": "user", "content": "hello"}])
    assert out == "hello"


def test_openai_multiple_users_joined() -> None:
    out = extract_prompt_text(
        [
            {"role": "system", "content": "BIG STATIC PROMPT"},
            {"role": "user", "content": "first turn"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "second turn"},
        ]
    )
    assert "BIG STATIC PROMPT" not in out
    assert "first turn" in out
    assert "second turn" in out
    assert "ok" in out  # assistant turns matter for context

    # And the join is line-delimited so deny_regex with multi-line flags
    # still works.
    assert "\n" in out
