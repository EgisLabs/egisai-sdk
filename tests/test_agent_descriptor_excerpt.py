"""The SDK ships a sanitised system-prompt excerpt for descriptor
generation — and only when the operator hasn't opted out.

Contract pinned here:

1. On first sight of a system-prompt-bearing call, the
   ``/v1/sdk/agents/ensure`` body carries a non-empty
   ``system_prompt_excerpt``.
2. The excerpt is PII-sanitised BEFORE egress — validated PII in the
   system prompt (email, SSN) never appears in the shipped text.
3. ``auto_describe=False`` (and the ``EGISAI_AUTO_DESCRIBE=0`` env
   var) suppress the excerpt entirely: no prompt text leaves the
   process.
"""

from __future__ import annotations

_SYSTEM_PROMPT = (
    "You are a specialist: Python Developer. Escalate to "
    "oncall@acmecorp.io or reference SSN 123-45-6789 when blocked."
)


def _gate_one(payload: dict) -> None:
    from egisai._patches._common import gate_call

    gate_call(
        source="anthropic",
        target="anthropic.messages.create",
        model="claude-3",
        prompt_text="…",
        stream=False,
        payload=payload,
        forward=lambda: "ok",
    )


def test_excerpt_shipped_and_sanitized(fake_backend) -> None:
    """First sight of a system prompt ships a sanitised excerpt."""
    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="orchestrator",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    _gate_one({"system": _SYSTEM_PROMPT})

    bodies = [b for b in fake_backend.ensure_requests if "system_prompt_excerpt" in b]
    assert bodies, "expected the ensure body to carry system_prompt_excerpt"
    excerpt = bodies[-1]["system_prompt_excerpt"]
    assert excerpt, "excerpt must be non-empty"
    # Role text (non-PII) survives so the backend can summarise it.
    assert "Python Developer" in excerpt
    # Validated PII must have been masked locally before egress.
    assert "oncall@acmecorp.io" not in excerpt
    assert "123-45-6789" not in excerpt


def test_excerpt_suppressed_when_auto_describe_off(fake_backend) -> None:
    """``auto_describe=False`` must keep ALL prompt text in-process —
    not even a sanitised excerpt may leave."""
    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="orchestrator",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        auto_describe=False,
    )

    _gate_one({"system": _SYSTEM_PROMPT})

    # The agent still registers (identity is unaffected) …
    assert fake_backend.ensure_requests, "agent should still be registered"
    # … but no ensure body may carry an excerpt.
    assert all(
        "system_prompt_excerpt" not in b for b in fake_backend.ensure_requests
    ), "no excerpt may be shipped when auto_describe is off"


def test_env_var_disables_excerpt(fake_backend, monkeypatch) -> None:
    """``EGISAI_AUTO_DESCRIBE=0`` is an alias for ``auto_describe=False``."""
    import egisai

    monkeypatch.setenv("EGISAI_AUTO_DESCRIBE", "0")
    egisai.init(
        api_key="egis_live_x",
        app="orchestrator",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    _gate_one({"system": _SYSTEM_PROMPT})

    assert all(
        "system_prompt_excerpt" not in b for b in fake_backend.ensure_requests
    )


def test_no_excerpt_without_system_prompt(fake_backend) -> None:
    """A call with no system prompt (e.g. app-fallback identity) ships
    no excerpt — there's nothing to summarise."""
    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="orchestrator",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    _gate_one({"messages": [{"role": "user", "content": "hello"}]})

    assert all(
        "system_prompt_excerpt" not in b for b in fake_backend.ensure_requests
    )
