"""Patches: gate_call's allow / block paths exercised against fake clients."""

from __future__ import annotations

import sys
import types

import pytest


def _install_fake_openai() -> tuple[type, type]:
    """Plant a minimal fake openai package in sys.modules."""
    fake = types.ModuleType("openai")
    res = types.ModuleType("openai.resources")
    chat = types.ModuleType("openai.resources.chat")
    completions = types.ModuleType("openai.resources.chat.completions")
    responses = types.ModuleType("openai.resources.responses")

    class Completions:
        def create(self, **kwargs):
            return {"id": "real", "kwargs": kwargs}

    class AsyncCompletions:
        async def create(self, **kwargs):
            return {"id": "real-async", "kwargs": kwargs}

    completions.Completions = Completions
    completions.AsyncCompletions = AsyncCompletions

    sys.modules.update(
        {
            "openai": fake,
            "openai.resources": res,
            "openai.resources.chat": chat,
            "openai.resources.chat.completions": completions,
            "openai.resources.responses": responses,
        }
    )
    return Completions, AsyncCompletions


def _pii_rule() -> dict:
    return {
        "id": 1,
        "name": "block-pii",
        "type": "pii_scan",
        "tenant": None,
        "config": {
            # Explicit block — 0.16.0 default flipped to sanitize.
            "action": "block",
            "threshold": 0.4,
            "kinds": ["ssn"],
            "message": "PII",
        },
    }


def test_openai_chat_blocks_ssn_with_raise(fake_backend) -> None:
    Completions, _ = _install_fake_openai()
    fake_backend.set_rules([_pii_rule()], etag='"pii"')

    import egisai

    egisai.init(
        api_key="egis_live_x", app="a", env="t",
        base_url="http://fake", enable_sse=False, on_block="raise",
    )

    c = Completions()
    with pytest.raises(PermissionError):
        c.create(model="gpt-4", messages=[{"role": "user", "content": "SSN 123-45-6789"}])


def test_openai_chat_allows_clean(fake_backend) -> None:
    Completions, _ = _install_fake_openai()
    fake_backend.set_rules([_pii_rule()], etag='"pii"')

    import egisai

    egisai.init(api_key="egis_live_x", app="a", env="t",
                base_url="http://fake", enable_sse=False)

    c = Completions()
    out = c.create(model="gpt-4", messages=[{"role": "user", "content": "hello"}])
    assert out["id"] == "real"


def test_openai_chat_stub_mode(fake_backend) -> None:
    Completions, _ = _install_fake_openai()
    fake_backend.set_rules([_pii_rule()], etag='"pii"')

    import egisai

    egisai.init(api_key="egis_live_x", app="a", env="t",
                base_url="http://fake", enable_sse=False, on_block="stub")

    c = Completions()
    out = c.create(model="gpt-4",
                   messages=[{"role": "user", "content": "SSN 123-45-6789"}])
    # Stub returns a SimpleNamespace with .choices[0].message.content explaining the block
    text = out.choices[0].message.content
    assert "[POLICY BLOCK]" in text or "block-pii" in text or "PII" in text
    assert getattr(out, "egis", {}).get("blocked") is True
