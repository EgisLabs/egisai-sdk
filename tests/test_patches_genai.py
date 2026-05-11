"""``google.genai`` patcher.

Mirrors ``test_patches.py`` for OpenAI: plants a fake ``google.genai``
package in ``sys.modules`` so the patcher can reach the surface it
expects, then drives it through allow / block / stub paths.

The ``google.generativeai`` patcher lives in ``egisai._patches.google``
and is unaffected by these tests.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


def _install_fake_genai() -> tuple[type, type]:
    """Plant a minimal fake ``google.genai`` in ``sys.modules``.

    Returns the ``Models`` and ``AsyncModels`` classes so tests can
    instantiate them and call ``generate_content`` directly.
    """
    google_pkg = types.ModuleType("google")
    google_pkg.__path__ = []  # type: ignore[attr-defined]
    genai_pkg = types.ModuleType("google.genai")
    genai_pkg.__path__ = []  # type: ignore[attr-defined]
    models_mod = types.ModuleType("google.genai.models")

    class Models:
        def generate_content(self, *, model: str, contents: Any, config: Any = None):
            return types.SimpleNamespace(
                text="ok",
                candidates=[
                    types.SimpleNamespace(
                        content=types.SimpleNamespace(
                            parts=[types.SimpleNamespace(text="ok")],
                            role="model",
                        ),
                        finish_reason="STOP",
                        index=0,
                    )
                ],
                usage_metadata=types.SimpleNamespace(
                    prompt_token_count=1,
                    candidates_token_count=1,
                    total_token_count=2,
                ),
                _model=model,
                _contents=contents,
                _config=config,
            )

        def generate_content_stream(
            self, *, model: str, contents: Any, config: Any = None
        ):
            return iter(
                [
                    types.SimpleNamespace(
                        candidates=[
                            types.SimpleNamespace(
                                content=types.SimpleNamespace(
                                    parts=[types.SimpleNamespace(text="ok")],
                                    role="model",
                                )
                            )
                        ]
                    )
                ]
            )

    class AsyncModels:
        async def generate_content(
            self, *, model: str, contents: Any, config: Any = None
        ):
            return types.SimpleNamespace(
                text="ok",
                candidates=[
                    types.SimpleNamespace(
                        content=types.SimpleNamespace(
                            parts=[types.SimpleNamespace(text="ok")],
                            role="model",
                        ),
                        finish_reason="STOP",
                        index=0,
                    )
                ],
                usage_metadata=types.SimpleNamespace(
                    prompt_token_count=1,
                    candidates_token_count=1,
                    total_token_count=2,
                ),
                _model=model,
                _contents=contents,
                _config=config,
            )

        async def generate_content_stream(
            self, *, model: str, contents: Any, config: Any = None
        ):
            async def _gen():
                yield types.SimpleNamespace(
                    candidates=[
                        types.SimpleNamespace(
                            content=types.SimpleNamespace(
                                parts=[types.SimpleNamespace(text="ok")],
                                role="model",
                            )
                        )
                    ]
                )

            return _gen()

    models_mod.Models = Models
    models_mod.AsyncModels = AsyncModels

    sys.modules.update(
        {
            "google": google_pkg,
            "google.genai": genai_pkg,
            "google.genai.models": models_mod,
        }
    )
    return Models, AsyncModels


def _pii_rule() -> dict:
    return {
        "id": "pii-1",
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


def test_genai_sync_blocks_ssn_with_raise(fake_backend) -> None:
    Models, _ = _install_fake_genai()
    fake_backend.set_rules([_pii_rule()], etag='"pii"')

    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )

    m = Models()
    with pytest.raises(PermissionError):
        m.generate_content(
            model="gemini-2.0-flash",
            contents="SSN 123-45-6789",
        )


def test_genai_sync_allows_clean(fake_backend) -> None:
    Models, _ = _install_fake_genai()
    fake_backend.set_rules([_pii_rule()], etag='"pii"')

    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    m = Models()
    out = m.generate_content(model="gemini-2.0-flash", contents="hello")
    assert out.text == "ok"
    assert out._model == "gemini-2.0-flash"


def test_genai_sync_stub_mode(fake_backend) -> None:
    Models, _ = _install_fake_genai()
    fake_backend.set_rules([_pii_rule()], etag='"pii"')

    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="stub",
    )

    m = Models()
    out = m.generate_content(
        model="gemini-2.0-flash",
        contents="SSN 123-45-6789",
    )
    assert "[POLICY BLOCK]" in out.text
    assert out.egis["blocked"] is True


@pytest.mark.asyncio
async def test_genai_async_blocks_ssn_with_raise(fake_backend) -> None:
    _, AsyncModels = _install_fake_genai()
    fake_backend.set_rules([_pii_rule()], etag='"pii"')

    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )

    am = AsyncModels()
    with pytest.raises(PermissionError):
        await am.generate_content(
            model="gemini-2.0-flash",
            contents="SSN 123-45-6789",
        )


@pytest.mark.asyncio
async def test_genai_async_allows_clean(fake_backend) -> None:
    _, AsyncModels = _install_fake_genai()
    fake_backend.set_rules([_pii_rule()], etag='"pii"')

    import egisai

    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    am = AsyncModels()
    out = await am.generate_content(
        model="gemini-2.0-flash", contents="hello"
    )
    assert out.text == "ok"


def test_genai_apply_is_idempotent(fake_backend) -> None:
    """``apply()`` must not double-wrap when called twice."""
    Models, AsyncModels = _install_fake_genai()
    fake_backend.set_rules([], etag='"empty"')

    import egisai
    from egisai._patches import genai as patch_genai

    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    # First apply happened during init(); calling again finds the
    # methods already wrapped and returns False.
    assert patch_genai.apply() is False

    sync = Models.generate_content
    asyn = AsyncModels.generate_content
    assert getattr(sync, "__egisai_wrapped__", False) is True
    assert getattr(asyn, "__egisai_wrapped__", False) is True


def test_genai_apply_returns_false_when_not_installed() -> None:
    """``apply()`` is a no-op when ``google.genai`` is absent."""
    # Make sure the module is NOT in sys.modules.
    for mod in [
        "google.genai.models",
        "google.genai",
    ]:
        sys.modules.pop(mod, None)

    from egisai._patches import genai as patch_genai

    assert patch_genai.apply() is False
