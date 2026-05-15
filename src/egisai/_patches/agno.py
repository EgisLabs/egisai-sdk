"""Identity patch for Agno (formerly Phidata).

Targets ``agno.agent.Agent.run`` / ``arun``. Agno agents have an
explicit ``name`` and a ``description`` — Tier 2A.

This module also installs a tiny stub-compat shim around
``agno.models.openai.chat.OpenAIChat._parse_provider_response``
so Agno survives the response object we return when
``on_block="stub"`` fires. See ``_wrap_openai_chat_parser`` for
why this is Agno-specific.
"""

from __future__ import annotations

from typing import Any

from egisai._auto_agent import IdentityRecord
from egisai._patches import has_module
from egisai._patches._framework import make_identity, patch_method

FRAMEWORK_SOURCE = "framework:agno"


def _derive(self_or_agent: Any, *args: Any, **kwargs: Any) -> IdentityRecord | None:
    agent = self_or_agent
    name = str(getattr(agent, "name", "") or "Agno Agent")
    description = str(getattr(agent, "description", "") or "")
    instructions = getattr(agent, "instructions", None) or ""
    if isinstance(instructions, list):
        instructions = "\n".join(str(i) for i in instructions)
    return make_identity(
        source=FRAMEWORK_SOURCE,
        display_name=name,
        bundle=("agno", name, description, str(instructions)),
    )


# --- Stub-compat shim -------------------------------------------------
#
# When ``on_block="stub"`` fires, ``_patches/openai.py`` returns a
# ``SimpleNamespace`` shaped after OpenAI's ``ChatCompletion``. The
# stub populates every field that frameworks read *generically*
# (``role``, ``content``, ``tool_calls``, ``usage.*_tokens_details``,
# …) and Pydantic-validates cleanly through every framework boundary
# we've audited — except Agno's ``OpenAIChat._parse_provider_response``,
# which reads two fields **unguarded**:
#
#   line 844:  response_audio = response_message.audio
#   line 884:  if response.model_extra: ...
#
# Both raise ``AttributeError`` on a ``SimpleNamespace`` that
# doesn't predeclare them. We don't want to bloat the shared
# OpenAI stub for every framework with fields only Agno reads, so
# the fix lives here, gated on the ``egis`` sentinel we stamp on
# every block-stub. Real ``ChatCompletion`` objects from a live
# OpenAI call carry no such marker and pass through untouched.
#
# The shim is idempotent — re-running ``apply()`` (which the
# init-time refresher does) never double-wraps.

_AGNO_STUB_DEFAULTS_MSG = (
    # Fields ``ChatCompletionMessage`` declares as Optional on
    # modern openai-python pins that Agno reads without a
    # ``hasattr`` guard. Pre-setting them to ``None`` is the same
    # value the upstream Pydantic model defaults to for a
    # no-audio / no-refusal / no-reasoning response.
    "audio",
    "function_call",
    "refusal",
    "reasoning",
    "reasoning_content",
    "annotations",
)


def _normalize_egis_block_stub(response: Any) -> None:
    """Inject the unguarded fields Agno reads on our block-stub.

    No-op on every response object that doesn't carry the
    ``egis`` marker our ``_stub_chat_completion`` stamps. Safe to
    call from a wrapped ``_parse_provider_response`` for *every*
    invocation (real and stubbed) — the marker check makes it a
    pass-through for real responses.
    """
    if not getattr(response, "egis", None):
        return  # real ChatCompletion — let Agno's reads succeed natively
    # ``if response.model_extra:`` raises AttributeError on a
    # SimpleNamespace that doesn't predeclare it. Pre-set to None.
    if not hasattr(response, "model_extra"):
        try:
            response.model_extra = None
        except (AttributeError, TypeError):
            pass
    for choice in getattr(response, "choices", []) or []:
        msg = getattr(choice, "message", None)
        if msg is None:
            continue
        for attr in _AGNO_STUB_DEFAULTS_MSG:
            if not hasattr(msg, attr):
                try:
                    setattr(msg, attr, None)
                except (AttributeError, TypeError):
                    pass


def _wrap_openai_chat_parser() -> bool:
    """Wrap ``OpenAIChat._parse_provider_response`` so our stub
    survives Agno's unguarded attribute reads.

    Returns ``True`` if the wrapping was applied this call,
    ``False`` if Agno's openai chat module isn't importable or
    the method was already wrapped (idempotent).
    """
    try:
        from agno.models.openai.chat import (  # type: ignore[import-not-found]
            OpenAIChat,
        )
    except Exception:
        return False
    target = getattr(OpenAIChat, "_parse_provider_response", None)
    if target is None or getattr(target, "__egis_wrapped__", False):
        return False

    def wrapped(self: Any, response: Any, response_format: Any = None) -> Any:
        _normalize_egis_block_stub(response)
        return target(self, response, response_format=response_format)

    wrapped.__egis_wrapped__ = True  # type: ignore[attr-defined]
    OpenAIChat._parse_provider_response = wrapped  # type: ignore[assignment]
    return True


def apply() -> bool:
    if not has_module("agno"):
        return False
    any_patched = False
    # ``run`` and ``arun`` are *polymorphic dispatchers* — plain ``def``
    # functions whose return type depends on the ``stream=`` kwarg:
    #   - ``run(stream=False)``  → RunOutput (plain value)
    #   - ``run(stream=True)``   → Iterator[RunOutputEvent]
    #   - ``arun(stream=False)`` → coroutine resolving to RunOutput
    #   - ``arun(stream=True)``  → AsyncIterator[RunOutputEvent]
    # The polymorphic wrapper inspects the runtime return and applies
    # the right scope-extension. Pre-0.17.5 we wrapped ``arun`` as
    # ``async``, which crashed ``stream=True`` users with
    # ``TypeError: object async_generator can't be used in 'await'``.
    if patch_method(
        "agno.agent", "Agent", "run", derive=_derive, kind="polymorphic"
    ):
        any_patched = True
    if patch_method(
        "agno.agent", "Agent", "arun", derive=_derive, kind="polymorphic"
    ):
        any_patched = True
    if patch_method(
        "agno.agent", "Agent", "print_response", derive=_derive, kind="sync"
    ):
        any_patched = True
    # Stub-compat wrapper for OpenAIChat — independent of the
    # identity patches above, so a failure here doesn't suppress
    # identity attribution.
    if _wrap_openai_chat_parser():
        any_patched = True
    return any_patched
