"""Identity + governance patch for AWS Bedrock's Converse API.

Bedrock is unique among the supported providers because it uses
boto3, not a vendor SDK. The ``bedrock-runtime`` client exposes
``converse`` / ``converse_stream`` — both take a ``system=`` list of
``{"text": "..."}`` blobs plus ``toolConfig`` and ``modelId``.

This patch:
  * Hooks ``boto3.client("bedrock-runtime")``'s ``converse`` and
    ``converse_stream`` methods at instantiation time (boto3 generates
    these methods dynamically from the service model, so we patch the
    client *instance* the first time the user constructs one).
  * Derives an identity hash from
    ``(system_text, sorted_tool_names, modelId)`` — Tier 5 path
    because Converse doesn't carry an agent name.
  * Forwards through ``gate_call`` so policies still fire.

Import-guarded on boto3; fail-open.
"""

from __future__ import annotations

import logging
from typing import Any

from egisai._auto_agent import _derive_identity_from_system, _hash_bundle
from egisai._evaluator import extract_anthropic_prompt
from egisai._patches import has_module
from egisai._patches._common import gate_call

LOGGER = logging.getLogger("egisai.patches.bedrock_runtime")

FRAMEWORK_SOURCE = "framework:bedrock_runtime"


def _system_text(system_arg: Any) -> str:
    if isinstance(system_arg, list):
        chunks = []
        for s in system_arg:
            if isinstance(s, dict):
                t = s.get("text")
                if isinstance(t, str):
                    chunks.append(t)
        return "\n".join(chunks).strip()
    if isinstance(system_arg, str):
        return system_arg.strip()
    return ""


def _tool_names(tool_config: Any) -> list[str]:
    if not isinstance(tool_config, dict):
        return []
    tools = tool_config.get("tools") or []
    names: list[str] = []
    for t in tools:
        if isinstance(t, dict):
            spec = t.get("toolSpec") or {}
            n = spec.get("name")
            if isinstance(n, str):
                names.append(n)
    return sorted(names)


def _extract_usage(response: Any) -> dict[str, Any]:
    if not isinstance(response, dict):
        return {}
    usage = response.get("usage") or {}
    return {
        "tokens_in": usage.get("inputTokens"),
        "tokens_out": usage.get("outputTokens"),
    }


def _make_call(orig: Any, model_id: str, kwargs: dict) -> Any:
    system_text = _system_text(kwargs.get("system"))
    tools = _tool_names(kwargs.get("toolConfig"))
    messages = kwargs.get("messages") or []

    # Pre-resolve a Tier 5-style identity from the composite bundle.
    # We don't directly push it here; the gate's resolver does the
    # push when it walks the tiers. Computing the digest upfront
    # would only help if we wanted to short-circuit Tier 5 — for
    # the Converse API, Tier 5 IS the right path.
    bundle = ("bedrock_runtime", system_text, tuple(tools), model_id)
    digest = _hash_bundle(bundle)

    if system_text:
        _, display_name = _derive_identity_from_system(system_text)
    else:
        display_name = f"bedrock-{model_id[:24]}"

    # Bedrock dictionaries use camelCase. We translate to the
    # gate's expected payload shape so prompt extraction picks up
    # both system + messages.
    prompt_text = extract_anthropic_prompt(messages, system_text)
    payload = {
        "messages": messages,
        "system": system_text,
        "tools": kwargs.get("toolConfig"),
        # Stamp the structural digest on the payload so a future
        # Bedrock-specific resolver tier can short-circuit. Current
        # resolver ignores unknown keys.
        "_bedrock_identity_hash": digest,
        "_bedrock_identity_name": display_name,
    }
    return gate_call(
        source="bedrock_runtime",
        target="bedrock.converse",
        model=model_id,
        prompt_text=prompt_text,
        stream=False,
        payload=payload,
        extract_usage=_extract_usage,
        forward=lambda: orig(**kwargs),
    )


def _wrap_converse(orig: Any) -> Any:
    """Wrap ``client.converse`` for one Bedrock client instance."""

    def wrapped(**kwargs: Any) -> Any:
        model_id = str(kwargs.get("modelId") or "unknown")
        return _make_call(orig, model_id, kwargs)

    setattr(wrapped, "__egisai_wrapped__", True)
    return wrapped


def _wrap_converse_stream(orig: Any) -> Any:
    """Wrap ``client.converse_stream`` for one Bedrock client instance."""

    def wrapped(**kwargs: Any) -> Any:
        model_id = str(kwargs.get("modelId") or "unknown")
        return _make_call(orig, model_id, dict(kwargs, _bedrock_stream=True))

    setattr(wrapped, "__egisai_wrapped__", True)
    return wrapped


_PATCHED_CLIENT_IDS: set[int] = set()


def patch_client_instance(client: Any) -> None:
    """Patch a single boto3 ``bedrock-runtime`` client in-place.

    Called by the ``boto3.client`` factory wrapper below. Idempotent
    per-instance — we track patched object ids so the same client
    isn't double-wrapped on accidental re-patches.
    """
    if id(client) in _PATCHED_CLIENT_IDS:
        return
    if hasattr(client, "converse") and not getattr(
        client.converse, "__egisai_wrapped__", False
    ):
        client.converse = _wrap_converse(client.converse)
    if hasattr(client, "converse_stream") and not getattr(
        client.converse_stream, "__egisai_wrapped__", False
    ):
        client.converse_stream = _wrap_converse_stream(client.converse_stream)
    _PATCHED_CLIENT_IDS.add(id(client))


def apply() -> bool:
    if not has_module("boto3"):
        return False
    try:
        import boto3  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return False
    if getattr(boto3.client, "__egisai_wrapped__", False):
        return True
    _orig_client = boto3.client

    def wrapped_client(*args: Any, **kwargs: Any) -> Any:
        client = _orig_client(*args, **kwargs)
        service_name = args[0] if args else kwargs.get("service_name")
        if service_name in ("bedrock-runtime",):
            patch_client_instance(client)
        return client

    setattr(wrapped_client, "__egisai_wrapped__", True)
    boto3.client = wrapped_client  # type: ignore[assignment]
    return True
