"""Identity patch for AWS Bedrock Agents (InvokeAgent).

Bedrock Agents are *managed* agents — AWS issues a stable
``agentId`` UUID at creation time. That's a Tier 1 server-issued
identifier, the same quality as OpenAI's ``assistant_id`` or
Gemini's ``cached_content``. We don't need to fingerprint the
prompt; we trust AWS's id and look up the friendly ``agentName``
once per process via ``bedrock-agent.get_agent``.

Patches the ``bedrock-agent-runtime`` client's ``invoke_agent``
method. Each invocation pushes an IdentityRecord keyed on
``(agentId, agentAliasId)`` so two aliases of the same agent
(``DRAFT`` vs ``LIVE``) deduplicate distinctly.

Import-guarded; fail-open.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from egisai._auto_agent import (
    IdentityRecord,
    _ensure_agent_id,
    _hash_bundle,
    identity_scope,
)
from egisai._patches import has_module
from egisai._patches._common import gate_call

LOGGER = logging.getLogger("egisai.patches.bedrock_agent")

FRAMEWORK_SOURCE = "framework:bedrock_agent"

_NAME_CACHE: dict[tuple[str, str], str] = {}
_NAME_LOCK = threading.Lock()


def _resolve_friendly_name(agent_id: str, alias_id: str) -> str:
    """Look up the human ``agentName`` from AWS Bedrock once per process.

    Cached forever — the agent's friendly name only changes when the
    operator renames it on the AWS console, and the next process
    restart picks up the new value. Fail-open: when AWS can't be
    reached (e.g. cross-account IAM denies ``GetAgent``) we fall
    back to ``bedrock:<agentId[:8]>``.
    """
    key = (agent_id, alias_id)
    cached = _NAME_CACHE.get(key)
    if cached:
        return cached
    with _NAME_LOCK:
        cached = _NAME_CACHE.get(key)
        if cached:
            return cached
        try:
            import boto3  # type: ignore[import-not-found]

            # boto3.client is already wrapped if bedrock_runtime
            # applied — that's fine, it just patches converse on
            # service=bedrock-runtime. ``bedrock-agent`` is a
            # different service entirely and goes through
            # unmodified.
            agent_client = boto3.client("bedrock-agent")
            resp = agent_client.get_agent(agentId=agent_id)
            agent_blob = resp.get("agent") or {}
            name = agent_blob.get("agentName") or ""
            if isinstance(name, str) and name.strip():
                _NAME_CACHE[key] = name.strip()
                return name.strip()
        except Exception:  # noqa: BLE001
            LOGGER.debug(
                "bedrock-agent.get_agent failed for agentId=%s — "
                "falling back to id-derived name",
                agent_id[:8],
                exc_info=True,
            )
        fallback = f"bedrock:{agent_id[:8]}"
        _NAME_CACHE[key] = fallback
        return fallback


def _wrap_invoke_agent(orig: Any) -> Any:
    def wrapped(**kwargs: Any) -> Any:
        agent_id = str(kwargs.get("agentId") or "")
        alias_id = str(kwargs.get("agentAliasId") or "")
        if not agent_id:
            return orig(**kwargs)
        display = _resolve_friendly_name(agent_id, alias_id)
        identity_key = f"{FRAMEWORK_SOURCE}:{agent_id}:{alias_id}"
        identity_hash = _hash_bundle(("bedrock_agent", agent_id, alias_id))
        ensured = _ensure_agent_id(
            display_name=display,
            identity_key=identity_key,
            identity_hash=identity_hash,
            source=FRAMEWORK_SOURCE,
        )
        if ensured is None:
            return orig(**kwargs)
        record = IdentityRecord(
            agent_id=ensured,
            display_name=display,
            identity_key=identity_key,
            identity_hash=identity_hash,
            source=FRAMEWORK_SOURCE,  # type: ignore[arg-type]
            push_to_stack=True,
        )
        # Run the gate around the call so input-side policies fire
        # (PII / deny_regex / allow_model / max_prompt_chars /
        # semantic_guard pre-model). ``inputText`` is the user prompt
        # leaving the Python process — same status as a raw model
        # call's prompt, so it gets the same input phase.
        #
        # We don't pass ``extract_output_signals``: ``invoke_agent``
        # returns an ``EventStream`` that must be iterated by the
        # caller exactly once. Wrapping it to extract output signals
        # would either consume the stream (breaking the caller) or
        # require building a replay proxy — out of scope for v1.
        # The model + tool calls inside the managed agent are still
        # audit-logged via the trace events the EventStream yields
        # (caller-driven), and the next user turn's input phase will
        # see whatever text comes back, but ``deny_tool_call`` on
        # the AWS-side execution is not enforceable from outside
        # the agent.
        input_text = kwargs.get("inputText")
        prompt_text = input_text if isinstance(input_text, str) else ""
        with identity_scope(record):
            return gate_call(
                source="bedrock_agent",
                target="bedrock_agent.invoke_agent",
                model=f"bedrock_agent:{agent_id}",
                prompt_text=prompt_text,
                stream=True,
                payload={"input": prompt_text, "agentId": agent_id},
                forward=lambda: orig(**kwargs),
            )

    wrapped.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


_PATCHED_CLIENT_IDS: set[int] = set()


def patch_client_instance(client: Any) -> None:
    if id(client) in _PATCHED_CLIENT_IDS:
        return
    if hasattr(client, "invoke_agent") and not getattr(
        client.invoke_agent, "__egisai_wrapped__", False
    ):
        client.invoke_agent = _wrap_invoke_agent(client.invoke_agent)
    _PATCHED_CLIENT_IDS.add(id(client))


def apply() -> bool:
    if not has_module("boto3"):
        return False
    try:
        import boto3  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return False
    orig_client = boto3.client
    # If bedrock_runtime already wrapped boto3.client, we need to
    # extend that wrapper rather than nest. Detect by sentinel.
    if getattr(orig_client, "__egisai_bedrock_agent_wrapped__", False):
        return True

    def wrapped_client(*args: Any, **kwargs: Any) -> Any:
        client = orig_client(*args, **kwargs)
        service_name = args[0] if args else kwargs.get("service_name")
        if service_name == "bedrock-agent-runtime":
            patch_client_instance(client)
        return client

    wrapped_client.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    wrapped_client.__egisai_bedrock_agent_wrapped__ = True  # type: ignore[attr-defined]
    boto3.client = wrapped_client  # type: ignore[assignment]
    return True
