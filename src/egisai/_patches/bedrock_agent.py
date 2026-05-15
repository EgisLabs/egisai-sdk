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

Enforcement limitation — **advisory only for tool calls AND
tool results**. Bedrock Agents execute Action Groups on
AWS-managed infrastructure outside this SDK's process. By the
time the response stream reaches us, AWS has already dispatched
the Action Group lambda AND the Action Group's result has
already been fed back to the model. This patch records what
happened (so the audit log has a row) for both phases, but it
cannot prevent the AWS-side tool execution and it cannot
substitute a sanitized result before the model sees it. Audit
rows for tool blocks stamp ``enforcement_status="advisory"`` to
honestly reflect this.

Compliance implication for SOC 2 / GDPR / HIPAA: a Bedrock
Agent whose Action Group returns PII (e.g. a CRM lookup) WILL
leak that PII to the model. The SDK cannot prevent it. If your
threat model requires pre-execution gating OR tool-result
sanitization, use one of:

* ``claude_agent_sdk`` — the SDK exposes ``PreToolUse`` AND
  ``PostToolUse`` hooks that the egisai patch wires into the
  policy engine for true pre-execution and pre-model-read
  enforcement (see ``claude_agent_sdk.py``).
* The standalone ``bedrock-runtime`` Converse API (the
  ``bedrock_runtime`` patch) with the agentic loop driven in
  Python — every tool dispatch routes through the SDK
  boundary and the next-call input phase scans returned tool
  results before they're sent back to the model.

A future revision may wire AWS's RETURN_CONTROL mode (where
the Action Group is returned to the caller rather than
executed by AWS) to convert this to ``enforced`` — both for
pre-execution gating and tool-result sanitization. Tracked in
the project roadmap.

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
from egisai._patches._framework import _RunScope

LOGGER = logging.getLogger("egisai.patches.bedrock_agent")

FRAMEWORK_SOURCE = "framework:bedrock_agent"
# Token stamped on the Run's ``framework`` field. AWS Bedrock Agents
# are a managed framework (AWS executes the loop server-side), so the
# audit row uses the same token shape as other framework wraps
# (``openai_agents``, ``langgraph``, …) rather than the raw ``legacy``
# fallback. This makes the dashboard's per-framework filter render
# Bedrock-managed runs alongside the rest of the agentic surface and
# unblocks the ``run.identity_source`` / ``run.identity_hash`` columns
# on the audit DB row (both are populated from the Run's
# :class:`IdentityRecord` at ``run.start`` ingest time).
FRAMEWORK_TOKEN = "bedrock_agent"

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
            import boto3  # type: ignore[import-not-found,import-untyped]

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
        # ``payload["input"]`` is the mutable seam ``mutate_prompt_text``
        # writes to on the sanitize verdict. Because ``inputText`` is
        # an immutable string in ``kwargs`` (not a list / dict that
        # would be mutated in place), the patch's forward lambda must
        # re-read the post-sanitization value from ``payload`` and
        # mirror it back into ``kwargs["inputText"]`` before
        # ``boto3.invoke_agent`` ships it to AWS — otherwise the raw
        # prompt would still leave the SDK boundary on a sanitize
        # verdict. Same root cause / fix shape as the
        # ``_patches.genai`` patch for ``contents="..."``.
        payload: dict[str, Any] = {
            "input": prompt_text,
            "agentId": agent_id,
        }

        def _forward() -> Any:
            sanitized = payload.get("input")
            if isinstance(sanitized, str) and sanitized != prompt_text:
                kwargs["inputText"] = sanitized
            return orig(**kwargs)

        # Open a Run BEFORE calling the gate so the framework patch
        # contract holds: every model call recorded under this
        # invocation lands as a step under one ``bedrock_agent`` Run
        # rather than as a legacy single-row event. The Run also
        # carries the identity record, which the backend's
        # ``_upsert_run_from_start`` ingester writes to
        # ``runs.identity_source`` + ``runs.identity_hash`` so the
        # Agent Identity card on the dashboard surfaces the same
        # provenance shown for every other framework wrap. Re-entry
        # guard on ``_RunScope`` keeps a nested ``invoke_agent`` call
        # (rare but possible from in-process Action Group
        # implementations using boto3 from the same process) riding on
        # the existing Run instead of spawning a duplicate.
        with _RunScope(FRAMEWORK_TOKEN, record), identity_scope(record):
            return gate_call(
                source="bedrock_agent",
                target="bedrock_agent.invoke_agent",
                model=f"bedrock_agent:{agent_id}",
                prompt_text=prompt_text,
                stream=True,
                payload=payload,
                forward=_forward,
            )

    wrapped.__egisai_wrapped__ = True  # type: ignore[attr-defined]
    return wrapped


def patch_client_instance(client: Any) -> None:
    """Patch a single boto3 ``bedrock-agent-runtime`` client in-place.

    Idempotency is keyed off the ``__egisai_wrapped__`` sentinel set
    on the wrapped ``invoke_agent`` function. We deliberately do
    NOT track ``id(client)`` in a module-level set — CPython
    recycles object ids the moment a previous client is GC'd, and
    a fresh client landing on a recycled id would be skipped by
    such a tracker, silently bypassing the gate. The sentinel-attr
    path is correct AND immune to id reuse.
    """
    if hasattr(client, "invoke_agent") and not getattr(
        client.invoke_agent, "__egisai_wrapped__", False
    ):
        client.invoke_agent = _wrap_invoke_agent(client.invoke_agent)


def apply() -> bool:
    if not has_module("boto3"):
        return False
    try:
        import boto3  # type: ignore[import-not-found,import-untyped]
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
