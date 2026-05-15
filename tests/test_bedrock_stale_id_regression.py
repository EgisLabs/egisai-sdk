"""Regression: ``bedrock_runtime`` / ``bedrock_agent`` must not key
idempotency off ``id(client)``.

Background
----------
Releases ≤ 0.25.14 of the SDK each kept a module-level
``_PATCHED_CLIENT_IDS: set[int]`` inside
``egisai._patches.bedrock_runtime`` and
``egisai._patches.bedrock_agent``. The intent was a fast-path
"already wrapped this client, skip" check inside
``patch_client_instance(client)``.

The bug: CPython recycles object addresses aggressively. The moment
the previous ``boto3.client("bedrock-runtime")`` instance is
garbage-collected, the next fresh client can land at the same
memory address. ``id()`` returns that address, so the set
``contains`` check would falsely report "already patched" and
``patch_client_instance`` would short-circuit before wrapping
``client.converse`` / ``client.invoke_agent``. The unwrapped method
then resolved to the class-level ``staticmethod`` (or, for real
boto3, to the underlying botocore-generated method), ``gate_call``
was never invoked, ``PermissionError`` was never raised for
``deny_tool_call`` / ``pii_block`` policies, and no audit event with
``verdict = allow`` was emitted.

This manifested as flaky CI: the same test would pass on one Python
matrix cell and fail on another depending on the allocator's
particular id-reuse pattern.

This test anchors the fix structurally — neither patch module may
re-introduce a module-level id tracker. The behavioural coverage
lives in ``tests/test_smoke_provider_battery.py::test_bedrock_*``
and ``tests/test_before_after_each_llm_and_tool.py::test_bedrock_*``,
which are themselves the canonical reproductions of the flake.
"""

from __future__ import annotations


def test_bedrock_runtime_has_no_stale_id_tracker() -> None:
    from egisai._patches import bedrock_runtime

    assert not hasattr(bedrock_runtime, "_PATCHED_CLIENT_IDS"), (
        "bedrock_runtime re-introduced a process-wide id() tracker. "
        "This causes a silent gate bypass when CPython recycles "
        "object ids between clients — the second client's converse "
        "method falls through to the raw provider, gate_call never "
        "runs, and policies don't fire. See egisai CHANGELOG 0.25.15."
    )


def test_bedrock_agent_has_no_stale_id_tracker() -> None:
    from egisai._patches import bedrock_agent

    assert not hasattr(bedrock_agent, "_PATCHED_CLIENT_IDS"), (
        "bedrock_agent re-introduced a process-wide id() tracker. "
        "This causes a silent gate bypass when CPython recycles "
        "object ids between clients — the second client's "
        "invoke_agent method falls through to the raw provider, "
        "gate_call never runs, and policies don't fire. See egisai "
        "CHANGELOG 0.25.15."
    )
