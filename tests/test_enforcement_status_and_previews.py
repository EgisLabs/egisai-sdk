"""Audit-honesty contracts: privacy of model responses,
``enforcement_status``, output-side policy latency, and multi-step
agentic emission.

These regression tests pin behavior that we ship as 0.19.0:

1. **Model responses are NEVER persisted.** No audit event
   carries ``response_preview``, no Run carries ``final_text``,
   no SSE step payload carries ``step_response_preview``. The
   text reaches output policies for evaluation, then goes out of
   scope. This is the platform's privacy stance: we store what
   the model **did** (verdict, tool calls, matched policy) — not
   what it **said**.

2. **Output-phase policy time + tokens land on the audit row.**
   ``semantic_guard`` (LLM-judge) running on the post-model side
   is a real LLM round-trip; its wall-clock + token spend must be
   reflected on ``policy_latency_ms`` / ``policy_tokens_*`` so the
   dashboard's "Policy (sum)" stat is honest.

3. **``enforcement_status`` honestly records what the SDK could do.**
   - Input-side block → ``enforced`` (prompt never forwarded).
   - Synchronous-path output-side block → ``enforced`` (stub
     returned / call raised before user code saw the response).
   - Agentic-framework output-side block → ``advisory`` (subprocess
     already ran tools). ``claude_agent_sdk`` is the canonical case.

4. **Agentic frameworks emit one step per tool call.**
   ``claude_agent_sdk``'s streaming receive path now dispatches
   one ``tool_call`` step per ``ToolUseBlock`` so the dashboard's
   timeline shows the full agent loop. Previously a 6-tool agentic
   run collapsed to a single ``model_call`` step on the dashboard.

The fixture/stub shape is shared with
``test_claude_agent_sdk_governance.py``; we keep tests in a separate
file because the behaviors above are cross-cutting (not just
claude-specific) and because failures here imply specific user-
visible regressions on the Requests page.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

# ── Test stubs identical to test_claude_agent_sdk_governance.py ─────


class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class ToolUseBlock:
    def __init__(
        self, name: str, input_: dict[str, Any] | None = None
    ) -> None:
        self.name = name
        self.input = input_ or {}
        self.id = f"tool_{name}_001"


class AssistantMessage:
    def __init__(self, content: list[Any]) -> None:
        self.content = content


class ResultMessage:
    def __init__(
        self,
        *,
        input_tokens: int = 12,
        output_tokens: int = 34,
        cost_usd: float = 0.0125,
    ) -> None:
        self.usage = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        self.total_cost_usd = cost_usd


class _Options:
    def __init__(
        self,
        *,
        system_prompt: str = "You are a helpful assistant.",
        allowed_tools: list[str] | None = None,
        permission_mode: str = "auto",
        model: str = "claude-3-5-sonnet",
        mcp_servers: dict[str, Any] | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.allowed_tools = allowed_tools or []
        self.permission_mode = permission_mode
        self.model = model
        self.mcp_servers = mcp_servers or {}


def _install_fake_module() -> tuple[types.ModuleType, type]:
    """Build + register a fake ``claude_agent_sdk`` module.

    Unlike the sister file's helper, the script is set on
    ``mod.__script__`` so a single test can swap scripts between
    multi-turn calls.
    """
    mod = types.ModuleType("claude_agent_sdk")

    async def _module_query(
        prompt: Any, options: Any = None
    ) -> AsyncIterator[Any]:
        for msg in mod.__script__:
            yield msg

    class _Client:
        def __init__(self, options: Any = None) -> None:
            self.options = options
            self._sent: list[Any] = []

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

        async def query(
            self, prompt: Any, session_id: str = "default"
        ) -> None:
            self._sent.append({"prompt": prompt, "session_id": session_id})

        async def receive_messages(self) -> AsyncIterator[Any]:
            for msg in mod.__script__:
                yield msg

        async def receive_response(self) -> AsyncIterator[Any]:
            async for msg in self.receive_messages():
                yield msg
                if isinstance(msg, ResultMessage):
                    return

    mod.query = _module_query
    mod.ClaudeSDKClient = _Client
    mod.AssistantMessage = AssistantMessage
    mod.TextBlock = TextBlock
    mod.ToolUseBlock = ToolUseBlock
    mod.ResultMessage = ResultMessage
    mod.__script__ = []
    sys.modules["claude_agent_sdk"] = mod
    return mod, _Client


@pytest.fixture
def fake_claude(
    fake_backend: Any,
) -> Iterator[tuple[Any, type, types.ModuleType]]:
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="claude-audit-test",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="stub",  # User's prod config — observe-only on output side.
    )
    mod, client_cls = _install_fake_module()

    from egisai._patches import claude_agent_sdk

    assert claude_agent_sdk.apply() is True

    yield fake_backend, client_cls, mod
    sys.modules.pop("claude_agent_sdk", None)


def _flush() -> None:
    from egisai import shutdown
    shutdown()


def _model_steps(events: list[dict]) -> list[dict]:
    return [
        e for e in events
        if e.get("kind") == "run.step" and e.get("step_kind") == "model_call"
    ]


def _tool_steps(events: list[dict]) -> list[dict]:
    return [
        e for e in events
        if e.get("kind") == "run.step" and e.get("step_kind") == "tool_call"
    ]


def _deny_tool_rule(pattern: str = r"^run_shell$") -> dict[str, Any]:
    return {
        "id": "r1",
        "name": "block-shell",
        "type": "deny_tool_call",
        "tenant": None,
        "config": {"patterns": [pattern]},
    }


def _deny_regex_input_rule() -> dict[str, Any]:
    return {
        "id": "r2",
        "name": "block-input",
        "type": "deny_regex",
        "tenant": None,
        "config": {"pattern": r"forbidden", "message": "no"},
    }


# ── 1. Model responses are NEVER persisted ──────────────────────────
#
# These are negative regression tests: previously (briefly, during
# 0.18-rc) the SDK stamped a label-redacted preview of the model
# output onto the audit event. The privacy contract is now that
# Egis NEVER stores model responses anywhere — they're evaluated
# by output policies and then go out of scope. Any future change
# that re-introduces ``response_preview`` (or any field carrying
# the response text) on the audit event MUST fail one of these
# tests.


def test_audit_event_never_carries_response_preview(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """No matter how chatty the model is, the model_call step
    event MUST NOT include a ``response_preview`` field."""
    fake_backend, client_cls, mod = fake_claude
    model_output = "Hello, here is your weather forecast: 72°F and sunny."
    mod.__script__ = [
        AssistantMessage([TextBlock(model_output)]),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("What's the weather?")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    for ev in _model_steps(fake_backend.events_received):
        assert ev.get("response_preview") is None, (
            f"response_preview MUST be absent from audit events "
            f"(got {ev.get('response_preview')!r})"
        )
        serialized = repr(ev)
        assert model_output not in serialized, (
            "model response text leaked into the audit event under "
            "some other field — privacy contract broken"
        )


def test_run_end_envelope_never_carries_final_text(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """The ``run.end`` envelope ships ``final_text`` as ``None``
    (or omits it) — the dashboard reads "Run completed" without
    needing the model's actual answer to render the modal."""
    fake_backend, client_cls, mod = fake_claude
    leak = "Confidential model reply: account holder is Maria."
    mod.__script__ = [
        AssistantMessage([TextBlock(leak)]),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("Look up that account.")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    end_envelopes = [
        e for e in fake_backend.events_received if e.get("kind") == "run.end"
    ]
    assert end_envelopes, "expected a run.end envelope on the wire"
    ev = end_envelopes[-1]
    assert ev.get("final_text") in (None, ""), (
        f"run.end must not carry the model's reply (got "
        f"{ev.get('final_text')!r})"
    )
    # And the leak text must not appear anywhere in the envelope
    # under a renamed field either.
    assert leak not in repr(ev), (
        "model response text leaked into the run.end envelope"
    )


def test_pii_in_model_response_never_reaches_backend(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """Even when the model output contains PII (e.g. a leaked SSN
    in its reply text), the audit log contains NO trace of the
    text. The reply is evaluated by output policies and then
    discarded; the wire payload carries only the verdict/metadata."""
    fake_backend, client_cls, mod = fake_claude
    leaked_ssn = "123-45-6789"
    mod.__script__ = [
        AssistantMessage(
            [TextBlock(f"User Maria's SSN is {leaked_ssn} on file.")]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("What is on file?")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    # Walk the full wire payload — every envelope, every field.
    # The literal SSN must not appear anywhere.
    for ev in fake_backend.events_received:
        assert leaked_ssn not in repr(ev), (
            f"raw SSN leaked into a wire event: kind={ev.get('kind')}, "
            f"fields={list(ev)}"
        )


# ── 2. Output-side policy latency lands on the audit row ────────────


def test_output_policy_latency_recorded_on_audit_row(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """An output-side policy that fires (here ``deny_tool_call``
    against a banned tool) MUST contribute to
    ``policy_latency_ms``. Pre-0.18.2 only the input phase booked
    policy latency — output-side ``semantic_guard`` rounds were
    silently invisible on the dashboard's "Policy (sum)" stat.
    """
    fake_backend, client_cls, mod = fake_claude
    fake_backend.set_rules([_deny_tool_rule()], etag='"deny-tool"')
    from egisai._policy_cache import replace_rules

    replace_rules('"deny-tool"', [_deny_tool_rule()])

    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock("calling shell"),
                ToolUseBlock("run_shell", {"cmd": "ls"}),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("do something")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    ev = _model_steps(fake_backend.events_received)[-1]
    # Both keys must be present and non-negative ints. We don't
    # assert > 0 because regex-only output policies are very fast
    # and may land on the 0-ms boundary; what we lock in is the
    # field is being populated rather than left None.
    assert isinstance(ev.get("policy_latency_ms"), int)
    assert ev["policy_latency_ms"] >= 0
    assert isinstance(ev.get("policy_tokens_in"), int)
    assert isinstance(ev.get("policy_tokens_out"), int)


# ── 3. enforcement_status semantics ─────────────────────────────────


def test_input_block_enforcement_is_enforced(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """Input-side block: prompt never reaches the subprocess →
    enforcement_status='enforced'. ``on_block='stub'`` doesn't
    matter on the input side; the claude_agent_sdk patch always
    raises for input blocks (the subprocess would otherwise
    receive the raw prompt)."""
    fake_backend, client_cls, mod = fake_claude
    fake_backend.set_rules(
        [_deny_regex_input_rule()], etag='"input-block"'
    )
    from egisai._policy_cache import replace_rules

    replace_rules('"input-block"', [_deny_regex_input_rule()])

    mod.__script__ = [
        AssistantMessage([TextBlock("never reached")]),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            with pytest.raises(PermissionError):
                await client.query("This is forbidden territory.")

    asyncio.run(run())
    _flush()

    ev = _model_steps(fake_backend.events_received)[0]
    assert ev["verdict"] == "block"
    assert ev.get("enforcement_status") == "enforced"


def test_output_block_with_on_block_stub_is_advisory(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """When ``on_block='stub'`` and the output policy fires,
    enforcement_status must be 'advisory' — the Node subprocess
    has already finished the agent loop, so the audit row honestly
    records "we observed but couldn't prevent"."""
    fake_backend, client_cls, mod = fake_claude
    fake_backend.set_rules([_deny_tool_rule()], etag='"deny-tool"')
    from egisai._policy_cache import replace_rules

    replace_rules('"deny-tool"', [_deny_tool_rule()])

    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock("calling shell"),
                ToolUseBlock("run_shell", {"cmd": "rm -rf /"}),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("clean it up")
            # on_block='stub' → no exception on output block.
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    ev = _model_steps(fake_backend.events_received)[-1]
    assert ev["verdict"] == "block"
    assert ev.get("enforcement_status") == "advisory", (
        "claude_agent_sdk output blocks under on_block=stub are "
        "post-hoc findings, not enforcement"
    )
    assert ev.get("matched_policy") == "block-shell"


def test_allow_path_enforcement_is_enforced(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """No policy fired → enforcement_status='enforced' (the SDK
    trivially didn't fail to enforce since there was nothing to
    enforce against)."""
    fake_backend, client_cls, mod = fake_claude
    mod.__script__ = [
        AssistantMessage([TextBlock("all clear")]),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("status?")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    ev = _model_steps(fake_backend.events_received)[-1]
    assert ev["verdict"] == "allow"
    assert ev.get("enforcement_status") == "enforced"


# ── 4. One ``tool_call`` step per ToolUseBlock ──────────────────────


def test_multi_tool_agentic_loop_emits_step_per_tool_use(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """A claude_agent_sdk turn that calls multiple tools must emit
    one ``tool_call`` step per ``ToolUseBlock`` — the dashboard's
    RunTimelineModal renders the agentic loop as a waterfall, and
    before 0.18.2 it collapsed multi-tool runs to a single
    ``model_call`` step.
    """
    fake_backend, client_cls, mod = fake_claude
    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock("Looking up account..."),
                ToolUseBlock(
                    "mcp__support__lookup_customer", {"id": "ACC-1"}
                ),
            ]
        ),
        AssistantMessage(
            [
                TextBlock("Issuing refund..."),
                ToolUseBlock(
                    "mcp__support__issue_refund", {"amount": 100}
                ),
                ToolUseBlock(
                    "mcp__support__send_email", {"to": "x@example.com"}
                ),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("process refund")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    tool_evs = _tool_steps(fake_backend.events_received)
    assert len(tool_evs) == 3, (
        f"expected 3 tool_call steps (one per ToolUseBlock), got "
        f"{len(tool_evs)}; events: "
        f"{[e.get('tool_name') for e in tool_evs]}"
    )
    names = {e.get("tool_name") for e in tool_evs}
    assert names == {
        "mcp__support__lookup_customer",
        "mcp__support__issue_refund",
        "mcp__support__send_email",
    }
    # Tool-call steps must be flagged advisory: the Node subprocess
    # had already executed the tool by the time we saw the block.
    for ev in tool_evs:
        assert ev.get("enforcement_status") == "advisory"


def test_tool_step_carries_per_tool_policy_match(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """When ``deny_tool_call`` matches one specific tool, the
    matching tool_call step gets verdict=block + the rule's name;
    sibling tool steps stay allow. This lets the operator point
    at the exact tool that tripped the policy."""
    fake_backend, client_cls, mod = fake_claude
    fake_backend.set_rules(
        [_deny_tool_rule(r"^mcp__support__send_email$")], etag='"deny-mail"'
    )
    from egisai._policy_cache import replace_rules

    replace_rules(
        '"deny-mail"',
        [_deny_tool_rule(r"^mcp__support__send_email$")],
    )

    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock("Looking up..."),
                ToolUseBlock("mcp__support__lookup_customer", {}),
            ]
        ),
        AssistantMessage(
            [
                TextBlock("Sending..."),
                ToolUseBlock(
                    "mcp__support__send_email", {"to": "x@example.com"}
                ),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("send email")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    tool_evs = _tool_steps(fake_backend.events_received)
    assert len(tool_evs) == 2
    by_name = {e["tool_name"]: e for e in tool_evs}
    assert by_name["mcp__support__lookup_customer"]["verdict"] == "allow"
    blocked = by_name["mcp__support__send_email"]
    assert blocked["verdict"] == "block"
    assert blocked.get("matched_policy") == "block-shell"
    assert blocked.get("enforcement_status") == "advisory"


def test_tool_step_input_label_redacted(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """The ``request_text`` of a tool_call step is the tool's input
    serialized as JSON and passed through ``label_redact``. If the
    model passed PII as a tool argument, the audit row must not
    persist it raw."""
    fake_claude_backend, client_cls, mod = fake_claude
    leak = "555-12-3456"
    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock("lookup"),
                ToolUseBlock(
                    "mcp__support__lookup_customer", {"ssn": leak}
                ),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("lookup")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    tool_evs = _tool_steps(fake_claude_backend.events_received)
    assert len(tool_evs) == 1
    text = tool_evs[0].get("request_text") or ""
    assert leak not in text, (
        f"raw SSN must not appear in tool_call request_text (got {text!r})"
    )
