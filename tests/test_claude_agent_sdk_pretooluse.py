"""PreToolUse hook tests for the Claude Agent SDK patch.

Pre-0.21 the ``claude_agent_sdk`` patch ran output policies on
``ToolUseBlock``/``ResultMessage`` AFTER the Node.js subprocess had
already executed the tool. Audit rows were stamped
``enforcement_status="advisory"`` to honestly reflect that the SDK
could observe but not enforce on the output side.

0.21 wires a ``PreToolUse`` hook into ``ClaudeAgentOptions.hooks``
so policy decisions fire IN OUR PYTHON PROCESS, BEFORE the CLI
dispatches the tool. ``deny_tool_call`` / ``deny_mcp_call`` /
``semantic_guard`` on tool calls become real pre-execution blocks.
The audit row's ``enforcement_status`` flips from ``"advisory"``
to ``"enforced"``.

These tests lock in the new contract at every level:

* **Hook callback unit semantics** — allow / deny / identity
  propagation / fail-open. Tested against the callback in isolation
  (without the SDK transport).
* **Hook injection composition** — ``options.hooks`` is mutated
  in place, our matcher is APPENDED to any user-supplied matchers
  (not replaced).
* **Per-tool step emission via hook** — when the hook fires, the
  ``tool_call`` step row lands with ``enforcement_status="enforced"``
  and the receive-side fallback emitter SKIPS the redundant row.
* **Fallback path** — when the SDK lacks the ``hooks`` field (older
  versions, custom transports), the patch quietly falls back to the
  legacy post-hoc advisory behavior. No regression on old SDKs.
* **End-of-turn audit row** — final ``model_call`` step stamps
  ``enforcement_status="enforced"`` for pure TEXT-only violations when
  hooks were active **and no ``tool_calls`` were replayed**. When aggregated
  OUTPUT evaluates structured tool payloads, blocks stamp ``advisory`` (see
  ``0.22.3`` CHANGELOG — subprocess timing honesty). Turns without injected
  hooks remain ``advisory`` end-to-end.

Tests intentionally do NOT invoke the real Node CLI — they stand
up a faithful in-process double that mirrors the SDK's attribute
shapes and bidirectional hook-callback protocol (``options.hooks``
gets read, callbacks get invoked with the documented input shape,
return values get inspected for ``permissionDecision``).
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

# ── Test stubs that mirror real upstream class shapes ───────────────


class TextBlock:
    def __init__(self, text: str) -> None:
        self.text = text


class ToolUseBlock:
    def __init__(
        self,
        name: str,
        input_: dict[str, Any] | None = None,
        id_: str | None = None,
    ) -> None:
        self.name = name
        self.input = input_ or {}
        # ID matches the upstream ``ToolUseBlock.id`` field; correlates
        # with the ``tool_use_id`` the hook callback receives.
        self.id = id_ or f"toolu_{name}_001"


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


@dataclass
class _HookMatcher:
    """Real upstream shape — appended to ``options.hooks[event]``."""

    matcher: str | None = None
    hooks: list[Any] = field(default_factory=list)
    timeout: float | None = None


@dataclass
class _ClaudeAgentOptions:
    """``ClaudeAgentOptions`` stand-in WITH the ``hooks`` field.

    The presence of the ``hooks`` field on this class is what
    ``_hooks_supported()`` detects — older SDKs without the field
    fall back to the legacy advisory path (covered by
    ``test_legacy_no_hooks_field_falls_back_to_advisory``).
    """

    system_prompt: str = "You are a helpful assistant."
    allowed_tools: list[str] = field(default_factory=list)
    permission_mode: str = "auto"
    model: str = "claude-3-5-sonnet"
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    hooks: dict[str, list[_HookMatcher]] | None = None


@dataclass
class _LegacyOptions:
    """Pre-hook ``ClaudeAgentOptions`` shape — NO ``hooks`` field.

    Used by the fallback test to assert we don't crash and we don't
    invent a ``hooks`` attribute on a legacy options dataclass.
    """

    system_prompt: str = "You are a helpful assistant."
    allowed_tools: list[str] = field(default_factory=list)
    permission_mode: str = "auto"
    model: str = "claude-3-5-sonnet"
    mcp_servers: dict[str, Any] = field(default_factory=dict)


def _install_fake_module(
    *, with_hooks: bool = True
) -> tuple[types.ModuleType, type, list[Any]]:
    """Build + register a fake ``claude_agent_sdk`` module.

    When ``with_hooks=True`` the module exposes ``HookMatcher`` and
    ``ClaudeAgentOptions.hooks`` so ``_hooks_supported()`` returns
    True. When False, neither symbol is present, which exercises
    the legacy fallback path.
    """
    mod = types.ModuleType("claude_agent_sdk")
    captured: list[Any] = []

    async def _module_query(prompt: Any, options: Any = None) -> AsyncIterator[Any]:
        captured.append({"prompt": prompt, "options": options})
        for msg in []:
            yield msg

    class _Client:
        def __init__(self, options: Any = None) -> None:
            self.options = options
            self._sent: list[Any] = []
            self._connected = False

        # Mirror upstream: ``__aenter__`` calls ``connect()``. The
        # egisai patch wraps ``connect`` to inject placeholder
        # PreToolUse / PostToolUse hooks BEFORE the (real) CLI
        # would freeze its matcher table at initialize-time. Tests
        # need this method so the wrap actually fires.
        async def connect(self, prompt: Any = None) -> None:
            self._connected = True

        async def __aenter__(self) -> _Client:
            await self.connect()
            return self

        async def __aexit__(self, *exc: Any) -> bool:
            return False

        async def query(
            self, prompt: Any, session_id: str = "default"
        ) -> None:
            self._sent.append({"prompt": prompt, "session_id": session_id})

        async def receive_messages(self) -> AsyncIterator[Any]:
            for msg in []:
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
    if with_hooks:
        mod.HookMatcher = _HookMatcher
        mod.ClaudeAgentOptions = _ClaudeAgentOptions
    else:
        mod.ClaudeAgentOptions = _LegacyOptions
    sys.modules["claude_agent_sdk"] = mod
    return mod, _Client, captured


@pytest.fixture
def fake_claude_with_hooks(
    fake_backend: Any,
) -> Iterator[tuple[Any, type, types.ModuleType]]:
    """Module that DOES expose hooks → modern path is exercised.

    The fake transport simulates the SDK's bidirectional protocol:
    when ``client.query(prompt)`` returns, the fake reads
    ``options.hooks["PreToolUse"]`` and invokes each callback once
    per ToolUseBlock in the configured script — mirroring how the
    real Node CLI fires the hook control message just before
    dispatching the tool. Hook responses are inspected: if any
    returns ``permissionDecision == "deny"``, the corresponding
    ToolUseBlock is REPLACED with a synthetic denial result in the
    receive stream so tests can assert end-to-end "tool never ran"
    semantics.
    """
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="claude-pretooluse-test",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )
    mod, client_cls, _captured = _install_fake_module(with_hooks=True)
    mod.__script__ = []
    mod.__hook_invocations__ = []  # tests inspect this for evidence

    async def _q(self: Any, prompt: Any, session_id: str = "default") -> None:
        self._sent.append({"prompt": prompt, "session_id": session_id})
        opts = getattr(self, "options", None)
        await _drive_hooks_for_script(opts, mod.__script__, mod.__hook_invocations__)

    async def _rm(self: Any) -> AsyncIterator[Any]:
        for msg in mod.__script__:
            yield msg

    client_cls.query = _q
    client_cls.receive_messages = _rm

    from egisai._patches import claude_agent_sdk

    assert claude_agent_sdk.apply() is True

    yield fake_backend, client_cls, mod
    sys.modules.pop("claude_agent_sdk", None)


@pytest.fixture
def fake_claude_legacy(
    fake_backend: Any,
) -> Iterator[tuple[Any, type, types.ModuleType]]:
    """Module that does NOT expose hooks → fallback path is exercised."""
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="claude-legacy-test",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )
    mod, client_cls, _captured = _install_fake_module(with_hooks=False)
    mod.__script__ = []

    async def _q(self: Any, prompt: Any, session_id: str = "default") -> None:
        self._sent.append({"prompt": prompt, "session_id": session_id})

    async def _rm(self: Any) -> AsyncIterator[Any]:
        for msg in mod.__script__:
            yield msg

    client_cls.query = _q
    client_cls.receive_messages = _rm

    from egisai._patches import claude_agent_sdk

    assert claude_agent_sdk.apply() is True

    yield fake_backend, client_cls, mod
    sys.modules.pop("claude_agent_sdk", None)


async def _drive_hooks_for_script(
    opts: Any,
    script: list[Any],
    invocations: list[dict[str, Any]],
) -> None:
    """Simulate the SDK's PreToolUse hook dispatch over the script.

    For each ``ToolUseBlock`` in each ``AssistantMessage``, invoke
    every registered PreToolUse callback (matchers are not honored —
    the fake mirrors the egisai matcher=None catch-all). Records
    each invocation's input + output in ``invocations`` so tests
    can assert the hook fired with the expected payload.

    The fake does NOT mutate the script based on deny results — the
    real CLI synthesizes a denial ToolResultBlock instead of the
    tool's output, but our tests assert on (a) the hook return
    value, (b) audit-row contents, (c) the hook invocations list,
    which is sufficient. End-to-end "tool was skipped" semantics
    are covered indirectly by asserting the hook returned ``deny``.
    """
    if opts is None:
        return
    hooks = getattr(opts, "hooks", None)
    if not isinstance(hooks, dict):
        return
    matchers = hooks.get("PreToolUse") or []
    for msg in script:
        if type(msg).__name__ != "AssistantMessage":
            continue
        for block in getattr(msg, "content", []) or []:
            if type(block).__name__ != "ToolUseBlock":
                continue
            hook_input = {
                "hook_event_name": "PreToolUse",
                "tool_name": block.name,
                "tool_input": block.input,
                "tool_use_id": block.id,
                "session_id": "default",
                "cwd": "/tmp",
                "permission_mode": "auto",
            }
            for matcher in matchers:
                for cb in getattr(matcher, "hooks", []) or []:
                    result = await cb(hook_input, block.id, {"signal": None})
                    invocations.append(
                        {
                            "tool_name": block.name,
                            "tool_use_id": block.id,
                            "input": hook_input,
                            "output": result,
                        }
                    )


def _flush() -> None:
    from egisai import shutdown

    shutdown()


def _step_events(events: list[dict], *, kind: str | None = None) -> list[dict]:
    return [
        e for e in events
        if e.get("kind") == "run.step"
        and (kind is None or e.get("step_kind") == kind)
    ]


# ── Policy helpers ──────────────────────────────────────────────────


def _deny_tool_rule(pattern: str = r"^run_shell$") -> dict[str, Any]:
    return {
        "id": "rt1",
        "name": "block-shell",
        "type": "deny_tool_call",
        "tenant": None,
        "config": {"patterns": [pattern]},
    }


def _deny_mcp_rule(pattern: str = r"^prod_") -> dict[str, Any]:
    return {
        "id": "rm1",
        "name": "block-prod-mcp",
        "type": "deny_mcp_call",
        "tenant": None,
        "config": {"patterns": [pattern]},
    }


def _load_rules(*rules: dict[str, Any]) -> None:
    from egisai._policy_cache import replace_rules

    etag = f'"r{len(rules)}"'
    replace_rules(etag, list(rules))


# ── 1. Hook injection ──────────────────────────────────────────────


def test_hook_injected_into_options_hooks_on_query(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """When SDK supports hooks, our PreToolUse matcher is appended
    to ``options.hooks["PreToolUse"]`` before the prompt is
    forwarded to the subprocess."""
    _, client_cls, mod = fake_claude_with_hooks
    mod.__script__ = [
        AssistantMessage([TextBlock("hi")]),
        ResultMessage(),
    ]

    captured_hooks: dict[str, Any] = {}

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Hello")
            captured_hooks["hooks"] = opts.hooks
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    hooks = captured_hooks["hooks"]
    assert isinstance(hooks, dict), "options.hooks must be a dict"
    assert "PreToolUse" in hooks
    matchers = hooks["PreToolUse"]
    assert len(matchers) == 1
    assert matchers[0].matcher is None  # catch-all
    assert len(matchers[0].hooks) == 1
    assert callable(matchers[0].hooks[0])


def test_hook_composes_with_existing_user_hooks(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """A user-supplied PreToolUse hook MUST remain in place; our
    matcher is APPENDED to the list, not substituted for it."""
    _, client_cls, mod = fake_claude_with_hooks
    mod.__script__ = [
        AssistantMessage([TextBlock("hi")]),
        ResultMessage(),
    ]

    user_hook_calls: list[Any] = []

    async def user_hook(hook_input: Any, tool_use_id: Any, context: Any) -> dict:
        user_hook_calls.append(hook_input.get("tool_name"))
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
            }
        }

    captured_hooks: dict[str, Any] = {}

    async def run() -> None:
        user_matcher = _HookMatcher(matcher="Bash", hooks=[user_hook])
        opts = _ClaudeAgentOptions(hooks={"PreToolUse": [user_matcher]})
        async with client_cls(options=opts) as client:
            await client.query("Hi")
            captured_hooks["hooks"] = opts.hooks
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    matchers = captured_hooks["hooks"]["PreToolUse"]
    assert len(matchers) == 2, "user matcher + ours = 2"
    # User's matcher (with matcher="Bash") preserved first; ours appended.
    assert matchers[0].matcher == "Bash"
    assert matchers[0].hooks == [user_hook]
    assert matchers[1].matcher is None  # ours


def test_no_hook_injection_when_options_missing(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """Without ``options=...`` we have nothing to mutate; the patch
    must NOT crash, just fall back to advisory mode for the turn."""
    _, client_cls, mod = fake_claude_with_hooks
    mod.__script__ = [
        AssistantMessage([TextBlock("hi")]),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls() as client:  # no options
            await client.query("Hello")
            async for _ in client.receive_response():
                pass

    # Should not raise.
    asyncio.run(run())
    _flush()


# ── 2. Hook callback semantics — allow path ─────────────────────────


def test_hook_allow_when_no_policies_loaded(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """No policies → hook returns allow, tool dispatch proceeds."""
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules()  # empty ruleset
    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock("Reading config."),
                ToolUseBlock("Read", {"path": "/tmp/x"}, id_="tu_001"),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Open config")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    invocations = mod.__hook_invocations__
    assert len(invocations) == 1
    out = invocations[0]["output"]
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"


def test_hook_deny_blocks_dangerous_tool(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """``deny_tool_call`` on ``run_shell`` → hook returns deny with
    a descriptive reason; subprocess never dispatches the tool."""
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules(_deny_tool_rule(r"^run_shell$"))
    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock("I'll clean up."),
                ToolUseBlock(
                    "run_shell", {"cmd": "rm -rf /"}, id_="tu_dangerous"
                ),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Clean up.")
            # The output-phase eval at ResultMessage may also fire
            # for deny_tool_call on the accumulated stream; we
            # catch that PermissionError to focus on hook semantics.
            try:
                async for _ in client.receive_response():
                    pass
            except PermissionError:
                pass

    asyncio.run(run())
    _flush()

    invocations = mod.__hook_invocations__
    assert len(invocations) == 1
    out = invocations[0]["output"]
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "[egisai]" in reason
    assert "block-shell" in reason


def test_hook_deny_blocks_mcp_target(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """MCP tool names are namespaced ``mcp__<server>__<tool>`` —
    ``deny_mcp_call`` matches on the ``<server>`` portion."""
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules(_deny_mcp_rule(r"^prod_db$"))
    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock("Fetching."),
                ToolUseBlock(
                    "mcp__prod_db__query",
                    {"sql": "SELECT * FROM users"},
                    id_="tu_mcp",
                ),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Query prod.")
            try:
                async for _ in client.receive_response():
                    pass
            except PermissionError:
                pass

    asyncio.run(run())
    _flush()

    invocations = mod.__hook_invocations__
    assert len(invocations) == 1
    out = invocations[0]["output"]
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_hook_records_decision_for_each_tool_use_id(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """Multiple tools in one assistant turn → hook fires once per
    tool, each receiving its distinct ``tool_use_id``."""
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules(_deny_tool_rule(r"^run_shell$"))
    mod.__script__ = [
        AssistantMessage(
            [
                ToolUseBlock("Read", {"path": "/x"}, id_="tu_A"),
                ToolUseBlock("run_shell", {"cmd": "ls"}, id_="tu_B"),
                ToolUseBlock("Read", {"path": "/y"}, id_="tu_C"),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Multi-step")
            try:
                async for _ in client.receive_response():
                    pass
            except PermissionError:
                pass

    asyncio.run(run())
    _flush()

    invocations = mod.__hook_invocations__
    assert len(invocations) == 3
    ids = [inv["tool_use_id"] for inv in invocations]
    assert ids == ["tu_A", "tu_B", "tu_C"]
    decisions = [
        inv["output"]["hookSpecificOutput"]["permissionDecision"]
        for inv in invocations
    ]
    assert decisions == ["allow", "deny", "allow"]


# ── 3. Audit-row enforcement semantics ─────────────────────────────


def test_tool_call_step_stamped_enforced_when_hook_active(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """Hook-gated tool_call step rows MUST stamp
    ``enforcement_status="enforced"`` (the headline UX change)."""
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules(_deny_tool_rule(r"^run_shell$"))
    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock("Running shell."),
                ToolUseBlock("run_shell", {"cmd": "rm"}, id_="tu_E"),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Do it.")
            try:
                async for _ in client.receive_response():
                    pass
            except PermissionError:
                pass

    asyncio.run(run())
    _flush()

    tool_steps = _step_events(fake_backend.events_received, kind="tool_call")
    assert len(tool_steps) == 1, "exactly one tool_call step expected"
    step = tool_steps[0]
    assert step["enforcement_status"] == "enforced"
    assert step["verdict"] == "block"
    assert step["matched_policy"] == "block-shell"
    assert step["tool_name"] == "run_shell"


def test_tool_call_step_carries_input_under_prompt_preview_key(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """Regression for 0.27.1 — every tool_call event MUST ship the
    tool input under the wire key ``prompt_preview``.

    The backend reads the audit row's preview text from
    ``ev.get("prompt_preview")`` (see ``app.routers.sdk
    ._build_request_log_row``). Pre-0.27.1 the Claude Agent SDK
    patch shipped tool_call inputs under ``request_text`` (the DB
    column name), which the backend ingest reader ignored —
    leaving every tool_call row's ``request_text`` NULL and
    collapsing the intent-summary LLM onto generic "Open ended
    assistant chat" / "General chat follow up question" labels.
    """
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules()  # allow-all so the hook emits with verdict=allow
    mod.__script__ = [
        AssistantMessage(
            [
                ToolUseBlock(
                    "lookup_customer_account",
                    {"customer_id": "ACME-001", "fields": ["status"]},
                    id_="tu_preview_A",
                ),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Look up ACME-001.")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    tool_steps = _step_events(fake_backend.events_received, kind="tool_call")
    assert len(tool_steps) == 1
    step = tool_steps[0]
    preview = step.get("prompt_preview")
    assert isinstance(preview, str) and preview, (
        "tool_call step must carry the tool input under "
        f"``prompt_preview`` (got {preview!r}; full keys="
        f"{sorted(step.keys())})"
    )
    assert "ACME-001" in preview
    # Defense in depth: the legacy key MUST NOT be set, or the
    # backend's compatibility fallback would still read it and
    # this test would mask a future regression.
    assert "request_text" not in step


def test_no_duplicate_tool_step_emitted_by_receive_when_hook_fired(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """The hook emits the tool_call step; ``receive_messages`` MUST
    NOT emit a second one for the same ``tool_use_id``."""
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules()  # allow-all
    mod.__script__ = [
        AssistantMessage(
            [
                ToolUseBlock("Read", {"path": "/a"}, id_="tu_dup_A"),
                ToolUseBlock("Read", {"path": "/b"}, id_="tu_dup_B"),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Multi-read")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    tool_steps = _step_events(fake_backend.events_received, kind="tool_call")
    # Exactly 2 — one per tool, emitted by the hook, never the
    # fallback emitter.
    assert len(tool_steps) == 2
    seen_ids = sorted(s.get("tool_name", "") for s in tool_steps)
    assert seen_ids == ["Read", "Read"]
    for s in tool_steps:
        assert s["enforcement_status"] == "enforced"


def test_final_model_call_step_stamped_enforced_with_hooks(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """End-of-turn ``model_call`` audit row stamps ``enforced`` when
    hooks were active (any output-text block is suppressed from the
    caller via ``PermissionError`` — identical to OpenAI/Anthropic
    direct patches)."""
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules()
    mod.__script__ = [
        AssistantMessage([TextBlock("Hello there.")]),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Hi")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    model_steps = _step_events(fake_backend.events_received, kind="model_call")
    assert len(model_steps) == 2
    assert model_steps[-1]["enforcement_status"] == "enforced"


# ── 4. Fallback path on legacy SDKs ────────────────────────────────


def test_legacy_no_hooks_field_falls_back_to_advisory(
    fake_claude_legacy: tuple[Any, type, types.ModuleType],
) -> None:
    """On an older SDK without the ``hooks`` field on
    ``ClaudeAgentOptions``, the patch must NOT crash; it falls back
    to the post-hoc advisory path (today's behavior pre-0.21)."""
    fake_backend, client_cls, mod = fake_claude_legacy
    _load_rules(_deny_tool_rule(r"^run_shell$"))
    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock("Running."),
                ToolUseBlock("run_shell", {"cmd": "rm"}, id_="tu_legacy"),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _LegacyOptions()
        async with client_cls(options=opts) as client:
            await client.query("Run it.")
            try:
                async for _ in client.receive_response():
                    pass
            except PermissionError:
                pass

    asyncio.run(run())
    _flush()

    tool_steps = _step_events(fake_backend.events_received, kind="tool_call")
    # Fallback emitter still fires (advisory mode).
    assert len(tool_steps) == 1
    assert tool_steps[0]["enforcement_status"] == "advisory"
    assert tool_steps[0]["verdict"] == "block"


def test_legacy_final_row_stamped_advisory(
    fake_claude_legacy: tuple[Any, type, types.ModuleType],
) -> None:
    """End-of-turn row on legacy SDKs stamps ``advisory`` — the
    SDK can only observe, not enforce, in subprocess-loop mode."""
    fake_backend, client_cls, mod = fake_claude_legacy
    _load_rules(_deny_tool_rule(r"^run_shell$"))
    mod.__script__ = [
        AssistantMessage(
            [
                ToolUseBlock("run_shell", {"cmd": "x"}, id_="tu_legacy_f"),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _LegacyOptions()
        async with client_cls(options=opts) as client:
            await client.query("Run.")
            try:
                async for _ in client.receive_response():
                    pass
            except PermissionError:
                pass

    asyncio.run(run())
    _flush()

    model_steps = _step_events(fake_backend.events_received, kind="model_call")
    assert len(model_steps) == 2
    assert model_steps[-1]["enforcement_status"] == "advisory"


# ── 5. Identity propagation into hook closure ──────────────────────


def test_hook_closure_carries_identity_into_step_row(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """The hook callback runs on a separate asyncio task — identity
    contextvars don't propagate automatically. The patch must enter
    ``identity_scope(record)`` inside the closure so the tool_call
    step's ``agent_id`` matches the registered agent."""
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules(_deny_tool_rule(r"^run_shell$"))
    mod.__script__ = [
        AssistantMessage(
            [
                ToolUseBlock("run_shell", {"cmd": "x"}, id_="tu_id"),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions(
            system_prompt="You are a tax filing specialist.",
            allowed_tools=["Read"],
        )
        async with client_cls(options=opts) as client:
            await client.query("Process")
            try:
                async for _ in client.receive_response():
                    pass
            except PermissionError:
                pass

    asyncio.run(run())
    _flush()

    tool_steps = _step_events(fake_backend.events_received, kind="tool_call")
    assert len(tool_steps) == 1
    # agent_id should be the same on every step row for this turn
    # (identity locks at run open and the hook closure carries it).
    model_steps = _step_events(fake_backend.events_received, kind="model_call")
    if model_steps:
        assert tool_steps[0]["agent_id"] == model_steps[-1]["agent_id"]
    assert tool_steps[0]["agent_id"]  # non-empty


# ── 6. Demo-critical: the "delete all users" scenario ──────────────


def test_destructive_db_call_via_mcp_is_hard_blocked(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """End-to-end demo scenario: a prompt-injection or agent error
    decides to wipe the database via an MCP tool. The PreToolUse
    hook MUST return ``deny`` so the Node CLI never dispatches the
    tool, and the audit row stamps ``enforced`` so an auditor can
    confirm the destructive action was prevented (not merely
    logged after the fact)."""
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules(
        {
            "id": "destructive-1",
            "name": "block-destructive-db-ops",
            "type": "deny_tool_call",
            "tenant": None,
            "config": {
                "patterns": [
                    r"^mcp__prod_db__delete_all_users$",
                    r"^mcp__prod_db__drop_table$",
                    r"^mcp__prod_db__truncate$",
                ],
                "message": "destructive database operations are forbidden",
            },
        }
    )
    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock(
                    "User asked to clean up old records; "
                    "I'll delete the users table."
                ),
                ToolUseBlock(
                    "mcp__prod_db__delete_all_users",
                    {"confirm": True},
                    id_="tu_destructive",
                ),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Clean up old user records")
            try:
                async for _ in client.receive_response():
                    pass
            except PermissionError:
                pass

    asyncio.run(run())
    _flush()

    invocations = mod.__hook_invocations__
    assert len(invocations) == 1, (
        "PreToolUse hook must fire exactly once for the destructive tool"
    )
    out = invocations[0]["output"]
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny", (
        "destructive tool must be denied BEFORE the subprocess "
        "dispatches it"
    )
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "destructive database operations" in reason

    tool_steps = _step_events(fake_backend.events_received, kind="tool_call")
    assert len(tool_steps) == 1
    step = tool_steps[0]
    assert step["verdict"] == "block"
    assert step["enforcement_status"] == "enforced", (
        "the destructive call was prevented at the hook layer; "
        "the audit row must say so"
    )
    assert step["matched_policy"] == "block-destructive-db-ops"
    assert step["tool_name"] == "mcp__prod_db__delete_all_users"


def test_destructive_bash_command_is_hard_blocked(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """Bash commands that destroy data must be gated by their
    ``tool_input.command`` content, not just the ``Bash`` tool name.
    A ``deny_output_regex`` policy matching on the input's command
    string catches it pre-execution.

    NB: ``deny_output_regex`` evaluates against the
    ``OutputCall.text`` field built from tool_calls' JSON-rendered
    arguments by the policy engine's tool-input matcher.
    """
    fake_backend, client_cls, mod = fake_claude_with_hooks
    # ``deny_tool_call`` with a name-based pattern blocks the
    # generic Bash tool — operator chooses how strict to be. For
    # this test we use the explicit shell deny.
    _load_rules(_deny_tool_rule(r"^Bash$"))
    mod.__script__ = [
        AssistantMessage(
            [
                ToolUseBlock(
                    "Bash",
                    {"command": "psql -c 'DROP TABLE users CASCADE'"},
                    id_="tu_bash_drop",
                ),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Clean things up")
            try:
                async for _ in client.receive_response():
                    pass
            except PermissionError:
                pass

    asyncio.run(run())
    _flush()

    invocations = mod.__hook_invocations__
    assert len(invocations) == 1
    out = invocations[0]["output"]
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"

    tool_steps = _step_events(fake_backend.events_received, kind="tool_call")
    assert len(tool_steps) == 1
    assert tool_steps[0]["verdict"] == "block"
    assert tool_steps[0]["enforcement_status"] == "enforced"


def test_chain_of_tools_each_independently_gated(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """Multi-step agent flow: read → process → write. The hook MUST
    fire once per tool, and a deny on any single tool MUST flip
    that tool's audit row to enforced/block while letting the
    allowed tools through with enforced/allow."""
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules(_deny_tool_rule(r"^write_file$"))
    mod.__script__ = [
        AssistantMessage(
            [
                ToolUseBlock("read_file", {"path": "/in"}, id_="tu_chain_1"),
                ToolUseBlock(
                    "process_data", {"data": "{...}"}, id_="tu_chain_2"
                ),
                ToolUseBlock(
                    "write_file", {"path": "/out"}, id_="tu_chain_3"
                ),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Process the file")
            try:
                async for _ in client.receive_response():
                    pass
            except PermissionError:
                pass

    asyncio.run(run())
    _flush()

    invocations = mod.__hook_invocations__
    assert len(invocations) == 3
    decisions = {
        inv["tool_name"]: inv["output"]["hookSpecificOutput"][
            "permissionDecision"
        ]
        for inv in invocations
    }
    assert decisions == {
        "read_file": "allow",
        "process_data": "allow",
        "write_file": "deny",
    }

    tool_steps = _step_events(fake_backend.events_received, kind="tool_call")
    assert len(tool_steps) == 3
    by_name = {s["tool_name"]: s for s in tool_steps}
    assert by_name["read_file"]["verdict"] == "allow"
    assert by_name["read_file"]["enforcement_status"] == "enforced"
    assert by_name["process_data"]["verdict"] == "allow"
    assert by_name["process_data"]["enforcement_status"] == "enforced"
    assert by_name["write_file"]["verdict"] == "block"
    assert by_name["write_file"]["enforcement_status"] == "enforced"
    assert by_name["write_file"]["matched_policy"] == "block-shell"


# ── 7. Fail-open contract — policy bugs MUST NOT brick the agent ────


def test_hook_fails_open_when_policy_eval_raises(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``evaluate_output`` raises (buggy policy / corrupted
    rule), the hook MUST fail open with ``permissionDecision="allow"``
    rather than denying or crashing. The user's agent must keep
    working even when our policy engine has a bad day."""
    _, client_cls, mod = fake_claude_with_hooks

    def _explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("policy engine crashed (simulated)")

    monkeypatch.setattr(
        "egisai._patches.claude_agent_sdk.evaluate_output", _explode
    )

    mod.__script__ = [
        AssistantMessage(
            [ToolUseBlock("Read", {"path": "/x"}, id_="tu_failopen")]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("ok")
            async for _ in client.receive_response():
                pass

    # MUST NOT raise.
    asyncio.run(run())
    _flush()

    invocations = mod.__hook_invocations__
    assert len(invocations) == 1
    out = invocations[0]["output"]
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow", (
        "policy eval failure → fail OPEN. A user's agent must "
        "never be bricked by our policy engine misbehaving."
    )


# ── 8. ``semantic_guard`` with ``targets: ["tool_calls"]`` ─────────
#
# The headline 0.24 capability: an operator writes a policy whose
# intent list describes forbidden agent BEHAVIOR in plain English
# ("block any lookup request", "wipe the production database"),
# sets ``targets: ["tool_calls"]``, and the PreToolUse hook asks
# the platform judge — for each tool the agent wants to invoke —
# whether THIS specific call matches THAT intent. The judge says
# match → hook returns deny → the Node CLI never dispatches the
# tool. ``deny_tool_call`` pattern lists alone can't cover the
# space of "new dangerous tool names a vendor will ship next
# month"; intent classification can.


def _install_mock_judge(verdict_for_text: dict[str, dict[str, Any]]) -> None:
    """Wire a deterministic in-memory transport into the
    process-wide ``SemanticBlocker`` so PreToolUse hooks that hit
    the judge get a scripted response instead of trying to reach a
    real platform.

    ``verdict_for_text`` maps a substring of the synthesized judge
    prompt → judge response body. The handler picks the first key
    that is a substring of the request body's ``prompt_text``;
    fallthrough is ``match=False`` so unmatched tools cleanly pass.
    """
    import httpx as _httpx

    from egisai import _evaluator
    from egisai.policy.semantic import SemanticBlocker

    def transport_handler(request: _httpx.Request) -> _httpx.Response:
        body = __import__("json").loads(request.content.decode())
        prompt_text = body.get("prompt_text", "")
        for needle, response in verdict_for_text.items():
            if needle in prompt_text:
                return _httpx.Response(200, json=response)
        return _httpx.Response(
            200,
            json={
                "match": False,
                "intent": "",
                "confidence": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
            },
        )

    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
    )
    blocker._http_client = _httpx.Client(
        transport=_httpx.MockTransport(transport_handler)
    )
    _evaluator._blocker = blocker


def _semantic_tool_rule(
    intent: str = "block any lookup request",
    message: str = "Blocked: forbidden agent intent.",
) -> dict[str, Any]:
    return {
        "id": "semantic-tool-1",
        "name": "forbid-destructive-actions",
        "type": "semantic_guard",
        "tenant": None,
        "config": {
            "intents": [intent],
            "targets": ["tool_calls"],
            "message": message,
        },
    }


def test_semantic_guard_with_tool_targets_blocks_first_tool_call(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """The reported scenario in one test: operator has a
    ``semantic_guard`` rule with intent "block any lookup request"
    and ``targets: ["tool_calls"]``. Agent decides to call
    ``mcp__support__lookup_customer`` on its first hop. PreToolUse
    fires, synthesizes the tool description, asks the judge, judge
    matches → hook returns deny → tool is NEVER dispatched and the
    audit row stamps ``enforced``.

    Pre-0.24 this would not have worked: ``semantic_guard`` only
    looked at text, the PreToolUse hook passed ``text=""``, and the
    rule was structurally invisible to the matcher for tool calls."""
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules(_semantic_tool_rule(intent="block any lookup request"))
    _install_mock_judge(
        {
            # Match any synthesized text mentioning the lookup tool.
            "lookup_customer": _stub_judge_match_dict(
                "block any lookup request"
            ),
        }
    )
    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock(
                    "I'll look up the customer account first."
                ),
                ToolUseBlock(
                    "mcp__support__lookup_customer",
                    {"query": "ACC-2847193", "query_type": "account_id"},
                    id_="tu_lookup",
                ),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query(
                "A customer named Maria Gonzalez (account #ACC-2847193) "
                "is threatening to cancel. Look up her account and "
                "issue a refund."
            )
            try:
                async for _ in client.receive_response():
                    pass
            except PermissionError:
                pass

    asyncio.run(run())
    _flush()

    invocations = mod.__hook_invocations__
    assert len(invocations) == 1, (
        "PreToolUse hook must fire exactly once for the lookup tool"
    )
    out = invocations[0]["output"]
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny", (
        "the lookup tool must be denied BEFORE the subprocess dispatches it"
    )
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "Blocked: forbidden agent intent." in reason
    assert "forbid-destructive-actions" in reason

    tool_steps = _step_events(fake_backend.events_received, kind="tool_call")
    assert len(tool_steps) == 1
    step = tool_steps[0]
    assert step["verdict"] == "block"
    assert step["enforcement_status"] == "enforced", (
        "lookup was prevented pre-execution; audit row must say enforced "
        "(not the advisory stamp the legacy text-only path produced)"
    )
    assert step["matched_policy"] == "forbid-destructive-actions"
    assert step["tool_name"] == "mcp__support__lookup_customer"


def test_semantic_guard_tool_target_passes_benign_tools(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """Conjugate of the prior test: when the judge says no-match
    for every tool, the hook lets the call through (verdict=allow,
    status=enforced). The rule must not false-positive on benign
    activity — operators trust the SDK to be quiet when nothing's
    actually wrong."""
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules(_semantic_tool_rule(intent="wipe the production database"))
    # No matching substrings — judge always says no-match.
    _install_mock_judge({})
    mod.__script__ = [
        AssistantMessage(
            [
                ToolUseBlock(
                    "Read", {"path": "/etc/hostname"}, id_="tu_safe"
                ),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("What's my hostname?")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    invocations = mod.__hook_invocations__
    assert len(invocations) == 1
    out = invocations[0]["output"]
    assert out["hookSpecificOutput"]["permissionDecision"] == "allow"

    tool_steps = _step_events(fake_backend.events_received, kind="tool_call")
    assert len(tool_steps) == 1
    assert tool_steps[0]["verdict"] == "allow"
    assert tool_steps[0]["enforcement_status"] == "enforced"


def test_semantic_guard_text_only_default_does_not_block_tool_call(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """Backwards-compat guard: a ``semantic_guard`` rule WITHOUT
    the ``targets`` field is identical in behavior to every release
    shipped before 0.24 — it judges TEXT only. The PreToolUse hook
    fires with ``text=""``, the matcher short-circuits, the judge
    is never consulted for the tool call, the tool runs.

    This is the test that catches the most dangerous possible
    regression: silently changing the meaning of every customer's
    pre-0.24 ``semantic_guard`` policy on upgrade.

    The judge IS called on the input prompt (legacy text behavior);
    we assert that the prompt path stays intact AND that no tool-
    call synthesized text is ever forwarded to the judge."""
    fake_backend, client_cls, mod = fake_claude_with_hooks
    # Legacy-shaped rule: NO ``targets`` field. Equivalent to every
    # rule shipped before 0.24.
    _load_rules(
        {
            "id": "legacy-semantic",
            "name": "legacy-rule",
            "type": "semantic_guard",
            "tenant": None,
            "config": {
                "intents": ["block any lookup request"],
                "message": "Blocked by intent.",
            },
        }
    )
    judge_call_bodies: list[str] = []

    import httpx as _httpx

    from egisai import _evaluator
    from egisai.policy.semantic import SemanticBlocker

    def _judge_handler(request: _httpx.Request) -> _httpx.Response:
        body = __import__("json").loads(request.content.decode())
        prompt_text = body.get("prompt_text", "")
        judge_call_bodies.append(prompt_text)
        # The judge stub returns no-match for everything so the input
        # phase doesn't block the prompt — the assertion below cares
        # only that NO call shows up with the synthesized tool-call
        # shape ("The agent is requesting to invoke tool …").
        return _httpx.Response(
            200,
            json={
                "match": False,
                "intent": "",
                "confidence": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
            },
        )

    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
    )
    blocker._http_client = _httpx.Client(
        transport=_httpx.MockTransport(_judge_handler)
    )
    _evaluator._blocker = blocker

    mod.__script__ = [
        AssistantMessage(
            [ToolUseBlock("lookup_customer", {"q": "x"}, id_="tu_legacy")]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Open a ticket")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    invocations = mod.__hook_invocations__
    assert len(invocations) == 1
    assert (
        invocations[0]["output"]["hookSpecificOutput"]["permissionDecision"]
        == "allow"
    ), (
        "legacy semantic_guard rule (no targets field) MUST NOT block "
        "tool calls — text-only behavior is the contract every "
        "pre-0.24 customer already depends on"
    )
    # The judge is invited to inspect the user PROMPT (legacy
    # behavior) — that's fine. What MUST NOT happen is the matcher
    # synthesizing a tool-call description and sending it. Search
    # every captured body for the synthesizer's signature substring.
    synthesized_calls = [
        b for b in judge_call_bodies if "invoke tool" in b
    ]
    assert synthesized_calls == [], (
        "the matcher MUST NOT round-trip the judge for a tool call "
        "when ``targets`` is absent — that would silently change the "
        "meaning of every customer's existing rules on upgrade. "
        f"Got: {synthesized_calls}"
    )


def _stub_judge_match_dict(intent: str) -> dict[str, Any]:
    """Helper mirroring the platform's /v1/sdk/judge match response."""
    return {
        "match": True,
        "intent": intent,
        "confidence": 0.94,
        "tokens_in": 220,
        "tokens_out": 10,
    }


def _install_counting_judge(
    verdict_for_text: dict[str, dict[str, Any]],
) -> list[str]:
    """Same as ``_install_mock_judge`` but records every request body.

    Returns the list the handler appends to so tests can assert how
    many judge round-trips actually fired during a turn — the linchpin
    of the dedupe regression (BUG 2): when PreToolUse already gated
    each tool individually, the parent ``_run_output_phase`` MUST NOT
    re-issue a second wave of judge calls against the same tool_calls
    list. Pre-fix this test would observe 2N round-trips for an N-tool
    turn; post-fix it must observe exactly N.
    """
    import httpx as _httpx

    from egisai import _evaluator
    from egisai.policy.semantic import SemanticBlocker

    bodies: list[str] = []

    def transport_handler(request: _httpx.Request) -> _httpx.Response:
        raw = request.content.decode()
        bodies.append(raw)
        body = __import__("json").loads(raw)
        prompt_text = body.get("prompt_text", "")
        for needle, response in verdict_for_text.items():
            if needle in prompt_text:
                return _httpx.Response(200, json=response)
        return _httpx.Response(
            200,
            json={
                "match": False,
                "intent": "",
                "confidence": 0.0,
                "tokens_in": 0,
                "tokens_out": 0,
            },
        )

    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
    )
    blocker._http_client = _httpx.Client(
        transport=_httpx.MockTransport(transport_handler)
    )
    _evaluator._blocker = blocker
    return bodies


def test_semantic_guard_tool_target_does_not_double_judge_when_hooks_active(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """Regression — BUG 2 dedupe: when PreToolUse hooks are wired,
    the per-turn ``_run_output_phase`` at ``ResultMessage`` MUST NOT
    re-evaluate the accumulated tool_calls. The PreToolUse hook
    already issued an authoritative per-tool judge round-trip and
    emitted a per-tool ``tool_call`` step row apiece; running the
    same tool_calls through ``_semantic_guard_match`` a second time
    in the parent phase doubles the wall-clock policy_latency_ms
    AND the policy_tokens_* charges on the dashboard.

    Test shape:

    * Operator has a ``semantic_guard`` rule with
      ``targets=["tool_calls"]``.
    * Agent emits 3 ``ToolUseBlock``s in a single turn, none of
      which match (judge replies match=False every time).
    * After the turn, the judge MUST have been called exactly 3
      times — once per PreToolUse hook invocation — NEVER 6.

    Pre-fix this would have measured 6 judge round-trips because
    the parent ``_run_output_phase`` iterated the same 3 tool_calls
    list and re-judged each one until first match. Post-fix the
    parent phase passes ``tool_calls=[]`` whenever ``hooks_active``
    is True.
    """
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules(_semantic_tool_rule(intent="wipe the production database"))
    judge_bodies = _install_counting_judge({})  # never matches

    mod.__script__ = [
        AssistantMessage(
            [
                ToolUseBlock("Read", {"path": "/etc/hostname"}, id_="tu1"),
                ToolUseBlock("Read", {"path": "/etc/release"}, id_="tu2"),
                ToolUseBlock("Read", {"path": "/proc/cpuinfo"}, id_="tu3"),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Tell me about this machine.")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    invocations = mod.__hook_invocations__
    assert len(invocations) == 3, (
        "PreToolUse must fire exactly once per tool — sanity check "
        "before we count judge calls."
    )

    synthesized_calls = [
        b for b in judge_bodies if "invoke tool" in b
    ]
    assert len(synthesized_calls) == 3, (
        f"Pre-fix: parent _run_output_phase re-iterated tool_calls "
        f"and double-judged each, producing 6 round-trips. Post-fix: "
        f"PreToolUse alone gates tool_calls and the parent phase "
        f"skips them. Got {len(synthesized_calls)} judge calls "
        f"(expected 3): {synthesized_calls}"
    )

    # Belt-and-braces: token spend on the parent model_call step
    # should reflect ONLY the user-prompt input-phase tokens (zero
    # in this fixture — the input rule wasn't a semantic_guard rule
    # against text). Per-tool tool_call step rows carry their own
    # PreToolUse policy_tokens_* tallies. The legacy double-counting
    # showed up as inflated tokens on the model_call row; this asserts
    # the row no longer carries the doubled charge.
    model_steps = _step_events(fake_backend.events_received, kind="model_call")
    assert model_steps, "expected a model_call step on the Run"
    parent = model_steps[-1]
    assert (parent.get("policy_tokens_in") or 0) == 0, (
        "the parent model_call step must not carry tool-call judge "
        "tokens — those belong on per-tool tool_call rows"
    )
    assert (parent.get("policy_tokens_out") or 0) == 0


def test_semantic_guard_text_block_still_fires_when_hooks_active(
    fake_claude_with_hooks: tuple[Any, type, types.ModuleType],
) -> None:
    """Counter-regression for BUG 2: dropping tool_calls from the
    parent ``_run_output_phase`` MUST NOT also drop the assistant
    text signal. ``deny_output_regex`` /
    ``semantic_guard.targets=["text"]`` rules fire on the
    ``TextBlock`` content the model streamed alongside its tool
    calls — that text was NEVER gated by PreToolUse, so the parent
    phase is the only place it gets evaluated.

    Test shape:

    * Operator has a ``deny_output_regex`` rule on a sentence the
      model says.
    * Agent emits an ``AssistantMessage`` with both a ``TextBlock``
      (carrying the matching phrase) and a ``ToolUseBlock``.
    * The output phase MUST still block on the matching text even
      though ``hooks_active=True`` and tool_calls were dropped.
    """
    fake_backend, client_cls, mod = fake_claude_with_hooks
    _load_rules(
        {
            "id": "regex-out-1",
            "name": "block-output-secrets",
            "type": "deny_output_regex",
            "tenant": None,
            "config": {
                # ``deny_output_regex`` reads its regex from
                # ``config["pattern"]`` (singular). ``patterns``
                # (plural) is the schema for ``deny_tool_call`` /
                # ``deny_mcp_call`` etc; using it here would silently
                # no-op the rule.
                "pattern": r"forbidden phrase",
                "message": "Blocked output content.",
            },
        }
    )

    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock(
                    "Sure, here's the forbidden phrase you asked for."
                ),
                ToolUseBlock("Read", {"path": "/etc/hostname"}, id_="tu_x"),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Help me with something.")
            try:
                async for _ in client.receive_response():
                    pass
            except PermissionError:
                pass

    asyncio.run(run())
    _flush()

    model_steps = _step_events(fake_backend.events_received, kind="model_call")
    assert model_steps, "expected a model_call step"
    parent = model_steps[-1]
    assert parent.get("verdict") == "block", (
        "text-side rules MUST still fire from the parent output "
        "phase even when tool signals are deduped — the assistant "
        "text was never evaluated by PreToolUse."
    )
    assert parent.get("matched_policy") == "block-output-secrets"
    # When hooks were active AND tool signals appeared, an
    # output-phase block stamps ``advisory`` (subprocess already
    # ran the tool turn even though the SDK refused the response
    # at the boundary).
    assert parent.get("enforcement_status") == "advisory"
