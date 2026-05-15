"""PostToolUse hook tests for the Claude Agent SDK patch.

Pre-0.22 the ``claude_agent_sdk`` patch governed tool *inputs* via
the ``PreToolUse`` hook but did NOT govern tool *responses*. A tool
that returned a CRM record with PII (email, SSN, credit card) would
feed the raw value straight back to Claude — the only enforcement
left was end-of-turn output-text scanning, which is too late (the
PII already round-tripped the provider) and attributes the
violation to ``model_call`` rather than the offending tool.

0.22 wires a ``PostToolUse`` hook into ``ClaudeAgentOptions.hooks``
so policy decisions fire IN OUR PYTHON PROCESS, AFTER the tool
executes but BEFORE Claude is shown the result. The hook uses the
SDK's ``updatedToolOutput`` / ``updatedMCPToolOutput`` substitution
contract to swap the result in place:

- ``pii_scan`` with ``action="sanitize"`` → tool result text is
  masked via ``pii.sanitize`` and the wrapper (MCP or built-in
  shape) is reconstructed around the masked text.
- ``pii_scan`` with ``action="block"`` (or any block verdict) →
  the result is replaced with a denial payload that tells the
  model the call was refused and which policy fired.
- ``allow`` → no substitution; the tool result passes through
  unchanged (the cheap path is the common path).

These tests lock in the contract at every level:

* **Hook callback unit semantics** — allow / sanitize / block /
  identity propagation / fail-open.
* **Hook injection composition** — ``options.hooks["PostToolUse"]``
  is mutated in place, our matcher is APPENDED to any
  user-supplied matchers.
* **Tool response shapes** — MCP-style ``{content: [{type:
  "text", text: ...}]}`` AND raw string AND opaque dict are all
  scanned, and the replacement payload mirrors the original
  shape.
* **Audit trail** — when a sanitize / block fires we emit a
  ``tool_call`` step row with ``target=...tool_result``,
  ``verdict``, ``matched_policy``, ``sanitizations``, and a
  POST-redaction ``request_text`` preview.
* **Privacy contract** — raw tool-response PII NEVER lands on
  the audit row. Sampled previews always reflect the
  post-sanitize / post-denial text.

Tests stand up an in-process double of the SDK that mirrors the
bidirectional hook protocol; no real Node CLI involved.
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import pytest

# ── Test stubs mirroring real upstream class shapes ───────────────


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
    matcher: str | None = None
    hooks: list[Any] = field(default_factory=list)
    timeout: float | None = None


@dataclass
class _ClaudeAgentOptions:
    """Stand-in with ``hooks`` field — modern SDK path."""

    system_prompt: str = "You are a helpful assistant."
    allowed_tools: list[str] = field(default_factory=list)
    permission_mode: str = "auto"
    model: str = "claude-3-5-sonnet"
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    hooks: dict[str, list[_HookMatcher]] | None = None


def _install_fake_module() -> tuple[types.ModuleType, type, list[Any]]:
    """Build + register a fake ``claude_agent_sdk`` module with hooks."""
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

        # Mirror the upstream SDK pattern: ``__aenter__`` calls
        # ``connect()``, which is where the real CLI subprocess
        # would be spun up and ``options.hooks`` shipped via the
        # ``initialize`` control message. This is the seam our
        # patch wraps to inject placeholder hooks BEFORE the CLI
        # freezes its matcher table. Without this method the
        # placeholder injection never fires and the tests would
        # only exercise the broken pre-fix code path.
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
    mod.HookMatcher = _HookMatcher
    mod.ClaudeAgentOptions = _ClaudeAgentOptions
    sys.modules["claude_agent_sdk"] = mod
    return mod, _Client, captured


@pytest.fixture
def fake_claude(
    fake_backend: Any,
) -> Iterator[tuple[Any, type, types.ModuleType]]:
    """Module that exposes hooks → PostToolUse path is exercised.

    The fake transport simulates the SDK's bidirectional protocol:
    for each scripted ToolUseBlock the fake invokes EVERY registered
    PreToolUse matcher first (allowing or denying), then — if the
    pre-decision is allow — invokes EVERY registered PostToolUse
    matcher with the scripted ``tool_response`` and collects any
    ``updatedToolOutput`` / ``updatedMCPToolOutput`` so tests can
    assert end-to-end substitution semantics.
    """
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="claude-posttooluse-test",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )
    mod, client_cls, _captured = _install_fake_module()
    mod.__script__ = []
    mod.__tool_responses__ = {}  # tool_use_id → tool_response
    mod.__pre_invocations__ = []
    mod.__post_invocations__ = []
    mod.__final_outputs__ = {}  # tool_use_id → what Claude WOULD see

    async def _q(self: Any, prompt: Any, session_id: str = "default") -> None:
        self._sent.append({"prompt": prompt, "session_id": session_id})
        opts = getattr(self, "options", None)
        await _drive_hooks_for_script(
            opts,
            mod.__script__,
            mod.__tool_responses__,
            mod.__pre_invocations__,
            mod.__post_invocations__,
            mod.__final_outputs__,
        )

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
    tool_responses: dict[str, Any],
    pre_invocations: list[dict[str, Any]],
    post_invocations: list[dict[str, Any]],
    final_outputs: dict[str, Any],
) -> None:
    """Simulate the SDK's hook dispatch flow over the script.

    For each ToolUseBlock in each AssistantMessage:

    1. Invoke every registered PreToolUse callback. If ANY returns
       deny, skip PostToolUse for that tool — the tool was never
       dispatched in the subprocess so no response exists. Record
       the denial reason as the ``final_output`` so tests can
       assert it.
    2. Otherwise invoke every registered PostToolUse callback with
       the scripted ``tool_response``. Merge any
       ``updatedToolOutput`` / ``updatedMCPToolOutput`` returned —
       last-write-wins (matches the SDK's documented merge
       semantics). Record the merged output as the ``final_output``.

    All invocations are recorded in the respective lists for
    assertions.
    """
    if opts is None:
        return
    hooks = getattr(opts, "hooks", None)
    if not isinstance(hooks, dict):
        return
    pre_matchers = hooks.get("PreToolUse") or []
    post_matchers = hooks.get("PostToolUse") or []

    for msg in script:
        if type(msg).__name__ != "AssistantMessage":
            continue
        for block in getattr(msg, "content", []) or []:
            if type(block).__name__ != "ToolUseBlock":
                continue

            # ── PreToolUse ─────────────────────────────────────────
            pre_input = {
                "hook_event_name": "PreToolUse",
                "tool_name": block.name,
                "tool_input": block.input,
                "tool_use_id": block.id,
                "session_id": "default",
                "cwd": "/tmp",
                "permission_mode": "auto",
            }
            pre_denied = False
            for matcher in pre_matchers:
                for cb in getattr(matcher, "hooks", []) or []:
                    result = await cb(pre_input, block.id, {"signal": None})
                    pre_invocations.append(
                        {
                            "tool_name": block.name,
                            "tool_use_id": block.id,
                            "input": pre_input,
                            "output": result,
                        }
                    )
                    if (
                        isinstance(result, dict)
                        and result.get("hookSpecificOutput", {}).get(
                            "permissionDecision"
                        )
                        == "deny"
                    ):
                        pre_denied = True
                        final_outputs[block.id] = {
                            "denied_by_pre": True,
                            "reason": result["hookSpecificOutput"].get(
                                "permissionDecisionReason"
                            ),
                        }

            if pre_denied:
                continue

            # ── PostToolUse ────────────────────────────────────────
            tool_response = tool_responses.get(block.id)
            if tool_response is None:
                continue
            post_input = {
                "hook_event_name": "PostToolUse",
                "tool_name": block.name,
                "tool_input": block.input,
                "tool_response": tool_response,
                "tool_use_id": block.id,
                "session_id": "default",
                "cwd": "/tmp",
                "permission_mode": "auto",
            }
            current_output: Any = tool_response
            for matcher in post_matchers:
                for cb in getattr(matcher, "hooks", []) or []:
                    result = await cb(post_input, block.id, {"signal": None})
                    post_invocations.append(
                        {
                            "tool_name": block.name,
                            "tool_use_id": block.id,
                            "input": post_input,
                            "output": result,
                        }
                    )
                    spec = (
                        result.get("hookSpecificOutput")
                        if isinstance(result, dict)
                        else None
                    )
                    if isinstance(spec, dict):
                        if "updatedMCPToolOutput" in spec:
                            current_output = spec["updatedMCPToolOutput"]
                        elif "updatedToolOutput" in spec:
                            current_output = spec["updatedToolOutput"]
            final_outputs[block.id] = current_output


def _flush() -> None:
    from egisai import shutdown

    shutdown()


def _step_events(events: list[dict], *, kind: str | None = None) -> list[dict]:
    return [
        e for e in events
        if e.get("kind") == "run.step"
        and (kind is None or e.get("step_kind") == kind)
    ]


def _post_step_events(events: list[dict]) -> list[dict]:
    """Step rows attributed to a PostToolUse-fired policy decision.

    Distinguished from PreToolUse rows by ``target`` ending in
    ``tool_result`` (vs. ``tool_call``).
    """
    return [
        e for e in _step_events(events, kind="tool_call")
        if str(e.get("target", "")).endswith(".tool_result")
    ]


# ── Policy helpers ──────────────────────────────────────────────────


def _pii_scan_rule(
    *,
    action: str = "sanitize",
    types_: list[str] | None = None,
    mask_char: str = "#",
) -> dict[str, Any]:
    cfg: dict[str, Any] = {"action": action, "threshold": 0.5}
    if types_ is not None:
        cfg["types"] = types_
    if mask_char != "#":
        cfg["mask_char"] = mask_char
    return {
        "id": "pii1",
        "name": "pii-output-scan",
        "type": "pii_scan",
        "tenant": None,
        "config": cfg,
    }


def _deny_output_regex_rule(pattern: str) -> dict[str, Any]:
    return {
        "id": "dor1",
        "name": "no-secret-tokens-in-output",
        "type": "deny_output_regex",
        "tenant": None,
        "config": {"pattern": pattern},
    }


def _load_rules(*rules: dict[str, Any]) -> None:
    from egisai._policy_cache import replace_rules

    etag = f'"r{len(rules)}"'
    replace_rules(etag, list(rules))


# ── 1. Hook injection ──────────────────────────────────────────────


def test_posttooluse_hook_injected_into_options_hooks(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """When SDK supports hooks, our PostToolUse matcher is appended
    to ``options.hooks["PostToolUse"]`` alongside the PreToolUse
    matcher we already inject."""
    _, client_cls, mod = fake_claude
    mod.__script__ = [
        AssistantMessage([TextBlock("hi")]),
        ResultMessage(),
    ]

    captured: dict[str, Any] = {}

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Hello")
            captured["hooks"] = opts.hooks
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    hooks = captured["hooks"]
    assert isinstance(hooks, dict)
    assert "PreToolUse" in hooks
    assert "PostToolUse" in hooks, (
        "PostToolUse hook MUST be injected so tool results are governed"
    )
    matchers = hooks["PostToolUse"]
    assert len(matchers) == 1
    assert matchers[0].matcher is None
    assert callable(matchers[0].hooks[0])


def test_hooks_injected_at_connect_time_not_query_time(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """REGRESSION (0.22.1): hooks MUST be present in
    ``options.hooks`` BEFORE ``client.query()`` is called.

    The upstream SDK reads ``options.hooks`` exactly once inside
    ``ClaudeSDKClient.connect()`` (which ``__aenter__`` calls for
    you) and ships the matcher table to the Node.js CLI. Any
    mutation to ``options.hooks`` AFTER ``connect()`` is a silent
    no-op — the CLI's matcher table is frozen.

    In 0.22.0 the patch wired hook injection inside
    ``client.query()``, which runs AFTER ``__aenter__`` →
    ``connect()`` already shipped an empty hook table to the CLI.
    The CLI then dispatched every tool with no governance round
    trip, and tool RESULTS were never policy-evaluated. PII in
    CRM lookups, file reads, etc. round-tripped Claude unmasked
    — the exact SOC 2 / ISO 27001 gap this patch is supposed to
    close.

    This test pins down the FIX (0.22.1+): the moment
    ``__aenter__`` returns (i.e. after ``connect()`` completed
    but BEFORE any ``query()`` call), both ``PreToolUse`` and
    ``PostToolUse`` matchers MUST already be present in
    ``options.hooks``. If a refactor moves injection back into
    ``query()``, this test fails immediately."""
    _, client_cls, mod = fake_claude
    mod.__script__ = [
        AssistantMessage([TextBlock("hi")]),
        ResultMessage(),
    ]

    captured: dict[str, Any] = {}

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            # Snapshot BEFORE the first query() call. If the
            # patch is correct, hooks are already wired here.
            captured["hooks_after_connect_before_query"] = opts.hooks
            await client.query("Hello")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    hooks = captured["hooks_after_connect_before_query"]
    assert isinstance(hooks, dict), (
        "options.hooks MUST be a dict after __aenter__/connect() — "
        "the patch must inject placeholder hooks BEFORE the CLI's "
        "matcher table is frozen, NOT during query()"
    )
    assert "PreToolUse" in hooks, (
        "PreToolUse hook MUST be in options.hooks immediately "
        "after connect(); otherwise the CLI dispatches tools "
        "with no governance"
    )
    assert "PostToolUse" in hooks, (
        "PostToolUse hook MUST be in options.hooks immediately "
        "after connect(); otherwise tool results round-trip "
        "the model unmasked"
    )
    # Verify these are real callables, not None/skeletons.
    pre_matchers = hooks["PreToolUse"]
    post_matchers = hooks["PostToolUse"]
    assert pre_matchers and callable(pre_matchers[0].hooks[0])
    assert post_matchers and callable(post_matchers[0].hooks[0])


def test_posttooluse_composes_with_existing_user_hooks(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """User-supplied PostToolUse hooks remain in place; our matcher
    is APPENDED to the list, not substituted for it."""
    _, client_cls, mod = fake_claude
    mod.__script__ = [
        AssistantMessage([TextBlock("hi")]),
        ResultMessage(),
    ]

    user_post_calls: list[str] = []

    async def user_post(hook_input: Any, _tuid: Any, _ctx: Any) -> dict:
        user_post_calls.append(hook_input.get("tool_name", ""))
        return {}

    captured: dict[str, Any] = {}

    async def run() -> None:
        user_matcher = _HookMatcher(matcher="Bash", hooks=[user_post])
        opts = _ClaudeAgentOptions(hooks={"PostToolUse": [user_matcher]})
        async with client_cls(options=opts) as client:
            await client.query("Hi")
            captured["hooks"] = opts.hooks
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    matchers = captured["hooks"]["PostToolUse"]
    assert len(matchers) == 2
    assert matchers[0].matcher == "Bash"
    assert matchers[0].hooks == [user_post]
    assert matchers[1].matcher is None  # ours appended


# ── 2. PII detection in tool result — MCP shape ─────────────────────


def test_pii_in_mcp_tool_result_is_sanitized_via_updatedMCPToolOutput(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """The headline SOC 2 scenario from the user's bug report:

    A CRM lookup MCP tool returns ``{"content": [{"type": "text",
    "text": "...email: maria.g@techcorp.io..."}]}``. The PostToolUse
    hook MUST detect the email, mask it via ``pii.sanitize``, and
    return ``updatedMCPToolOutput`` with the masked content so the
    raw email never reaches Claude.
    """
    fake_backend, client_cls, mod = fake_claude
    _load_rules(_pii_scan_rule(action="sanitize", types_=["email"]))
    mod.__script__ = [
        AssistantMessage(
            [
                ToolUseBlock(
                    "mcp__support__lookup_customer",
                    {"query": "ACC-2847193"},
                    id_="tu_lookup",
                ),
            ]
        ),
        ResultMessage(),
    ]
    mod.__tool_responses__["tu_lookup"] = {
        "content": [
            {
                "type": "text",
                "text": (
                    '{"customer_id": "ACC-2847193", '
                    '"name": "Maria Gonzalez", '
                    '"email": "maria.g@techcorp.io"}'
                ),
            }
        ]
    }

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Look up account ACC-2847193")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    # The PostToolUse hook fired exactly once for this tool.
    posts = mod.__post_invocations__
    assert len(posts) == 1
    spec = posts[0]["output"].get("hookSpecificOutput") or {}
    assert spec.get("hookEventName") == "PostToolUse"
    assert "updatedMCPToolOutput" in spec, (
        "MCP-shaped responses MUST be replaced via updatedMCPToolOutput, "
        "not updatedToolOutput"
    )

    # The replacement payload preserves the MCP shape.
    replaced = spec["updatedMCPToolOutput"]
    assert isinstance(replaced, dict)
    assert isinstance(replaced.get("content"), list)
    assert replaced["content"][0]["type"] == "text"
    masked = replaced["content"][0]["text"]
    assert "maria.g@techcorp.io" not in masked, (
        "raw email MUST NOT survive sanitization"
    )
    # The non-PII fields survive — the agent can still reason about them.
    assert "ACC-2847193" in masked
    # Sanitizer's exact replacement varies between Presidio (mask_char
    # multiplication) and the regex fallback (label-style
    # ``[email-redacted]``). Both are acceptable — what we assert is
    # that the raw value is gone AND something visible took its place
    # (so the model knows the field existed but was redacted).
    assert masked != (
        '{"customer_id": "ACC-2847193", '
        '"name": "Maria Gonzalez", '
        '"email": "maria.g@techcorp.io"}'
    )

    # Audit row stamped with the right verdict and shape.
    post_steps = _post_step_events(fake_backend.events_received)
    assert len(post_steps) == 1
    step = post_steps[0]
    assert step["verdict"] == "sanitize"
    assert step["enforcement_status"] == "enforced"
    assert step["tool_name"] == "mcp__support__lookup_customer"
    # Sanitizations audit records the type + count, never the value.
    sans = step.get("sanitizations") or []
    assert any(s.get("type") == "email" for s in sans)
    assert all("maria.g@techcorp.io" not in (s.get("pattern") or "") for s in sans)


def test_pii_in_mcp_tool_result_blocks_when_action_is_block(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """When the operator's ``pii_scan`` is configured with
    ``action="block"``, the PostToolUse hook MUST replace the tool
    result with a denial payload (not just mask) — the model sees a
    refusal it can recover from."""
    fake_backend, client_cls, mod = fake_claude
    _load_rules(_pii_scan_rule(action="block", types_=["ssn"]))
    mod.__script__ = [
        AssistantMessage(
            [
                ToolUseBlock(
                    "mcp__support__lookup_customer",
                    {"query": "ACC-1"},
                    id_="tu_block",
                ),
            ]
        ),
        ResultMessage(),
    ]
    raw_ssn = "123-45-6789"
    mod.__tool_responses__["tu_block"] = {
        "content": [
            {"type": "text", "text": f"customer SSN is {raw_ssn}"}
        ]
    }

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("look it up")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    posts = mod.__post_invocations__
    assert len(posts) == 1
    spec = posts[0]["output"]["hookSpecificOutput"]
    replaced = spec["updatedMCPToolOutput"]
    replacement_text = replaced["content"][0]["text"]
    # Denial payload mentions the tool + the policy so the model can recover.
    assert raw_ssn not in replacement_text
    assert "withheld" in replacement_text.lower()
    assert "pii-output-scan" in replacement_text

    post_steps = _post_step_events(fake_backend.events_received)
    assert len(post_steps) == 1
    assert post_steps[0]["verdict"] == "block"
    assert post_steps[0]["enforcement_status"] == "enforced"
    assert post_steps[0]["matched_policy"] == "pii-output-scan"
    # SOC 2 audit: raw PII never lands on the audit row.
    preview = post_steps[0].get("request_text") or ""
    assert raw_ssn not in preview


# ── 3. Non-MCP tool result shapes (Bash / Read / opaque dict) ───────


def test_string_tool_response_replaced_via_updatedToolOutput(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """A built-in tool (Bash, Read, etc.) whose response is a raw
    string MUST be replaced via ``updatedToolOutput`` (NOT
    ``updatedMCPToolOutput``)."""
    fake_backend, client_cls, mod = fake_claude
    _load_rules(_pii_scan_rule(action="sanitize", types_=["ssn"]))
    mod.__script__ = [
        AssistantMessage(
            [
                ToolUseBlock(
                    "Bash",
                    {"command": "grep ssn /tmp/log"},
                    id_="tu_bash",
                ),
            ]
        ),
        ResultMessage(),
    ]
    raw_ssn = "555-12-3456"
    mod.__tool_responses__["tu_bash"] = (
        f"found record: SSN={raw_ssn}\n"
    )

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("scan logs")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    spec = mod.__post_invocations__[0]["output"]["hookSpecificOutput"]
    assert "updatedToolOutput" in spec, (
        "non-MCP tool result MUST use updatedToolOutput field"
    )
    assert "updatedMCPToolOutput" not in spec
    replaced = spec["updatedToolOutput"]
    assert isinstance(replaced, str)
    assert raw_ssn not in replaced


def test_opaque_dict_tool_response_serialized_and_replaced(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """An opaque non-MCP dict response (some custom tool) MUST still
    get scanned — the extractor falls back to JSON serialization
    and the replacement returns through ``updatedToolOutput``."""
    fake_backend, client_cls, mod = fake_claude
    _load_rules(_pii_scan_rule(action="block", types_=["ssn"]))
    mod.__script__ = [
        AssistantMessage(
            [ToolUseBlock("custom_tool", {}, id_="tu_custom")]
        ),
        ResultMessage(),
    ]
    # NB: SSNs starting with 9 are invalid per SSA rules and the
    # detector correctly skips them. Use a valid prefix so the test
    # exercises the block path rather than a silent allow.
    raw_ssn = "456-78-1234"
    mod.__tool_responses__["tu_custom"] = {
        "rows": [{"ssn": raw_ssn, "id": 1}],
        "total": 1,
    }

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("query")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    spec = mod.__post_invocations__[0]["output"]["hookSpecificOutput"]
    assert "updatedToolOutput" in spec
    replaced = spec["updatedToolOutput"]
    # Denial payload, raw SSN gone.
    assert raw_ssn not in str(replaced)


# ── 4. Allow path — cheap, no substitution, no extra audit row ──────


def test_allow_path_no_substitution_no_extra_step(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """When the PostToolUse evaluator returns allow (no PII / no
    matching rule), the callback MUST return ``{}`` so the SDK
    passes the original response through unchanged, AND it MUST
    NOT emit a tool_result step row (avoids audit churn for the
    99% allow path)."""
    fake_backend, client_cls, mod = fake_claude
    _load_rules(_pii_scan_rule(action="block", types_=["email"]))
    mod.__script__ = [
        AssistantMessage(
            [ToolUseBlock("Read", {"path": "/tmp/x"}, id_="tu_clean")]
        ),
        ResultMessage(),
    ]
    mod.__tool_responses__["tu_clean"] = {
        "content": [{"type": "text", "text": "Plain log line, no PII"}]
    }

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("read")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    posts = mod.__post_invocations__
    assert len(posts) == 1
    assert posts[0]["output"] == {}, "allow → no substitution"

    # PreToolUse emits 1 row (verdict=allow). PostToolUse emits 0.
    post_steps = _post_step_events(fake_backend.events_received)
    assert len(post_steps) == 0


def test_empty_response_skipped(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """A tool that returns nothing (None / empty content list) MUST
    NOT crash the hook and MUST NOT emit an audit row."""
    fake_backend, client_cls, mod = fake_claude
    _load_rules(_pii_scan_rule(action="block"))
    mod.__script__ = [
        AssistantMessage(
            [ToolUseBlock("Read", {"path": "/tmp/y"}, id_="tu_empty")]
        ),
        ResultMessage(),
    ]
    mod.__tool_responses__["tu_empty"] = {"content": []}

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("read")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    posts = mod.__post_invocations__
    assert len(posts) == 1
    assert posts[0]["output"] == {}
    assert len(_post_step_events(fake_backend.events_received)) == 0


# ── 5. deny_output_regex on tool result ─────────────────────────────


def test_deny_output_regex_fires_on_tool_result(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """``deny_output_regex`` scans tool result text — useful for
    catching secrets / API keys / proprietary identifiers that
    aren't standard PII types."""
    fake_backend, client_cls, mod = fake_claude
    _load_rules(_deny_output_regex_rule(r"sk-[A-Za-z0-9]{20,}"))
    mod.__script__ = [
        AssistantMessage(
            [ToolUseBlock("Read", {"path": "/tmp/k"}, id_="tu_key")]
        ),
        ResultMessage(),
    ]
    mod.__tool_responses__["tu_key"] = (
        "API_KEY=sk-abcdefghijklmnopqrstuvwxyz"
    )

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("read keys")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    spec = mod.__post_invocations__[0]["output"]["hookSpecificOutput"]
    # Replaced — string shape → updatedToolOutput
    assert "updatedToolOutput" in spec
    assert "sk-abcdefghij" not in str(spec["updatedToolOutput"])

    post_steps = _post_step_events(fake_backend.events_received)
    assert len(post_steps) == 1
    assert post_steps[0]["verdict"] == "block"
    assert post_steps[0]["matched_policy"] == "no-secret-tokens-in-output"


# ── 6. Multi-tool turn — independent gating per tool ────────────────


def test_multi_tool_each_response_independently_gated(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """Three tools fire in one assistant turn:
    - tool A returns PII → sanitize
    - tool B returns clean text → allow (no substitution)
    - tool C returns a secret → block

    The audit trail MUST carry exactly two PostToolUse step rows
    (A and C), correctly attributed by tool_name."""
    fake_backend, client_cls, mod = fake_claude
    _load_rules(
        _pii_scan_rule(action="sanitize", types_=["ssn"]),
        _deny_output_regex_rule(r"SECRET-[0-9]+"),
    )
    mod.__script__ = [
        AssistantMessage(
            [
                ToolUseBlock("tool_a", {}, id_="tu_a"),
                ToolUseBlock("tool_b", {}, id_="tu_b"),
                ToolUseBlock("tool_c", {}, id_="tu_c"),
            ]
        ),
        ResultMessage(),
    ]
    mod.__tool_responses__["tu_a"] = "user ssn is 111-22-3333"
    mod.__tool_responses__["tu_b"] = "all good, no sensitive data"
    mod.__tool_responses__["tu_c"] = "ID is SECRET-9876"

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("do it")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    posts = mod.__post_invocations__
    assert len(posts) == 3
    outputs = {p["tool_use_id"]: p["output"] for p in posts}
    assert outputs["tu_a"].get("hookSpecificOutput", {}).get(
        "updatedToolOutput"
    )
    assert outputs["tu_b"] == {}
    assert outputs["tu_c"].get("hookSpecificOutput", {}).get(
        "updatedToolOutput"
    )

    post_steps = _post_step_events(fake_backend.events_received)
    by_tool = {s["tool_name"]: s for s in post_steps}
    assert set(by_tool.keys()) == {"tool_a", "tool_c"}, (
        "tool_b returned no PII / no match — no PostToolUse step row"
    )
    assert by_tool["tool_a"]["verdict"] == "sanitize"
    assert by_tool["tool_c"]["verdict"] == "block"


# ── 7. PreToolUse + PostToolUse interplay ───────────────────────────


def test_pre_denied_skips_post(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """If PreToolUse denies, the subprocess never dispatches the
    tool → there's no response → PostToolUse never fires for this
    tool. The audit trail has exactly one row (the PreToolUse
    block), not two."""
    fake_backend, client_cls, mod = fake_claude
    _load_rules(
        {
            "id": "rt1",
            "name": "block-shell",
            "type": "deny_tool_call",
            "tenant": None,
            "config": {"patterns": [r"^run_shell$"]},
        },
        _pii_scan_rule(action="sanitize"),
    )
    mod.__script__ = [
        AssistantMessage(
            [ToolUseBlock("run_shell", {"cmd": "x"}, id_="tu_denied")]
        ),
        ResultMessage(),
    ]
    mod.__tool_responses__["tu_denied"] = "should never reach here"

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("clean up")
            try:
                async for _ in client.receive_response():
                    pass
            except PermissionError:
                pass

    asyncio.run(run())
    _flush()

    assert len(mod.__post_invocations__) == 0, (
        "PostToolUse MUST NOT fire when PreToolUse already denied "
        "the tool"
    )

    post_steps = _post_step_events(fake_backend.events_received)
    assert len(post_steps) == 0


# ── 8. Identity propagation into PostToolUse closure ────────────────


def test_posttooluse_carries_identity_into_step_row(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """The PostToolUse callback runs on a separate asyncio task —
    identity contextvars don't propagate automatically. The
    closure must enter ``identity_scope(record)`` so the
    tool_result step's ``agent_id`` matches the registered agent.
    """
    fake_backend, client_cls, mod = fake_claude
    _load_rules(_pii_scan_rule(action="block", types_=["ssn"]))
    mod.__script__ = [
        AssistantMessage([ToolUseBlock("t", {}, id_="tu_id")]),
        ResultMessage(),
    ]
    mod.__tool_responses__["tu_id"] = "ssn: 222-33-4444"

    async def run() -> None:
        opts = _ClaudeAgentOptions(
            system_prompt="You are a tax filing specialist.",
            allowed_tools=["t"],
        )
        async with client_cls(options=opts) as client:
            await client.query("Process")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    post_steps = _post_step_events(fake_backend.events_received)
    assert len(post_steps) == 1
    assert post_steps[0]["agent_id"], "agent_id must be non-empty"
    # Pre + post share the same agent_id (identity locks at run open).
    pre_steps = [
        e for e in _step_events(fake_backend.events_received, kind="tool_call")
        if str(e.get("target", "")).endswith(".tool_call")
    ]
    if pre_steps:
        assert pre_steps[0]["agent_id"] == post_steps[0]["agent_id"]


# ── 9. Fail-open on policy crash ────────────────────────────────────


def test_posttooluse_fails_open_when_eval_raises(
    fake_claude: tuple[Any, type, types.ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``evaluate_output`` raises during PostToolUse (buggy
    policy, corrupted rule), the hook MUST return ``{}`` rather
    than denying or crashing. The user's agent must keep working
    even when our policy engine has a bad day.
    """
    _, client_cls, mod = fake_claude

    real = sys.modules["egisai._patches.claude_agent_sdk"].evaluate_output

    def _explode(call: Any) -> Any:
        # Only blow up on the PostToolUse path (target ends with
        # .tool_result). PreToolUse must still work or we'd fail
        # the wrong invariant.
        if str(getattr(call, "target", "")).endswith(".tool_result"):
            raise RuntimeError("simulated policy crash")
        return real(call)

    monkeypatch.setattr(
        "egisai._patches.claude_agent_sdk.evaluate_output", _explode
    )

    mod.__script__ = [
        AssistantMessage([ToolUseBlock("t", {}, id_="tu_crash")]),
        ResultMessage(),
    ]
    mod.__tool_responses__["tu_crash"] = "any data"

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("ok")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    posts = mod.__post_invocations__
    assert len(posts) == 1
    assert posts[0]["output"] == {}, (
        "policy crash → fail OPEN. The agent's tool result must "
        "pass through unchanged."
    )


# ── 10. Privacy contract — raw PII never on audit ───────────────────


def test_audit_row_preview_is_post_sanitize(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """``security-and-compliance.mdc`` §1 + §5: the audit row's
    ``request_text`` preview MUST be sampled from the
    post-sanitize / post-denial text, never the raw tool
    response."""
    fake_backend, client_cls, mod = fake_claude
    raw_ssn = "777-88-9999"
    _load_rules(_pii_scan_rule(action="sanitize", types_=["ssn"]))
    mod.__script__ = [
        AssistantMessage([ToolUseBlock("t", {}, id_="tu_priv")]),
        ResultMessage(),
    ]
    mod.__tool_responses__["tu_priv"] = f"customer ssn: {raw_ssn}"

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("ok")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    post_steps = _post_step_events(fake_backend.events_received)
    assert len(post_steps) == 1
    step = post_steps[0]
    preview = step.get("request_text") or ""
    assert raw_ssn not in preview, (
        "Privacy contract: raw PII MUST NOT appear in the "
        "audit row preview"
    )

    # Also verify it's not in the sanitizations entries (count/type only).
    for s in step.get("sanitizations") or []:
        assert raw_ssn not in str(s)


# ── 11. MCP image-only response — non-text parts survive ────────────


def test_mcp_response_with_image_only_skipped(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """A tool that returns only image content (no text part) has
    nothing to scan — the hook MUST return ``{}`` and the
    response passes through untouched."""
    fake_backend, client_cls, mod = fake_claude
    _load_rules(_pii_scan_rule(action="sanitize"))
    mod.__script__ = [
        AssistantMessage([ToolUseBlock("screenshot", {}, id_="tu_img")]),
        ResultMessage(),
    ]
    mod.__tool_responses__["tu_img"] = {
        "content": [{"type": "image", "data": "base64-blob"}]
    }

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("snap")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    posts = mod.__post_invocations__
    assert len(posts) == 1
    assert posts[0]["output"] == {}
    assert len(_post_step_events(fake_backend.events_received)) == 0


# ── 12. Multi-text MCP single-replace (regression invariant) ───────


def test_mcp_response_with_multiple_text_parts_collapses_to_one(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """An MCP tool returns three ``{type: "text"}`` parts, two
    carrying PII. After PostToolUse sanitize, the replacement
    payload MUST contain a SINGLE text part with the post-sanitize
    concatenation — never multiple text parts that re-introduce the
    raw PII.

    Why this matters: pre-0.22.x the rewriter walked the content
    list and replaced each text part independently. If
    ``pii.sanitize`` was called per-part on un-joined chunks the
    detector could miss spans that crossed part boundaries (a SSN
    split across two adjacent text parts), and the audit row's
    sanitization counts wouldn't reflect the global view. 0.22.x
    settled on "extract ALL text -> scan as one blob -> drop every
    text part except the first, which gets the post-scan
    replacement"; this test pins that contract.

    Non-text MCP parts (image, audio) MUST survive the rewrite
    unchanged — they're independently consumable by the model and
    can't be PII-scanned anyway.
    """
    fake_backend, client_cls, mod = fake_claude
    raw_a = "234-56-7891"
    raw_b = "345-67-8901"
    _load_rules(_pii_scan_rule(action="sanitize", types_=["ssn"]))
    mod.__script__ = [
        AssistantMessage([ToolUseBlock("lookup", {}, id_="tu_multi")]),
        ResultMessage(),
    ]
    mod.__tool_responses__["tu_multi"] = {
        "content": [
            {"type": "text", "text": f"Customer A SSN {raw_a}"},
            {"type": "image", "data": "base64-blob"},
            {"type": "text", "text": f"Customer B SSN {raw_b}"},
            {"type": "text", "text": "End of report."},
        ]
    }

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("audit")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    posts = mod.__post_invocations__
    assert len(posts) == 1
    spec = posts[0]["output"]["hookSpecificOutput"]
    assert "updatedMCPToolOutput" in spec, (
        "MCP-shape responses must use updatedMCPToolOutput, not "
        f"updatedToolOutput: keys={list(spec)}"
    )
    new_doc = spec["updatedMCPToolOutput"]
    new_content = new_doc["content"]

    text_parts = [p for p in new_content if p.get("type") == "text"]
    assert len(text_parts) == 1, (
        f"multi-text MCP rewrite produced {len(text_parts)} text parts; "
        f"contract is exactly 1. content={new_content!r}"
    )

    image_parts = [p for p in new_content if p.get("type") == "image"]
    assert image_parts == [
        {"type": "image", "data": "base64-blob"}
    ], f"image part lost or corrupted: {image_parts!r}"

    rendered = repr(new_doc)
    assert raw_a not in rendered, (
        f"raw SSN A leaked into MCP replacement: {rendered!r}"
    )
    assert raw_b not in rendered, (
        f"raw SSN B leaked into MCP replacement: {rendered!r}"
    )

    post_steps = _post_step_events(fake_backend.events_received)
    assert len(post_steps) == 1
    sans = post_steps[0].get("sanitizations") or []
    ssn_sans = [s for s in sans if s.get("type") == "ssn"]
    assert ssn_sans, f"no ssn sanitization recorded: {sans!r}"
    assert ssn_sans[0]["count"] == 2, (
        f"multi-part scan missed a span: counts={ssn_sans!r}"
    )
