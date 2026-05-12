"""End-to-end governance tests for the Claude Agent SDK patch.

The 0.17.5 patch wrapped only the identity boundary on
``ClaudeSDKClient.query``; no audit row was ever produced and no
input/output policy fired. This was the bug the user reported as
"agents register on the dashboard but no requests come through and no
policies are enforced over MCP / tool calls".

0.17.6 governs the Python-visible boundary instead. The tests here
lock in three contracts:

1. **Audit emission** — every ``query`` call produces exactly one
   audit row (input verdict on ``query``, finalized on the trailing
   ``ResultMessage`` from ``receive_messages``).
2. **Input policies fire** — ``deny_regex`` / ``pii_scan`` / etc.
   run on the prompt; sanitization mutates the forwarded prompt;
   block raises ``PermissionError`` and forwards nothing.
3. **Output policies fire** — ``deny_tool_call``,
   ``deny_mcp_call``, and ``deny_output_regex`` run on the
   accumulated stream signals (text + tool names + MCP targets)
   and stamp the audit row.

Because the real upstream pipes JSON to a Node.js subprocess, we
hand-roll a faithful in-process double: ``ClaudeSDKClient`` with
``options``, ``query`` (coroutine), and ``receive_messages``
(``async def … yield``) plus ``AssistantMessage`` / ``TextBlock``
/ ``ToolUseBlock`` / ``ResultMessage``. The shapes mirror the real
package's attribute names (``content`` is a list of blocks,
``ToolUseBlock.name`` / ``.input``, ``ResultMessage.usage`` is
``{"input_tokens": …, "output_tokens": …}``).
"""

from __future__ import annotations

import asyncio
import sys
import types
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest

# ── Test stubs that mirror real upstream class shapes ───────────────


# NB: the class NAMES below (``TextBlock``, ``ToolUseBlock``,
# ``AssistantMessage``, ``ResultMessage``) must match the real
# upstream class names exactly — the patch duck-types on
# ``type(message).__name__`` so renaming these will silently break
# the output gate.


class TextBlock:
    """Real claude_agent_sdk shape: ``content`` block with ``.text``."""

    def __init__(self, text: str) -> None:
        self.text = text


class ToolUseBlock:
    """Real claude_agent_sdk shape: ``.name``, ``.input``, ``.id``."""

    def __init__(self, name: str, input_: dict[str, Any] | None = None) -> None:
        self.name = name
        self.input = input_ or {}
        self.id = f"tool_{name}_001"


class AssistantMessage:
    """One assistant turn carrying TextBlock / ToolUseBlock children."""

    def __init__(self, content: list[Any]) -> None:
        self.content = content


class ResultMessage:
    """End-of-turn marker carrying tokens + cost."""

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
    """ClaudeAgentOptions stand-in (matches attribute names)."""

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


def _install_fake_module(
    script: list[Any],
) -> tuple[types.ModuleType, type, list[Any]]:
    """Build + register a fake ``claude_agent_sdk`` module.

    ``script`` is the list of ``Message`` objects the fake client
    yields from ``receive_messages``. The same list backs every
    invocation; tests that need different scripts per call should
    instantiate one client per call.
    """
    mod = types.ModuleType("claude_agent_sdk")
    captured: list[Any] = []

    async def _module_query(prompt: Any, options: Any = None) -> AsyncIterator[Any]:
        captured.append({"prompt": prompt, "options": options})
        for msg in script:
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
            # Real upstream is ``async def → None`` (sends prompt to
            # subprocess and returns; iteration is via
            # receive_messages).
            self._sent.append({"prompt": prompt, "session_id": session_id})

        async def receive_messages(self) -> AsyncIterator[Any]:
            for msg in script:
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
    sys.modules["claude_agent_sdk"] = mod
    return mod, _Client, captured


@pytest.fixture
def fake_claude(
    fake_backend: Any,
) -> Iterator[
    tuple[Any, type, types.ModuleType]
]:
    """Yield a (fake_backend, ClientClass, module) tuple.

    The script can be set per-test by replacing ``module.__script__``
    BEFORE instantiating the client.
    """

    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="claude-test",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        on_block="raise",
    )
    mod, client_cls, captured = _install_fake_module([])
    mod.__script__ = []  # tests assign to this
    mod.__captured__ = captured

    # Make every client read the latest script at iteration time.
    async def _q(self: Any, prompt: Any, session_id: str = "default") -> None:
        self._sent.append({"prompt": prompt, "session_id": session_id})

    async def _rm(self: Any) -> AsyncIterator[Any]:
        for msg in mod.__script__:
            yield msg

    client_cls.query = _q
    client_cls.receive_messages = _rm

    # Now apply the patch to the freshly-installed module.
    from egisai._patches import claude_agent_sdk

    assert claude_agent_sdk.apply() is True

    yield fake_backend, client_cls, mod
    sys.modules.pop("claude_agent_sdk", None)


def _flush() -> None:
    """Drain the SDK's logger queue so events_received is populated."""
    from egisai import shutdown

    shutdown()


def _step_events(events: list[dict]) -> list[dict]:
    """Return only the model-step events from a wire-event stream.

    Since 0.18.0 the SDK wraps every audit row in a ``run.start`` /
    ``run.step`` / ``run.end`` envelope so the dashboard can render
    live timelines. Test assertions about "the audit row" should
    look at the step event(s), not the envelope events.
    """
    return [
        e for e in events
        if e.get("kind") == "run.step" and e.get("step_kind") == "model_call"
    ]


# ── 1. Audit emission contract ──────────────────────────────────────


def test_query_emits_one_audit_row_per_turn(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """Allow path: one ``query`` + one ``receive_messages`` produces
    exactly one audit row with both phases populated."""
    fake_backend, client_cls, mod = fake_claude
    mod.__script__ = [
        AssistantMessage([TextBlock("Hello, how can I help?")]),
        ResultMessage(input_tokens=10, output_tokens=20, cost_usd=0.005),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("What's the weather?")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    events = _step_events(fake_backend.events_received)
    assert len(events) == 1, f"expected 1 event, got {len(events)}"
    ev = events[0]
    assert ev["source"] == "claude_agent_sdk"
    assert ev["model"] == "claude-3-5-sonnet"
    assert ev["verdict"] == "allow"
    assert ev["prompt_chars"] == len("What's the weather?")
    assert ev["tokens_in"] == 10
    assert ev["tokens_out"] == 20
    assert ev["cost_usd"] == pytest.approx(0.005, rel=1e-6)
    assert ev["latency_ms"] >= 0
    assert ev.get("agent_id"), "audit row must carry the resolved agent_id"


def test_query_audit_row_carries_identity(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """Tier-2B identity (system_prompt + tools + model bundle) lands
    on the audit row's ``agent_id`` field."""
    fake_backend, client_cls, mod = fake_claude
    mod.__script__ = [
        AssistantMessage([TextBlock("ok")]),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(
            options=_Options(
                system_prompt="You are a tax filing agent.",
                allowed_tools=["Read", "Bash"],
            )
        ) as client:
            await client.query("Hi")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    ev = _step_events(fake_backend.events_received)[0]
    assert ev["agent_id"] != ""
    # Ensure the agent registered through the ensure pipe.
    assert len(fake_backend.ensured_agents) >= 1


# ── 2. Input-policy enforcement ─────────────────────────────────────


def _deny_regex_rule(pattern: str = r"forbidden") -> dict[str, Any]:
    return {
        "id": "r1",
        "name": "block-forbidden-prompt",
        "type": "deny_regex",
        "tenant": None,
        "config": {"pattern": pattern, "message": "forbidden term"},
    }


def _pii_sanitize_rule() -> dict[str, Any]:
    return {
        "id": "r2",
        "name": "sanitize-pii",
        "type": "pii_scan",
        "tenant": None,
        "config": {"action": "sanitize"},
    }


def _pii_block_rule() -> dict[str, Any]:
    return {
        "id": "r3",
        "name": "block-pii",
        "type": "pii_scan",
        "tenant": None,
        "config": {"action": "block"},
    }


def test_input_policy_block_raises_and_emits_event(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """``deny_regex`` on the prompt: ``query`` must raise
    ``PermissionError`` BEFORE the subprocess sees the prompt, and
    an audit row must land with ``verdict=block``."""
    fake_backend, client_cls, mod = fake_claude
    fake_backend.set_rules([_deny_regex_rule()], etag='"deny-regex"')
    from egisai._policy_cache import replace_rules

    replace_rules('"deny-regex"', [_deny_regex_rule()])

    mod.__script__ = [
        AssistantMessage([TextBlock("never reached")]),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            with pytest.raises(PermissionError):
                await client.query("This message contains forbidden text.")

    asyncio.run(run())
    _flush()

    events = _step_events(fake_backend.events_received)
    assert len(events) == 1
    ev = events[0]
    assert ev["verdict"] == "block"
    assert ev["matched_policy"] == "block-forbidden-prompt"
    # Subprocess (or the fake's ``_sent`` list) must NOT have
    # received the prompt — the block fired pre-forward.
    # ``client._sent`` is internal but a useful invariant to pin.


def test_input_policy_sanitize_mutates_forwarded_prompt(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """``pii_scan`` action=sanitize replaces the prompt with masked
    text before forwarding to the subprocess + records the
    sanitization on the audit row.

    We pin SSN masking specifically (the default pii_scan kind list
    is conservative; what matters is that *some* masking happens and
    the masked copy — not the original — is what reaches the
    subprocess. Per security-and-compliance.mdc §1 raw PII must
    never leave the SDK boundary.)
    """
    fake_backend, client_cls, mod = fake_claude
    fake_backend.set_rules([_pii_sanitize_rule()], etag='"san-pii"')
    from egisai._policy_cache import replace_rules

    replace_rules('"san-pii"', [_pii_sanitize_rule()])

    mod.__script__ = [
        AssistantMessage([TextBlock("ok")]),
        ResultMessage(),
    ]

    raw_ssn = "123-45-6789"
    captured_client: dict[str, Any] = {}

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            captured_client["c"] = client
            await client.query(
                f"Patient SSN {raw_ssn} requires a follow-up call."
            )
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    client = captured_client["c"]
    assert len(client._sent) == 1
    sent_prompt = client._sent[0]["prompt"]
    assert raw_ssn not in sent_prompt, (
        f"raw SSN must never reach the subprocess (got {sent_prompt!r})"
    )
    assert "#" in sent_prompt or "*" in sent_prompt, (
        "masked output must contain a mask character"
    )

    ev = _step_events(fake_backend.events_received)[0]
    assert ev["verdict"] == "sanitize"
    assert ev.get("sanitizations"), "sanitize verdict must record details"


def test_input_policy_block_short_circuits_subprocess(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """Input-side block must never write the prompt to the
    subprocess (the subprocess would otherwise still send the raw
    text to Anthropic — a leak)."""
    fake_backend, client_cls, mod = fake_claude
    fake_backend.set_rules([_pii_block_rule()], etag='"block-pii"')
    from egisai._policy_cache import replace_rules

    replace_rules('"block-pii"', [_pii_block_rule()])

    captured_client: dict[str, Any] = {}

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            captured_client["c"] = client
            with pytest.raises(PermissionError):
                await client.query(
                    "Patient SSN: 123-45-6789, MRN: AB-991-22"
                )

    asyncio.run(run())
    _flush()

    client = captured_client["c"]
    assert client._sent == [], (
        "blocked prompt must never reach the subprocess"
    )
    ev = _step_events(fake_backend.events_received)[0]
    assert ev["verdict"] == "block"


# ── 3. Output-policy enforcement on streamed messages ───────────────


def _deny_tool_rule() -> dict[str, Any]:
    return {
        "id": "r4",
        "name": "block-shell",
        "type": "deny_tool_call",
        "tenant": None,
        "config": {"patterns": [r"^run_shell$"]},
    }


def _deny_mcp_rule() -> dict[str, Any]:
    return {
        "id": "r5",
        "name": "block-prod-mcp",
        "type": "deny_mcp_call",
        "tenant": None,
        "config": {"patterns": [r"^prod_"]},
    }


def _deny_output_regex_rule() -> dict[str, Any]:
    return {
        "id": "r6",
        "name": "block-secret",
        "type": "deny_output_regex",
        "tenant": None,
        "config": {"pattern": r"sk-[A-Za-z0-9]{16,}"},
    }


def test_output_policy_blocks_on_tool_call(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """``ToolUseBlock`` with a banned name (``run_shell``) makes the
    output phase stamp ``verdict=block`` on the audit row + raise
    once the ``ResultMessage`` arrives."""
    fake_backend, client_cls, mod = fake_claude
    fake_backend.set_rules([_deny_tool_rule()], etag='"deny-tool"')
    from egisai._policy_cache import replace_rules

    replace_rules('"deny-tool"', [_deny_tool_rule()])

    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock("I'll run a shell command."),
                ToolUseBlock("run_shell", {"cmd": "rm -rf /"}),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("Clean up the system.")
            with pytest.raises(PermissionError):
                async for _ in client.receive_response():
                    pass

    asyncio.run(run())
    _flush()

    ev = _step_events(fake_backend.events_received)[0]
    assert ev["verdict"] == "block"
    assert ev["matched_policy"] == "block-shell"
    assert "response_decision" in ev
    assert ev["response_decision"]["verdict"] == "block"


def test_output_policy_blocks_on_mcp_call(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """MCP tool names are namespaced ``mcp__<server>__<tool>``;
    ``deny_mcp_call`` matches on the ``<server>`` portion."""
    fake_backend, client_cls, mod = fake_claude
    fake_backend.set_rules([_deny_mcp_rule()], etag='"deny-mcp"')
    from egisai._policy_cache import replace_rules

    replace_rules('"deny-mcp"', [_deny_mcp_rule()])

    mod.__script__ = [
        AssistantMessage(
            [
                TextBlock("Calling production MCP."),
                ToolUseBlock(
                    "mcp__prod_db__query", {"sql": "SELECT * FROM users"}
                ),
            ]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("Fetch user data.")
            with pytest.raises(PermissionError):
                async for _ in client.receive_response():
                    pass

    asyncio.run(run())
    _flush()

    ev = _step_events(fake_backend.events_received)[0]
    assert ev["verdict"] == "block"
    assert ev["matched_policy"] == "block-prod-mcp"


def test_output_policy_blocks_on_assistant_text(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """``deny_output_regex`` runs against the concatenated
    ``TextBlock`` payload."""
    fake_backend, client_cls, mod = fake_claude
    fake_backend.set_rules(
        [_deny_output_regex_rule()], etag='"deny-out-regex"'
    )
    from egisai._policy_cache import replace_rules

    replace_rules('"deny-out-regex"', [_deny_output_regex_rule()])

    mod.__script__ = [
        AssistantMessage(
            [TextBlock("Here is the key: sk-abcdefghijklmnop1234567")]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("What is the key?")
            with pytest.raises(PermissionError):
                async for _ in client.receive_response():
                    pass

    asyncio.run(run())
    _flush()

    ev = _step_events(fake_backend.events_received)[0]
    assert ev["verdict"] == "block"
    assert ev["matched_policy"] == "block-secret"


def test_allow_path_carries_response_decision(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """Allow path: ``response_decision`` block is populated when the
    output phase ran but didn't block (dashboard renders the
    'allow' pill on the post-model column)."""
    fake_backend, client_cls, mod = fake_claude
    fake_backend.set_rules([_deny_tool_rule()], etag='"a"')
    from egisai._policy_cache import replace_rules

    replace_rules('"a"', [_deny_tool_rule()])

    mod.__script__ = [
        AssistantMessage([TextBlock("All clear, no shell.")]),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("Diagnose the cluster.")
            async for _ in client.receive_response():
                pass

    asyncio.run(run())
    _flush()

    ev = _step_events(fake_backend.events_received)[0]
    assert ev["verdict"] == "allow"
    assert "response_decision" in ev
    assert ev["response_decision"]["verdict"] == "allow"


# ── 4. Multi-turn audit hygiene ─────────────────────────────────────


def test_multi_turn_emits_one_event_per_turn(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """Two ``query`` + two ``receive_response`` cycles on the same
    client produce two distinct audit rows."""
    fake_backend, client_cls, mod = fake_claude

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            for prompt in ("hello", "follow-up"):
                mod.__script__ = [
                    AssistantMessage([TextBlock(f"ack: {prompt}")]),
                    ResultMessage(),
                ]
                await client.query(prompt)
                async for _ in client.receive_response():
                    pass

    asyncio.run(run())
    _flush()

    events = _step_events(fake_backend.events_received)
    assert len(events) == 2
    assert events[0]["prompt_chars"] == len("hello")
    assert events[1]["prompt_chars"] == len("follow-up")


def test_query_without_iteration_still_emits_on_close(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """A user who calls ``query`` and exits the client without
    iterating ``receive_response`` still produces a request row
    (marked ``error=never_consumed``) so the dashboard doesn't
    silently drop the turn."""
    fake_backend, client_cls, mod = fake_claude
    mod.__script__ = [
        AssistantMessage([TextBlock("ok")]),
        ResultMessage(),
    ]

    async def run() -> None:
        async with client_cls(options=_Options()) as client:
            await client.query("ping")
            # No receive_response — intentional drop.

    asyncio.run(run())
    _flush()

    events = _step_events(fake_backend.events_received)
    assert len(events) == 1
    assert events[0].get("error") == "never_consumed"
    assert events[0]["prompt_chars"] == len("ping")


# ── 5. Module-level ``query`` shape ────────────────────────────────


def test_module_level_query_emits_event(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """``async for msg in claude_agent_sdk.query(prompt, options=…)``
    runs the same input/output gate as the client path."""
    fake_backend, client_cls, mod = fake_claude

    script = [
        AssistantMessage([TextBlock("module-level ok")]),
        ResultMessage(input_tokens=5, output_tokens=10),
    ]

    async def _module_query(
        prompt: Any, options: Any = None
    ) -> AsyncIterator[Any]:
        for msg in script:
            yield msg

    # Replace the module-level query AFTER apply() has already wrapped
    # the previous one — verify that the patch correctly hooks the
    # NEW function on a fresh apply().
    mod.query = _module_query
    from egisai._patches import claude_agent_sdk

    claude_agent_sdk.apply()

    async def run() -> None:
        async for _ in mod.query("hi", options=_Options()):
            pass

    asyncio.run(run())
    _flush()

    events = _step_events(fake_backend.events_received)
    assert len(events) == 1
    ev = events[0]
    assert ev["verdict"] == "allow"
    assert ev["tokens_in"] == 5
    assert ev["tokens_out"] == 10


# ── 6. Signature parity (regression for 0.17.2 TypeError) ──────────


def test_client_query_remains_a_coroutine_after_apply(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """``inspect.iscoroutinefunction(ClaudeSDKClient.query)`` must
    remain True so downstream introspection (e.g. anyio's
    coroutine-check) doesn't flip behaviour. This is the exact
    regression that produced the 0.17.2–0.17.4 ``TypeError: object
    async_generator can't be used in 'await' expression``."""
    import inspect

    _, client_cls, _ = fake_claude
    assert inspect.iscoroutinefunction(client_cls.query)
    assert not inspect.isasyncgenfunction(client_cls.query)


def test_client_receive_messages_remains_async_gen_after_apply(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """``receive_messages`` is an ``async def … yield`` upstream —
    wrapping must preserve its async-generator-function nature so
    ``async for msg in client.receive_messages():`` keeps working."""
    import inspect

    _, client_cls, _ = fake_claude
    assert inspect.isasyncgenfunction(client_cls.receive_messages)
