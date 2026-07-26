"""Identity propagation into side effects ("On-Behalf-Of" stamping).

Covers, in order:

1. The ``_identity_stamp`` helpers in isolation — trailer building
   (name sanitization), the conservative Bash ``git commit``
   allowlist, GitHub MCP body/message stamping, idempotence, and
   every "leave it alone" branch.
2. End-to-end through the Claude Agent SDK PreToolUse hook: the
   allow response carries ``updatedInput`` iff the operator opted
   in via ``stamp_identity`` AND the tool is allowlisted AND the
   verdict is allow; the audit row's preview reflects the stamped
   (post-mutation) input per the audit-before-persist contract.

The end-to-end tests reuse the fake ``claude_agent_sdk`` module
double from ``test_claude_agent_sdk_pretooluse`` (imported, not
copy-pasted) so the two files can't drift apart on the SDK shape.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

from egisai._patches._identity_stamp import (
    TRAILER_KEY,
    build_trailer,
    stamp_tool_input,
)

AGENT_NAME = "Code Generation Agent"
AGENT_ID = "5f1c0000-0000-0000-0000-000000000042"


# ── 1. Trailer building ─────────────────────────────────────────────


class TestBuildTrailer:
    def test_basic_shape(self) -> None:
        assert (
            build_trailer(AGENT_NAME, AGENT_ID)
            == f"On-Behalf-Of: {AGENT_NAME} (egis:{AGENT_ID})"
        )

    def test_hostile_name_is_sanitized(self) -> None:
        # Quotes, shell metacharacters, and newlines are dropped so
        # the name can never break out of the single-quoted shell
        # argument it lands in.
        trailer = build_trailer("Evil'; rm -rf / #\nAgent", AGENT_ID)
        assert "'" not in trailer.split(": ", 1)[1].rsplit(" (egis:", 1)[0]
        assert ";" not in trailer
        assert "\n" not in trailer
        assert "/" not in trailer.rsplit(" (egis:", 1)[0]

    def test_empty_name_falls_back(self) -> None:
        assert build_trailer("", AGENT_ID).startswith(
            "On-Behalf-Of: egisai agent"
        )

    def test_missing_id_omits_suffix(self) -> None:
        assert build_trailer(AGENT_NAME, "") == f"On-Behalf-Of: {AGENT_NAME}"

    def test_long_name_is_truncated(self) -> None:
        trailer = build_trailer("A" * 500, AGENT_ID)
        assert len(trailer) < 200


# ── 2. Bash git-commit stamping ─────────────────────────────────────


def _stamp(tool: str, tool_input: Any) -> dict[str, Any] | None:
    return stamp_tool_input(
        tool, tool_input, agent_name=AGENT_NAME, agent_id=AGENT_ID
    )


class TestBashStamping:
    def test_plain_git_commit_gets_trailer_flag(self) -> None:
        out = _stamp("Bash", {"command": 'git commit -m "fix the bug"'})
        assert out is not None
        assert out["command"] == (
            'git commit -m "fix the bug" '
            f"--trailer 'On-Behalf-Of: {AGENT_NAME} (egis:{AGENT_ID})'"
        )

    def test_original_input_is_not_mutated(self) -> None:
        original = {"command": "git commit -m 'x'"}
        out = _stamp("Bash", original)
        assert out is not None
        assert original["command"] == "git commit -m 'x'"

    def test_git_dash_c_commit_is_stamped(self) -> None:
        out = _stamp(
            "Bash", {"command": "git -C /repo commit -am 'quick fix'"}
        )
        assert out is not None
        assert "--trailer" in out["command"]

    def test_compound_command_left_alone(self) -> None:
        for cmd in (
            "git add -A && git commit -m 'x'",
            "git commit -m 'x'; git push",
            "git commit -m 'x' | tee log",
            "git commit -m 'x'\ngit push",
        ):
            assert _stamp("Bash", {"command": cmd}) is None, cmd

    def test_non_commit_git_left_alone(self) -> None:
        for cmd in ("git push origin main", "git status", "ls -la"):
            assert _stamp("Bash", {"command": cmd}) is None, cmd

    def test_idempotent_when_trailer_present(self) -> None:
        cmd = (
            "git commit -m 'x' --trailer 'On-Behalf-Of: Someone (egis:abc)'"
        )
        assert _stamp("Bash", {"command": cmd}) is None

    def test_garbage_shapes_left_alone(self) -> None:
        assert _stamp("Bash", {"command": 42}) is None
        assert _stamp("Bash", {"command": ""}) is None
        assert _stamp("Bash", {}) is None
        assert _stamp("Bash", "git commit -m x") is None
        assert _stamp("Bash", None) is None


# ── 3. GitHub MCP stamping ──────────────────────────────────────────


class TestMcpStamping:
    def test_create_pull_request_body_gets_trailer(self) -> None:
        out = _stamp(
            "mcp__github__create_pull_request",
            {"title": "Fix", "body": "Closes #12."},
        )
        assert out is not None
        assert out["body"] == (
            f"Closes #12.\n\nOn-Behalf-Of: {AGENT_NAME} (egis:{AGENT_ID})"
        )
        assert out["title"] == "Fix"

    def test_create_issue_body_gets_trailer(self) -> None:
        out = _stamp("mcp__github__create_issue", {"title": "Bug", "body": ""})
        assert out is not None
        assert out["body"] == f"On-Behalf-Of: {AGENT_NAME} (egis:{AGENT_ID})"

    def test_push_files_message_gets_trailer(self) -> None:
        out = _stamp(
            "mcp__github__push_files",
            {"message": "chore: sync", "files": [{"path": "a"}]},
        )
        assert out is not None
        assert out["message"].endswith(f"(egis:{AGENT_ID})")
        assert TRAILER_KEY in out["message"]

    def test_create_or_update_file_message_gets_trailer(self) -> None:
        out = _stamp(
            "mcp__github__create_or_update_file",
            {"message": "docs: update", "path": "README.md"},
        )
        assert out is not None
        assert TRAILER_KEY in out["message"]

    def test_missing_body_becomes_trailer_only(self) -> None:
        out = _stamp("mcp__github__create_pull_request", {"title": "Fix"})
        assert out is not None
        assert out["body"] == f"On-Behalf-Of: {AGENT_NAME} (egis:{AGENT_ID})"

    def test_idempotent_when_trailer_present(self) -> None:
        assert (
            _stamp(
                "mcp__github__create_issue",
                {"body": "On-Behalf-Of: Someone (egis:abc)"},
            )
            is None
        )

    def test_non_string_field_left_alone(self) -> None:
        assert (
            _stamp("mcp__github__create_pull_request", {"body": ["x"]})
            is None
        )

    def test_non_allowlisted_tools_left_alone(self) -> None:
        for tool in (
            "Read",
            "Write",
            "mcp__github__get_issue",
            "mcp__gitlab__create_pull_request",  # not the github server
            "mcp__slack__post_message",
        ):
            assert _stamp(tool, {"body": "x", "command": "git commit"}) is None


# ── 4. End-to-end through the PreToolUse hook ───────────────────────
#
# Reuse the fake claude_agent_sdk module double + hook driver from
# the PreToolUse suite so the SDK shape stays defined in one place.

from test_claude_agent_sdk_pretooluse import (  # noqa: E402
    AssistantMessage,
    ResultMessage,
    ToolUseBlock,
    _ClaudeAgentOptions,
    _drive_hooks_for_script,
    _install_fake_module,
)


@pytest.fixture
def fake_claude(fake_backend: Any) -> Any:
    """Fake claude_agent_sdk with hooks; init'd WITHOUT stamping
    (individual tests flip ``stamp_identity`` via update_config)."""
    import egisai

    egisai.init(
        api_key="egis_live_test",
        app="stamp-test",
        env="t",
        base_url="http://fake",
        enable_sse=False,
        quiet=True,
    )
    mod, client_cls, _captured = _install_fake_module(with_hooks=True)
    mod.__script__ = []
    mod.__hook_invocations__ = []

    async def _q(self: Any, prompt: Any, session_id: str = "default") -> None:
        self._sent.append({"prompt": prompt, "session_id": session_id})
        opts = getattr(self, "options", None)
        await _drive_hooks_for_script(
            opts, mod.__script__, mod.__hook_invocations__
        )

    async def _rm(self: Any) -> Any:
        for msg in mod.__script__:
            yield msg

    client_cls.query = _q
    client_cls.receive_messages = _rm

    from egisai._patches import claude_agent_sdk

    assert claude_agent_sdk.apply() is True
    yield fake_backend, client_cls, mod
    sys.modules.pop("claude_agent_sdk", None)


def _enable_stamping() -> None:
    from egisai._config import update_config

    update_config(stamp_identity=True)


def _run_script(client_cls: type, script_prompt: str = "Ship it") -> None:
    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query(script_prompt)
            async for _ in client.receive_response():
                pass

    asyncio.run(run())


def _flush() -> None:
    from egisai import shutdown

    shutdown()


def test_hook_returns_updated_input_when_stamping_enabled(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    fake_backend, client_cls, mod = fake_claude
    _enable_stamping()
    mod.__script__ = [
        AssistantMessage(
            [ToolUseBlock("Bash", {"command": "git commit -m 'fix'"}, id_="tu1")]
        ),
        ResultMessage(),
    ]

    _run_script(client_cls)
    _flush()

    (invocation,) = mod.__hook_invocations__
    out = invocation["output"]["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"
    updated = out["updatedInput"]
    assert updated["command"].startswith("git commit -m 'fix' --trailer ")
    assert TRAILER_KEY in updated["command"]
    assert "(egis:" in updated["command"]


def test_no_updated_input_when_stamping_disabled(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """Default-off contract: without the opt-in, the hook response
    is byte-identical to the pre-stamping shape."""
    _, client_cls, mod = fake_claude
    mod.__script__ = [
        AssistantMessage(
            [ToolUseBlock("Bash", {"command": "git commit -m 'fix'"}, id_="tu1")]
        ),
        ResultMessage(),
    ]

    _run_script(client_cls)
    _flush()

    (invocation,) = mod.__hook_invocations__
    out = invocation["output"]["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"
    assert "updatedInput" not in out


def test_non_allowlisted_tool_not_stamped_even_when_enabled(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    _, client_cls, mod = fake_claude
    _enable_stamping()
    mod.__script__ = [
        AssistantMessage(
            [ToolUseBlock("Read", {"path": "/tmp/x"}, id_="tu1")]
        ),
        ResultMessage(),
    ]

    _run_script(client_cls)
    _flush()

    (invocation,) = mod.__hook_invocations__
    out = invocation["output"]["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"
    assert "updatedInput" not in out


def test_blocked_tool_is_never_stamped(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """A denied invocation must carry no ``updatedInput`` — stamping
    only decorates work that is actually allowed to happen."""
    fake_backend, client_cls, mod = fake_claude
    _enable_stamping()
    from egisai._policy_cache import replace_rules

    replace_rules(
        '"stamp-block"',
        [
            {
                "id": "b1",
                "name": "no-bash",
                "type": "deny_tool_call",
                "tenant": None,
                "config": {"patterns": ["^Bash$"]},
            }
        ],
    )
    mod.__script__ = [
        AssistantMessage(
            [ToolUseBlock("Bash", {"command": "git commit -m 'fix'"}, id_="tu1")]
        ),
        ResultMessage(),
    ]

    async def run() -> None:
        opts = _ClaudeAgentOptions()
        async with client_cls(options=opts) as client:
            await client.query("Ship it")
            try:
                async for _ in client.receive_response():
                    pass
            except PermissionError:
                pass

    asyncio.run(run())
    _flush()

    (invocation,) = mod.__hook_invocations__
    out = invocation["output"]["hookSpecificOutput"]
    assert out["permissionDecision"] == "deny"
    assert "updatedInput" not in out


def test_audit_preview_reflects_stamped_input(
    fake_claude: tuple[Any, type, types.ModuleType],
) -> None:
    """Audit-before-persist: the tool_call step row previews the
    post-mutation input (with the trailer) and carries the additive
    ``identity_stamped`` marker."""
    fake_backend, client_cls, mod = fake_claude
    _enable_stamping()
    mod.__script__ = [
        AssistantMessage(
            [ToolUseBlock("Bash", {"command": "git commit -m 'fix'"}, id_="tu1")]
        ),
        ResultMessage(),
    ]

    _run_script(client_cls)
    _flush()

    tool_steps = [
        e
        for e in fake_backend.events_received
        if e.get("step_kind") == "tool_call"
        and e.get("tool_name") == "Bash"
    ]
    assert tool_steps, "no Bash tool_call step row shipped"
    step = tool_steps[0]
    assert step.get("identity_stamped") is True
    assert TRAILER_KEY in (step.get("prompt_preview") or "")


def test_stamp_failure_fails_open(
    fake_claude: tuple[Any, type, types.ModuleType],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the stamping helper explodes, the tool proceeds with its
    original input — never an error, never a deny."""
    _, client_cls, mod = fake_claude
    _enable_stamping()

    from egisai._patches import _identity_stamp

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("stamping exploded")

    monkeypatch.setattr(_identity_stamp, "stamp_tool_input", _boom)
    mod.__script__ = [
        AssistantMessage(
            [ToolUseBlock("Bash", {"command": "git commit -m 'fix'"}, id_="tu1")]
        ),
        ResultMessage(),
    ]

    _run_script(client_cls)
    _flush()

    (invocation,) = mod.__hook_invocations__
    out = invocation["output"]["hookSpecificOutput"]
    assert out["permissionDecision"] == "allow"
    assert "updatedInput" not in out
