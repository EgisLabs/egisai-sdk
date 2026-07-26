"""Identity propagation into agent side effects ("On-Behalf-Of").

Opt-in (``egisai.init(stamp_identity=True)`` / ``EGISAI_STAMP_IDENTITY=1``).
When enabled, tool invocations that create durable artifacts outside
the platform — git commits, GitHub pull requests, GitHub issues —
get the calling agent's identity appended as a trailer::

    On-Behalf-Of: Code Generation Agent (egis:5f1c…)

so the attribution survives in the artifact itself (git history, PR
body) even where the dashboard can't reach. The idea is borrowed
from network-layer gateways that rewrite GitHub commits with an
``On-Behalf-Of`` trailer; here the interception point is the Claude
Agent SDK's PreToolUse hook, which supports rewriting the tool input
via ``hookSpecificOutput.updatedInput`` on current CLI versions
(older CLIs ignore the unknown key — the tool then runs with its
original input, i.e. the feature degrades to a silent no-op, never
an error).

Safety contract (every rule is enforced in code below and covered by
``tests/test_identity_stamp.py``):

* **Allowlist only.** Only the tools named in ``_BASH_TOOL`` /
  ``_MCP_FIELD_BY_TOOL`` are ever touched. Everything else returns
  ``None`` ("no change") immediately.
* **Idempotent.** If an ``On-Behalf-Of:`` trailer is already present
  in the target field the stamp is skipped — re-invocations and
  user-supplied trailers never double up.
* **Conservative Bash parsing.** Shell strings are stamped only when
  the entire command is a single plain ``git commit`` invocation
  (no ``&&`` / ``;`` / ``|`` / newline chaining) — appending a
  ``--trailer`` flag to a compound command could attach it to the
  wrong sub-command. Compound commands are left untouched.
* **Sanitized identity.** The agent name is reduced to a safe
  character set before it is embedded anywhere (defense against a
  hostile display name breaking out of the shell quoting).
* **Never raises.** Any unexpected shape returns ``None`` and the
  original input is forwarded unchanged (fail-open, same posture as
  every other patch).

The stamp itself carries only the agent's display name and its Egis
agent UUID — no prompt text, no PII (compliance: the identity module
already guarantees display names never contain raw prompt content).
"""

from __future__ import annotations

import copy
import re
from typing import Any

TRAILER_KEY = "On-Behalf-Of"

# Characters allowed in the embedded agent name. Anything else is
# dropped — the name ends up inside single-quoted shell arguments
# and markdown bodies, so the set is deliberately tight.
_NAME_SAFE_RE = re.compile(r"[^A-Za-z0-9 ._\-]+")
_MAX_NAME_LEN = 80

# A single plain ``git commit …`` command — optional leading
# whitespace, no command chaining anywhere in the string. ``git -C
# <path> commit`` is also accepted (common in agent-generated
# commands).
_GIT_COMMIT_RE = re.compile(
    r"^\s*git\s+(?:-C\s+\S+\s+)?commit\b"
)
_SHELL_CHAIN_RE = re.compile(r"[;&|\n]")

# Claude Code's built-in shell tool.
_BASH_TOOL = "Bash"

# GitHub MCP tools (official ``github`` MCP server namespace) that
# create durable artifacts, mapped to the input field carrying the
# human-readable body/message the trailer belongs in.
_MCP_FIELD_BY_TOOL: dict[str, str] = {
    "mcp__github__create_pull_request": "body",
    "mcp__github__create_issue": "body",
    "mcp__github__create_or_update_file": "message",
    "mcp__github__push_files": "message",
}


def build_trailer(agent_name: str, agent_id: str) -> str:
    """``On-Behalf-Of: <name> (egis:<id>)`` with a sanitized name."""
    name = _NAME_SAFE_RE.sub("", agent_name or "").strip()[:_MAX_NAME_LEN]
    if not name:
        name = "egisai agent"
    ident = (agent_id or "").strip()
    suffix = f" (egis:{ident})" if ident else ""
    return f"{TRAILER_KEY}: {name}{suffix}"


def stamp_tool_input(
    tool_name: str,
    tool_input: Any,
    *,
    agent_name: str,
    agent_id: str,
) -> dict[str, Any] | None:
    """Return a stamped **copy** of ``tool_input``, or ``None``.

    ``None`` means "forward the original input unchanged" — the tool
    isn't allowlisted, the input shape is unexpected, the trailer is
    already present, or stamping failed. The caller must treat
    ``None`` as a no-op, never as an error.
    """
    try:
        if not isinstance(tool_input, dict):
            return None
        trailer = build_trailer(agent_name, agent_id)

        if tool_name == _BASH_TOOL:
            return _stamp_bash(tool_input, trailer)

        field = _MCP_FIELD_BY_TOOL.get(tool_name)
        if field is not None:
            return _stamp_text_field(tool_input, field, trailer)

        return None
    except Exception:  # noqa: BLE001
        return None


def _stamp_bash(tool_input: dict[str, Any], trailer: str) -> dict[str, Any] | None:
    """Append ``--trailer '…'`` to a plain single ``git commit`` command.

    ``git commit --trailer`` (git ≥ 2.32) appends the trailer to the
    commit message regardless of how ``-m`` was quoted, which
    sidesteps every quote-context pitfall of rewriting the message
    string itself.
    """
    command = tool_input.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    if not _GIT_COMMIT_RE.match(command):
        return None
    if _SHELL_CHAIN_RE.search(command):
        # Compound command — the flag could bind to the wrong
        # sub-command. Leave it alone (conservative allowlist).
        return None
    if TRAILER_KEY in command:
        return None
    # ``build_trailer`` sanitized the name to a set that cannot
    # contain a single quote, so the single-quoted argument below is
    # closed-form safe.
    updated = dict(tool_input)
    updated["command"] = f"{command.rstrip()} --trailer '{trailer}'"
    return updated


def _stamp_text_field(
    tool_input: dict[str, Any], field: str, trailer: str
) -> dict[str, Any] | None:
    """Append the trailer as a final paragraph of a text field."""
    value = tool_input.get(field)
    if value is None:
        value = ""
    if not isinstance(value, str):
        return None
    if TRAILER_KEY in value:
        return None
    updated = copy.deepcopy(tool_input)
    updated[field] = f"{value.rstrip()}\n\n{trailer}" if value.strip() else trailer
    return updated
