"""Framework-specific output signal extractors.

Pulls the four pieces of structured information that output-side
policies care about — assistant text, tool definition names, tool
calls the model emitted, and MCP target names — from each
provider's response shape.

Each extractor is best-effort: if the response is partial,
streamed, or malformed, the function returns whatever it could
recover and an empty value for the rest. The output evaluator
treats empty signals as "nothing to enforce".
"""

from __future__ import annotations

from typing import Any

OutputSignals = tuple[str, list[str], list[dict[str, Any]], list[str]]
"""``(text, tool_names, tool_calls, mcp_targets)``.

* ``text``         — assistant message body, joined across content
  parts. Empty string when the model produced only tool calls.
* ``tool_names``   — tool *definition* names from the request
  payload (lets ``deny_tool_call`` block on registration even
  when the model didn't actually invoke the tool yet).
* ``tool_calls``   — list of ``{"name": str, "arguments": str}``
  for tools the model asked to invoke.
* ``mcp_targets``  — flattened list of MCP target identifiers,
  combining definition-side (``tools[*].mcp.server``) and
  call-side (``tool_calls[*].mcp_target``) signals.
"""


def _read(obj: Any, *names: str) -> Any:
    """Best-effort attribute / dict lookup over a response-shaped value."""
    for name in names:
        if isinstance(obj, dict):
            v = obj.get(name)
        else:
            v = getattr(obj, name, None)
        if v is not None:
            return v
    return None


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _coerce_arguments(arguments: Any) -> str:
    """``arguments`` is sometimes a JSON string, sometimes a dict."""
    if arguments is None:
        return ""
    if isinstance(arguments, str):
        return arguments
    try:
        import json

        return json.dumps(arguments, ensure_ascii=False, sort_keys=True)
    except Exception:  # noqa: BLE001
        return str(arguments)


def _tool_definition_names(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    out: list[str] = []
    for tool in _as_list(payload.get("tools")):
        if not isinstance(tool, dict):
            continue
        # OpenAI v1: ``{"type": "function", "function": {"name": "..."}}``.
        if isinstance(tool.get("function"), dict):
            name = tool["function"].get("name")
            if isinstance(name, str) and name:
                out.append(name)
                continue
        # Anthropic / OpenAI legacy: ``{"name": "..."}``.
        name = tool.get("name")
        if isinstance(name, str) and name:
            out.append(name)
    return out


def _mcp_targets_from_payload(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []
    out: list[str] = []
    for tool in _as_list(payload.get("tools")):
        if not isinstance(tool, dict):
            continue
        mcp = tool.get("mcp")
        if isinstance(mcp, dict):
            server = mcp.get("server") or mcp.get("server_url")
            if isinstance(server, str) and server:
                out.append(server)
    return out


# ── OpenAI Chat Completions ────────────────────────────────────────────


def extract_openai_chat(response: Any, payload: Any) -> OutputSignals:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    mcp_targets: list[str] = list(_mcp_targets_from_payload(payload))

    for choice in _as_list(_read(response, "choices")):
        message = _read(choice, "message")
        content = _read(message, "content")
        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for part in content:
                t = _read(part, "text") or _read(part, "content")
                if isinstance(t, str):
                    text_parts.append(t)

        for call in _as_list(_read(message, "tool_calls")):
            fn = _read(call, "function") or call
            name = _read(fn, "name") or ""
            args = _coerce_arguments(_read(fn, "arguments"))
            if isinstance(name, str) and name:
                tool_calls.append({"name": name, "arguments": args})
            mcp = _read(call, "mcp_target") or _read(call, "mcp_server")
            if isinstance(mcp, str) and mcp:
                mcp_targets.append(mcp)

    return (
        "\n".join(p for p in text_parts if p),
        _tool_definition_names(payload),
        tool_calls,
        mcp_targets,
    )


# ── OpenAI Responses API ───────────────────────────────────────────────


def extract_openai_responses(response: Any, payload: Any) -> OutputSignals:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    mcp_targets: list[str] = list(_mcp_targets_from_payload(payload))

    for item in _as_list(_read(response, "output")):
        item_type = _read(item, "type") or ""
        if item_type == "message":
            for part in _as_list(_read(item, "content")):
                t = _read(part, "text")
                if isinstance(t, str):
                    text_parts.append(t)
        elif item_type in ("tool_call", "function_call", "tool_use"):
            name = _read(item, "name")
            args = _coerce_arguments(_read(item, "arguments") or _read(item, "input"))
            if isinstance(name, str) and name:
                tool_calls.append({"name": name, "arguments": args})

    direct_text = _read(response, "output_text")
    if isinstance(direct_text, str) and direct_text:
        text_parts.append(direct_text)

    return (
        "\n".join(p for p in text_parts if p),
        _tool_definition_names(payload),
        tool_calls,
        mcp_targets,
    )


# ── Anthropic Messages ────────────────────────────────────────────────


def extract_anthropic(response: Any, payload: Any) -> OutputSignals:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    mcp_targets: list[str] = list(_mcp_targets_from_payload(payload))

    for block in _as_list(_read(response, "content")):
        block_type = _read(block, "type") or ""
        if block_type == "text":
            t = _read(block, "text")
            if isinstance(t, str):
                text_parts.append(t)
        elif block_type == "tool_use":
            name = _read(block, "name")
            args = _coerce_arguments(_read(block, "input"))
            if isinstance(name, str) and name:
                tool_calls.append({"name": name, "arguments": args})

    return (
        "\n".join(p for p in text_parts if p),
        _tool_definition_names(payload),
        tool_calls,
        mcp_targets,
    )


# ── Google Gemini ─────────────────────────────────────────────────────


def extract_google(response: Any, payload: Any) -> OutputSignals:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    mcp_targets: list[str] = list(_mcp_targets_from_payload(payload))

    for candidate in _as_list(_read(response, "candidates")):
        content = _read(candidate, "content")
        for part in _as_list(_read(content, "parts")):
            t = _read(part, "text")
            if isinstance(t, str):
                text_parts.append(t)
            fc = _read(part, "function_call")
            if fc is not None:
                name = _read(fc, "name")
                args = _coerce_arguments(_read(fc, "args"))
                if isinstance(name, str) and name:
                    tool_calls.append({"name": name, "arguments": args})

    direct_text = _read(response, "text")
    if isinstance(direct_text, str) and direct_text:
        text_parts.append(direct_text)

    return (
        "\n".join(p for p in text_parts if p),
        _tool_definition_names(payload),
        tool_calls,
        mcp_targets,
    )
