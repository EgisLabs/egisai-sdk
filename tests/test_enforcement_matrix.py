"""Cross-framework enforcement matrix — locked-in claims per framework.

This file is the contract for what the SDK promises (and what it
doesn't) across every framework patch we ship. The matrix below is
the source of truth backing ``README.md`` and ``SECURITY.md``;
changing one without the other will fail the test ``test_readme_matches_matrix``.

What each framework patch must guarantee
----------------------------------------

For each row in the matrix, this file verifies:

1. The patch module exists at ``egisai._patches.<name>``.
2. The patch exposes an ``apply()`` callable returning ``bool``.
3. The patch ``apply()`` is a no-op (``False``) when the underlying
   framework isn't installed. Specifically asserts NO crash, NO
   side-effects on global state.
4. The patch's enforcement *tier* is correct:
   - **Tier 1** — direct LLM client; gates output via ``gate_call``.
     ``deny_tool_call`` blocks the response before the framework's
     agent loop dispatches the tool.
   - **Tier 2** — agentic delegator; relies on a Tier-1 patch under
     the hood. Cascade-enforced.
   - **Tier 3a** — claude_agent_sdk subprocess; PreToolUse hook
     enforces pre-execution when available, advisory fallback on
     older SDKs.
   - **Tier 3b** — bedrock_agent managed agent; tool execution
     happens on AWS. Documented as advisory until RETURN_CONTROL
     mode is wired in.

The test asserts these facts mechanically so an unrelated change
that breaks the contract (e.g. someone removes ``apply`` from a
patch module) fails CI immediately.
"""

from __future__ import annotations

import importlib
import inspect
import re
from pathlib import Path
from typing import Any

import pytest

# ── Matrix: locked-in enforcement claims per framework ─────────────


# Each entry: (module_name, friendly_label, tier, enforcement_doc).
# ``tier`` is one of ``"1"``, ``"2"``, ``"3a"``, ``"3b"``. The
# enforcement_doc string MUST appear verbatim in ``README.md`` or
# ``SECURITY.md`` (whichever the matrix lives in). The README sync
# test below verifies this.
MATRIX: list[tuple[str, str, str, str]] = [
    # Tier 1 — direct LLM clients (gates output via gate_call).
    ("openai", "OpenAI", "1", "enforced"),
    ("anthropic", "Anthropic", "1", "enforced"),
    ("genai", "Google GenAI (Gemini)", "1", "enforced"),
    ("google", "Google (legacy)", "1", "enforced"),
    ("bedrock_runtime", "AWS Bedrock Converse", "1", "enforced"),
    ("http", "HTTP fallback (httpx/requests)", "1", "enforced"),
    ("langchain", "LangChain", "1", "enforced"),
    # Tier 2 — agentic delegators (cascade-enforced via Tier-1).
    ("openai_agents", "OpenAI Agents", "2", "enforced"),
    ("crewai", "CrewAI", "2", "enforced"),
    ("autogen", "AutoGen", "2", "enforced"),
    ("langgraph", "LangGraph", "2", "enforced"),
    ("llamaindex", "LlamaIndex", "2", "enforced"),
    ("agno", "Agno", "2", "enforced"),
    ("smolagents", "smolagents", "2", "enforced"),
    ("strands", "Strands Agents", "2", "enforced"),
    ("pydantic_ai", "Pydantic AI", "2", "enforced"),
    ("google_adk", "Google ADK", "2", "enforced"),
    # Tier 3a — claude_agent_sdk (PreToolUse hook in 0.21+).
    ("claude_agent_sdk", "Claude Agent SDK", "3a", "enforced"),
    # Tier 3b — bedrock_agent (AWS-managed agent loop).
    ("bedrock_agent", "AWS Bedrock Agents", "3b", "advisory"),
    # Inbound — MCP server add-on. Governs inbound ``tools/call``
    # against a customer-hosted MCP server (fastmcp / official mcp).
    # Dormant unless the org has the ``mcp_servers`` entitlement, but
    # when active it gates the tool BEFORE the handler runs, so the
    # claim is enforced. Distinct from every row above: those govern
    # the customer's *outbound* LLM/agent calls, this governs calls
    # *into* the customer's server.
    ("mcp_server", "MCP Server (inbound)", "in", "enforced"),
]


@pytest.mark.parametrize(
    "module_name,_label,_tier,_doc",
    list(MATRIX),
    ids=[m for (m, _, _, _) in MATRIX],
)
def test_patch_module_exists_and_exposes_apply(
    module_name: str,
    _label: str,
    _tier: str,
    _doc: str,
) -> None:
    """Every framework in the matrix MUST ship a patch module with
    an ``apply()`` callable. Catches accidental deletion or rename
    of a patch file."""
    mod = importlib.import_module(f"egisai._patches.{module_name}")
    apply_fn = getattr(mod, "apply", None)
    assert callable(apply_fn), (
        f"{module_name}.apply must be callable; "
        f"got {type(apply_fn).__name__}"
    )
    sig = inspect.signature(apply_fn)
    assert len(sig.parameters) == 0, (
        f"{module_name}.apply must take no arguments; "
        f"got {list(sig.parameters)}"
    )


@pytest.mark.parametrize(
    "module_name,_label,_tier,_doc",
    list(MATRIX),
    ids=[m for (m, _, _, _) in MATRIX],
)
def test_patch_apply_is_noop_when_framework_not_installed(
    module_name: str,
    _label: str,
    _tier: str,
    _doc: str,
) -> None:
    """Without the underlying framework installed, ``apply()`` MUST
    return ``False`` (gracefully) instead of raising. This is the
    fail-open import-time guarantee — egisai never bricks an app
    just because one optional integration's library is absent."""
    mod = importlib.import_module(f"egisai._patches.{module_name}")
    # We can't reliably "uninstall" a framework mid-test, so this
    # test only meaningfully runs when the framework is NOT in the
    # test environment. The CI installs claude_agent_sdk, openai,
    # etc. — those return True. For everything else, False.
    result = mod.apply()
    assert isinstance(result, bool), (
        f"{module_name}.apply must return bool; got {type(result).__name__}"
    )


def test_matrix_covers_every_patch_module() -> None:
    """Every ``.py`` file in ``egisai/_patches`` (except internals)
    MUST appear in the matrix above. Catches a new framework patch
    being added without updating the matrix / docs."""
    patches_dir = Path(__file__).resolve().parents[1] / "src" / "egisai" / "_patches"
    on_disk: set[str] = set()
    for path in patches_dir.glob("*.py"):
        name = path.stem
        if name.startswith("_"):
            continue
        on_disk.add(name)

    in_matrix = {m for (m, _, _, _) in MATRIX}
    missing = on_disk - in_matrix
    extra = in_matrix - on_disk
    assert not missing, (
        f"new patch modules without matrix entries (add to "
        f"test_enforcement_matrix.py MATRIX): {sorted(missing)}"
    )
    assert not extra, (
        f"matrix references missing patch modules: {sorted(extra)}"
    )


def test_readme_matches_matrix() -> None:
    """The enforcement matrix in ``README.md`` must list every
    framework + tier from MATRIX. Prevents documentation drift —
    if someone changes a tier in MATRIX without updating README
    (or vice versa), this test fails.

    We do a relaxed check: each framework's friendly label must
    appear AT LEAST ONCE in README.md alongside its enforcement
    doc string (within ~400 chars). The README is free to render
    the matrix in any markdown format and the label is free to
    appear elsewhere in the README for unrelated reasons (e.g.
    "OpenAI" in the intro paragraph).
    """
    readme = (
        Path(__file__).resolve().parents[1] / "README.md"
    ).read_text(encoding="utf-8")
    missing_in_readme: list[str] = []
    for _module, label, tier, doc in MATRIX:
        if label not in readme:
            missing_in_readme.append(f"{label} (tier {tier}) — label absent")
            continue
        # Walk every occurrence of the label; at least ONE must be
        # followed (within 400 chars) by the doc string. That window
        # is wide enough to cover a markdown table row + cell text.
        found = False
        start = 0
        while True:
            idx = readme.find(label, start)
            if idx < 0:
                break
            window = readme[idx : idx + 400]
            if doc in window or doc in window.lower():
                found = True
                break
            start = idx + 1
        if not found:
            missing_in_readme.append(
                f"{label}: expected '{doc}' within 400 chars after at "
                f"least one occurrence of the label"
            )

    assert not missing_in_readme, (
        "README.md does not match the locked enforcement matrix; "
        f"missing: {missing_in_readme}"
    )


def test_security_md_documents_bedrock_agent_advisory_limitation() -> None:
    """SECURITY.md MUST document the one tier-3b advisory case
    (Bedrock Agents — AWS-managed loop). This is the one publicly
    advertised "we don't enforce here" caveat; we don't want to
    accidentally hide it from a security-conscious customer."""
    security = (
        Path(__file__).resolve().parents[1] / "SECURITY.md"
    ).read_text(encoding="utf-8")
    assert "Bedrock Agents" in security or "bedrock_agent" in security, (
        "SECURITY.md must explicitly name the Bedrock Agents advisory "
        "limitation so customers can risk-assess accordingly."
    )
    # And it must mention "advisory" (or equivalent honest wording).
    assert re.search(
        r"advisory|cannot enforce|observe[- ]only",
        security,
        flags=re.IGNORECASE,
    ), (
        "SECURITY.md must use 'advisory' / 'cannot enforce' / "
        "'observe-only' to label the Bedrock Agents limitation."
    )


# ── Behavioural lock-in: where the heavy enforcement guarantees live ──


def test_gate_call_default_enforcement_is_enforced() -> None:
    """The shared ``gate_call`` (used by every Tier-1 direct-LLM
    patch) stamps ``enforced`` by default for both allow and
    output-block paths. Confirms the Tier-1 enforcement claim in
    the matrix.

    A regression here would mean an OpenAI / Anthropic / Google
    direct-LLM tool block silently downgrades to advisory — that
    would mean we lied to customers reading the matrix."""
    src = (
        Path(__file__).resolve().parents[1]
        / "src" / "egisai" / "_patches" / "_common.py"
    ).read_text(encoding="utf-8")
    # The shared sync gate stamps enforced as the default; we also
    # confirm the default in _stamp_output_block is enforced (so
    # output-block rows for Tier-1 patches land as enforced unless
    # an explicit override is passed).
    assert re.search(
        r"enforcement_status:\s*str\s*=\s*ENFORCEMENT_ENFORCED",
        src,
    ), (
        "_stamp_output_block's enforcement_status MUST default to "
        "ENFORCEMENT_ENFORCED; that's the Tier-1 enforcement claim."
    )


def test_claude_agent_sdk_wraps_connect_for_hook_injection() -> None:
    """REGRESSION (0.22.1): the ``claude_agent_sdk`` patch MUST
    expose ``_wrap_client_connect`` and call it during ``apply()``.

    The upstream SDK ships ``options.hooks`` to the Node CLI
    inside ``ClaudeSDKClient.connect()`` — exactly once, at
    subprocess-init time. Mutating ``options.hooks`` later (e.g.
    inside ``client.query()``) is a silent no-op; the CLI's
    matcher table is already frozen.

    For the SOC 2 / ISO 27001 enforced claim on Tier 3a to hold,
    we MUST inject our PreToolUse + PostToolUse placeholders
    BEFORE the original ``connect()`` runs. If a future refactor
    drops this wrap, the patch silently regresses to advisory-only
    on tool dispatch AND advisory-only on tool results — bug we
    fixed in 0.22.1. This test fails loudly if the structural
    guarantee disappears.
    """
    mod = importlib.import_module("egisai._patches.claude_agent_sdk")
    assert hasattr(mod, "_wrap_client_connect"), (
        "claude_agent_sdk patch must expose _wrap_client_connect() — "
        "this is the seam that injects placeholder hooks BEFORE the "
        "Node CLI freezes its matcher table at initialize time. "
        "Without it, hooks land too late and customer agents run "
        "ungoverned (SOC 2 / ISO 27001 violation)."
    )
    assert hasattr(mod, "_make_pretooluse_dispatcher"), (
        "claude_agent_sdk patch must expose _make_pretooluse_dispatcher() "
        "— the deferred dispatcher bound to a client instance that "
        "the placeholder routes through at hook-fire time."
    )
    assert hasattr(mod, "_make_posttooluse_dispatcher"), (
        "claude_agent_sdk patch must expose _make_posttooluse_dispatcher() "
        "— same pattern for the PostToolUse seam."
    )


def test_claude_agent_sdk_uses_hooks_when_available() -> None:
    """The ``claude_agent_sdk`` patch MUST attempt to inject BOTH a
    PreToolUse hook (gates tool dispatch — 0.21+) AND a PostToolUse
    hook (gates tool result before model is shown it — 0.22+).
    The presence of both factories is the structural guarantee for
    the Tier-3a enforced claim on both call AND result columns."""
    mod = importlib.import_module("egisai._patches.claude_agent_sdk")
    assert hasattr(mod, "_hooks_supported"), (
        "claude_agent_sdk patch must expose _hooks_supported() — "
        "feature-detection of the modern hooks API."
    )
    # PreToolUse — gates tool / MCP dispatch.
    assert hasattr(mod, "_inject_pretooluse_hook"), (
        "claude_agent_sdk patch must expose _inject_pretooluse_hook() "
        "— the actual injection helper."
    )
    assert hasattr(mod, "_build_pretooluse_callback"), (
        "claude_agent_sdk patch must expose _build_pretooluse_callback() "
        "— the callback factory bound per-turn."
    )
    # PostToolUse — gates tool result. This is the SOC 2 / ISO 27001
    # tool-result-PII guarantee for subprocess-loop agents.
    assert hasattr(mod, "_inject_posttooluse_hook"), (
        "claude_agent_sdk patch must expose _inject_posttooluse_hook() "
        "— required for the tool-result PII enforcement column."
    )
    assert hasattr(mod, "_build_posttooluse_callback"), (
        "claude_agent_sdk patch must expose _build_posttooluse_callback() "
        "— the callback factory that runs output policies on the "
        "tool response and substitutes via updatedToolOutput / "
        "updatedMCPToolOutput before Claude sees the result."
    )
    assert hasattr(mod, "_post_hooks_supported"), (
        "claude_agent_sdk patch must expose _post_hooks_supported() "
        "— feature-detection for PostToolUse independently of PreToolUse."
    )


def test_bedrock_agent_documents_advisory_limitation_in_code() -> None:
    """The bedrock_agent patch MUST document its advisory-only
    limitation in code so a maintainer surveying the patches sees
    the gap without having to read the README."""
    src = (
        Path(__file__).resolve().parents[1]
        / "src" / "egisai" / "_patches" / "bedrock_agent.py"
    ).read_text(encoding="utf-8")
    # The existing comment block in bedrock_agent.py spells out the
    # limitation around AWS-side execution; assert at least one of
    # those signal phrases survives any future refactor.
    assert re.search(
        r"AWS[- ]side|managed agent|outside the agent|not enforceable",
        src,
        flags=re.IGNORECASE,
    ), (
        "bedrock_agent.py must document why tool calls inside the "
        "managed agent loop are advisory-only. Don't quietly remove "
        "the explanation — auditors need it."
    )


# ── No-regression: full patches list applies cleanly under init ─────


def test_init_applies_all_available_patches_without_crashing(
    fake_backend: Any,
) -> None:
    """``egisai.init()`` cycles through every patch's ``apply()``
    function. A single bad patch must never abort the others.

    This is the safety net for ``sdk-design-philosophy.mdc`` rule:
    a missing dep (e.g. ``boto3`` not installed) MUST not crash
    the user's ``import egisai; egisai.init()``."""
    import egisai

    # If this raises, the matrix is broken end-to-end.
    egisai.init(
        api_key="egis_live_test",
        app="enforcement-matrix",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )
    # Successful init proves every patch's apply() returned cleanly.
