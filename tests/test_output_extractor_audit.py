"""Structural audit: every LLM-level patch must wire ``extract_output_signals``.

The two-phase contract documented in ``.cursor/rules/security-and-
compliance.mdc`` requires both an input gate AND an output gate on
every model boundary. ``gate_call`` only invokes the output phase
when its ``extract_output_signals`` argument is non-None — so a patch
that forgets to pass it silently skips ``deny_tool_call``,
``deny_mcp_call``, ``deny_output_regex``, and the post-model
``semantic_guard``.

Before 0.18.x this gap actually existed for ``bedrock_runtime`` and
was caught by this test (after we added it). Keep this test green:
any new ``_patches/*.py`` that calls ``gate_call`` / ``async_gate_call``
MUST pass ``extract_output_signals=`` next to ``forward=``, with one
documented exception — ``bedrock_agent`` (managed agent, stream is
caller-iterated and we can't replay it without breaking the user's
loop).
"""

from __future__ import annotations

import ast
from pathlib import Path

# Patches that intentionally do NOT pass extract_output_signals.
# Each entry needs a short justification in the comment below.
PATCHES_WITHOUT_OUTPUT_EXTRACTOR: set[str] = {
    # InvokeAgent returns an EventStream the caller iterates exactly
    # once; we can't transparently replay it to run an output gate.
    # Input-side enforcement is still in place; the gap is documented
    # in the patch's module docstring.
    "bedrock_agent.py",
}


def _patches_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "src" / "egisai" / "_patches"


def _gate_call_sites(path: Path) -> list[ast.Call]:
    """Return every ``gate_call`` / ``async_gate_call`` invocation in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    targets = {"gate_call", "async_gate_call"}
    sites: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in targets:
                sites.append(node)
            elif isinstance(fn, ast.Attribute) and fn.attr in targets:
                sites.append(node)
    return sites


def _has_kwarg(call: ast.Call, name: str) -> bool:
    return any(
        isinstance(kw, ast.keyword) and kw.arg == name for kw in call.keywords
    )


def test_every_gate_call_site_wires_output_extractor() -> None:
    """Each ``gate_call`` invocation outside the exemption list must
    pass an ``extract_output_signals=`` keyword. Skipping it silently
    drops the entire post-model policy phase.
    """
    failures: list[str] = []
    for patch_file in sorted(_patches_dir().glob("*.py")):
        if patch_file.name.startswith("_"):
            # Skip internal helpers (``_common.py``, ``_framework.py``).
            continue
        if patch_file.name in PATCHES_WITHOUT_OUTPUT_EXTRACTOR:
            continue
        for call in _gate_call_sites(patch_file):
            if not _has_kwarg(call, "extract_output_signals"):
                failures.append(
                    f"{patch_file.name}:{call.lineno} — "
                    f"gate_call without extract_output_signals "
                    f"(output phase will silently skip — see "
                    f"security-and-compliance.mdc rule §2)"
                )
    assert not failures, (
        "Output-phase gap regression — add extract_output_signals "
        "to every flagged call, or document the exemption in "
        "PATCHES_WITHOUT_OUTPUT_EXTRACTOR with the architectural "
        "reason. Failures:\n  " + "\n  ".join(failures)
    )


def test_bedrock_agent_exemption_is_real() -> None:
    """Verify the documented exemption file exists. If someone deletes
    bedrock_agent.py, the exemption set is stale and we should know.
    """
    for name in PATCHES_WITHOUT_OUTPUT_EXTRACTOR:
        assert (_patches_dir() / name).exists(), (
            f"Exempted patch file {name!r} is missing — clean up the "
            "PATCHES_WITHOUT_OUTPUT_EXTRACTOR set."
        )
