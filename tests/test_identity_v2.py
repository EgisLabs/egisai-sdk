"""Identity v2 — model-invariant, prompt-evolution-aware identity.

The contract under test, end to end:

* **Model invariance** — neither the model id in a framework bundle
  nor a model name embedded in prompt *text* ("powered by GPT-5")
  may change an agent's identity hash. Model switches (user choice
  or Smart Model Routing) are observed metadata, never identity.
* **Anchor dominance** — a framework's declared name is the whole
  identity; instruction/tool edits on a named agent are revisions.
* **Continuity** — every changed recipe ships the exact v1 hash as
  ``identity_hash_legacy`` so the backend re-stamps existing agents
  in place instead of forking on SDK upgrade.
* **Reconciliation signals** — canonical-prompt SimHash + tool-
  bundle hash ride the ensure payload; both are non-reversible
  digests (privacy contract unchanged).
"""

from __future__ import annotations

from typing import Any

import egisai
from egisai._auto_agent import (
    IDENTITY_ALGO_VERSION,
    _hash_bundle,
    _payload_tool_names,
    canonicalize_identity_text,
    derive_prompt_identity,
    hamming64,
    resolve_identity,
    simhash64_hex,
    tool_bundle_hash_from_names,
)

_CURSOR_GPT = (
    "You are an AI coding assistant, powered by GPT-5. "
    "You operate in Cursor. Follow the user's instructions carefully."
)
_CURSOR_CLAUDE = (
    "You are an AI coding assistant, powered by Claude Sonnet 4.5. "
    "You operate in Cursor. Follow the user's instructions carefully."
)


def _init(fake_backend: Any) -> None:
    egisai.init(
        api_key="egis_live_x",
        app="identity-v2-app",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )


# ── Canonicalizer: model-token masking ──────────────────────────────


def test_canonicalizer_masks_model_families_identically() -> None:
    """Every way of naming a model collapses to the same placeholder."""
    variants = [
        _CURSOR_GPT,
        _CURSOR_CLAUDE,
        _CURSOR_GPT.replace("GPT-5", "o3-mini"),
        _CURSOR_GPT.replace("GPT-5", "gemini-2.5-pro"),
        _CURSOR_GPT.replace("GPT-5", "llama-3.1-70b-instruct"),
        _CURSOR_GPT.replace("GPT-5", "deepseek-r1"),
        _CURSOR_GPT.replace(
            "GPT-5", "anthropic.claude-3-haiku-20240307-v1:0"
        ),
    ]
    canonicals = {canonicalize_identity_text(v) for v in variants}
    assert len(canonicals) == 1, canonicals
    assert "<model>" in next(iter(canonicals))


def test_canonicalizer_preserves_ordinary_prose() -> None:
    """Weak family words in prose are NOT masked (no digit tail)."""
    for text in (
        "Write a sonnet about the sea.",
        "Compute the phi coefficient for the dataset.",
        "The opus was performed at the palm court.",
        "Run the command and check the titan crane.",
    ):
        assert canonicalize_identity_text(text) == text


def test_canonicalizer_masks_weak_families_with_versions() -> None:
    assert "<model>" in canonicalize_identity_text("Use Sonnet 4.5 here.")
    assert "<model>" in canonicalize_identity_text("Prefer opus-4 for this.")


def test_canonicalizer_sentence_punctuation_parity() -> None:
    """A version tail must never eat the sentence period — otherwise
    "GPT-5." and "Claude." would canonicalize differently."""
    a = canonicalize_identity_text("powered by GPT-5.")
    b = canonicalize_identity_text("powered by Claude.")
    assert a == b == "powered by <model>."


def test_canonicalizer_does_not_mangle_names_containing_families() -> None:
    """"Claudette" must survive — family tokens only match whole
    words (or with digit-led version tails)."""
    text = "You are Claudette, an accountant."
    assert canonicalize_identity_text(text) == text


def test_canonicalizer_distinct_personas_stay_distinct() -> None:
    a = canonicalize_identity_text("You are a Researcher. Dig deep.")
    b = canonicalize_identity_text("You are a Copywriter. Be witty.")
    assert a != b


# ── derive_prompt_identity: dual hash ───────────────────────────────


def test_prompt_identity_model_swap_same_digest_different_legacy() -> None:
    """The v2 digest is model-invariant; the v1 legacy digest is not
    (that's exactly why v1 forked agents on model switches)."""
    a = derive_prompt_identity(_CURSOR_GPT)
    b = derive_prompt_identity(_CURSOR_CLAUDE)
    assert a.digest == b.digest
    assert a.legacy_digest != b.legacy_digest
    assert len(a.digest) == 64


def test_prompt_identity_no_model_tokens_digests_converge() -> None:
    """A prompt without model chrome hashes identically under both
    recipes — the SDK then skips shipping a redundant legacy hash."""
    pid = derive_prompt_identity("You are a Payment Agent. Verify accounts.")
    assert pid.digest == pid.legacy_digest


# ── SimHash ─────────────────────────────────────────────────────────


def test_simhash_locality_light_edit_within_threshold() -> None:
    base = (
        "You are the Payment Execution Agent. Verify beneficiaries, "
        "check limits, and submit payments to core banking. Payments "
        "over 10000 EUR stage for human approval."
    )
    edited = (
        "You are the Payment Execution Agent. Verify beneficiaries, "
        "check limits carefully, and submit payments to core banking "
        "systems. Payments over 10000 EUR stage for human approval."
    )
    unrelated = (
        "You are the Fraud Monitoring Agent. Score transaction risk "
        "and compare behaviour to historical baselines. Raise alerts "
        "with evidence and freeze suspect activity."
    )
    s_base = simhash64_hex(canonicalize_identity_text(base))
    s_edit = simhash64_hex(canonicalize_identity_text(edited))
    s_other = simhash64_hex(canonicalize_identity_text(unrelated))
    assert s_base and s_edit and s_other
    # Same threshold the backend reconciliation uses
    # (SIMHASH_MAX_HAMMING = 14 of 64).
    assert hamming64(s_base, s_edit) <= 14
    assert hamming64(s_base, s_other) > 14


def test_simhash_short_prompts_return_none() -> None:
    """< 8 tokens → no simhash: too few shingles for a meaningful
    fingerprint (measured: unrelated short prompts collide within
    any usable threshold). Short prompts never auto-reconcile."""
    assert simhash64_hex("You are a Researcher.") is None


def test_simhash_is_deterministic_hex16() -> None:
    text = "one two three four five six seven eight nine ten"
    a = simhash64_hex(text)
    b = simhash64_hex(text)
    assert a == b
    assert a is not None and len(a) == 16
    int(a, 16)  # valid hex


# ── Tool-bundle hash ────────────────────────────────────────────────


def test_tool_bundle_hash_order_and_duplicates_invariant() -> None:
    a = tool_bundle_hash_from_names(["read_file", "write_file"])
    b = tool_bundle_hash_from_names(["write_file", "read_file", "read_file"])
    assert a == b
    assert tool_bundle_hash_from_names([]) != a


def test_payload_tool_names_openai_anthropic_bedrock_shapes() -> None:
    openai_shape = {
        "tools": [{"type": "function", "function": {"name": "get_weather"}}],
    }
    anthropic_shape = {"tools": [{"name": "get_weather"}]}
    bedrock_shape = {
        "toolConfig": {"tools": [{"toolSpec": {"name": "get_weather"}}]},
    }
    assert _payload_tool_names(openai_shape) == ["get_weather"]
    assert _payload_tool_names(anthropic_shape) == ["get_weather"]
    assert _payload_tool_names(bedrock_shape) == ["get_weather"]


# ── Tier 5 resolver: model invariance end-to-end ────────────────────


def test_tier5_model_in_prompt_does_not_fork_agent(fake_backend: Any) -> None:
    """THE bug this release fixes: Cursor-style prompts that embed
    the model name resolve to ONE agent across model switches."""
    _init(fake_backend)
    r1 = resolve_identity(
        {"system": _CURSOR_GPT, "model": "gpt-5"},
        auto_stack_hints="off",
    )
    r2 = resolve_identity(
        {"system": _CURSOR_CLAUDE, "model": "claude-sonnet-4-5"},
        auto_stack_hints="off",
    )
    assert r1 is not None and r2 is not None
    assert r1.identity_hash == r2.identity_hash
    assert r1.agent_id == r2.agent_id
    # Exactly one ensure round-trip for the shared identity.
    ensures = [
        b for b in fake_backend.ensure_requests
        if b.get("identity_source") == "hash"
    ]
    assert len(ensures) == 1


def test_tier5_ensure_ships_v2_continuity_fields(fake_backend: Any) -> None:
    _init(fake_backend)
    resolve_identity(
        {
            "system": _CURSOR_GPT,
            "model": "gpt-5",
            "tools": [
                {"type": "function", "function": {"name": "run_terminal"}},
            ],
        },
        auto_stack_hints="off",
    )
    body = next(
        b for b in fake_backend.ensure_requests
        if b.get("identity_source") == "hash"
    )
    assert body.get("identity_version") == IDENTITY_ALGO_VERSION
    # The prompt embeds a model token, so v1 and v2 digests differ
    # and the legacy hash MUST ride along for continuity.
    assert body.get("identity_hash_legacy")
    assert body["identity_hash_legacy"] != body["identity_hash"]
    assert body.get("identity_simhash") and len(body["identity_simhash"]) == 16
    assert body.get("tool_bundle_hash") == tool_bundle_hash_from_names(
        ["run_terminal"]
    )
    assert body.get("model") == "gpt-5"
    # Privacy: no raw prompt text beyond the sanitized excerpt field.
    assert _CURSOR_GPT not in repr(
        {k: v for k, v in body.items() if k != "system_prompt_excerpt"}
    )


def test_tier5_legacy_hash_matches_v1_recipe(fake_backend: Any) -> None:
    """The shipped legacy hash must be byte-identical to what a v1
    SDK would have computed — otherwise backend continuity misses."""
    import hashlib
    import re
    import unicodedata

    _init(fake_backend)
    resolve_identity({"system": _CURSOR_GPT}, auto_stack_hints="off")
    body = next(
        b for b in fake_backend.ensure_requests
        if b.get("identity_source") == "hash"
    )
    v1_normalized = re.sub(
        r"\s+", " ", unicodedata.normalize("NFKC", _CURSOR_GPT)
    ).strip()
    v1_digest = hashlib.sha256(v1_normalized.encode("utf-8")).hexdigest()
    assert body["identity_hash_legacy"] == v1_digest


# ── Framework bundles: model invariance ─────────────────────────────


def test_claude_agent_sdk_bundle_model_and_permission_invariant() -> None:
    from egisai._patches.claude_agent_sdk import _bundle_from_options

    class _Opts:
        def __init__(self, model: str, permission_mode: str) -> None:
            self.system_prompt = (
                "You are the AML Compliance Agent. Screen transactions "
                "against sanctions lists and file SARs."
            )
            self.allowed_tools = ["Read", "Grep"]
            self.permission_mode = permission_mode
            self.model = model
            self.mcp_servers = {"core-banking": object()}

    _, _, bundle_a, legacy_a, tools_a, model_a = _bundle_from_options(
        _Opts("claude-sonnet-4-5", "acceptEdits")
    )
    _, _, bundle_b, legacy_b, _, _ = _bundle_from_options(
        _Opts("claude-opus-4", "bypassPermissions")
    )
    # v2: same identity across model + permission_mode changes.
    assert _hash_bundle(bundle_a) == _hash_bundle(bundle_b)
    # legacy: still distinct (that's what it reproduces).
    assert _hash_bundle(legacy_a) != _hash_bundle(legacy_b)
    assert tools_a == ["Grep", "Read"]
    assert model_a == "claude-sonnet-4-5"


def test_claude_agent_sdk_distinct_prompts_stay_distinct() -> None:
    from egisai._patches.claude_agent_sdk import _bundle_from_options

    class _Opts:
        def __init__(self, prompt: str) -> None:
            self.system_prompt = prompt
            self.allowed_tools: list[str] = []
            self.permission_mode = ""
            self.model = ""
            self.mcp_servers: dict[str, Any] = {}

    _, _, a, _, _, _ = _bundle_from_options(
        _Opts("You are the KYC Onboarding Agent. Verify identity documents.")
    )
    _, _, b, _, _, _ = _bundle_from_options(
        _Opts("You are the Fraud Monitoring Agent. Score transaction risk.")
    )
    assert _hash_bundle(a) != _hash_bundle(b)


# ── Framework patches: anchor + model invariance via the wire ──────


def test_smolagents_model_swap_same_identity(fake_backend: Any) -> None:
    from egisai._patches.smolagents import _derive

    _init(fake_backend)

    class _Model:
        def __init__(self, model_id: str) -> None:
            self.model_id = model_id

    class _Agent:
        name = "Research Crawler"

        def __init__(self, model_id: str) -> None:
            self.model = _Model(model_id)
            self.tools = {"web_search": object()}

    r1 = _derive(_Agent("gpt-4o"))
    r2 = _derive(_Agent("claude-sonnet-4-5"))
    assert r1 is not None and r2 is not None
    assert r1.identity_hash == r2.identity_hash


def test_pydantic_ai_model_swap_same_identity(fake_backend: Any) -> None:
    from egisai._patches.pydantic_ai import _derive

    _init(fake_backend)

    class _Model:
        def __init__(self, name: str) -> None:
            self.model_name = name
            self.name = name

    class _Agent:
        name = ""
        system_prompt = (
            "You are the Loan Underwriting Agent. Score applications "
            "against the credit policy."
        )

        def __init__(self, model: str) -> None:
            self.model = _Model(model)

    r1 = _derive(_Agent("openai:gpt-4o"))
    r2 = _derive(_Agent("anthropic:claude-sonnet-4-5"))
    assert r1 is not None and r2 is not None
    assert r1.identity_hash == r2.identity_hash


def test_crewai_role_anchor_survives_goal_rewrite(fake_backend: Any) -> None:
    from egisai._patches.crewai import _derive

    _init(fake_backend)

    class _Agent:
        role = "Senior Market Analyst"

        def __init__(self, goal: str) -> None:
            self.goal = goal
            self.backstory = "Veteran analyst."
            self.tools: list[Any] = []

    r1 = _derive(_Agent("Analyse EU markets"))
    r2 = _derive(_Agent("Analyse EU markets with a focus on energy"))
    assert r1 is not None and r2 is not None
    assert r1.identity_hash == r2.identity_hash


def test_framework_ensure_ships_legacy_hash(fake_backend: Any) -> None:
    """A named framework agent still ships its v1 composite hash so
    the backend can migrate the existing row in place."""
    from egisai._patches.crewai import _derive

    _init(fake_backend)

    class _Agent:
        role = "Compliance Officer"
        goal = "Review filings"
        backstory = "Ten years in audit."
        tools: list[Any] = []

    _derive(_Agent())
    body = next(
        b for b in fake_backend.ensure_requests
        if b.get("name") == "Compliance Officer"
    )
    legacy = _hash_bundle(
        ("crewai", "Compliance Officer", "Review filings",
         "Ten years in audit.", ()),
    )
    assert body.get("identity_hash_legacy") == legacy
    assert body.get("identity_hash") == _hash_bundle(
        ("crewai", "Compliance Officer"),
    )
    assert body.get("identity_version") == IDENTITY_ALGO_VERSION
