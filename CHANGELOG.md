# Changelog

All notable changes to `egisai` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.13.1] — 2026-05-06

### Added

- **Runtime fingerprint shipped on `/v1/sdk/handshake`.** When the
  API key is bound to a specific agent, the handshake now stamps
  the platform-side runtime blob onto that agent's Provenance card
  immediately — without waiting for the first `set_context(agent=…)`
  call. Sub-agents continue to be captured via
  `/v1/sdk/agents/ensure` as before. Older backends ignore the
  field; older SDKs against new backends behave identically to
  pre-0.13.1.

### Changed

- `egisai._backend.handshake()` accepts an optional `runtime`
  kwarg. Internal API; user code is unaffected.

---

## [0.13.0] — 2026-05-10

### Added

- **Agent Identity capture on `/v1/sdk/agents/ensure`.** Every
  call to `set_context(agent="…")` now ships a small platform-side
  *runtime fingerprint* (Python version, OS, framework versions,
  container / serverless hints, SDK version) alongside the agent
  name. The platform stamps it onto the agent's Provenance card
  on the dashboard, refreshes it on every redeploy, and uses
  deltas to detect `runtime_change` anomalies. Privacy: see the
  `egisai/_runtime.py` module docstring — no hostname, no IP, no
  env vars, no user paths leave the process.
- **`set_context(end_user_id="…")`.** New optional context field
  that ties a governed call to an opaque end-user identifier.
  The platform hashes it on intake; the SDK encourages callers to
  ship a SHA-256 already (e.g.
  `hashlib.sha256(customer_id.encode()).hexdigest()`) so a real
  customer-id never lands in a network call. Powers per-end-user
  behavioral roll-ups inside the new Agent Identity modal on the
  dashboard.
- **Agent codename + glyph (server-side).** The platform now
  derives a deterministic, human-friendly codename
  (e.g. *Crimson-Falcon*) and a stable visual glyph seed from
  every agent's UUID. Both surface on the dashboard's Agents
  table and the new Agent Identity modal. No SDK changes
  required — the SDK's own contract is unchanged for callers
  not interested in identity surfaces.

### Changed

- **`ensure_agent` SDK helper signature.**
  `egisai._backend.ensure_agent(name, description=None)` gains an
  optional `runtime: dict | None = None` parameter. Older backends
  silently ignore unknown payload keys, so calling 0.13.0 against
  a 0.12.x platform is safe (the runtime blob is dropped on the
  floor, identity gracefully falls back to UUID-derived defaults).

### Notes

- This release adds *capture* — the platform-side analyzer
  (anomaly detection, twin detection, behavioral classification)
  ships in the same platform release (0030) and reads only the
  post-sanitization fields (`payload_preview`, `response_preview`,
  `policy_reason`, `verdict`, `model`) per the security contract
  in `security-and-compliance.mdc`. No raw prompt or response
  text leaves the SDK boundary, ever.

---

## [0.12.5] — 2026-05-06

### Changed

- **Post-model evaluation now runs deterministic-first, LLM-second.**
  `evaluate_output_policies` was refactored to mirror
  `evaluate_policies` exactly: local checks (`pii_scan`,
  `deny_output_regex`, `max_prompt_chars`, `allow_model`,
  `deny_tool_call`, `deny_bash_command`, `deny_mcp_call`) all
  run as Phase 1 against the response, and `semantic_guard`
  runs as Phase 2 only if Phase 1 didn't already block. Same
  security contract the prompt side has always honored
  (`security-and-compliance.mdc` §2): once a deterministic rule
  refuses a response, the LLM judge is never consulted — no
  network call, no token spend, no chance of the response
  reaching an external model. List order is irrelevant; the
  split is entirely type-driven.
- **Phase × type matrix is fully open.** Every rule type now
  accepts every phase (`pre_model`, `post_model`, `both`).
  Operators can target any rule on either side of a call without
  the dashboard refusing the combination. The engine evaluates
  each rule on whichever side it has meaningful signals for and
  silently no-ops the rest, so the freedom can't break the gate.
- **Phase-symmetric evaluators.** Rule types that look at text or
  the model name now fire on either side, with side-specific
  reason codes so the audit narrative reads correctly:

  - `pii_scan` — runs on the response too (`pii_in_output`
    reason). `action="sanitize"` on the response side is coerced
    to block; the SDK can't safely rewrite provider response
    payloads, so the operator's intent is preserved by refusing
    the response.
  - `deny_regex` / `deny_output_regex` — interchangeable; on the
    prompt side both emit `prompt_blocked`, on the response side
    both emit `output_blocked`.
  - `max_prompt_chars` — caps response size when scoped to
    `post_model` (`output_too_large` reason).
  - `allow_model` — identical check on either side
    (`model_not_allowed` reason).
  - `semantic_guard` — already symmetric; unchanged.

  Tool/bash/MCP rules (`deny_tool_call`, `deny_bash_command`,
  `deny_mcp_call`) still need response-side signals, so they
  silently no-op when an operator targets them on `pre_model`.
  They fire normally whenever the phase includes `post_model`.

### Internal

- New `tests/test_cross_side_evaluators.py` pins the symmetry
  contract: 18 assertions covering each type × side combination,
  the side-specific reason codes, and the silent no-ops on the
  prompt side for tool/bash/MCP rules.
- New `tests/test_post_model_two_phase.py` pins the deterministic-
  before-LLM contract on the response side: a recording stub
  blocker proves the judge is never invoked when Phase 1 blocks
  via `pii_scan`, `deny_output_regex`, or `deny_tool_call`, and
  is invoked exactly once when Phase 1 allows. Order of rules in
  the policy list is varied to confirm the split is type-driven.

---

## [0.12.4] — 2026-05-05

### Added

- **Two-phase policy enforcement.** Each policy now carries a `phase`
  field that selects which side of a model call it runs on:
  `"pre_model"` (the prompt, before the call), `"post_model"` (the
  response, after the call), or `"both"`. Operators can scope a
  single rule to either side or keep the legacy "wherever it
  applies" behaviour. Older platform responses that don't carry the
  field default to `"both"`, preserving every previous deployment's
  semantics.
- **Per-phase decision blocks on the audit event.** The SDK now
  emits two structured blocks alongside the legacy top-level fields:

  - `prompt_decision` — verdict, reason, and matched policies for
    the pre-model phase. Always present.
  - `response_decision` — verdict, reason, and matched policies for
    the post-model phase. Present only when the phase actually ran
    (the model returned and an output extractor produced signals to
    evaluate). Absent when the prompt was blocked, since the model
    was never called.

  The legacy `verdict` / `matched_policy` / `matched_policies`
  fields stay for backward compatibility with existing backends.

### Changed

- `evaluate_policies` and `evaluate_output_policies` now filter
  rules by `phase` before walking them. A rule scoped to
  `"post_model"` will not fire on the prompt side, and vice versa,
  even when the rule's *type* is technically valid on the
  un-scoped side (e.g. `semantic_guard`).

### Internal

- New tests cover phase filtering across both evaluators, the wire
  parser's default behaviour, and the dual-decision audit shape on
  allow / pre-model-block / post-model-block paths.

---

## [0.11.1] — 2026-05-05

### Added

- **Support for `google-genai`.** The Google Gen AI SDK
  (`from google import genai`) is now patched directly. Both
  `client.models.generate_content(...)` and the async sibling on
  `client.aio.models.generate_content(...)` are governed end-to-end, including
  the streaming variants. The `google-generativeai` patcher continues to
  operate alongside it.

### Changed

- The `google` extra now installs `google-genai`. Use
  `pip install "egisai[google]"` for `google-genai` or
  `pip install "egisai[google-legacy]"` for `google-generativeai`. Both
  extras are independent and can be installed together.
- Documentation and integration guides updated to cover both Google SDKs.

### Internal

- New patcher module `egisai._patches.genai`, mirroring the structure of
  the existing patchers and registered alongside them in `egisai.init()`.
- Tests added for sync, async, allow, block (raise), block (stub), idempotent
  re-apply, and the no-op behavior when `google.genai` is not installed.

---

## [0.11.0] — 2026-05-05

### Added

- **Output-side policy enforcement.** Rules of type `deny_tool_call`,
  `deny_mcp_call`, `deny_output_regex`, and `deny_bash_command` now run on the
  model's response in addition to the request. The framework patchers extract
  assistant text, tool invocations, and MCP targets from OpenAI (Chat
  Completions and Responses), Anthropic, and Google Gemini responses before
  the call returns to your code.
- **Async `semantic_guard`.** `SemanticBlocker.acheck()` uses an
  `httpx.AsyncClient` so async model calls no longer block the event loop while
  the platform's intent judge evaluates the prompt.
- **Configurable fail-closed for `semantic_guard`.** New `semantic_on_outage`
  option on `egisai.init()` (`"allow"` by default, `"block"` to fail closed)
  controls how the SDK behaves when the intent-judge endpoint is unavailable.
- **Audit-event drop accounting.** When the audit queue is full (sustained
  platform outage with no drain), the oldest events are dropped instead of the
  newest, a counter is incremented, and a warning is logged at exponential
  thresholds (1, 10, 100, 1 000, …).
- **`egisai.diagnostics()`.** Returns a small dict describing the SDK runtime
  health (initialised yes/no, queue depth, drop counter, configured
  integrations, policy count). Useful as a `/healthz` data source.
- **Reserved-domain handling for email PII.** Addresses that use the
  RFC 2606 / 6761 reserved domains (`*.test`, `*.example`, `*.invalid`,
  `*.localhost`, `example.com`, `example.net`, `example.org`) are now
  consistently treated as documentation samples rather than personal data.

### Changed

- **Stronger ReDoS guard.** The regex pre-compile validator now rejects
  patterns with five or more optional / star quantifiers in a short window
  (`a?a?a?a?a?aaaaa`-shaped runaway patterns), in addition to nested-quantifier
  shapes. The runtime watchdog around `re.search` keeps protecting calls that
  release the GIL.
- **SHA-256 for agent identity fingerprints.** Replaces SHA-1 to remove the
  weak hash from the SDK code path, even though the value is non-security
  (deduplicating fingerprints, not authenticating callers).
- **Library-grade logging.** Failure paths previously printed to stderr are now
  logged through `logging.getLogger("egisai.*")` so they can be routed,
  captured, or silenced by the host application.

### Fixed

- `_handle_sse` no longer eagerly parses the SSE event payload — only the
  trigger is needed and the body is treated as opaque.

### Internal

- Tests added for output-side policy wiring, async `semantic_guard`,
  fail-closed semantics, drop accounting, the `diagnostics()` helper, and the
  reserved-email-domain matrix.

---

## [0.10.0] — 2026-05-05

First public release of the **egisai** Python SDK from the open-source
repository at [`EgisLabs/egisai-sdk`](https://github.com/EgisLabs/egisai-sdk).

The SDK provides runtime governance for AI agents: import-time patching of
OpenAI, Anthropic, Google Gemini, and common HTTP clients so governed calls
are evaluated against policies from the EgisAI platform, with deterministic
local checks (PII, regex, size, model allow-list) and audit telemetry that
mirror operator-defined rules.
