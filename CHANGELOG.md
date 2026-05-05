# Changelog

All notable changes to `egisai` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
