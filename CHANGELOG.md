# Changelog

All notable changes to `egisai` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.39.0] — 2026-07-21

### Added

- Access-inventory capture for the Claude Agent SDK. Tools declared
  via `ClaudeAgentOptions` (`mcp_servers` in-process SDK servers with
  full metadata, `allowed_tools` built-ins) and the CLI's `init`
  system message runtime toolset now report to the dashboard's
  per-agent Access tab. MCP-served tools ship under their runtime
  invocation name (`mcp__<server>__<tool>`) plus a `server_name`
  field so declared inventory joins observed usage. Previously,
  agents governed through `claude_agent_sdk` showed an empty Access
  tab because this framework never declares tools in the request
  payload.
- `server_name` on payload-extracted MCP-tagged tool items.
- Multi-source declarations merge monotonically per agent (options +
  init message union) so the bundle hash converges instead of
  re-reporting every turn.

---

## [0.38.0] — 2026-07-21

### Added

- **Agent access-inventory reporting** for the dashboard's new
  per-agent **Access** tab. The SDK now derives a metadata-only
  inventory of what each agent can reach from the tool bundle it
  already sees on every call — tool names, PII-sanitized
  descriptions, a SHA-256 hash of each declared input schema, and
  parameter *names* (never schemas, never call arguments) — and
  reports it to `POST /v1/sdk/agents/access` the first time an
  agent's bundle hash is seen. Steady-state cost is one dict lookup
  per call; the report runs on a daemon thread and fails open, so a
  backend outage never touches the customer's call path. Covers
  OpenAI (v1 + legacy), Anthropic, Google, Bedrock Converse
  (`toolConfig`), and MCP-tagged tool definitions.

---

## [0.37.0] — 2026-07-16

### Added

- **BYOK vault mode for `egisai.Client` — `provider_key` is now
  optional.** Store your provider keys once on the dashboard's
  Gateway page (encrypted at rest) and construct the client with just
  your Egis key: `egisai.Client(api_key="egis_live_…")`. The Gateway
  resolves the right provider key for each request from the model
  name and forwards it upstream — no provider key in your code, no
  custom headers. This is the same server-side vault that lets
  header-less platforms (Cursor, n8n, low-code tools) connect to the
  Gateway with only their Egis key in the `Authorization` bearer.
  When `provider_key` is supplied it is forwarded untouched exactly
  as before (passthrough); `OPENAI_API_KEY` still works as the legacy
  default-provider fallback. Fully backward compatible — existing
  code that passes `provider_key` is unaffected.

## [0.36.0] — 2026-07-15

### Added

- **Per-agent governance opt-out ("monitor only" mode).** Operators
  can now turn policy enforcement off for a specific agent from the
  dashboard without stopping its traffic. Calls attributed to an
  ungoverned agent skip every policy phase — no deterministic
  checks, no semantic-guard LLM judge, no sanitization — and flow
  to the model untouched, while event logging stays on so the
  Requests page keeps full visibility. The SDK learns the set via
  a new `ungoverned_agent_ids` field on the `/v1/sdk/policies`
  snapshot (ETag-versioned in lockstep with the rule list; flips
  take effect within ~50 ms via the existing `agent.changed` SSE
  ping). The operator pause kill switch takes precedence: a paused
  agent is refused even if it is also ungoverned. Older backends
  that don't ship the field behave exactly as before — every agent
  stays governed, the safe (enforcing) direction.

## [0.35.1] — 2026-07-11

### Fixed

- Release only, no code change. The 0.35.0 release never reached
  PyPI: the release tag was mis-named (`0.35.0.0.0`, which the
  ``v*``-triggered workflow ignores) and the corrected tag name then
  collided with a stale local ref in the mirror checkout. This
  version carries the full 0.35.0 payload below.

## [0.35.0] — 2026-07-09

### Added

- **Gateway calls now carry the full `set_context` request context.**
  Previously only the explicit agent identity crossed the Gateway wire
  (`X-Egis-Agent`); the other context fields applied to in-process
  governance only, so gateway-audited runs showed an empty Context
  section on the dashboard. `inject_headers` now ships `user_id` /
  `user_role` / `session_id` / `workflow_id` / `end_user_id` as
  `X-Egis-User` / `X-Egis-User-Role` / `X-Egis-Session` /
  `X-Egis-Workflow` / `X-Egis-End-User`. Values are percent-encoded
  UTF-8 on the wire (HTTP header values must be latin-1-safe or the
  transport would raise inside the customer's call, violating
  fail-open) and are capped to the platform's column widths before
  encoding. The platform decodes on intake, re-hashes the end-user id
  (the raw value is never persisted or logged), stamps the run's
  Context section, and folds the observation into the per-end-user
  roll-up — identical to SDK-audited traffic. Caller-supplied
  `X-Egis-*` headers still win on conflict, and context collection
  fails open: a header that can't be built degrades the audit row's
  metadata, never the call.

## [0.34.2] — 2026-07-08

### Changed

- Documentation only, no code change. README: the "Transparent
  integration" overview row now presents the inline Gateway as the
  network-path alternative to in-process patching ("no mandatory
  proxy") and cross-links the Gateway mode section. SECURITY.md: the
  vulnerability-report scope now explicitly includes the inline
  Gateway endpoint (`/v1/chat/completions`) that `egisai.Client`
  and `init(gateway=True)` send traffic through.

## [0.34.1] — 2026-07-07

### Fixed

- Release CI: `mypy` failed on `egisai/_client.py` in environments
  without the optional `openai` extra installed (the exact setup the
  public release pipeline uses). The lazy `import openai` inside
  `egisai.Client` now carries the same `import-not-found` ignore as
  the openai patch. No runtime behavior change; 0.34.0 never
  published because of this failure, so this release carries the
  full 0.34.0 payload below.

## [0.34.0] — 2026-07-07

### Added

- **`egisai.Client` / `egisai.AsyncClient` — the Gateway-first
  client.** `import egisai` is now the only import needed to send
  governed traffic: `egisai.Client(api_key="egis_live_…",
  provider_key="sk-ant-…")` exposes the familiar
  `.chat.completions.create(...)` surface (streaming included) and
  always talks to the platform's Gateway, which evaluates policies,
  sanitizes/blocks inline, routes to the right provider from the
  model name, and writes the audit row server-side. No `base_url`,
  no header wiring, no provider import in your code. `init()` is
  optional — the client carries its own keys — but when `init()` is
  active, per-call context (`egisai.set_context(agent=…)` /
  `with egisai.agent(…)`) rides along as `X-Egis-Agent`
  automatically. Requires the `egisai[openai]` extra (the Gateway's
  wire format); a missing dependency raises a clear install hint.
- The openai patch now recognises *any* client pointed at the
  Gateway (`egisai.Client`, `init(gateway=True)` reroutes, or a
  hand-configured `base_url`) and skips the local gate for those
  calls — governance and audit happen exactly once, server-side —
  while still injecting per-call context headers.

---

## [0.33.0] — 2026-07-07

### Added

- **Gateway mode — `egisai.init(gateway=True)`.** One flag routes
  OpenAI chat-completions calls through the platform's inline
  Gateway instead of evaluating policies in-process. The calling
  convention is unchanged (`client.chat.completions.create(...)` on
  your own client, your provider key untouched); the SDK injects
  `X-Egis-Api-Key` automatically and translates an explicit
  `egisai.set_context(agent=…)` / `with egisai.agent(...)` into the
  `X-Egis-Agent` header, so identity context works identically in
  both modes. Enforcement, sanitization, and the audit row happen
  server-side; the local gate is skipped for rerouted calls so
  nothing is governed twice. Also settable via `EGISAI_GATEWAY=1`.
  Requires the org's `inline_gateway` feature.
- Scope guardrails for the reroute: only `chat.completions.create`
  is carried (the Gateway's surface). The Responses API, Anthropic /
  Google / Bedrock SDKs, agent frameworks, and MCP keep the normal
  in-process governance path, and Azure OpenAI clients are never
  rerouted (deployment-based URLs). If the reroute can't be
  constructed the call falls back to in-process governance — fail
  open, never fail the customer's call.
- `diagnostics()` now reports `gateway_mode`.

---

## [0.32.0] — 2026-07-07

### Added

- **Call-relative phase vocabulary.** `PolicyRule.phase` now speaks
  `request` / `response` / `both` — names that read correctly for
  every governed surface (model calls, tool calls, MCP calls,
  gateway traffic), replacing the model-centric `pre_model` /
  `post_model`. The legacy spellings are still accepted on the wire
  and normalized on parse, so the SDK works against both old and
  new platform versions with identical behavior.
- **`applies_to` surface scoping.** Rules can be scoped to specific
  call surfaces (`model`, `tool`, `mcp`) via the new
  `PolicyRule.applies_to` field. Empty (the default, and the shape
  of every existing rule) means "all surfaces" — nothing changes
  for existing policies. The engine's `evaluate_policies` /
  `evaluate_output_policies` take a new optional `surfaces` keyword
  naming what the evaluation covers; single-surface gates (the
  `claude_agent_sdk` per-tool hooks, the MCP-server `tools/call`
  gate) narrow it so a rule scoped to `tool` never fires on a plain
  model prompt. Unknown surfaces from a future backend are dropped
  on parse — the rule stays active on the surfaces this SDK
  understands (over-application, the safe direction).

### Notes

- No behavioral change for existing policies: the default `surfaces`
  on both evaluators covers everything the respective phase covered
  before, and empty `applies_to` matches every surface.

---

## [0.31.0] — 2026-06-30

### Added

- **MCP Servers add-on (SDK side).** When the org has the
  `mcp_servers` add-on enabled (reported on the `init()` handshake),
  the SDK now auto-detects a hosted MCP server (FastMCP v2 or the
  official `mcp` SDK's `FastMCP`), fingerprints it from its name +
  tool schema + transport, auto-registers it via
  `POST /v1/sdk/mcp-servers/ensure` (cached after first sight), and
  reports its tool inventory — exactly mirroring how agents are
  auto-detected and named, with no new user code.
- **Inbound `tools/call` governance.** Each inbound tool call against
  the server is evaluated with the existing policy engine
  (`evaluate_output_policies`) scoped to that server: `block` raises a
  tool error so the tool never runs, `sanitize` masks PII in the call
  arguments in place, `allow` proceeds. Every outcome is reported as a
  `source_kind="mcp_server"` audit event so it shows up on the
  dashboard's Requests page and the server's profile.
- **Policy scope for MCP servers.** `PolicyRule` now carries
  `mcp_server_ids`; rules can be targeted at specific MCP servers (or
  left org-wide) and the new MCP gate honours that scope.

### Notes

- **Zero impact when the add-on is off.** The MCP patch is fully
  dormant unless the handshake reports the add-on is enabled — it
  never wraps the customer's server, registers anything, or emits
  events. Customers without the add-on get byte-for-byte the same
  behaviour as 0.30.0.
- **Fail-open.** Any unexpected error in the MCP gate falls through to
  the original tool handler so a hosted MCP server keeps serving even
  when egisai is unhappy.

## [0.30.0] — 2026-06-22

### Added

- **Auto-generated agent description + business function.** The
  first time an agent is auto-registered, the SDK now ships a
  PII-sanitised, truncated (≤ 2 KB) excerpt of its system prompt so
  the platform can generate a human-readable description (e.g.
  "Automates AML and sanctions investigations by screening
  counterparties…") and a free-form business function ("Anti Money
  Laundering (AML) And Financial Crime Compliance") in place of the
  old `Auto-detected by SDK (framework:…) identity=…` placeholder.
  Generation runs entirely in the background **on the server, after
  the SDK has its response** — zero added latency on your agent's
  call path, and no LLM dependency on the SDK side.
- **`auto_describe` opt-out.** `init(auto_describe=False)` (or
  `EGISAI_AUTO_DESCRIBE=0`) disables the excerpt entirely: no prompt
  text, even sanitised, leaves the process. The agent keeps the local
  placeholder description and its business function is inferred from
  anonymised behavioural telemetry instead.

### Security

- The system-prompt excerpt is scrubbed by the SDK's own PII engine
  **before** it leaves the process, capped at 2 KB, used transiently
  for a single server-side LLM call, and never persisted or logged on
  the backend. See `SECURITY.md` → "Agent descriptor".

---

## [0.29.2] — 2026-06-21

### Fixed

- **`claude_agent_sdk` no longer hides failed model calls as
  clean zero-token runs.** When the Claude Code CLI can't run the
  requested model — the model id isn't available to the
  `ANTHROPIC_API_KEY`, a 429/5xx overload, a provider outage — it
  returns `ResultMessage.is_error` with an `api_error_status` and a
  diagnostic string, *but still* reports `subtype="success"` and a
  zeroed `usage` block. The wrapper read `usage` (so tokens were
  `0`) but ignored `is_error`, recording the turn as a clean
  `verdict=allow` run that "completed successfully" with `0 in /
  0 out` and `$0`. That is the single most common cause of "why are
  my Claude Agent SDK tokens 0?" — the model never ran, the tokens
  were never spent. The terminal handler now copies `is_error` /
  `api_error_status` / the diagnostic onto the audit row
  (`error` + `api_error_status`) and threads it into `close_run`,
  so the dashboard renders an errored run with an actionable reason
  instead of a silent zero. `result` is read only on the error path
  (a CLI diagnostic string, never a model completion), so the
  never-persist-model-output contract is unaffected.
- **`claude_agent_sdk` runs no longer under-report model tokens.**
  The Claude Agent SDK prompt-caches the system prompt and tool
  schema aggressively, so a turn's uncached `input_tokens` is
  routinely a single-digit remainder while the real prompt the
  model processed lives in `cache_read_input_tokens` /
  `cache_creation_input_tokens`. `_stamp_usage_from_result`
  previously read only `input_tokens`, so the dashboard's
  "Model In" showed e.g. `6` for a turn that actually fed the
  model tens of thousands of tokens. Input tokens are now the
  sum of the uncached count and both cache counters.
- **Token capture is robust to a zeroed aggregate `usage` block.**
  When `ResultMessage.usage` is missing or all-zero, the stamp now
  falls back to the authoritative per-model `model_usage`
  breakdown (camelCase `inputTokens` / `outputTokens` /
  `cacheReadInputTokens` / `cacheCreationInputTokens` / `costUSD`)
  before giving up on the row. Cost falls back the same way
  (`total_cost_usd` → summed `costUSD`).

### Note

- A `claude_agent_sdk` run that still shows `0 in / 0 out` and
  `$0` is the signature of a model call that **never actually ran**
  — typically the selected model id is not available to the
  `ANTHROPIC_API_KEY` in use (the CLI returns `is_error` with a
  zeroed `usage` block and an empty `model_usage`), or a transient
  provider error. It is not a lost token count; verify model
  access and retry.

---

## [0.29.0] — 2026-06-02

This release is entirely about cutting policy enforcement
latency. A field report of a 12-second `policy_latency_ms` row
on the dashboard kicked off an audit of every code path between
the SDK's call gate and the platform's `/v1/sdk/judge` endpoint.
The audit found five compounding issues — duplicated work, a
serialised inner loop, blocking calls on the asyncio event loop,
unbounded retry sleeps, and a one-shot library cold-start
silently bundled into per-call governance time. All five ship
fixed here. Combined effect on the worst-case turn that
prompted the report: ~12 s → ~1 s. Steady-state cost on the
common no-match path drops from O(N × P50) to O(P50). Behaviour
is fully backwards-compatible — every fix is internal to the
SDK and no public API or wire-shape changed.

### Performance

- **`claude_agent_sdk` no longer double-judges tool calls.** When
  `PreToolUse` hooks are wired (the default for `0.21+` against
  the modern `claude_agent_sdk`), the post-turn `_run_output_phase`
  used to re-evaluate the same `tool_calls` list a `semantic_guard`
  rule with `targets=["tool_calls"]` had already gated per-tool.
  An N-tool turn paid 2N judge round-trips. The output phase now
  drops `tool_calls` / `tool_names` / `mcp_targets` when hooks
  were active — text-side rules
  (`deny_output_regex` / `semantic_guard.targets=["text"]` /
  `pii_scan` on output) still fire because the assistant's
  streamed `TextBlock` content was never gated by `PreToolUse`.
- **Per-tool `semantic_guard` judge calls run in parallel.**
  `_semantic_guard_match` previously walked `tool_calls`
  sequentially — N tools meant N back-to-back blocking
  round-trips to `/v1/sdk/judge`. The matcher now submits each
  tool's judge call to a bounded `ThreadPoolExecutor` (max 8
  concurrent), collapsing wall-clock from `sum(t_i)` to `max(t_i)`.
  First-match-by-input-order semantics on the audit row
  (`matched_policy.message` names the FIRST matching tool) are
  preserved verbatim.
- **Async patches no longer block the event loop on the judge
  HTTP call.** Both `_async_gate_call_inner` and the
  `claude_agent_sdk` async receive loop wrap the synchronous
  `evaluate` / `evaluate_output` body in `asyncio.to_thread`, so
  a slow judge round-trip no longer pins every concurrent
  coroutine on the same loop. Per-tool `PreToolUse` and
  `PostToolUse` hook callbacks parallelize the same way.
  ContextVar state (identity, policy usage accumulator, init-
  latency accumulator) is propagated into the worker thread by
  the standard `asyncio.to_thread` contract.
- **Judge HTTP timeout default lowered from 20.0 s to 8.0 s.**
  The backend's own OpenAI-judge timeout is 15 s, so anything
  much higher than that just lets a stuck backend silently
  widen the SDK's stall window. Operators who want more
  headroom (regulated workloads with `semantic_on_outage="block"`
  preferring a long timeout to a fail-open) can override via
  `EGISAI_JUDGE_TIMEOUT_SECS`.
- **`Retry-After` header is clamped to a configurable maximum
  (default 5.0 s).** A misconfigured upstream proxy could ship
  `Retry-After: 90` and freeze every governed call for 90 s ×
  3 retry attempts = 270 s. The clamp guarantees a single retry
  costs at most this many seconds; override via
  `EGISAI_JUDGE_RETRY_AFTER_MAX_SECS`.

### Changed

- **`policy_latency_ms` no longer includes the one-shot PII NER
  warm-up wait.** The Presidio + spaCy analyzer load happens once
  per fresh process and can take up to
  `EGISAI_PII_WARMUP_TIMEOUT_SECS` (default 2 s). Previously this
  cold-start was bundled into call-#1's `policy_latency_ms`,
  making the dashboard's "policy enforcement latency" column
  look permanently inflated for short-lived workloads. The wait
  is now booked separately on `ev["init_latency_ms"]` so the
  cold-start cost is still visible without misattribution.
  `policy_latency_ms` after this release reflects only per-call
  governance work.

### Added

- **`EGISAI_JUDGE_TIMEOUT_SECS` env var.** Override the per-
  request timeout for `/v1/sdk/judge`. Resolution order:
  constructor arg → env → 8.0 s default. Clamped at 0.5 s.
- **`EGISAI_JUDGE_RETRY_AFTER_MAX_SECS` env var.** Cap the
  honoured `Retry-After` value on HTTP 429 responses. Resolution
  order: constructor arg → env → 5.0 s default. Clamped at 0.1 s.
- **`init_latency_ms` event field.** Carries the per-call
  one-shot library cold-start cost (today: PII NER load) that
  used to bleed into `policy_latency_ms`. The backend silently
  ignores unknown fields, so this is forward-compatible — a
  future schema migration can promote it to a first-class column
  without breaking older SDKs.

---

## [0.28.0] — 2026-05-27

This release closes one acute regression and adds the supporting
infrastructure so the same class of "shipped a broken
`pip install`" never goes unnoticed again. The user-facing
changes ship in two halves: a direct dep bump that repairs fresh
installs (the immediate fix); and a small, opt-in-via-default
startup-telemetry hop that lets the operator's dashboard surface
SDK-side install warnings before the next customer pings them.

### Fixed

- **`pip install egisai` now brings `click` on its own.** A fresh
  install on a clean Python 3.10+ environment was broken since
  `typer 0.26.0` vendored its copy of click into `typer/_click/`
  and dropped `click` from `Requires-Dist`. spaCy still imports
  the external `click` directly in `spacy/cli/_util.py`
  (`from click import NoSuchOption`), and `spacy/__init__.py`
  eagerly loads that submodule on every plain `import spacy`, so
  the missing transitive dep caused the Presidio analyzer to
  fail to load on first `init()` with:

      [egisai] PII NER analyzer failed to load
      (ModuleNotFoundError: No module named 'click') — falling
      back to regex+checksum detection.

  The SDK kept running (failed-open, regex+checksum PII detection
  stayed on), but name / location / GDPR-special-category text
  was silently un-flagged on fresh installs until the operator
  added `click` by hand. `egisai`'s `pyproject.toml` now declares
  `click>=8.0` as a direct runtime dep so pip resolves it
  regardless of typer's vendoring choice.

  Existing installs that already had `click` in `site-packages`
  (any environment with flask, uvicorn, mypy, black, pip-tools,
  or even pip itself) were never affected; this only repairs
  greenfield installs.

  Operators stuck on `0.27.2` can unblock themselves immediately
  with `pip install click` — no need to wait for the upgrade.

### Added

- **Startup-warning telemetry — dashboard now shows when an
  install reports an SDK init-time issue.** Previously, when a
  customer's SDK hit a non-fatal init-time problem (today: the
  PII NER analyzer failing to load), the only signal was a single
  log line on the customer's own stderr. We — the platform
  operator — had no way of knowing until the customer asked.

  The SDK now fires a one-shot fire-and-forget POST to
  `/v1/sdk/telemetry/startup-warning` whenever a warning fires.
  The dashboard's new "SDK install warnings" banner (silent until
  `total_recent > 0`) surfaces a count + per-code rollup + the
  most recent N events, so a regression is visible the next time
  the operator loads the dashboard instead of weeks later via a
  support ticket.

  **Privacy contract:** the payload carries operator-controlled
  diagnostic columns only — a stable `code`, the exception class
  name, a *sanitized* one-line error message (the SDK strips
  obvious filesystem home-directory paths and truncates to 256
  chars before transmission), the SDK version, the Python
  version, and the OS family. No prompt text, no API key
  material, no agent display names, no customer-identifying
  value ever crosses this boundary. See
  `egisai._backend.post_startup_warning` for the SDK-side
  contract and `app/db/models/sdk_startup_warning.py` for the
  backend storage shape.

  **Reliability contract:** the send is fire-and-forget. Every
  failure mode — backend down, 4xx, 5xx, slow network, missing
  config, malformed exception — is swallowed silently inside
  the telemetry function so the user's `egisai.init()` and
  first model call are never delayed or blocked by this
  diagnostic hop.

  **No retries, one event per process.** Re-emitting on every
  restart would inflate dashboard counts and bury new signals
  under repeats.

### Internal

- **Pipeline hardening — never silently ship a broken
  `pip install egisai` again.** Three new gates land alongside
  the dep bump above:

  1. **`ci.yml: fresh-install` matrix job.** Every PR / push now
     builds the wheel, installs it into a brand-new venv (no
     `[dev]` extras, no editable install) on each supported
     Python × OS cell, and exercises the exact import chain a
     customer hits on first `init()`. The class of failure that
     bit us this week — "every existing dev venv has the dep
     transitively, but a fresh `pip install` doesn't" — would
     have been a red CI check from the moment of the regression.
  2. **`nightly-install.yml` scheduled workflow.** Once a day
     at 07:00 UTC, a fresh runner installs `egisai` from PyPI
     (both `pinned` and `--upgrade-strategy eager` strategies)
     and runs the same import smoke. Catches the case where our
     code is unchanged but an upstream dep regression breaks the
     already-published wheel for new customers. This is the
     time-based safety net the click bug would have needed.
  3. **`release.yml: smoke-install` gate.** A new step between
     `build` and `sign` that installs the just-built wheel into
     a fresh venv on a fresh runner and exercises the same
     import smoke. The publish step now depends on this — a
     wheel that can't be installed will never reach PyPI.

  All three gates share the same script body (intentionally
  inlined, not factored into a helper, so the mirror repo
  stays standalone). The smoke covers `import egisai`,
  `import spacy`, `import click`, the full
  `presidio_analyzer` chain, and the public SDK surface.

---

## [0.27.2] — 2026-05-26

### Changed

- **README points to the canonical legal documents.** The
  "Privacy and security" section now explicitly links to the
  authoritative [Privacy Policy](https://egisai.co/privacy) and
  [Terms of Service](https://egisai.co/terms) at `egisai.co` —
  the SDK is Apache 2.0 but use of the hosted control plane is
  governed by those documents, and customers reading the PyPI
  page should be able to find them in one click.

- **README ships the same enterprise disclaimers as the website.**
  Two paragraphs added next to the architecture-review summary:
  a *certification-status* disclaimer (EgisAI is not currently
  SOC 2, ISO 27001, HIPAA, FedRAMP, or PCI DSS certified unless
  the website Security page says otherwise with current evidence)
  and a *no professional advice* disclaimer (the Service provides
  technical controls and audit-supporting evidence, not legal /
  regulatory / compliance / security advice). Both paragraphs are
  kept in sync with `homepage/terms.html`,
  `homepage/privacy.html`, the homepage compliance section, and
  `docs/security.mdx` from a single source of truth in
  `legal/snippets.md` (propagated by `scripts/sync_legal.py`),
  so future legal-review cycles update every surface in one
  step.

No code or runtime behavior changed in this release — README-only.

---

## [0.27.1] — 2026-05-17

### Fixed

- **`tool_call` step events now ship the tool input under the
  correct wire key.** Pre-0.27.1, every per-tool step row built
  by `_patches/_common.py:_dispatch_per_tool_steps` and by the
  Claude Agent SDK's PreToolUse / PostToolUse / post-hoc
  fallback paths stamped the tool input under the dict key
  `"request_text"` (matching the DB column name). The backend
  ingest reader at `app.routers.sdk._build_request_log_row`
  reads the audit row's preview text from
  `ev.get("prompt_preview")`, so every `tool_call` row landed
  with `request_text = NULL`. The dashboard's intent-summary
  LLM then ran against an empty prompt and collapsed those
  rows onto the generic "Open ended assistant chat" /
  "General chat follow up question" fallback shapes (or, when
  the LLM was unavailable, onto the `"Governed agent run"`
  seed). Affected every framework that emits per-tool steps —
  OpenAI Agents (sync + responses), Anthropic, Google
  Generative AI, LangChain / LangGraph / LlamaIndex / Bedrock
  via the shared `_common.gate_call` waterfall, and every
  flavor of the Claude Agent SDK path. The fix renames the
  wire key in all four sites to `"prompt_preview"`; no DB
  schema change. Backend hardening lands alongside: ingest
  now falls back to `ev.get("request_text")` so dashboards on
  a fresh backend heal immediately for customers still on
  ≤ 0.27.0. Two regression tests pin the new contract — one
  on the OpenAI per-tool waterfall, one on the Claude Agent
  SDK PreToolUse hook — and both assert that the legacy
  `"request_text"` key is no longer set on the event (so the
  backend's compatibility fallback can't mask a future
  regression).

---

## [0.27.0] — 2026-05-16

### Changed

- **`semantic_guard` policies no longer support a per-policy
  `judge_model` override.** The platform's judge `SYSTEM_PROMPT`
  is calibrated against a single OpenAI model and the per-rule
  `threshold` knob (default 0.75) behaves as documented only
  against that calibration. Allowing operators to swap the model
  per policy silently skewed the threshold semantics every other
  rule on the workspace assumed — a foot-gun masquerading as a
  knob. Removed end-to-end:
  1. `SemanticBlocker._prepare()` (`egisai/policy/semantic.py`)
     stops appending `judge_model` to the `/v1/sdk/judge` request
     body. Existing policies that still carry the field stay
     valid (no schema error); the field is silently ignored.
  2. The first time a `semantic_guard` rule with a `judge_model`
     entry is evaluated, the SDK logs one `WARNING` per process
     pointing the operator at the cleanup. After that the field
     is silent.
  3. Platform-side, `/v1/sdk/judge` still accepts `judge_model`
     in the request body (for back-compat with SDK ≤ 0.26.x)
     but drops the value before invoking the judge.

  Migration: remove `judge_model` from any `semantic_guard`
  policy config in your dashboard. No code change required;
  behavior is unchanged unless you were intentionally pointing
  a policy at a different model — in which case the warning
  surfaces it.

---

## [0.26.0] — 2026-05-15

### Added

- **Operator pause / resume — runtime kill switch for an agent.**
  When an operator clicks *Pause* on the dashboard's Agents page,
  the platform marks the agent paused (`agents.paused_at` set;
  `paused_by` records the operator), bumps the org-scoped policy
  ETag, and emits an `agent.changed` SSE event. The SDK reacts in
  three ways:
  1. `_policy_cache` now stores a `paused_agent_ids: frozenset`
     in lockstep with the rule list. A 200 response from
     `/v1/sdk/policies` carries the new field; a 304 leaves
     both pieces of state untouched.
  2. `_refresher` subscribes to the new `agent.*` SSE prefix in
     addition to `policy.*`, so a pause takes effect SDK-side
     within ≈ 50 ms (or one `refresh_interval_seconds` tick on
     the polling fallback).
  3. `_evaluator.evaluate` and `evaluate_output` gain a **Phase 0**
     gate that runs **before** Phase 1's deterministic checks —
     a paused agent's call short-circuits with a synthetic
     `PolicyDecision.deny` (`reason_code = "agent_paused"`,
     `matched_policy = "Agent paused"`) so no LLM token is spent
     on a `semantic_guard` judge, no raw prompt is even hashed,
     and the audit pipeline records the block in the same shape
     every other policy block already uses.

  Backwards-compatible: older backends that don't ship
  `paused_agent_ids` are treated as "no agents are paused", which
  matches their pre-rollout Behavior. The `replace_rules()`
  helper takes the field as an optional keyword argument so
  third-party callers (and the existing test suite) keep
  compiling unchanged.



### Fixed

- **Bedrock Converse + Bedrock Agent: drop the stale-id tracker
  that caused a silent gate bypass on the second-and-later
  client in a process.** Both
  `egisai/_patches/bedrock_runtime.py` and
  `egisai/_patches/bedrock_agent.py` kept a module-level
  `_PATCHED_CLIENT_IDS: set[int]` and used `id(client) in
  _PATCHED_CLIENT_IDS` as a fast-path "already patched" check
  inside `patch_client_instance`. The intent was idempotency,
  but `id()` in CPython is just the object's address — addresses
  are aggressively recycled the moment the previous occupant is
  garbage-collected. In long-lived processes that constructed
  multiple `boto3.client("bedrock-runtime")` /
  `("bedrock-agent-runtime")` instances over time (the public
  mirror's CI pytest run is the canonical reproduction:
  back-to-back `bedrock_smoke` and `bedrock_converse_with_tool`
  fixtures freshly allocate a `_BedrockClient` whose `id()`
  often lands on the slot the previous test's instance just
  vacated), the second call would short-circuit the wrap. The
  client's `converse` / `converse_stream` / `invoke_agent`
  method would then resolve to the underlying raw method,
  `gate_call` would never run, and policies wouldn't fire —
  blocked tool calls silently succeeded, PII calls didn't
  raise `PermissionError`, and no audit event with `verdict =
  allow|block` was emitted. The fix is to delete the
  `_PATCHED_CLIENT_IDS` set entirely and rely on the
  `__egisai_wrapped__` sentinel attribute that the wrapper
  functions already carry: it's an attribute on the bound
  method, not on the client object, so it's immune to id reuse
  and gives the same "don't double-wrap" guarantee that the
  set was meant to provide. The five flaky tests in
  `tests/test_smoke_provider_battery.py::test_bedrock_converse_*`
  and `tests/test_before_after_each_llm_and_tool.py::test_bedrock_converse_deny_tool_call_refuses_BEFORE_tool_dispatch`
  now pass reliably across every Python / OS matrix cell, not
  just the one that happened to allocate fresh memory for the
  first client in each fixture.

---

## [0.25.14] — 2026-05-15

### Fixed

- **Close the `ci.yml` mypy gate that 0.25.13 left half-fixed.**
  0.25.13 added `# type: ignore[import-not-found]` to twelve
  guarded imports of `openai.types.…` inside
  `src/egisai/_patches/openai.py` so mypy would tolerate the
  missing optional `openai` extra on the public mirror's CI.
  After landing those edits, `ruff --fix` re-formatted four of
  the imports (`PromptTokensDetails`, `CompletionTokensDetails`,
  `InputTokensDetails`, `OutputTokensDetails`) from single-line
  to parenthesised multi-line form because the single-line
  version exceeded the project's 100-char limit. The reformat
  parked the `# type: ignore[import-not-found]` directive on
  the *symbol-name* line, but mypy emits the
  `Cannot find implementation or library stub` error on the
  `from … import (` line — so on those four sites the ignore
  was a no-op. Only one of the four bubbled up as a CI error
  (the others were silently masked by mypy's per-module
  "module-missing" caching: once `openai.types.completion_usage`
  was flagged on its first guarded import, subsequent imports
  of the same module didn't re-flag). 0.25.13 still shipped
  successfully because `release.yml`'s gate matrix is
  `ruff + pytest`, not mypy — `mypy` lives only in `ci.yml`'s
  `type-check` job, which is non-blocking for PyPI publish but
  red on the repository badge.

  0.25.14 moves the directive to the `from … import (` line on
  all four sites, matching the pattern already in use on the
  `response_output_message` / `response_output_text` siblings
  (those weren't affected because their imports had multiple
  symbols and ruff's reformat naturally put the ignore on the
  outer line). No runtime Behavior change. Wheel + sdist for
  0.25.14 are byte-equivalent-modulo-comments to 0.25.13's
  artefacts.

---

## [0.25.13] — 2026-05-15

### Fixed

- **CI gate parity for the public mirror's no-`openai`
  environment** — both the `pytest` and `mypy` gates. The
  mirror's CI bootstraps with `pip install -e ".[dev]"`; the
  `dev` extra intentionally does **not** pull the optional
  `openai` / `anthropic` / `google-genai` packages so the
  release build proves the SDK still works when a customer
  installs the bare `egisai` distribution. Two separate gates
  were red on 0.25.12's release pipeline:

  1. **`pytest`** — four contract tests in
     `tests/test_block_stub_provider_sdk_shape.py` (the OpenAI
     Responses `usage` / output-list shape pins, and the Chat
     Completions `.model_dump()` / autogen-unpack pins)
     imported `openai.types.…` at top-level inside the test
     body and called `.model_dump()` on the stub. Both
     contracts only apply when the optional `openai` extra is
     installed. The four tests now use
     `pytest.importorskip("openai")` / `try / except
     ImportError: return` to skip cleanly when the extra is
     absent, matching the pattern already in place on the
     LangChain-OpenAI and openai-agents sibling tests in the
     same file.

  2. **`mypy`** — twelve guarded imports of `openai.types.…`
     inside `try / except` blocks in
     `src/egisai/_patches/openai.py` (the Pydantic-typed stub
     factories for `ChatCompletion`, `ChatCompletionChunk`,
     `CompletionUsage`, `ResponseOutputMessage`,
     `ResponseOutputText`, and the `*TokensDetails`
     sub-objects) didn't carry the
     `# type: ignore[import-not-found]` directive the rest of
     the optional-extra imports in the SDK already use
     (`_patches/genai.py`, `_patches/agno.py`,
     `_patches/google.py`). mypy doesn't honour
     `try: import x; except ImportError: …` for missing
     modules, so the in-`try`-body imports tripped seven
     `Cannot find implementation or library stub`
     errors. Each import now carries the project-standard
     ignore directive.

  No runtime Behavior change — the installed bytes are
  identical to what 0.25.12 would have shipped. This release
  exists because 0.25.12's PyPI publish was blocked by the
  `pytest` gate inside `release.yml`'s `test` job (the
  `build` job has `needs: test`), so the Agent-Identity
  audit-provenance fix from 0.25.12 reaches customers as the
  first installable 0.25.x in this series after 0.25.11. The
  parallel `mypy` failure in `ci.yml`'s `type-check` job —
  same root cause, separate workflow, not on the publish-
  critical path — is fixed here too so the main-branch CI
  badge stays green.

---

## [0.25.12] — 2026-05-14

> **Note**: 0.25.12 was never published to PyPI — its release
> pipeline was blocked at the `pytest -q` gate by the test-only
> regression fixed in 0.25.13. The Fixed entry below describes
> the actual code change that 0.25.13 carries forward.

### Fixed

- **Legacy single-row audit events now stamp `identity_source` +
  `identity_hash` on the audit event**, so the backend's synth-Run
  ingest path (the `else` branch in `ingest_events` when
  `ev.get("run_id") is None`) can carry the Agent Identity
  provenance through onto the `runs` row it materialises. Before
  this fix every Bedrock Converse / raw OpenAI Chat / raw
  Anthropic call landed with NULL `identity_source` /
  `identity_hash` on the synth Run — the Agent Identity card on
  the dashboard rendered blank, and the agents-test validator's
  `run.identity_source set` / `run.identity_hash set` checks
  failed for every direct-LLM harness. Framework-wrapped Runs
  were unaffected because they ship a `run.start` envelope that
  carries the same two fields through `_upsert_run_from_start`.
  The new helper `_stamp_identity_provenance` on
  `egisai._patches._common` lives at a single seam
  (`_attribute_event`) so the stamp lands on the event regardless
  of whether the identity record came from the locked Run, the
  pushed identity stack, or the fresh 7-tier resolver call.
  `identity_source` is a controlled-vocabulary token and
  `identity_hash` is a SHA-256 digest, so this is
  compliance-safe — no raw prompt content leaks via either
  field. Pairs with the matching backend change that copies
  these onto `synth_run.identity_source` /
  `synth_run.identity_hash` at ingest time. Regression coverage:
  the agents-test validator (`agents-test/bedrock_converse_agentic.py`,
  `agents-test/openai_direct_agentic.py`, `agents-test/anthropic_direct_agentic.py`)
  now passes the two Agent Identity checks end-to-end.

---

## [0.25.11] — 2026-05-14

### Fixed

- **`_RunScope` now skips opening a duplicate Run when re-entered by
  the *same* logical agent identity**, fixing a long-standing
  double-row issue in LangGraph harnesses. `Pregel.invoke` calls
  `self.stream` internally; both methods sit behind separate
  `_RunScope` wraps, and before this fix each layer materialised
  its own `runs` row. The outer (invoke) Run ended up empty
  (`step_count=0`, `prompt_text=""`, `verdict="allow"` even when
  the inner call blocked) while the inner (stream) Run held the
  real step. The validator's "average step count" tile, billing
  token roll-up, and SOC 2 "what actually happened" timeline all
  double-counted the same trace as a result. The guard keys on
  `identity_hash` so a *true* sub-agent / handoff (different
  bundle → different hash) still opens a child Run with
  `parent_run_id` wired up — the parent→child topology contract
  is preserved end-to-end. Same code path also covers any future
  framework whose user-facing entry point internally dispatches
  through another wrapped entry point on the same `self`
  (LlamaIndex's `AgentWorkflow.run` → `Workflow._astream`, etc.).
  Regression test pinned in
  `tests/test_run_per_framework.py::test_same_identity_nested_wraps_emit_one_run`.

- **SDK-raised block `PermissionError` no longer pollutes `run.error`
  with the full exception repr.** When `_block_response` (in
  `_patches/_common.py`) refuses a call by raising
  `PermissionError("[egisai] …")`, the step it already dispatched
  carries the full verdict + matched-policy context on
  `prompt_decision` / `response_decision`. `_RunScope.__exit__` used
  to stamp the propagating PermissionError's `repr()` onto
  `run.error`, which made every refused Bedrock-Agent / Bedrock-
  Runtime turn look like an uncaught-exception crash in the
  dashboard (and failed the agents-test validator's "`run.error`
  is None on block" check). The exit path now recognises the SDK's
  own block-raise (PermissionError whose message starts with the
  `[egisai]` prefix) and stamps the short canonical reason
  `"policy block"` instead — matching the allowed-list the
  validator and `claude_agent_sdk`'s own `close_run` sites already
  use (`"input policy block"`, `"output policy block"`). Real
  unexpected exceptions (framework crashes, network errors,
  programming bugs) still stamp their full `repr()` as before, so
  the "what actually broke" diagnostic path is untouched. Regression
  test pinned in
  `tests/test_run_per_framework.py::test_sdk_block_permission_error_stamps_short_reason_not_full_repr`.

---

## [0.25.10] — 2026-05-14

### Fixed

- **Bedrock managed agents (`bedrock-agent-runtime`) now open a real
  Run on every `InvokeAgent` call** so the audit row carries
  `framework="bedrock_agent"`, `identity_source="framework:bedrock_agent"`,
  and the bundled `identity_hash` — the same shape every other
  framework wrap stamps. Before this release the patch only pushed
  the identity onto the stack and called `gate_call` directly, so
  no `RunContext` was open, `_dispatch_step` fell through to the
  legacy single-row `enqueue` path, and the backend synthesised a
  Run with `framework="legacy"` and `identity_source=NULL` /
  `identity_hash=NULL`. The dashboard's Agent Identity card on a
  Bedrock managed run was therefore missing the provenance row it
  shows for every other framework. The fix wraps the gate call in
  the shared `_RunScope("bedrock_agent", record)` (same primitive
  used by the OpenAI Agents / LangGraph / Agno / Crew patches) so
  the Run lifecycle, identity stamping, and re-entry guard all
  match. No change to the advisory enforcement contract — input
  policies still fire pre-`boto3`; output / tool-side enforcement
  is still `advisory` because AWS executes the agent loop
  server-side.

---

## [0.25.9] — 2026-05-14

### Fixed

- **LangGraph / LangChain `create_agent` / classic `AgentExecutor`
  no longer crash with `AttributeError: 'ChatCompletion' object
  has no attribute 'parse'` on blocked turns.**
  `langchain-openai>=1.2`'s `ChatOpenAI._generate` (and its async
  / streaming siblings) routes every non-streaming model call
  through the raw-response code path —
  `self.client.with_raw_response.create(**payload)` followed by
  `response = raw_response.parse()`. The upstream
  `to_raw_response_wrapper` injects the
  `X-Stainless-Raw-Response: true` marker into `extra_headers` so
  the OpenAI SDK returns a `LegacyAPIResponse` whose `.parse()`
  yields the real `ChatCompletion`. Before this release the egisai
  OpenAI patch returned the synthesised `ChatCompletion` block stub
  directly back into that call site, which then crashed at the
  `.parse()` step. Allow turns also silently dropped `tokens_in`,
  `tokens_out`, and `cost_usd` on the audit row because our
  usage extractor saw a `LegacyAPIResponse` instead of the parsed
  body. We now sniff the marker header, wrap the block stub in a
  `LegacyAPIResponse`-shaped object (`_RawResponseStub` — exposes
  `.parse()`, `.headers`, `.http_response`, `.status_code`,
  `.request_id`, `.content`, `.text`, `.elapsed`,
  `.retries_taken`), and route the gate's usage / output-signal
  extractors through `.parse()` so the audit row carries real
  token counts on allow turns. Covers both Chat Completions and
  Responses APIs, sync and async. The non-raw path is byte-for-byte
  unchanged.

---

## [0.25.8] — 2026-05-14

### Fixed

- **Streamed OpenAI Chat Completions now record real `tokens_in`,
  `tokens_out`, and `cost_usd` on the audit row instead of
  zeros / `None`.** Upstream OpenAI only emits a final usage
  chunk on streamed responses when the caller passes
  `stream_options={"include_usage": True}` — but several agentic
  frameworks (notably `llama-index-llms-openai`'s `_stream_chat`
  / `_astream_chat`) never set this. Our streaming wrapper now
  injects `include_usage=True` into `stream_options` when the
  caller didn't already opt in (or out), so the materialised
  replay's aggregated `response.usage` surface carries real
  token counts. Existing keys in `stream_options` are merged
  rather than overwritten, and an explicit
  `include_usage=False` is honoured. Affects every streamed
  call routed through `openai.chat.completions.create` (sync
  and async), which is what every LlamaIndex agent run hits
  through the OpenAI LLM adapter.

---

## [0.25.7] — 2026-05-14

### Fixed

- **LlamaIndex `FunctionAgent` / `ReActAgent` / `CodeActAgent` /
  `AgentWorkflow` runs no longer crash on `TypeError: 'async for'
  requires an object with __aiter__ method, got _StreamReplay`.**
  `llama-index-llms-openai`'s `OpenAI._astream_chat` consumes the
  return value of `await aclient.chat.completions.create(stream=True)`
  via `async with stream as r: async for chunk in r: ...`. Our
  `_StreamReplay` wrapper (used on both the block-stub and
  allow-replay streaming paths) only implemented the **sync**
  iteration / context-manager protocol, so any LlamaIndex agent
  whose LLM hit the streaming code path crashed on the first
  chunk. The replay now satisfies both halves of the async
  protocol (`__aiter__`, `__anext__`, `__aenter__`, `__aexit__`,
  `aclose`) via a new `_StreamReplayAsyncIter` helper that walks
  the same materialised chunk list the sync path uses, so a
  single replay can be consumed by either iteration flavour.

- **LlamaIndex agent runs are now correctly grouped into a single
  `RunContext` instead of fragmenting into a wrap-side empty run
  plus N "legacy" inner runs.** `FunctionAgent.run()` returns
  immediately with a `WorkflowHandler`, but the actual LLM calls
  happen later on the handle's internal `_result_task`. The
  previous `kind="sync"` patch closed the `RunContext` as soon as
  `run()` returned the handle — every inner LLM call then saw
  `current_run() is None`, opened its own ephemeral run with
  `framework="legacy"`, and the dashboard's run waterfall lost
  the workflow's structure entirely. A new `kind="handler"` wrap
  in `egisai._patches._framework` opens the run + pushes
  identity, calls `orig()` (which constructs the workflow's
  internal asyncio tasks under our captured contextvars), then
  hooks `add_done_callback` on `handle._result_task` to finalise
  the run when the workflow actually completes. The parent task's
  contextvars are restored before returning so user code after
  `agent.run(...)` sees a clean stack. Falls back to sync close
  when `_result_task` is absent (older LlamaIndex, fakes used in
  unit tests). Inner LLM calls now correctly attribute to the
  framework-stamped Run with the workflow's identity, matching
  the per-Run waterfall every other framework wrap produces.

## [0.25.6] — 2026-05-14

### Fixed

- **AutoGen ``AssistantAgent.run`` survives ``on_block="stub"``
  without crashing on ``'types.SimpleNamespace' object has no
  attribute 'model_dump'``.** ``autogen-ext.models.openai`` calls
  ``response.model_dump()`` directly to populate its
  ``LLMCallEvent`` log payload after every chat completion (see
  ``_openai_client.py`` line 719) — a bare ``SimpleNamespace``
  stub crashed with ``AttributeError`` between the gate returning
  the verdict and the agent surfacing it to the operator. The
  OpenAI patch's non-streaming block stub now builds a real
  upstream Pydantic ``ChatCompletion`` (with
  ``ChatCompletionMessage``, ``Choice``, ``CompletionUsage``
  inside), so ``.model_dump()`` round-trips cleanly while every
  existing attribute path the SDK contract tests pin
  (``response.choices[0].message.content``,
  ``response.usage.completion_tokens_details.reasoning_tokens``,
  ``response.model``, ``response.system_fingerprint``,
  ``response.service_tier``, …) keeps working. The streaming
  block stub (added in v0.25.5 for the LangChain ``AgentExecutor``
  path) already used real ``ChatCompletionChunk`` instances and
  is unaffected. The ``SimpleNamespace`` fallback path remains
  for the rare case where Pydantic construction fails (very old
  ``openai`` pin or future shape change), matching the posture of
  every other ``_build_*`` helper in this file.

## [0.25.5] — 2026-05-14

### Fixed

- **LangChain ``AgentExecutor.invoke`` survives ``on_block="stub"``
  without crashing on ``'types.SimpleNamespace' object does not
  support the context manager protocol``.** The classic
  ``AgentExecutor`` (both LangChain 0.x and ``langchain-classic``
  on 1.x) defaults to ``stream_runnable=True``, which forces the
  inner ``ChatOpenAI`` call down ``_stream`` →
  ``self.client.create(stream=True)`` → ``with response as
  response: for chunk in response: ...``. The OpenAI patch's
  block-stub used to return a bare ``SimpleNamespace`` regardless
  of ``stream=``, so the ``with`` step raised ``TypeError`` before
  the policy verdict ever reached the operator. We now detect
  ``stream=True`` and return a streaming-shaped replay that
  satisfies both the context-manager and iteration protocols and
  yields real upstream ``ChatCompletionChunk`` objects (with
  ``.model_dump()``) so the framework's chunk-to-generation
  converter keeps working unchanged.

- **Streaming OpenAI calls now record real ``tokens_in`` /
  ``tokens_out`` / ``cost_usd`` on the audit row.** When the
  caller asked for ``stream=True`` (LangChain's default agentic
  path, plus any direct streaming user code), the gate previously
  ran ``_extract_chat_usage`` against the upstream ``Stream``
  before iteration had drained it — ``.usage`` wasn't yet
  populated, so every streamed turn shipped to the dashboard with
  zero tokens and a NULL cost. The same blind spot affected
  ``extract_openai_chat`` on the output policy phase, which only
  saw an empty ``choices`` accessor and silently skipped
  evaluation. The OpenAI patch now materialises the upstream
  ``Stream`` into a re-iterable replay that exposes the
  aggregated ``choices[0].message`` (content + reassembled
  ``tool_calls`` across deltas) and the final-chunk ``usage`` on
  its response surface — so per-tool waterfall emission, output
  semantic_guard / deny_tool_call / deny_output_regex, and token
  accounting all keep working on streaming turns. ``AsyncStream``
  follows the same path via the async sibling.

## [0.25.4] — 2026-05-14

### Fixed

- **LangChain patch covers both ``langchain.agents.AgentExecutor``
  (LangChain 0.x) and ``langchain_classic.agents.AgentExecutor``
  (LangChain 1.x).** LangChain 1.0 removed ``AgentExecutor`` from
  ``langchain.agents`` and shipped the pre-1.0 agentic surface as
  the dedicated back-compat package ``langchain-classic``. Users
  on LangChain 1.x who install ``langchain-classic`` to keep the
  classic ``AgentExecutor.invoke`` /  ``ainvoke`` / ``stream``
  surface now get their executors patched and attributed with
  ``identity_source="framework:langchain"`` exactly as on 0.x.
  The patch tries each home independently — neither installed →
  silent no-op (legacy Behavior); only one installed → that one
  gets patched; both installed (transition envs) → both get
  patched. The modern ``langchain.agents.create_agent`` path is
  unchanged and continues to be covered transparently by the
  ``langgraph`` patches via MRO on its returned
  ``CompiledStateGraph``.

## [0.25.3] — 2026-05-13

### Fixed

- **Agno agents survive ``on_block="stub"`` without crashing on
  ``'types.SimpleNamespace' object has no attribute 'audio'``.**
  Agno's ``OpenAIChat._parse_provider_response`` reads
  ``response_message.audio`` and ``response.model_extra``
  *unguarded* (no ``hasattr`` check on either site), so any
  block-stub returned to it raised ``AttributeError`` the moment
  a policy fired. The fix lives entirely in
  ``_patches/agno.py``: a tiny shim wraps
  ``OpenAIChat._parse_provider_response`` and pre-populates the
  unguarded fields with ``None`` *only* when the response carries
  our ``egis`` block-stub sentinel — real ``ChatCompletion``
  objects pass through untouched. No other framework's stub
  contract changes. The wrap is idempotent and applied once per
  ``init()``.

## [0.25.2] — 2026-05-13

### Fixed

- **HTTP fallback no longer enqueues phantom audit rows for the
  OpenAI Agents SDK's tracing uploads.** Every time
  ``Runner.run`` returned, the ``openai-agents`` SDK's
  ``BackendSpanExporter`` POSTed its trace payload to
  ``https://api.openai.com/v1/traces/ingest`` via its own
  ``httpx.Client``. Our ``_patches/http`` fallback's
  ``_looks_like_model_call`` matched the URL on host alone
  (``"api.openai.com" in url``), so each tracing upload became a
  second audit event with ``model="unknown"``,
  ``verdict="allow"``, no prompt preview, and the app name as the
  agent — surfacing on the dashboard as a ghost run alongside
  every legitimate ``Runner.run`` invocation. The fallback now
  requires BOTH a known LLM-provider host AND a known model-call
  path token (``/chat/completions``, ``/responses``,
  ``/messages``, ``:generateContent``, ``/openai/deployments``,
  …) before logging, which silently drops tracing, files, audio,
  images, threads, moderations, and any other ancillary
  endpoint that happens to share the host. Same posture applies
  to the Anthropic, Google Gemini, Together, Groq, Cohere, and
  Mistral hosts. Dedicated provider patches still cover the
  in-band model call — the fallback only matters for transports
  we don't have a first-class adapter for.

---

## [0.25.1] — 2026-05-13

### Fixed

- **Block-stub responses now mirror the upstream provider SDK's
  attribute shape.** When a policy fires with `on_block="stub"`,
  every patched provider returns a `types.SimpleNamespace` posing
  as the real SDK response. Agentic frameworks
  (`openai-agents`, `langchain-openai`, LangGraph, CrewAI, …)
  consume that stub as if it were a real response — they walk
  `response.usage.<field>`, `response.output[].content[].text`,
  etc. directly. Several fields the upstream SDKs read
  unconditionally were missing from our stubs, so any agentic
  loop that hit a blocking policy crashed with
  `AttributeError` on the *next* statement after `egisai`
  returned. Concretely:
  - **OpenAI Responses API** (`gpt-5`, `gpt-4o`, …):
    `usage.input_tokens_details` and
    `usage.output_tokens_details` are non-Optional on
    `openai.types.responses.ResponseUsage`, and the
    `openai-agents` Runner reads both at
    `agents/models/openai_responses.py:495`. A stub without them
    crashed the Runner with
    `AttributeError: 'types.SimpleNamespace' object has no
    attribute 'input_tokens_details'` the moment a blocking
    policy fired. *Both* sub-objects are now constructed as
    real upstream `InputTokensDetails` / `OutputTokensDetails`
    Pydantic instances — a bare `SimpleNamespace` with the right
    attribute names was still rejected by the agents-SDK's own
    `Usage` Pydantic dataclass (which validates via
    `isinstance`). Same treatment applied to the `output[]`
    items: real `ResponseOutputMessage` / `ResponseOutputText`
    instances so the subsequent
    `ModelResponse(output=response.output, ...)` Pydantic
    construction succeeds. The stub also now carries
    `output_text` / `incomplete_details` / `error` for
    frameworks that read those convenience fields.
  - **OpenAI Chat Completions API**:
    `usage.prompt_tokens_details` (`audio_tokens`,
    `cached_tokens`) and `usage.completion_tokens_details`
    (`accepted_prediction_tokens`, `audio_tokens`,
    `reasoning_tokens`, `rejected_prediction_tokens`) sub-objects
    — constructed as real upstream `PromptTokensDetails` /
    `CompletionTokensDetails` Pydantic instances for the same
    reason — plus `message.tool_calls`, `choice.logprobs`,
    `system_fingerprint`, `service_tier`, `created`.
  - **Anthropic Messages API**: `usage.cache_creation_input_tokens`,
    `usage.cache_read_input_tokens`, `usage.server_tool_use`,
    `usage.service_tier`, plus top-level `stop_sequence` and
    `container`. Fixes LangChain `ChatAnthropic` and CrewAI's
    Anthropic adapter.
  - **Google `google-genai` and `google.generativeai`**:
    `usage_metadata.cached_content_token_count`,
    `thoughts_token_count`, `tool_use_prompt_token_count`, the
    `*_tokens_details` sub-objects, plus `function_calls`
    aggregator, `model_version`, `response_id`. Fixes
    LangChain `ChatGoogleGenerativeAI` and Vertex Agent
    Builder shims.
  Behavior of the gate itself is unchanged — same verdicts, same
  audit fields, same `egis.blocked` marker on every stub. Only
  the shape of the synthetic response object grew.

### Added

- **Contract test `test_block_stub_provider_sdk_shape.py`.** Pins
  the exact attribute access pattern every supported framework
  performs against a blocked stub. When a future upstream SDK
  adds another required field, the failing assertion points
  straight at the stub factory to patch — no more debugging
  through 5-frame stack traces in customer agent loops.

---

## [0.25.0] — 2026-05-13

### Added

- **First-call gate for the Presidio + spaCy PII analyzer.** Before
  0.25, a model call that fired within ~1 s of `egisai.init()` raced
  the background daemon thread that warms the NER analyzer — meaning
  the **first call of a fresh process could silently drop name /
  address / GDPR-special-category detection**, falling back to the
  regex chain (which only catches SSN / credit card / IBAN / email /
  phone / API key / IP / MAC). Long-running services were unaffected
  (the analyzer is warm long before request #1), but test harnesses,
  demo scripts, and AWS Lambda cold starts routinely hit this. Worse,
  the regression was invisible — the call returned `allow`, the
  audit row showed no PII findings, and the operator believed their
  policy was on.

  0.25 wires a one-shot gate inside `egisai._evaluator.evaluate()`
  and `evaluate_output()` that, on the **first** policy phase of the
  process, briefly blocks waiting for the analyzer to warm — but
  **only when** the org has at least one active `pii_scan` rule
  scoped to this call. After call #1 the gate is permanently off
  (the analyzer is either warm or has hard-failed; waiting again
  serves no one).

  Conditions for the wait to actually fire:

  1. No previous call in this process burned the one-shot.
  2. A `pii_scan` rule is in scope after `_scope_filter` (semantic-
     only orgs pay zero overhead).
  3. The analyzer isn't already warm (steady-state services skip
     the wait entirely on a single attribute read).
  4. `EGISAI_PII_WARMUP_TIMEOUT_SECS` > 0 (operators on Lambda /
     latency-sensitive paths can opt out by setting it to `0`).

  Default cap: **2.0 seconds**. Tunable via the new env var
  `EGISAI_PII_WARMUP_TIMEOUT_SECS` (read on every gate invocation,
  so a runtime change takes effect without a restart). On timeout
  OR load failure the SDK falls through to the regex chain — the
  privacy contract for structured PII (SSN / CC / IBAN / email /
  phone / API key) is unaffected, only the NER-derived entities
  (names, addresses, GDPR special categories) are deferred.

  Observability: routed through ``logging.getLogger("egisai.evaluator")``
  rather than ``sys.stderr`` so the success path is silent by
  default (the gate firing in the happy case is invisible — the
  operator's terminal stays clean). Ops who want to track cold-
  start cost in a structured log pipeline (Datadog, CloudWatch,
  Loki) attach a handler at INFO level and they'll see one
  ``waited N ms`` line per process. The **timeout** branch logs at
  WARNING level so default logging configs DO catch it — a
  timeout is an honest degradation (THIS call ran with regex-only
  PII detection on call #1) and operators need to know.

  Why we did NOT take the obvious alternatives:

  * A blocking `init()` kwarg (`wait_for_pii_engine_secs=2.0`) was
    rejected — it regresses `init()` from instant to ~2 s for every
    Lambda cold start AND every CLI tool that imports egisai, even
    when the first call wouldn't have needed the analyzer. The
    in-evaluator gate fires only when a PII rule is actually about
    to run, so semantic-only workloads stay fast.
  * Always waiting on every call was rejected — call #2 in a
    long-running service must never pay this cost.

### Changed

- `egisai.policy._pii_loader.wait_for_warm(timeout_secs)` is now
  part of the public-internal surface. The function is `Event`-based
  (no busy-polling), returns `True` if the analyzer settled warm,
  `False` on timeout or hard failure. Available for callers that
  want to drive their own cold-start coordination (custom test
  harnesses, health endpoints).

---

## [0.24.0] — 2026-05-13

### Added

- **`semantic_guard` policies can now intent-classify tool calls,
  not just text.** Operators can describe forbidden agent behavior
  in plain English — `"delete all users"`, `"block any lookup
  request"`, `"wipe the production database"` — and the SDK asks
  the platform judge, **before** the tool dispatches, whether THIS
  specific call matches THAT intent. The match returns `block` ⇒
  the PreToolUse hook returns `permissionDecision: deny` ⇒ the
  Node CLI never executes the tool ⇒ audit row stamps
  `enforcement_status="enforced"`. This closes the gap that
  `deny_tool_call` (pattern-only) and prose-side `semantic_guard`
  (text-only) couldn't cover — a model that decides to call a
  destructive tool the operator never enumerated by name still
  gets caught by an intent rule. Opt-in via a new `targets` field
  on the rule config:

  ```json
  {
    "type": "semantic_guard",
    "name": "Forbid destructive actions",
    "config": {
      "intents": ["wipe the production database", "delete all users"],
      "targets": ["text", "tool_calls"],
      "message": "Refused: agent attempted a destructive action."
    }
  }
  ```

  `targets` defaults to `["text"]`, which preserves byte-for-byte
  behavior of every pre-0.24 `semantic_guard` rule (no silent
  meaning change on upgrade). When `targets` includes
  `"tool_calls"`, the matcher synthesizes one natural-language
  sentence per pending tool call —

      "The agent is requesting to invoke tool 'X' with arguments {...}"

  — sends each to the existing `/v1/sdk/judge` endpoint, and
  short-circuits on the first match for cost control. The blame
  attribution in the audit row's `matched_policy` message names
  the specific tool that tripped the rule.

- **Privacy contract for the new path.** Tool arguments are
  PII-label-redacted via `pii_scanner.label_redact` **before**
  they reach the judge, per `security-and-compliance.mdc` §1
  ("PII never leaves the SDK boundary in raw form, including our
  own LLM-based policy judges"). Intent classification accuracy
  is preserved because the judge cares about the verb/noun shape
  ("the agent is deleting `<NAME>`"), not the exact identifier
  values. The two-phase contract from §2 still holds:
  deterministic Phase 1 rules (`deny_tool_call`, etc.) get the
  first look at every tool call; the judge is only consulted
  when Phase 1 didn't already block.

- **No framework patch changes required.** The `claude_agent_sdk`
  PreToolUse hook, every OpenAI / Anthropic / Google / Bedrock
  end-of-turn output policy evaluation, and every other path
  through `evaluate_output` already pass `tool_calls` into the
  evaluator. The matcher change alone wires `targets:
  ["tool_calls"]` into all of them.

### Test coverage

- New `tests/test_semantic_guard_tool_calls.py` (15 tests) —
  matcher behavior on every targets shape, short-circuit on first
  match, multi-tool blame attribution, backwards-compat for
  pre-0.24 rules, PII redaction of tool args, malformed input
  defense, output-side eval routing.
- Extended `tests/test_claude_agent_sdk_pretooluse.py` (+ 3
  tests) — the reported scenario in one test: a `semantic_guard`
  rule with intent "block any lookup request" + `targets:
  ["tool_calls"]` denies `mcp__support__lookup_customer` on its
  first hop, with `enforcement_status="enforced"`. A conjugate
  test pins that benign tools pass cleanly. A third test pins the
  most dangerous regression — a legacy rule (no `targets` field)
  MUST NOT round-trip the judge for tool calls.

---

## [0.23.0] — 2026-05-12

### Added

- **Per-tool waterfall on every direct LLM provider** — The
  multi-step ``model_call`` → ``tool_call`` audit timeline
  previously emitted only on the OpenAI Chat / Responses
  patches is now consistent across every direct LLM provider:
  Anthropic Messages, Google GenAI (``google.genai``), Google
  legacy (``google.generativeai``), and AWS Bedrock Converse.
  An operator reading a Run never sees "this provider collapses
  tools, that one doesn't" — the timeline reads identically
  end-to-end regardless of which model the agent's call
  fanned out to.

### Fixed — privacy contract

- **``payload_preview`` is now label-redact aware** —
  ``egisai._events.safe_preview`` shipped raw ``repr(payload)``
  to the wire on the input-side block path (the path that
  raises ``PermissionError``, where no later code re-set the
  preview). A prompt with an unflagged SSN inside an
  immutable payload could land on the audit envelope. The
  preview now passes through ``label_redact`` before
  truncation, so even an inadvertent leak surfaces as
  ``<SSN>`` / ``<EMAIL>`` / ``<CREDIT_CARD>``. Same
  treatment for sanitize / allow paths; the fix is centralized
  so every framework patch benefits.
- **Gemini ``contents="..."`` sanitize was a no-op on the
  forward** — ``mutate_prompt_text`` updated
  ``payload["contents"]`` in place, but Gemini's ergonomic
  ``contents="..."`` shape passes the same string as a
  kwarg; strings are immutable so the upstream SDK kept
  shipping the raw bytes to Google. The genai + google
  legacy patches now mirror the post-sanitization value
  back into the forwarded kwargs. Same fix shape for
  OpenAI Responses (``input="..."``) and AWS Bedrock
  Agents (``inputText="..."``).
- **``extract_payload_text`` / ``mutate_prompt_text`` handle
  top-level string ``contents``** — Previously only the
  list-of-dicts shape was scanned; the top-level string was
  invisible to PII detectors. Fixed at the evaluator seam so
  every patch that uses Gemini-style shapes picks up the
  improvement transparently.

### Added — battle-tested smoke battery

Six new test files lock in the privacy + enforcement contracts
across the full integration matrix. Run any of them in CI to
verify the SDK still honors the runtime-governance posture:

- ``tests/test_smoke_provider_battery.py`` (25 tests) —
  per-provider battery: allow / sanitize / block / tool-block /
  per-tool waterfall / privacy contract for OpenAI Chat,
  Anthropic Messages, Google GenAI, and Bedrock Converse.
- ``tests/test_smoke_framework_cascade.py`` (18 tests) —
  end-to-end cascade for every Tier-2 framework
  (``openai_agents`` / ``langgraph`` / ``langchain`` / ``crewai``
  / ``autogen`` / ``agno`` / ``strands`` / ``pydantic_ai``)
  composed with ``openai``: identity push + sanitize/block at
  the downstream provider seam.
- ``tests/test_smoke_privacy_contract.py`` (13 tests) —
  cross-cutting "no raw bytes ever leave the SDK boundary"
  invariants. Asserts the locked contract for both
  ``payload_preview`` and the never-persisted model
  response across all Tier-1 providers.
- ``tests/test_smoke_bedrock_agent.py`` (7 tests) — pins the
  advisory-but-honest contract for AWS Bedrock managed
  agents. Includes the regression for the
  ``inputText="..."`` sanitize bug fixed above.
- Strengthened ``tests/test_claude_agent_sdk_governance.py``
  with three new invariants: concurrent multi-client isolation
  (no cross-talk between two ``ClaudeSDKClient`` instances),
  same-identity concurrent independence (two parallel turns
  with the same agent_id emit independent audit rows), and an
  end-to-end zero-leak lifecycle probe (raw PII + model text
  never appear on any audit envelope).
- Strengthened ``tests/test_claude_agent_sdk_posttooluse.py``
  with the multi-text MCP single-replace invariant: an MCP
  tool returning multiple ``{type:text}`` parts that contain
  PII collapses to a SINGLE post-sanitize text part (the
  contract change of 0.22.x — never multiple post-sanitize
  parts that could re-introduce the raw spans).

## [0.22.3] — 2026-05-12

### Changed

- **`claude_agent_sdk` OUTPUT-phase audit stamping (SOC 2 / ISO honesty)** —
  When aggregated output evaluation at ``ResultMessage`` replays structured
  ``tool_calls`` from the subprocess, a blocking verdict now stamps
  ``enforcement_status="advisory"`` on the parent ``model_call`` row instead
  of claiming full pre-execution enforcement. Pure assistant-text violations
  with PreTool hooks wired remain ``enforced``. Per-tool Hook blocks /
  substitutions keep their existing ``tool_call`` / ``PostToolUse`` signals.
  Documented in [README.md](./README.md) and [SECURITY.md](./SECURITY.md).

## [0.22.2] — 2026-05-12

### Fixed

- **Claude Agent SDK run timeline** — Emit seq-0 ``model_call`` as soon as
  input policy clears (before tool rows) and **finalize** that row when
  ``ResultMessage`` arrives instead of appending the model step last, so the
  dashboard reads Input policy → Model → tools → … in true chronological order.
- **Duplicate “Allowed” tool steps** — Skip the stream-time
  ``_dispatch_tool_call_step`` fallback when PreToolUse hooks are active; the
  hook emits the authoritative row after ``AssistantMessage`` was processed but
  before ``hook_decisions`` was populated, which doubled allow rows.
- **Platform ingest** — Upsert ``request_logs`` rows when the same
  ``(run_id, step_seq)`` arrives again (terminal merge) and recompute Run
  aggregates from all steps so token/latency totals stay correct.

## [0.22.1] — 2026-05-12

### Fixed — `claude_agent_sdk` hook injection timing (SOC 2 / ISO 27001)

Critical fix: 0.22.0 wired the `PreToolUse` + `PostToolUse`
hook injection inside `_wrap_client_query`, which runs AFTER
the user's `async with ClaudeSDKClient(...)` block has
already triggered `__aenter__` → `connect()` → CLI
subprocess init. Upstream `claude_agent_sdk.client.ClaudeSDKClient`
reads `self.options.hooks` exactly ONCE inside `connect()`,
ships the matcher table to the Node.js CLI as part of the
`initialize` control message, and the CLI never re-reads it.
Mutating `options.hooks` after `connect()` returned was a
silent no-op — the CLI dispatched every tool with no
governance round trip, and tool RESULTS were never
policy-evaluated. The "Allowed" tool rows customers saw on
the dashboard came from the legacy post-hoc
`_dispatch_tool_call_step` fallback inside
`receive_messages`, not from real hook decisions.

Effect on customers: every tool call AND tool result on a
`ClaudeSDKClient` path on 0.22.0 ran ungoverned. PII in CRM
lookups, file reads, database rows, etc. round-tripped Claude
unmasked. Module-level `claude_agent_sdk.query()` was
unaffected because its wrapper injects hooks before calling
the original (no `connect()` had run yet at that point).

Fix: inject placeholder dispatchers into `options.hooks` at
`connect()` time, BEFORE the upstream's matcher table is
shipped. The placeholders are bound to the client instance
and read the real per-turn callback off `self` at hook-fire
time. The real callback is built inside `_wrap_client_query`
(unchanged eager-binding logic) and stashed on
`self.__egisai_pre_cb__` / `self.__egisai_post_cb__`. The CLI
sees one stable placeholder ID per matcher; the callable it
points to refreshes every turn. Both deny / sanitize paths
are now genuinely enforced — verified by the regression test
`test_hooks_injected_at_connect_time_not_query_time` which
asserts hooks are present in `options.hooks` immediately
after `__aenter__` returns and BEFORE the first `query()`.

- **`_wrap_client_connect()`** — new wrapper for
  `ClaudeSDKClient.connect`. Wraps connect (not `__aenter__`)
  because users may call `await client.connect()` directly;
  `__aenter__` internally calls `self.connect()` so wrapping
  connect covers both code paths in one patch.
- **`_make_pretooluse_dispatcher(client_self)`** +
  **`_make_posttooluse_dispatcher(client_self)`** —
  placeholder async callbacks bound to a client instance. At
  fire time they look up `self.__egisai_pre_cb__` /
  `__egisai_post_cb__` and delegate; if no turn is in flight,
  they fail open with `{}` (no decision) per
  `sdk-design-philosophy.mdc` §5.
- **`INFLIGHT_PRE_CALLBACK_ATTR`** + **`INFLIGHT_POST_CALLBACK_ATTR`**
  — new client-instance attributes the dispatchers read.
  Cleared on every `_clear_inflight()` so callbacks from a
  closed turn can't accidentally fire on a fresh turn.
- **`tests/test_claude_agent_sdk_posttooluse.py::test_hooks_injected_at_connect_time_not_query_time`**
  — regression test asserts that immediately after
  `__aenter__` (i.e., after `connect()` completed but BEFORE
  any `query()` call), both `PreToolUse` and `PostToolUse`
  matchers are present in `options.hooks`. If a future
  refactor moves injection back into `query()`, this test
  fails loudly.
- **`tests/test_enforcement_matrix.py::test_claude_agent_sdk_wraps_connect_for_hook_injection`**
  — structural guarantee: the patch MUST expose
  `_wrap_client_connect`, `_make_pretooluse_dispatcher`,
  and `_make_posttooluse_dispatcher`. Locks the API surface
  for the SOC 2 / ISO 27001 enforced claim on Tier 3a.
- **Test fakes updated**:
  `tests/test_claude_agent_sdk_pretooluse.py` and
  `tests/test_claude_agent_sdk_posttooluse.py` now mirror the
  real SDK pattern — `__aenter__` calls `connect()` — so the
  new wrap actually fires in tests. Without this the tests
  would only exercise the broken pre-0.22.1 code path.

Customers on 0.22.0 should upgrade immediately. There is no
migration step beyond `pip install --upgrade egisai`; the
public API is unchanged.

---

## [0.22.0] — 2026-05-12

### Added — `claude_agent_sdk` PostToolUse hook gates tool *results*

The 0.21 release wired `PreToolUse` into the policy engine so
tool / MCP **inputs** could be hard-gated before the Node.js
subprocess dispatched them. 0.22 closes the symmetric SOC 2 /
ISO 27001 / GDPR gap on the **output** side: the
`claude_agent_sdk` patch now also injects a `PostToolUse` hook
that fires AFTER the tool runs but BEFORE Claude is shown the
result. Output-side `pii_scan` / `deny_output_regex` /
`semantic_guard` policies evaluate the tool's response text and
the SDK swaps the result in place via the CLI's
`updatedToolOutput` / `updatedMCPToolOutput` substitution
contract — so a CRM lookup that returns a customer's email or
SSN gets masked (sanitize verdict) or replaced with a
recoverable denial payload (block verdict) before the model
ever sees the raw bytes.

This was the only seam in the matrix where a tool result could
round-trip the model unmasked. Bug originally reported by a
customer running a multi-agent harness against the SDK: every
tool step showed `verdict=allow` on the dashboard even when the
tool returned validated PII, because policy evaluation happened
only on the tool's **input args** (PreToolUse) and on the
model's **assistant text** at end-of-turn. The tool's **result**
itself was never policy-evaluated. After 0.22, the audit
trail carries a dedicated `tool_result` step row per affected
tool with `verdict=sanitize|block`, `matched_policy`,
`sanitizations`, and a post-redaction `request_text` preview —
so SOC 2 auditors querying "what tool results did we refuse?"
get a faithful answer attributed to the offending tool, not to
the end-of-turn `model_call`.

- **`_build_posttooluse_callback()`** — builds a hook closure
  bound to the turn's identity record + Run context. Extracts
  the tool response text from MCP-shaped (`{"content": [{"type":
  "text", "text": "..."}]}`), raw-string, or opaque-JSON
  responses; runs `evaluate_output(OutputCall(text=...,
  allow_sanitize=True))`; on sanitize masks via `pii.sanitize`,
  on block builds a denial payload, on allow returns `{}` (the
  cheap-path is the common path). Emits a `tool_call` step row
  with `target=...tool_result` only when the verdict is not
  allow, so the dashboard timeline stays clean.
- **`_extract_tool_response_text()`** + **`_rewrite_tool_response()`**
  — handle the three response shapes (`mcp`, `string`, `json`)
  with shape-preserving substitution. Non-text MCP parts
  (images, audio) pass through unmodified so we don't
  accidentally strip a tool's legitimate non-text content.
- **`_inject_posttooluse_hook()`** — mirrors the PreToolUse
  injector: shallow-copies any existing `options.hooks` dict,
  appends our matcher to the `PostToolUse` slot, writes back.
  User-supplied PostToolUse hooks remain in place; the CLI
  runs all of them.
- **`OutputPolicyContext.allow_sanitize`** + **`OutputCall.allow_sanitize`**
  (new field, default `False`) — flips `pii_scan` from "always
  block on the output side" to "honour the operator's action
  config". Only the PostToolUse path sets it to `True`; every
  other output caller's behavior is byte-for-byte identical to
  0.21. This is what makes `pii_scan` with `action="sanitize"`
  actually mask the email in the tool result (vs. coerce to
  block as on response-text paths that have no atomic
  mutation surface).
- **Privacy contract preserved** — the step row's
  `request_text` is sampled from the POST-sanitize / POST-denial
  text. Raw PII goes out of scope as soon as the policy
  decision is computed (per `security-and-compliance.mdc` §1, §5).
- **Fail-open** — any exception in the PostToolUse callback
  returns `{}` so a buggy policy never bricks the customer's
  agent (per `sdk-design-philosophy.mdc` §5).

The patch covers both `ClaudeSDKClient.query` (streaming) and
the module-level `claude_agent_sdk.query()` one-shot, mirroring
the PreToolUse wiring exactly.

### Changed — `bedrock_agent.py` docstring tightened to call out tool-result advisory gap

AWS Bedrock Agents run Action Groups on AWS-managed
infrastructure with no equivalent of `PostToolUse`. The
docstring now explicitly tells SOC 2 / GDPR / HIPAA-bound
customers that PII in Action Group results is leaked to the
model and lists two escape hatches: `claude_agent_sdk` for
agentic workloads (PostToolUse enforced), or the standalone
`bedrock-runtime` Converse API with the agentic loop driven in
Python (next-call input phase catches tool-result PII).

### Added — `tests/test_claude_agent_sdk_posttooluse.py`

15 tests covering the full PostToolUse contract:

- Hook injected alongside PreToolUse on every `query()`.
- User-supplied PostToolUse hooks compose without clobbering.
- PII in MCP-shaped tool result → masked via
  `updatedMCPToolOutput`, raw value gone.
- PII in MCP-shaped tool result with `action="block"` →
  denial payload via `updatedMCPToolOutput`, model can recover.
- Raw-string tool result (Bash, Read, …) → replaced via
  `updatedToolOutput` (NOT `updatedMCPToolOutput` — the field
  the CLI looks at depends on the shape).
- Opaque-dict tool result → JSON-serialized for scanning,
  replaced via `updatedToolOutput`.
- Allow path: no substitution, no extra step row (cheap path
  stays cheap).
- Empty / image-only / `None` responses skip evaluation
  without crashing.
- `deny_output_regex` fires on tool result text (catches
  secrets / API keys / proprietary identifiers that aren't
  standard PII).
- Multi-tool turn: each tool's result independently gated;
  only non-allow tools emit step rows.
- `PreToolUse` deny → `PostToolUse` never fires for that
  tool (subprocess never dispatched it).
- Identity scope propagates into the hook closure (separate
  asyncio task; contextvars don't carry).
- Fail-open on policy crash.
- Raw PII never lands on the audit row preview.

### Added — `tests/test_tool_result_extraction_cross_framework.py`

7 regression tests locking in the cross-framework guarantee:
for every patched framework whose agentic loop runs in Python
(OpenAI / Anthropic / Google GenAI direct + every framework
that delegates to one of them — LangChain, LangGraph, CrewAI,
Pydantic-AI, LlamaIndex, AutoGen, Agno, Smolagents, Google
ADK, Strands, OpenAI Agents), tool results round-trip through
the next call's input phase. The tests pin
`extract_prompt_text` / `extract_anthropic_prompt` /
`extract_gemini_prompt` / `extract_payload_text` so they don't
silently stop walking tool-result blocks on a refactor.

End-to-end glue test
`test_pii_scan_fires_on_anthropic_next_call_with_tool_result`
verifies a `pii_scan` rule on the input side actually blocks a
follow-up call whose `messages` contain a `tool_result` block
with an SSN — i.e. the SOC 2 contract holds for the direct-LLM
case even without the new PostToolUse hook.

---

## [0.21.0] — 2026-05-12

### Added — `claude_agent_sdk` PreToolUse hook hard-gates tool / MCP calls

The `claude_agent_sdk` patch now injects a `PreToolUse` hook into
`ClaudeAgentOptions.hooks` at every `query()` / `ClaudeSDKClient`
boundary. The hook routes every tool / MCP dispatch through our
policy engine **before** the Node.js subprocess runs the tool, so
`deny_tool_call` / `deny_mcp_call` / `semantic_guard` on tool calls
become real pre-execution enforcement rather than post-hoc audit.

Previously, `claude_agent_sdk` was the only framework in the matrix
that stamped tool-block audit rows as `enforcement_status="advisory"`
because the CLI executed tools in a subprocess before Python ever
saw the `ToolUseBlock`. With 0.21, the hook fires on the control
channel before dispatch; we evaluate policies, return `permissionDecision`,
and the CLI either runs the tool or synthesizes a denial result.
Audit rows for these calls now stamp `enforcement_status="enforced"`,
matching every other framework in the matrix.

- **Pre-execution gate** — `_build_pretooluse_callback()` builds a
  closure that runs `evaluate_output` on the tool dispatch, captures
  the identity scope and run context (contextvars don't propagate
  to the SDK's hook task), emits the per-tool step row, and returns
  `{"hookSpecificOutput": {"permissionDecision": "allow" | "deny", ...}}`
  to the CLI.
- **User-hook composition** — If the user already passed their own
  `PreToolUse` hooks in `options.hooks`, our matcher is **appended**
  to the list rather than replacing it. The SDK runs all matchers;
  any one returning `deny` denies the call.
- **Feature-detected** — `_hooks_supported()` checks for the
  `hooks` field on `ClaudeAgentOptions` and the `HookMatcher` class.
  Older SDK versions without the hook API fall back to the legacy
  post-hoc advisory mode; audit rows in fallback mode are honestly
  labelled `enforcement_status="advisory"`.
- **Fail-open on policy errors** — A buggy policy that raises in
  the hook is caught and treated as `allow`; the user's agent never
  gets bricked by our policy engine misbehaving (per
  `sdk-design-philosophy.mdc`).
- **No double-emission** — When the hook ships a `tool_call` step,
  the receive-side fallback emitter sees the `tool_use_id` in the
  in-flight decisions dict and skips its redundant row. Audit logs
  show exactly one step per tool dispatch.

The patch covers both `ClaudeSDKClient.query` (streaming) and the
module-level `claude_agent_sdk.query()` one-shot.

### Added — cross-framework enforcement matrix in README

`README.md` now ships a locked enforcement matrix documenting which
of the 19 supported frameworks gate tool / MCP calls
pre-execution (`enforced`) versus post-hoc audit (`advisory`).
After this release, **only** AWS Bedrock Agents stamps advisory —
and `SECURITY.md` explicitly names that limitation so customers
can risk-assess accordingly. A new test
(`tests/test_enforcement_matrix.py`) asserts the matrix in code
stays in sync with the README and SECURITY.md so doc drift fails CI.

### Added — comprehensive `PreToolUse` hook test suite

`tests/test_claude_agent_sdk_pretooluse.py` (17 tests) covers:

- Hook injection composes with user-supplied `PreToolUse` matchers
  without clobbering them.
- Hook returns `allow` on no-policies / safe tools.
- Hook returns `deny` on `deny_tool_call` matches with a descriptive
  `[egisai] …` reason embedded for the model's next-turn context.
- MCP namespaced tool names (`mcp__<server>__<tool>`) are parsed
  and matched against `deny_mcp_call` rules.
- Each `tool_use_id` is gated independently in a multi-tool turn.
- Per-tool `tool_call` step rows stamp `enforcement_status="enforced"`
  when the hook fires; no duplicate rows from the receive-side
  fallback emitter.
- End-of-turn `model_call` row stamps `enforced` (matching OpenAI /
  Anthropic / Google direct patches) when hooks were active for
  the turn.
- Legacy `ClaudeAgentOptions` without the `hooks` field falls back
  to the pre-0.21 advisory path; existing audit semantics preserved.
- Identity propagates correctly into the hook closure (contextvars
  don't cross the SDK's task boundary; we re-enter `identity_scope`
  inside the callback).
- **Demo scenario**: "agent decides to delete all users via MCP"
  is hard-blocked at the hook before AWS receives the call.
- Bash commands that match a `deny_tool_call` on `^Bash$` are
  blocked pre-execution.
- Chain-of-tools: read → process → write where `write_file` is
  denied leaves the first two with `allow`/enforced and the third
  with `block`/enforced — independently gated per tool.
- Policy engine raising in the hook fails OPEN, the agent keeps
  running.

### Changed — `_run_output_phase` enforcement-status semantics

End-of-turn `model_call` audit rows in `claude_agent_sdk` now stamp:

- `enforced` on the allow path (no policy fired — there was nothing
  to enforce against). Unchanged from 0.20.
- `enforced` on the block path **when hooks were active for the
  turn**. New in 0.21.
- `advisory` on the block path **when hooks were NOT active**
  (older SDK / custom transport bypass). Unchanged from 0.20.

Sanitize paths and allow paths remain `enforced` in all cases.

### Removed — nothing externally visible

No public API removed. The pre-existing single-row advisory
behavior is preserved as the fallback path for legacy SDK
versions.

---

## [0.20.0] — 2026-05-12

### Added — per-tool `tool_call` steps for OpenAI

OpenAI integrations (Chat Completions and the Responses API,
sync and async) now emit one `tool_call` step per tool the
model invokes, in addition to the parent `model_call` step.
The dashboard's run timeline now reads top-to-bottom as a clear
multi-step waterfall:

```
Prompt received
  → Input policy   (Allowed / Sanitized / Blocked)
  → Model call
  → Output policy  (Allowed / Sanitized / Blocked)
  → Tool · lookup_customer
  → Tool · send_email
  → Input policy
  → Model call
  → Output policy
  → … → Returned to your code
```

Each per-tool step carries the tool's name and a
**label-redacted** preview of its arguments (`label_redact` + 2
KB truncation, identical to the existing tool-input sanitizer
used by `claude_agent_sdk`). The SDK never persists raw tool
arguments — if the model emitted a free-text PII argument, the
operator sees `<EMAIL>` / `<SSN>` / etc. on the timeline.

`enforcement_status` on these per-tool steps is `"enforced"`:
the parent `model_call`'s output policy had a chance to refuse
the tool request before this response left the gate. (Contrast
with `claude_agent_sdk`, whose per-tool steps are marked
`"advisory"` because the Node.js subprocess executes tools
before Python sees them.)

The per-tool emission only fires when an agentic framework
(`Runner.run`, the auto-detected agent loop, …) has opened a
Run above the gate. Raw `client.chat.completions.create()`
calls without an agent wrap continue to emit a single legacy
event the backend synthesises into a one-step Run on ingest —
that preserves wire compatibility for every pre-0.20 SDK
running in production today.

### Changed — every policy enforcement layer is visible on the dashboard

The run timeline modal now **always** renders the Input policy
box and Output policy box for every `model_call` step, even
on a plain `allow` with no matched rule. Pre-0.20 the policy
boxes were collapsed away on allow paths, which made it hard
to confirm at a glance that a clean run had in fact been
evaluated.

The visual contract is now:

- **Allowed** — small green shield box with "Prompt cleared
  every input-side rule" / "Response cleared every output-side
  rule". The matched-policy chip is omitted because nothing
  fired; the box's presence alone communicates "policy ran,
  result was clean".
- **Sanitized** — yellow box with the matched rule + redaction
  summary (`Redacted: 2 SSNs and 1 email`). On the input side
  the box also carries the masked prompt body.
- **Blocked (enforced)** — red box with the matched rule. When
  the **input** phase enforces a block, the timeline terminates
  immediately at the red box (no follow-up Model or Output
  boxes) — "if it's red, that's the end of the flow".
- **Blocked (advisory)** — yellow box with an `advisory` chip
  and a one-paragraph explanation that the framework had
  already executed in a subprocess by the time the policy could
  fire.

Each step on the timeline is also numbered (`Step 1 · Input
policy · Allowed` / `Step 1 · Model · gpt-4o` / `Step 2 · Tool
· lookup_customer` / …) so the operator can reason about the
agent's loop position even on long runs.

This change is dashboard-only — no SDK code or schema column
moved.

---

## [0.19.0] — 2026-05-12

### Privacy — model responses are NEVER persisted

Egis now treats the **model's response text as ephemeral**.
The SDK evaluates the response against output policies
(`deny_output_regex`, `deny_tool_call`, `deny_mcp_call`,
`semantic_guard`, …) — and then the text goes out of scope.
No audit event carries a `response_preview`. No `run.end`
envelope carries a `final_text`. No SSE step payload carries a
`step_response_preview`. The dashboard's "Model response" panel
has been removed from `RunTimelineModal`; what the model
**said** is intentionally not part of the visible audit record.

What we DO still record:

- the verdict, matched policy, and reason of any output-side
  decision (so operators can see that, e.g.,
  `deny_output_regex='credit_card'` matched without ever
  storing the matching string),
- token usage and wall-clock cost of the response,
- per-tool steps for agentic runs — including each tool's
  name and label-redacted input arguments — so the run
  timeline still shows what the agent **did**.

Why: model outputs are far less constrained than their
inputs (the model can emit anything from PHI to API keys to
free-text PII the input sanitizer never had a chance to see).
SOC 2 / GDPR / HIPAA / ISO 27001 auditors consistently prefer
"we never had it" over "we redacted it well". Storing model
responses is a perpetual leak surface; not storing them
eliminates the surface entirely.

The `response_preview` column on `RequestLog` and the
`final_text` column on `Run` are retained on the schema for
historic-row compatibility but are written as `NULL` on
every new row. Defense in depth: the ingest endpoint coerces
incoming `response_preview` / `final_text` fields to `NULL`,
so even if a misbehaving SDK ships them they don't get
persisted.

### Added — `enforcement_status` on every audit event

A new top-level field on every audit event the SDK emits records
whether the policy decision was **actually enforced** or merely
**advisory**. This is independent from `verdict` so the audit row
can honestly answer two questions at once: "what did the policy
decide?" and "did the SDK actually prevent the data flow?".

- `"enforced"` (default for every event): the SDK actively
  prevented the call from reaching its full destination. Every
  input-side block is `enforced` (the prompt is never forwarded),
  and every output-side block on the synchronous patches
  (`openai`, `anthropic`, `google.genai`, `bedrock`, the
  `httpx`/`requests` fallback) is `enforced` (the gate either
  returns a synthesized stub or raises `PermissionError` before
  user code sees the model's real output).
- `"advisory"`: the policy decided block, but by the time the
  SDK could observe the response, the agentic framework had
  already executed the call end-to-end (tools fired, side
  effects landed). `claude_agent_sdk` with `on_block="stub"` is
  the canonical case: Anthropic's Node.js CLI runs the entire
  agent loop in a subprocess we don't own, so an output-side
  policy firing on `ResultMessage` is a post-hoc finding, not
  an enforcement.

The dashboard's RunTimelineModal renders advisory blocks in a
distinct warning tone with the copy *"Policy fired — call not
prevented"* and explains in the empty-state hint that "the agent
framework runs its loop in a subprocess we cannot stop
mid-flight". SOC 2 / GDPR / HIPAA / ISO 27001 all require the
audit trail to distinguish "control evaluated" from "control
enforced"; conflating them inside `verdict` was an audit-finding
risk we now eliminate.

### Added — output-side policy latency + tokens land on the audit row

`policy_latency_ms`, `policy_tokens_in`, and `policy_tokens_out`
are now **additive across both phases** of policy evaluation.
Pre-0.19 only the input phase booked these fields, so the
dashboard's "Policy (sum)" stat was 0 ms for any call where the
only matching policy was an output-side `semantic_guard` —
hiding the real cost (LLM judge wall-clock + token spend) of
post-model governance.

The output-phase contribution is summed on top of whatever the
input phase booked, so a call that pays for both an input-side
deterministic regex and an output-side LLM judge reports the
correct combined total without overwriting either phase.

### Added — agentic frameworks emit one step per tool use

`claude_agent_sdk`'s streaming receive path now dispatches one
`tool_call` step per `ToolUseBlock` it sees in an
`AssistantMessage`, so the dashboard's RunTimelineModal renders
the actual agentic loop (assistant turn → its tools → next turn
→ its tools → …) instead of collapsing a 6-tool customer-support
agent into a single `model_call` step.

Each per-tool step carries its own:
- per-tool output-policy verdict (so `deny_tool_call` /
  `deny_mcp_call` pinpoint which tool tripped them),
- label-redacted `request_text` of the tool's input (so PII a
  model passed as a tool argument never reaches the audit row
  raw),
- `enforcement_status="advisory"` (the Node subprocess had
  already executed the tool by the time the `ToolUseBlock`
  reached Python).

### Fixed — model_call step now renders as separate per-phase boxes

Before: a `pii_scan` policy with `action="sanitize"` runs on
**both** sides of the call. On the input side it sanitizes the
prompt and forwards it; on the output side it gets coerced to
`block` (the SDK can't safely rewrite a provider's response —
see `policy/engine.py::_pii_scan_match`) which on
`claude_agent_sdk` becomes `advisory` because the Node
subprocess had already executed. The audit row is honest about
all of this — `prompt_decision.verdict="sanitize"`,
`response_decision.verdict="block"`,
`enforcement_status="advisory"` — but the `RunTimelineModal`'s
`StepBox` rolled the two phases into a single visual unit that
mixed "redact" + "blocked" + "forwarded to model" into one box,
which was visually contradictory ("blocked AND forwarded?") even
after an earlier fix tried to split the verdicts into chips
inside that same single box.

The `RunTimelineModal` now expands each `model_call` step into
up to **three sequential boxes** so the operator can read the
flow top-to-bottom — input policy → model → output policy:

    ┌── Input policy check ─── [sanitize] [Redact…] [Redacted: 2 PII] ──┐
    │   Egis redacted regulated data before forwarding to the model    │
    │   Sanitized prompt (forwarded to model): SECURITY REPORT…        │
    └──────────────────────────────────────────────────────────────────┘
    ┌── Model · claude-opus-4-8 ─── 47015 ms · 62 in / 2422 out ───────┐
    │   Model called and returned — see output policy box below        │
    └──────────────────────────────────────────────────────────────────┘
    ┌── Output policy check ─── [block] [advisory] [Redact…] ──────────┐
    │   Policy decided block — the framework had already executed in  │
    │   a subprocess. The audit row records this as an advisory       │
    │   finding — investigate whether an input-side guard …           │
    └──────────────────────────────────────────────────────────────────┘

Implementation:

- `renderModelCallBoxes(step)` is the new dispatcher: it
  inspects `prompt_decision` / `response_decision` /
  `enforcement_status` on the step row and returns the right
  set of `<FlowBox>` elements for this step's actual shape.
- `PolicyPhaseBox` is the new shared component for either
  pre- or post-model policy boxes. It renders the matched
  policy name, the verdict badge, an "advisory" tag for the
  output-side advisory case, the redaction-type chip, and —
  for output-side advisory blocks — the long explanatory
  paragraph that pre-0.19 lived on the run-level terminator.
- `ModelCallBox` is the new dedicated middle box that just
  describes the API call itself (model name, latency, tokens).
  When input was enforced-blocked it switches to a danger
  tone and reads "Call refused before reaching the model".
  When the call ran but output policy fired advisory, it
  reads "Model called and returned — see output policy box
  below" so the operator's eye is pulled to the actual
  finding.
- The run-level terminator copy simplifies to "Run completed
  with advisory finding" (one line) since the advisory
  explanation is now anchored to the specific step that
  produced it. Pre-0.19 the terminator carried that whole
  paragraph at the bottom of the timeline, which made it
  unclear *which* step had fired.
- Tool calls in `claude_agent_sdk` have only one policy phase
  (per-tool output evaluation) so they continue to render as
  a single box — splitting them would be empty noise.

For live-streaming steps the per-phase signal is now on
the `run.step.added` SSE wire as `step_prompt_decision` and
`step_response_decision`, mirroring the JSONB columns the
backend already persists. Previously only the rolled-up
`step_verdict` reached the SSE stream, so freshly-arriving
rows could only render the legacy single-box view until a
page reload.

### Added — redaction-type chip on every step of the run timeline

The `RunTimelineModal`'s per-step box now renders a `Redacted: 2
SSNs and 1 email`-style chip whenever the SDK masked PII on that
step. The chip shows only the **types and counts** of fields
that were redacted — never the original values, never the
positions, never anything that could be reverse-engineered into
the source PII. This restores the visibility that the
RequestDetailModal already had, but at the per-step granularity
the run timeline needs (e.g. one tool call masked an email, the
next masked an SSN).

To make this work for live-streaming steps without a page
reload, the backend's `run.step.added` SSE event now also
carries the canonical `step_sanitizations`,
`step_matched_policy`, and `step_policy_reason` fields. The
shape mirrors the persisted `RequestLog.sanitizations` column
(`[{type, count, pattern}]`) — counts and types only.

### Changed — intent_summary headlines are now ≤ 70 chars

The LLM-generated **Decision Analysis** headline on the
Requests table and the `RunTimelineModal` H2 is now strictly
short: 4–8 words, ≤ 70 characters. Pre-0.19 the cap was 220
chars, which let the LLM produce paragraph-length intent
descriptions that broke the table layout and buried the
operator-facing signal under restated prompt content.

Changes:
- `_MAX_LEN` in `intent_summarizer.py` lowered from 220 → 70.
- `_SYSTEM_PROMPT` rewritten to demand 4–8 words / ≤ 60 chars
  with an explicit example list and an explicit ban on
  prompt-specific identifiers (account numbers, customer
  names, dollar amounts, dates).
- `max_tokens` lowered from 64 → 24 as a hard ceiling.
- `template_summary` fallback strings rewritten as short
  imperative headlines (e.g. `Tried to send regulated PII`,
  `Tried restricted MCP server`, `Allowed by policies`).
- `_truncate` now strips trailing terminal punctuation and
  prefers a word boundary when cutting (mid-word breaks were
  invisible at 220 chars; they're glaring at 70).
- The frontend's `RunTimelineModal` H2 also truncates the
  prompt-text fallback to 80 chars one-liner so a still-
  pending intent_summary doesn't briefly render a whole
  paragraph as the title.

### Fixed — `claude_agent_sdk` audit row no longer says "Blocked"
###     when it was actually a post-hoc finding

The combination of the changes above means a successful
multi-tool `claude_agent_sdk` run with `on_block="stub"` no
longer shows the dashboard a single "Blocked" step. Instead it
shows: the model_call's `enforcement_status="advisory"` (with
the original policy match preserved) and one `tool_call` step
per real tool invocation. Operators auditing the run get an
honest answer to "did this run succeed?": *the tools executed
because the SDK couldn't stop the subprocess mid-flight; the
policy that fired is recorded as advisory; investigate whether
an input-side guard or a synchronous SDK path would close the
gap*.

### Compliance

This release is a SOC 2 (CC7.2 / CC4.2), GDPR Art. 30 / 32,
HIPAA §164.308, and ISO 27001 A.12.4 alignment update on three
axes:

1. **Data minimization** — model responses are no longer
   retained in any data store the platform owns. The
   auditable trail shows what the agent *did* (verdict, tool
   calls, matched policies), not what it *said*.
2. **Control evidence** — `enforcement_status` distinguishes
   "control evaluated" from "control enforced" as a queryable
   column rather than implicit text.
3. **Operator UX** — short, deterministic intent headlines
   make the Decision Analysis column a usable governance
   signal at fleet scale rather than an essay-length echo of
   the prompt.

No behavioural change for callers; the audit shape is additive
and older backends ingest 0.19 events cleanly (every field has
a backend-side `server_default`).

---

## [0.18.1] — 2026-05-12

### Fixed — Bedrock output-phase enforcement gap

`bedrock_runtime` (AWS Bedrock Converse API) was the one LLM-level
adapter that forwarded through `gate_call` *without* an
`extract_output_signals` extractor. The result: every Bedrock-hosted
model call silently skipped Phase 3 of the policy engine —
`deny_tool_call`, `deny_mcp_call`, `deny_output_regex`, and the
post-model `semantic_guard` policies never fired against Bedrock
traffic. OpenAI / Anthropic / Google calls were always fully gated;
Bedrock callers had input-side enforcement only.

This release closes the gap:

- New `extract_bedrock_converse` (in `egisai/_output_signals.py`)
  parses Bedrock's `output.message.content` blocks for assistant
  text + `toolUse` invocations, plus `toolConfig.tools[*].toolSpec.name`
  for definition-side denylist matches. The shape mirrors
  Anthropic's Messages API because Bedrock normalises across
  providers (Anthropic / Mistral / Cohere / Meta / Amazon) onto one
  envelope.
- `_patches/bedrock_runtime.py` passes `extract_output_signals=
  extract_bedrock_converse` so every `client.converse(...)` and
  `client.converse_stream(...)` call runs the output phase.
- `_patches/bedrock_agent.py` (`InvokeAgent` for AWS-managed
  agents) now runs `gate_call` so input-side policies (PII,
  `deny_regex`, `max_prompt_chars`, pre-model `semantic_guard`)
  fire on the `inputText` parameter before it reaches AWS.
  Output-side gating remains exempted for this one adapter — the
  response is a caller-iterated `EventStream` we can't replay
  without breaking the user's iteration loop. The exemption is
  declared in `PATCHES_WITHOUT_OUTPUT_EXTRACTOR` and pinned by the
  audit test below.

### Added — structural audit test that prevents this gap from regressing

`tests/test_output_extractor_audit.py` walks every `_patches/*.py`
via AST and asserts that any `gate_call` / `async_gate_call` site
either passes `extract_output_signals=` or appears in an explicit
exemption list with a justification. This is the test that would
have caught the original Bedrock gap on its first commit. Any
future LLM-level patch that forgets to wire the output extractor
fails the audit at unit-test time.

End-to-end coverage added in `tests/test_output_side_policies.py`:
`test_gate_call_blocks_bedrock_response_with_banned_tool_call`
verifies a Bedrock Converse response carrying a denylisted tool
invocation now raises `PermissionError` exactly like the OpenAI
path does.

### Fixed — Compliance rule §5 leak in `ClaudeSDKClient.query`

`_patches/claude_agent_sdk.py` was calling `open_run(prompt_text=
prompt_text[:280])` **before** the input policy phase had run,
which meant the streaming `run.start` envelope shipped the raw
prompt to the platform before any sanitization layer had a chance
to redact PII. That violates rule §5 (audit before persist) and
rule §1 (no raw PII over the wire) from the security & compliance
guide.

`open_run` is now called with `prompt_text=None` for the
`ClaudeSDKClient.query` path. The backend's `_apply_step_to_run`
promotes the first step's **post-sanitize** `prompt_preview` onto
the Run's `prompt_text` once the input phase has finished — so the
dashboard still shows the prompt within the SDK's normal flush
window (typically sub-second), but the payload that crosses the
wire never carries the un-redacted original.

The module-level `claude_agent_sdk.query` path was already correct
(it called `open_run` *after* the input phase, so the value it
passed was already sanitized). No change there.

### Internal

- SDK suite: **421 passed, 28 skipped** (real-framework audits
  require the audit venv with all 14 libraries pre-installed).
- `ruff check .` and `mypy src/egisai` both clean.

---

## [0.18.0] — 2026-05-11

### Architecture — Runs & Steps: one row per logical agent task

Solves the "4 tool calls = 4 agents + 5 requests" bug across **all 14
supported frameworks** (OpenAI, Anthropic, Google Gemini, openai-agents,
Claude Agent SDK, LangGraph, LangChain, CrewAI, AutoGen, Agno, Strands,
smolagents, LlamaIndex, Pydantic AI, plus Google ADK, raw httpx/requests/
boto3). Each framework entry-point invocation
(`Runner.run(...)`, `Pregel.invoke(...)`, `ClaudeSDKClient.query(...)`,
`Agent.arun(...)`, `Workflow.run(...)`, …) now ships as **one Run** with
N **Steps** beneath it instead of N independent audit rows.

Concretely:

- **New SDK module `egisai/_run.py`** — defines `RunContext`,
  `RunStep`, a `ContextVar`-backed `_current_run`, and an
  `open_run` / `append_step` / `close_run` lifecycle that's async-
  / thread-safe via Python's standard context-variable propagation.
  Identity is **locked at run open**: every inner model call shares
  the same `agent_id`, even when subsequent payloads would otherwise
  re-derive a different one. Trace IDs are minted once per Run.
- **Generic `_framework.py` wrappers** (`wrap_sync_entrypoint`,
  `wrap_async_entrypoint`, `wrap_async_iter_entrypoint`,
  `wrap_sync_iter_entrypoint`, `wrap_polymorphic_entrypoint`) all
  open a Run on entry, run the original framework code inside an
  `identity_scope` + a `_RunScope`, and close the Run on exit
  (including on early `break`, exception, `aclose`, GC).
- **Claude Agent SDK** (`_patches/claude_agent_sdk.py`) opens the
  Run **before** input policy evaluation so even input-blocked calls
  emit a complete `run.start → run.step → run.end` triplet — the
  dashboard never sees an invisible refusal.
- **Streaming wire protocol** — events are now framed as one of:
  - `run.start`  — emitted at framework entry (≤ 1 ms after invoke).
  - `run.step`   — emitted as each LLM / tool call completes.
  - `run.end`    — emitted when the framework entry-point exits.
  Long-running agentic tasks (5–50 steps) paint themselves on the
  dashboard step-by-step instead of waiting for the terminal event.

### Wire format

`POST /v1/sdk/events` now accepts a heterogeneous batch of the
three envelope kinds above. The backend dispatches on `kind`:

- `run.start` → upserts a `runs` row + publishes `run.logged` SSE.
- `run.step`  → inserts the `request_logs` row, updates the parent
                 Run's aggregates (worst-of verdict, sum tokens /
                 cost / latency, max `step_seq + 1` as step_count),
                 publishes `run.step.added` SSE with both the
                 step-local fields and the latest cumulative
                 Run aggregates.
- `run.end`   → finalizes Run aggregates from the SDK's canonical
                 totals + publishes `run.summarized` SSE.

Legacy single-row events from pre-0.18 SDKs are transparently
synthesised into one-step Runs so dashboards reading `/v1/runs`
see a uniform shape regardless of SDK age.

### Schema additions (`backend/alembic/versions/20260601_0000_runs_and_steps.py`)

- New `runs` table partitioned by month on `started_at`. Aggregates
  step_count, worst-of verdict + risk_level, sum tokens / cost /
  latency, primary model, first-step prompt preview, last-model-
  step response preview, plus `parent_run_id` for sub-agent /
  handoff linkage.
- New columns on `request_logs`: `run_id`, `step_seq`, `step_kind`
  (`model_call` | `tool_call` | `sub_agent_spawn` | `policy_check`),
  `framework`, `tool_name`. The existing column set is otherwise
  unchanged.
- Chunked SQL backfill (7-day windows) folds historic
  `request_logs` rows into one-step Runs.

### Dashboard

- **Requests page** is now a list of Runs — one row per logical
  agent task, with step count, primary model, aggregated tokens /
  latency / cost, and worst-of verdict.
- **RunTimelineModal** (click any row) shows the prompt → policy
  → model → policy → tool → … → final waterfall as a vertical
  timeline, paint-as-you-go via SSE.
- **Overview → Recent governed actions** widget mirrors the Requests page.
- **/v1/runs** + **/v1/runs/{run_id}** endpoints land alongside
  the still-supported `/v1/requests` (back-compat shim for one
  release).

### Tests

- 18 SDK lifecycle tests (`tests/test_run_lifecycle.py`).
- 6 per-framework integration tests
  (`tests/test_run_per_framework.py`) — assert exactly one Run
  with N steps across OpenAI Agents, LangGraph, Agno, sync, async,
  streaming, polymorphic, and nested-wrap scenarios.
- 14 backend ingest helper unit tests
  (`tests/routers/test_sdk_runs_ingest.py`).
- SDK suite: 417 green.
- Backend suite: 614 green (600 existing + 14 new).

### Compatibility

- 0.17.x SDK clients keep working unchanged — their single-row
  events are synthesised into one-step Runs at ingest time.
- `/v1/requests` keeps its old contract for at least one release.

---

## [0.17.6] — 2026-05-11

### Fixed — Claude Agent SDK policy enforcement (gap fix)

Closes the second half of the Claude Agent SDK bug reported after
0.17.5: agents registered on the dashboard, but no audit row ever
landed and no policy ever fired against an `async with
ClaudeSDKClient(...)` session. The 0.17.5 patch only wrapped the
*identity* boundary on `ClaudeSDKClient.query`; the LLM call itself
happens inside a Node.js subprocess (`claude` CLI) that the Python
package pipes JSON into, so `gate_call` was never invoked — there was
no `httpx`/`requests` round-trip for our SDK to intercept.

This release moves governance up to the Python-visible boundary:

- **`ClaudeSDKClient.query(prompt)`** now runs the full input gate
  (`deny_regex`, `pii_scan`, `semantic_guard`, …) on the prompt
  BEFORE the JSON is shipped to the subprocess. Blocks raise
  `PermissionError` and the raw prompt never leaves the Python
  process. Sanitize verdicts mutate the forwarded prompt to the
  masked copy — security-and-compliance.mdc §1 (raw PII never
  leaves the SDK boundary) is honoured even though the LLM call
  lives in another process.
- **`ClaudeSDKClient.receive_messages()`** accumulates the streamed
  `AssistantMessage` (text + `ToolUseBlock`) and `ResultMessage`
  per turn, runs output policies (`deny_tool_call`,
  `deny_mcp_call`, `deny_output_regex`) on the accumulated
  signals, stamps the audit event with tokens / cost / latency,
  and enqueues it for the dashboard. Output-side blocks raise
  before yielding the `ResultMessage` so iterating loops
  terminate at the violation.
- **`ClaudeSDKClient.__aexit__`** flushes any in-flight event when
  the client context closes without iterating — incomplete turns
  surface as `error="never_consumed"` in the audit log rather
  than silently dropping.
- **Module-level `claude_agent_sdk.query(...)`** (the
  single-call async generator API) runs the same Phase 1 → forward
  → Phase 2 pipeline inline.

MCP tools (`mcp__<server>__<tool>` namespacing) are recognised in
the output gate as `mcp_targets` so `deny_mcp_call` rules match on
the server portion. Note: tool execution happens *inside* the Node.js
subprocess before Python sees the `ToolUseBlock`, so blocks are
detect-and-stop rather than pre-execution; pre-execution gating
would require forking the MCP transport.

### Tests

- **`tests/test_claude_agent_sdk_governance.py`** — 14 new tests
  covering audit emission, input policies (block / sanitize /
  allow), output policies (tool call, MCP target, assistant text),
  multi-turn audit hygiene, the `never_consumed` flush, and the
  signature-parity regression check from 0.17.5.

The Claude Agent SDK stays import-guarded and fail-open. Every other
framework patch is unchanged.

---

## [0.17.5] — 2026-05-11

### Fixed — framework patch correctness audit

The Identity v1 patches that shipped in 0.17.2 misclassified the call
shape of several upstream entry points. Where the wrap-kind didn't
match the real upstream's signature, calling the patched method
either crashed with ``TypeError: object async_generator can't be used
in 'await' expression`` (Claude Agent SDK, Agno streaming) or
silently swallowed the framework's returned handle inside a
coroutine (LlamaIndex). Both shipped because the in-repo test stubs
were hand-rolled and didn't mirror the real upstream's signature
shape — so the tests passed on the stub but the real packages broke.

This release re-audits every patched entry point against the real
upstream signature, fixes the mismatches, adds a polymorphic wrapper
for the dispatcher entry points, and locks the regression class out
of CI with a signature-parity gate (40+ new test cases) plus an
opt-in real-libraries audit (``EGIS_AUDIT_REAL_FRAMEWORKS=1``).

**No user-facing API changes** — same ``egisai.init()`` call, same
``set_context`` / ``agent`` / ``register_agent`` helpers, same
backend payload shape. Existing users get the fixes for free on
upgrade.

#### Framework patch fixes

- **Claude Agent SDK** —
  ``ClaudeSDKClient.query`` was wrapped as ``kind="async_iter"``;
  the real method is a coroutine (``async def query(self, prompt)``),
  not an async generator. ``await client.query(prompt)`` now works
  again. Fixes the ``TypeError`` reported by users running the
  Anthropic agentic example. Module-level ``claude_agent_sdk.query``
  (which IS an async generator) is unchanged.

- **Agno** —
  Both ``Agent.run`` and ``Agent.arun`` are plain ``def`` polymorphic
  dispatchers: ``stream=False`` returns a value / coroutine,
  ``stream=True`` returns a sync / async iterator. Pre-0.17.5 they
  were wrapped as ``sync`` / ``async`` respectively, so any caller
  using ``stream=True`` crashed (``arun``) or lost identity scope
  (``run``). Now wrapped with the new ``polymorphic`` kind that
  resolves the return shape at call time.

- **smolagents** —
  ``MultiStepAgent.run`` / ``ToolCallingAgent.run`` / ``CodeAgent.run``
  also support ``stream=True``; same polymorphic wrap.

- **LlamaIndex** —
  ``FunctionAgent.run`` (and ``ReActAgent.run``, ``CodeActAgent.run``,
  ``AgentWorkflow.run``) is a plain ``def`` returning a
  ``WorkflowHandler`` — an awaitable handle whose ``.stream_events()``
  is the streaming API. Pre-0.17.5 we wrapped it as ``async`` which
  swallowed the handle inside a coroutine, breaking
  ``async for ev in agent.run(...).stream_events()``. Now wrapped as
  ``sync`` and the handle is returned directly. Also added explicit
  patches for ``ReActAgent`` / ``CodeActAgent`` / ``AgentWorkflow``
  (only ``FunctionAgent`` was covered previously). Removed the dead
  ``AgentRunner`` patch reference (the class was removed upstream).

- **LangChain 1.x** —
  No code change required. ``AgentExecutor`` was removed in
  LangChain 1.0 in favour of ``langchain.agents.create_agent``, which
  returns a ``CompiledStateGraph``. That class inherits from
  ``langgraph.pregel.Pregel``, so our existing LangGraph patches
  (``invoke`` / ``ainvoke`` / ``stream`` / ``astream``) transparently
  cover ``create_agent`` calls via Python's MRO. Documented in the
  patch module's docstring so future maintainers don't try to "fix"
  the apparent silent no-op.

#### New: polymorphic wrapper

``egisai._patches._framework.wrap_polymorphic_entrypoint`` handles
the broad class of upstream entry points that are plain ``def``
functions whose runtime return value depends on kwargs:

  - returns a coroutine          → identity stays on the stack across ``await``
  - returns an async iterator    → identity stays in scope per ``async for`` yield
  - returns a sync iterator      → identity stays in scope per ``for`` yield
  - returns a plain value/handle → identity was on the stack during the call

Adds ``kind="polymorphic"`` to ``patch_method``.

#### Tests

- **``tests/test_framework_patches.py``** rewritten so every stub
  faithfully mirrors the real upstream signature shape — async
  generator vs. coroutine, sync generator vs. plain ``def``,
  polymorphic dispatcher with ``stream=`` kwarg, ``WorkflowHandler``
  returns. Added stream-toggle paths for Agno / smolagents. Added
  ``await client.query(prompt)`` regression for the original bug.
- **``tests/test_framework_signatures.py``** (new, 10 tests) is a
  signature-parity gate: for every patched method, the wrapped
  attribute MUST keep the upstream's ``iscoroutinefunction`` /
  ``isasyncgenfunction`` / ``isgeneratorfunction`` shape. Any future
  wrong-kind regression fails this gate before the SDK ships.
- **``tests/test_framework_audit.py``** (new, 28 cases, opt-in via
  ``EGIS_AUDIT_REAL_FRAMEWORKS=1``) imports the actual third-party
  libraries — ``claude-agent-sdk``, ``agno``, ``langgraph``,
  ``langchain``, ``autogen-agentchat``, ``crewai``, ``openai-agents``,
  ``pydantic-ai``, ``llama-index-core``, ``smolagents``,
  ``strands-agents``, ``google-adk`` — and verifies the patches don't
  change the upstream's call shape against the real library. Catches
  both wrong-kind regressions AND upstream signature drift.

---

## [0.17.2] — 2026-05-11

### Fixed

- Re-publish of 0.17.0. The original ``v0.17.0`` tag fired against
  a stale commit on the public mirror before the Agent Identity v1
  changes had been rsynced over, so PyPI received an older wheel
  under the ``0.17.0`` name. ``0.17.1`` was skipped because that
  version number was burned during the recovery attempt. The
  ``0.17.2`` wheel contains every bit of the intended Agent
  Identity v1 release (see the ``[0.17.0]`` entry below for the
  full feature list); no other source changes versus that entry.

---

## [0.17.0] — 2026-05-11

### Added — Agent Identity v1

A full rewrite of how the SDK identifies which agent is making a call.
Pre-0.17 the SDK leaned almost entirely on a regex-style hash of the
system prompt; that broke for any flow where users named their agents
explicitly (we'd shadow their name with `agent-0e431168`) or where two
agents shared a system prompt but differed in tools / permissions.
Identity v1 introduces a 7-tier ladder so every flow — chat-style raw
LLM calls *and* agent-framework runs — is attributed correctly, with
the same agent never double-counted across processes or async tasks.

The 7 tiers, in priority order:

| Tier | Source                                       | Stable across calls? |
|------|----------------------------------------------|----------------------|
| 0    | Explicit `set_context` / `egisai.agent()`    | yes — user-supplied  |
| 0.5  | OpenTelemetry `gen_ai.agent.{id,name}` span attrs | yes — span-scoped |
| 1    | Server-issued stable id (OpenAI Responses API `prompt_id`, Gemini `cached_content`, Bedrock `invoke_agent` `agentId`) | yes — server-issued |
| 2A   | Framework patch reading explicit `agent.name` (OpenAI Agents SDK, Google ADK, AutoGen, Agno, Strands, CrewAI, smolagents, LangGraph) | yes |
| 2B   | Framework patch fingerprinting a composite bundle (Claude Agent SDK, LlamaIndex, PydanticAI, legacy LangChain `AgentExecutor`) | yes |
| 3    | Stack-frame hint (`__egisai_agent__`, `agent_name` locals) — opt-in via `auto_stack_hints` | per-call |
| 4    | Class-name introspection (`self.__class__.__name__` ending in `Agent` / `Bot` / `Worker` / `Specialist` / `Assistant`) | per-call |
| 5    | System-prompt SHA-256 + spaCy NER name (NER-first, hash fallback) | yes within process |
| 6    | Init-time `app=` fallback                    | yes within process |

New public surfaces:

- **`egisai.agent("Triage")`** — context manager that pins an explicit
  identity onto the resolver stack for the duration of the `with`
  block. Inner LLM calls auto-inherit. Replaces every place where
  users were tempted to call `set_context(agent=…)` per call.
- **`egisai.register_agent("Triage")`** — eager one-shot registration
  for code paths that want the agent row created up-front (e.g. before
  the first call ever fires). Returns the agent_id or `None` on
  failure (fail-open).
- **`auto_stack_hints="strict" | "loose" (default) | "off"`** — new
  init kwarg controlling Tier 3 stack-frame inspection. `"strict"`
  only respects the explicit `__egisai_agent__` marker; `"loose"`
  also picks up natural `agent_name` / `agent` locals; `"off"`
  disables Tier 3 entirely for security-sensitive deployments that
  don't want any stack walking.

New framework patches (auto-installed on `egisai.init()` when the
framework is importable; silent no-op otherwise):

- `openai_agents.py` — OpenAI Agents SDK
- `claude_agent_sdk.py` — Anthropic Claude Agent SDK
- `langgraph.py` — LangGraph (Pregel.invoke)
- `bedrock_runtime.py` — AWS Bedrock Converse API
- `bedrock_agent.py` — AWS Bedrock InvokeAgent (server-issued agentId)
- `google_adk.py` — Google Agent Development Kit
- `autogen.py` — Microsoft AutoGen
- `crewai.py` — CrewAI
- `agno.py` — Agno (formerly Phidata)
- `strands.py` — AWS Strands Agents
- `smolagents.py` — HuggingFace smolagents
- `langchain.py` — LangChain legacy `AgentExecutor`
- `llamaindex.py` — LlamaIndex `FunctionAgent`
- `pydantic_ai.py` — PydanticAI

Each framework patch wraps the framework's documented entry point,
derives identity using either explicit name (Tier 2A) or a composite
SHA-256 bundle hash (Tier 2B), and pushes the resolved identity onto
a `ContextVar` stack. Inner LLM calls during that invocation read the
parent's identity instead of re-deriving from a (possibly empty)
inner-call system prompt. Idempotent: calling `apply()` twice never
double-wraps.

Backend changes (additive, no breaking schema changes):

- **`agents.identity_hash` `VARCHAR(64) NULL`** — SHA-256 hex of the
  identity bundle the SDK chose. Used as the primary dedup key by
  `POST /v1/sdk/agents/ensure`. Nullable so legacy SDKs (< 0.17)
  keep working unchanged.
- **`agents.identity_source` `VARCHAR(32) NULL`** — controlled-vocab
  detection-tier token surfaced in the dashboard's Provenance card.
- **`agents.name_normalized`** — Postgres `GENERATED ALWAYS AS
  (lower(btrim(name))) STORED` column with a unique index. Stops the
  "I created the same agent twice with different cases" failure.
- **Partial unique index** `(org_id, identity_hash) WHERE
  identity_hash IS NOT NULL` — the canonical dedup contract for new
  rows.
- **`POST /v1/sdk/agents/ensure`** now accepts `identity_hash` +
  `identity_source` in the payload, prefers them for lookup, and
  backfills existing rows when a legacy agent gets re-identified
  under the new scheme.

Frontend:

- `Agent` / `AgentIdentity` types extended with `identity_source` and
  `identity_hash_prefix` (full hash never crosses the API boundary).
- Agent Identity modal's Identity section now renders a Provenance
  row that maps each `identity_source` token to a plain-English
  explanation an operator (or SOC 2 reviewer) can act on.

Stress / coverage:

- 360 SDK tests pass, including 12 new stress tests (concurrent
  threads, async tasks, nested scopes, async generators, idempotent
  `apply()`, fail-open under garbage payloads) and 24 mock-based
  per-framework patch tests.
- 600 backend tests pass, including new schema + repository tests
  pinning `identity_hash` / `identity_source` round-trip and the
  prefix-derivation contract on `AgentIdentityOut`.

### Fixed

- **Policy attribution gap closed.** Pre-0.17, scoped policy rules
  (`target_agents=[…]`) only matched when the user had explicitly
  called `set_context(agent=…)`. The resolver now runs *before*
  policy evaluation inside `gate_call` / `async_gate_call`, so any
  auto-detected identity is visible to scoped rules without user
  intervention. Existing `set_context` callers see no Behavior
  change (Tier 0 still wins).
- **HTTP fallback attribution.** The `httpx` / `requests` model-host
  fallback now also runs identity resolution against the request
  body before enqueueing the audit event — previously those events
  inherited only the init-time `agent_id`.

---

## [0.16.0] — 2026-05-10

### Changed

- **``pii_scan`` policies now default to ``action: "sanitize"`` instead
  of ``"block"``.** Sanitize is the less-destructive choice: the
  user's call still reaches the model, just with the regulated
  values masked locally (``#`` by default, configurable via
  ``mask_char``). The raw PII never leaves the SDK boundary, so the
  audit row, the policy decision, and the model payload are all
  PII-free. Operators who need a hard refusal — for example a
  compliance bar that forbids credit-card text in prompts — opt
  into ``action: "block"`` explicitly on the policy config; the
  dashboard's PII policy modal surfaces both options and the new
  default pre-selects sanitize.

  Existing rules unaffected: every policy that has ``action`` set
  explicitly (the default for any rule created via the dashboard's
  modal) keeps its current Behavior. Only policies that OMITTED
  ``action`` — typically API-created rules — see the new default,
  and that change is desirable: the previous Behavior would refuse
  the call even when masking would have preserved the user's
  workflow without risking a leak.

  On the response side ``action: "sanitize"`` is automatically
  coerced to a block — we never rewrite provider responses — so the
  default-flip does NOT change response-phase enforcement.

---

## [0.15.1] — 2026-05-10

### Changed

- **Quieter PII engine startup.** Removed the ``✓ [egisai] PII engine
  ready (Presidio + spaCy …)`` confirmation line that printed once
  per process after the analyzer finished warming up. The main
  ``✓ [egisai] active …`` banner already confirms the SDK is alive;
  the PII engine is an implementation detail and shouldn't add a
  second startup line. Load failures still surface via the existing
  warning path so misconfigurations remain visible.
- **Removed the per-call "unknown PII types" warning** from the
  policy engine. Unknown ids are still filtered out at runtime
  (the membership check against the canonical taxonomy is
  unchanged), and the platform's policy create/update endpoint
  already rejects unknown types with a ``422`` at write time —
  so live policies are vetted before they hit the SDK. Removing
  the stderr print stops the noise on every prompt for orgs that
  still carry legacy ``pii_scan`` rows with stray strings in
  ``config.kinds``.

---

## [0.15.0] — 2026-05-10

### Added

- **Microsoft Presidio is now the default PII engine.** Ships ~60
  checksum-validated detectors out of the box — SSN, passport, IBAN,
  driver's license, national ID, bank account, crypto wallet,
  multi-country tax IDs, medical IDs, vehicle plates, plus spaCy
  Named Entity Recognition for names / locations / addresses.
  Everything runs LOCALLY inside the SDK process — no network calls
  reach Microsoft or any third party at detection time. The
  ``en_core_web_lg`` spaCy model auto-downloads on first
  ``egisai.init()`` in a background daemon thread; until it's warm a
  regex + checksum fast path keeps the hot path online so detection
  never blocks the user's call.

- **Canonical PII type taxonomy** exposed through a new
  ``GET /v1/sdk/pii-types`` backend endpoint. The dashboard's
  ``pii_scan`` policy modal now renders the catalog as a
  category-grouped checkbox grid (Identity / Contact / Financial /
  Medical / Credentials / Network / Vehicle), so an operator can
  never silently configure a policy against a type the engine
  doesn't know how to detect — that's the bug that produced zero
  detections in production when "passport" was typed into the old
  free-form ``kinds`` field.

- **Custom recognizers ported from the legacy engine.** API-key
  Shannon-entropy detection, reserved-domain email allowlist
  (``example.com`` and the RFC 6761 test domains), date-of-birth
  filtering, and word-form digit detection (``"one two three…"`` →
  SSN / credit card) all run inside Presidio's pipeline; nothing
  the old engine caught regresses.

### Changed

- **Renamed ``kind`` → ``type`` across the PII surface** so the
  SDK, the backend, the dashboard, and the persisted JSONB all
  speak one vocabulary. The legacy attribute / parameter / config
  keys are kept as aliases for one release (~3 months):
  - ``Sanitization.kind`` is now a ``@property`` on top of
    ``Sanitization.type``.
  - ``pii.sanitize(text, types=[...])`` is the canonical
    signature; ``kinds=[...]`` is still accepted.
  - ``pii_scan`` policy config keys on ``config.types``;
    ``config.kinds`` is still accepted on the wire.
  - ``MatchedPolicyRecord.sanitize_types`` is the canonical
    field; ``sanitize_kinds`` is a property alias.
- Audit-event payloads ship ``sanitizations[*].type`` exclusively
  on this release going forward.

### Deprecated

- The ``kind`` / ``kinds`` / ``sanitize_kinds`` field names will
  be removed in 0.16.x. New code should use ``type`` / ``types`` /
  ``sanitize_types``.

---

## [0.14.0] — 2026-05-07

### Added

- **Cloud-provider auto-detection in the runtime fingerprint.**
  ``init()`` now probes a small set of platform-set env vars
  (``AWS_LAMBDA_FUNCTION_NAME``, ``K_SERVICE``,
  ``WEBSITE_SITE_NAME``, ``VERCEL``, ``FLY_APP_NAME``, ``DYNO``,
  …) and emits a stable ``cloud`` token (``aws`` / ``gcp`` /
  ``azure`` / ``vercel`` / ``netlify`` / ``fly`` / ``railway`` /
  ``render`` / ``heroku`` / ``digitalocean``) in the runtime
  blob shipped to the backend on every handshake / agent
  registration. The backend uses this token to populate the
  agent's ``first_seen_asn`` field (the ASN chip in the Identity
  modal header and the Provenance card row), which previously
  rendered ``—`` on every customer because no path computed it.

  Detection is purely env-var-based — no network calls, no IMDS
  probes, no DNS — so the SDK design philosophy's "no network
  calls in init()" rule is preserved. Customers running on
  unrecognised platforms (bare metal, on-prem, an exotic cloud)
  see ``cloud: null`` and the field stays empty rather than
  showing a fabricated value.

  The new ``cloud`` key is purely additive to the runtime blob;
  no public API change, no Behavior change for existing code.

---

## [0.13.4] — 2026-05-07

### Added

- **Runtime governance expansion** — argument-aware enforcement for
  the post-model side. Five policy types now reason about *what*
  the model is asking the agent to do, not just *which* tool it's
  invoking:

  - **`deny_tool_call` (extended)** — three independent matching
    axes per rule. ``patterns`` continues to match tool names;
    ``argument_patterns`` is a new regex list run against each
    live tool call's serialized arguments (catches dangerous use
    of an otherwise-legitimate tool, e.g. ``http_get`` pointed at
    an internal IP); ``argument_max_chars`` caps the size of any
    single tool call's argument blob.
  - **`deny_bash_command` (extended)** — set
    ``block_dangerous_defaults: true`` to union in a curated
    preset of high-confidence patterns (recursive ``rm -rf``,
    fork bombs, ``curl … | sh``, ``sudo``, ``chmod +s``, ``dd
    if=``, …) without re-discovering them from first principles.
    Operator patterns still take precedence in evaluation order.
  - **`deny_mcp_call` (extended)** — adds a deny-by-default
    ``allowed_servers`` allowlist (substring match) plus a
    separate ``denied_resources`` regex axis scoped to MCP
    resource paths. The original ``patterns`` denylist still
    works; the three axes can be combined freely.
  - **`deny_db_query` (new)** — content-based detection of
    SQL-shaped tool calls. Works regardless of which tool wraps
    the query (``run_sql``, ``execute_query``, ``db_run``…).
    Three axes: ``query_patterns`` (operator regex against
    arguments), ``denied_tables`` (word-boundary table-name
    matching that tolerates backticks / quoted / bracketed /
    backslash-escaped identifiers), and ``dangerous_operations``
    (default-on list of DROP / TRUNCATE / DELETE / ALTER /
    GRANT / REVOKE / CREATE USER / DROP USER, with multi-word
    op support). Tool-name scoping via ``tool_patterns`` is
    optional.
  - **`deny_financial_action` (new)** — block tool calls that
    look like money movement above operator-defined risk
    appetite. Four axes: ``action_patterns`` (regex against
    tool name; defaults to a curated set of payment verbs that
    handles snake_case + camelCase via letter-boundary regex),
    ``amount_threshold`` + ``amount_field`` (recursive walk of
    parsed JSON arguments to find amount-shaped values, even
    when nested), ``denied_destinations`` (regex against
    serialized arguments), and ``allowed_currencies`` (case-
    insensitive currency allowlist applied to ``currency`` keys
    anywhere in the arguments tree).

  All five policy kinds remain in **Phase 1** of the two-phase
  policy contract — pure-Python regex + JSON walking, no network,
  no LLM judge. They short-circuit cleanly so a Phase 1 block
  never lets a request reach Phase 2 (`semantic_guard`),
  preserving the security-and-compliance.mdc §2 contract.

### Backwards compatibility

- Pure additions. Existing rules with no new keys behave
  identically. ``deny_bash_command`` defaults remain
  operator-supplied unless ``block_dangerous_defaults`` is set;
  ``deny_db_query`` and ``deny_financial_action`` default to
  their curated lists when the corresponding config is omitted
  (operators turn them off explicitly with empty lists).

---

## [0.13.3] — 2026-05-07

### Fixed

- **Runtime fingerprint cache is now thread-safe.** Two threads
  racing to populate the cache on the first auto-registered
  sub-agent could each walk `importlib.metadata` and write the
  cache concurrently. The cache is now guarded by a lock with a
  double-checked-read on the hot path, so steady-state lookups stay
  lock-free while the first miss is serialized. No behavioural
  change for single-threaded apps.

---

## [0.13.2] — 2026-05-07

### Fixed

- **Auto-detected agents now show full Provenance.** Sub-agents
  registered via system-prompt fingerprinting (the most common path
  in any multi-agent app) were calling `/v1/sdk/agents/ensure`
  *without* the runtime blob, so their Provenance card on the
  dashboard stayed blank — no Python version, no OS, no framework
  versions, no host-class badge. Fixed: every agent registration
  path (`set_context`, system-prompt auto-detect, handshake) now
  ships the same fingerprint.
- **Runtime fingerprint cached for the SDK process.** Walking
  `importlib.metadata` for four framework names + reading
  `/proc/1/cgroup` is no longer repeated on every per-prompt
  registration; the values can't change inside a process so we
  collect once, return defensive copies thereafter.

### Added

- **Debug breadcrumb on agent registration.** `egisai.backend`
  logger now emits a one-line DEBUG record per
  `/v1/sdk/agents/ensure` call listing the runtime keys shipped.
  Off by default; turn on with `logging.getLogger("egisai.backend")
  .setLevel("DEBUG")` to verify from the SDK side that the
  fingerprint left the building.

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
  applies" Behavior. Older platform responses that don't carry the
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
  parser's default Behavior, and the dual-decision audit shape on
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
