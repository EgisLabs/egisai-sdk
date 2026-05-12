# Changelog

All notable changes to `egisai` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Overview → Recent runs** widget mirrors the Requests page.
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
  intervention. Existing `set_context` callers see no behaviour
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
  modal) keeps its current behaviour. Only policies that OMITTED
  ``action`` — typically API-created rules — see the new default,
  and that change is desirable: the previous behaviour would refuse
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
  no public API change, no behaviour change for existing code.

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
