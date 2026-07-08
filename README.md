# egisai — Runtime governance for AI agents

[![PyPI version](https://img.shields.io/pypi/v/egisai.svg)](https://pypi.org/project/egisai/)
[![Python versions](https://img.shields.io/pypi/pyversions/egisai.svg)](https://pypi.org/project/egisai/)
[![License](https://img.shields.io/pypi/l/egisai.svg)](https://github.com/EgisLabs/egisai-sdk/blob/main/LICENSE)

**Production guardrails for Python AI applications.** Install the SDK, call `egisai.init()`, and continue using OpenAI, Anthropic, Google Gemini, or plain HTTP clients as you do today—policy evaluation and audit logging wrap those calls automatically.

This document is the canonical SDK guide for **[PyPI](https://pypi.org/project/egisai/)** and mirrors what we publish at **[docs.egisai.co](https://docs.egisai.co)**.

---

## Overview

| Capability | What it means for you |
|------------|------------------------|
| **Central policies** | Operators configure rules in the [EgisAI dashboard](https://app.egisai.co). The SDK loads them at runtime and refreshes them continuously—no redeploy to tighten controls. |
| **Transparent integration** | No proxy layer and no wrapper objects you must remember to use. `egisai.init()` patches every supported library that is importable in the current environment; calls through their normal API go through governance without further wiring. |
| **Broad framework coverage** | Direct provider SDKs (OpenAI, Anthropic, Google Gemini, AWS Bedrock Converse) plus the major agent frameworks (LangChain, LangGraph, OpenAI Agents, CrewAI, AutoGen, Agno, Strands, smolagents, LlamaIndex, Pydantic AI, Google ADK, Claude Agent SDK) are governed transparently. |
| **Audit trail** | Governed calls emit structured events to your org so teams can review verdicts, latency, tool calls, and usage in one place. |
| **Local-first sensitive checks** | Pattern-based PII handling and other deterministic rules run entirely inside your process before traffic leaves your environment. |
| **Automatic agent identity** | Each logical agent in your process is auto-detected and registered on the dashboard. Sub-agents fingerprinted by system prompt, framework hook, or explicit name show up as distinct rows for attribution. |

---

## What you need

1. **Python 3.11+**
2. An **[EgisAI](https://app.egisai.co)** account and an **SDK API key** (dashboard → **API Keys** → create). Keys look like `egis_live_…`.
3. The AI SDK(s) you already use (`openai`, `anthropic`, `google-genai`, …).

---

## Installation

```bash
pip install "egisai[all]"
```

Optional extras (smaller installs):

```bash
pip install "egisai[openai]"
pip install "egisai[anthropic]"
pip install "egisai[google]"          # google-genai
pip install "egisai[google-legacy]"   # google-generativeai
```

Only frameworks present in your environment are activated at runtime. The
`google` and `google-legacy` extras are independent and can both be installed
when an application uses both SDKs.

---

## Getting started

### 1. Initialize once per process

Call `egisai.init()` as early as possible in your application lifecycle (for example right after loading configuration). Use your SDK API key from the dashboard.

```python
import egisai

egisai.init(
    api_key="egis_live_…",   # or set EGISAI_API_KEY in the environment
    app="customer-support-bot",
    env="production",
)
```

### 2. Use your LLM client normally

No changes to your calling convention—the SDK intercepts supported APIs after initialization.

**OpenAI**

```python
from openai import OpenAI

client = OpenAI()
response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Hello"}],
)
```

**Anthropic**

```python
import anthropic

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-opus-4-8",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hello"}],
)
```

**Google Gemini**

```python
from google import genai

client = genai.Client()  # picks up GEMINI_API_KEY from the environment
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Hello",
)
```

### 3. Review activity

Open **[Dashboard → Requests](https://app.egisai.co/dashboard)** to see governed calls, verdicts, and supporting metadata for your organization.

---

## How governance fits your call path

Each governed call is evaluated in two phases—once before the model runs and once after it returns—so policies can intervene on either side independently.

1. **Request evaluation** — Before the upstream model runs, the SDK applies your organization’s active policies (cached locally) to the prompt. Local deterministic rules — PII detection, regex denylists, model allowlists, prompt-size caps — run first; intent-oriented `semantic_guard` is consulted only if every local rule allowed the prompt.
2. **Outcomes (request)** — A call may be **allowed**, **sanitized** (payload adjusted per policy, then forwarded), or **blocked**. Blocked calls never reach the provider when enforcement raises or returns a stub, depending on configuration (see below).
3. **Response evaluation** — When the model responds, output-side policies run against the assistant’s text, tool invocations, and connector targets, using **the same two-phase split**. Local rules run first; `semantic_guard` runs only if no local rule already blocked. A blocked response is suppressed before it reaches your code.
4. **Telemetry** — The audit event records each phase’s decision independently (`prompt_decision`, `response_decision`) so your dashboard can show exactly which side fired and which rules matched. Delivery is non-blocking.

Each policy carries a **phase** that selects which side it runs on:

| Phase | When the rule fires |
|-------|--------------------|
| `request`  | Only on the inbound payload, before the call is made. |
| `response` | Only on the outbound payload, after the call returns. |
| `both`     | On both sides where the rule type has signals to evaluate. |

The phase names are call-relative on purpose — they read correctly for every governed surface: model calls, tool calls, MCP calls. The legacy wire spellings `pre_model` / `post_model` are still accepted and normalize to `request` / `response`.

Operators choose the phase in the dashboard when they create or edit a rule, and every rule type accepts every phase. The engine evaluates each rule on whichever side it has meaningful signals for: text-content rules (PII detection, regex deny-lists, prompt-size caps, semantic guard) fire symmetrically on prompt and response; tool / shell / connector rules need response-side signals and silently no-op when an operator targets them on `request` only. Older platform deployments that haven’t shipped the field yet behave as if every rule were `both`, preserving previous semantics.

A policy can also carry an **`applies_to`** list that scopes it to specific call surfaces (`model`, `tool`, `mcp`); empty means every surface, which is the behavior all rules had before surface scoping existed.

Sensitive pattern detection intended to catch regulated data is performed locally so raw values are not sent to third-party models as part of governance — on **both** the prompt and the response. Intent-oriented policies (`semantic_guard`) are consulted only after the applicable local checks have run on the text that will be judged, and never run at all when a local rule has already refused the call. The same ordering applies to whichever side the operator scoped the rule to.

---

## When a call is blocked

| `on_block` | Behavior |
|------------|----------|
| `"raise"` *(default)* | Raises `PermissionError` if a policy blocks the call. |
| `"stub"` | Returns a framework-shaped refusal object so applications that cannot tolerate exceptions keep running; the refusal is clearly identifiable in your logs and on the dashboard. |

Configure at init:

```python
egisai.init(..., on_block="stub")
```

---

## Configuration reference

### Initialization parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | — | **Required** unless `EGISAI_API_KEY` is set. Your EgisAI SDK key (`egis_live_…`). |
| `app` | `"default"` | Logical application name; appears as an **Agent** in the dashboard for attribution. |
| `env` | `"production"` | Environment label (for example `staging`, `prod`). Free-form string for your own segmentation. |
| `base_url` | Hosted control plane | Override only when directed by EgisAI (for example dedicated regions or enterprise deployments). |
| `on_block` | `"raise"` | `"raise"` or `"stub"` — see above. |
| `semantic_on_outage` | `"allow"` | What `semantic_guard` rules do if the intent judge can't be reached. `"allow"` fails open (preserves availability); `"block"` fails closed (refuses the call when the judge is unreachable). |
| `refresh_interval_seconds` | `10` | How often to poll for policy updates if live streaming is unavailable. |
| `enable_sse` | `True` | Subscribe to live policy and configuration updates when supported. |
| `enable_http_fallback` | `True` | Optional patching of `httpx` / `requests` for broader HTTP visibility where enabled. |
| `auto_stack_hints` | `"loose"` | Controls the stack-frame inspector used by the agent identity resolver. `"loose"` (default) honors common conventions; `"strict"` requires an explicit `__egisai_agent__` marker; `"off"` disables stack inspection. |
| `gateway` | `False` | Route OpenAI chat-completions calls through the platform's inline Gateway instead of evaluating policies in-process — see "Gateway mode" below. |
| `quiet` | `False` | Set `True` to suppress the one-line startup banner on stderr. |

### Environment variables

| Variable | Purpose |
|----------|---------|
| `EGISAI_API_KEY` | SDK API key if not passed as `api_key=`. |
| `EGISAI_BASE_URL` | Control plane base URL override when supplied by EgisAI. |
| `EGISAI_GATEWAY` | Set `1` to enable Gateway mode without a code change. |

Treat API keys as secrets—use environment variables or a secrets manager, never commit them to source control.

---

## Gateway mode (optional)

If your organization uses the platform's inline Gateway, the SDK offers two ways to use it. The simplest is `egisai.Client` — one import, no provider SDK in your code, no URL or header wiring:

```python
import egisai

client = egisai.Client(
    api_key="egis_live_…",       # your Egis key
    provider_key="sk-ant-…",     # forwarded to the provider untouched, never stored
)

response = client.chat.completions.create(
    model="claude-sonnet-4-5",   # the model name picks the provider
    messages=[{"role": "user", "content": "Hello"}],
)
```

The client speaks the familiar chat-completions surface (streaming included) and always sends to the Gateway, which evaluates policies, sanitizes/blocks inline, routes to OpenAI / Anthropic / Google / Mistral / xAI / DeepSeek from the model name, and writes the audit row server-side. `egisai.AsyncClient` is the async sibling. `init()` is optional; when it's active, `egisai.set_context(agent=…)` rides along as `X-Egis-Agent` per call. Requires `pip install "egisai[openai]"` (the Gateway's wire format).

For an existing codebase that already uses the OpenAI client everywhere, one flag reroutes it without touching call sites:

```python
import egisai
egisai.init(api_key="egis_live_…", gateway=True)

from openai import OpenAI
client = OpenAI()  # your provider key, forwarded untouched
egisai.set_context(agent="Support Copilot")  # still works — becomes X-Egis-Agent

response = client.chat.completions.create(
    model="gpt-4.1",
    messages=[{"role": "user", "content": "Hello"}],
)
```

What changes and what doesn't:

- **Same calling convention.** You keep your own OpenAI client and provider key. The SDK reroutes `chat.completions.create` to the Gateway and injects `X-Egis-Api-Key` (and `X-Egis-Agent` when you set an explicit identity) automatically.
- **Same policies, evaluated server-side.** The Gateway runs the identical engine, sanitizes/blocks inline, and writes the audit row itself; the local gate is skipped for rerouted calls so nothing is governed twice.
- **Everything else stays local.** The Responses API, Anthropic / Google / Bedrock SDKs, agent frameworks, and MCP keep the normal in-process governance path. Azure OpenAI clients are never rerouted.
- **Fail open.** If the reroute can't be constructed, the call falls back to in-process governance from the locally cached policies — your call path never breaks because of the mode switch.

---

## Policies (operator concepts)

Organizations configure policies in the dashboard. Typical categories include:

| Category | Purpose (high level) |
|----------|----------------------|
| **PII & secrets** | Detect and block or mask categories such as government identifiers, payment data, and credential-shaped strings before model calls. |
| **Content patterns** | Allow or deny prompts or outputs matching operator-defined patterns. |
| **Models & size** | Restrict which model names may be called or cap prompt size. |
| **Intent** | Block prompts, responses, *or* tool calls that match dangerous or out-of-scope *intent* even when phrased obliquely or in another language. |
| **Tools, shell, MCP & connectors** | Restrict tool, shell, MCP, database, or financial actions when the model returns structured tool or command requests. |

Exact rule JSON and ordering are managed in the product; the SDK consumes the published configuration and does not require you to embed policy documents in your repository.

---

## Advanced: explicit context (optional)

The SDK auto-detects an agent identity for each call (system-prompt fingerprint, framework signal, or explicit name) so you usually don't need to wire anything up. When you do want to override it explicitly, three primitives are public:

```python
import egisai

# 1) Per-task metadata. Writes to a ContextVar — async/threads inherit.
egisai.set_context(
    agent="billing-agent",       # logical sub-agent name
    user_id="u_123",
    user_role="customer",
    session_id="s_abc",
    workflow_id="wf_42",
    end_user_id="hash_of_customer_id",
)

# 2) Block-scoped agent identity. Wins outright over auto-detection
# inside the block; the previous identity is restored on exit.
with egisai.agent("Triage"):
    client.chat.completions.create(...)

# 3) Eager registration so the Agent row exists on the dashboard
# before any traffic flows. Useful at process startup.
egisai.register_agent("billing-agent")
```

The `set_context` and `agent()` paths use `ContextVar`, so asyncio tasks and child threads inherit the values cleanly without leaking across requests. Explicit overrides always win over auto-detection.

### Health snapshot for `/healthz` endpoints

```python
import egisai

snapshot = egisai.diagnostics()
# {"initialized": True, "sdk_version": "0.28.0", "app": "...",
#  "env": "...", "policy_etag": "...", "policy_rule_count": 12,
#  "audit_queue_size": 0, "audit_dropped_total": 0, ...}
```

`diagnostics()` is a JSON-serializable dict suitable for exposing on your own health endpoint or dashboard so you can confirm the SDK is initialized, policies are loaded, and audit delivery is keeping up.

---

## Performance and availability

- **Steady-state overhead** is designed to stay on the order of a fraction of a millisecond for policy lookup per call after initialization and cache warm-up.
- **Control plane connectivity** — If the SDK cannot reach EgisAI at startup, your process can still run; policy enforcement may be limited until a successful connection and policy fetch. PII and other local checks remain in force where the engine can evaluate them. For your specific deployment’s behavior, refer to your contract and [SECURITY.md](https://github.com/EgisLabs/egisai-sdk/blob/main/SECURITY.md).
- **Audit delivery** is asynchronous so network latency to EgisAI does not sit on the critical path of every model call.

---

## Privacy and security

- Do **not** embed secrets in repository copies of this README.
- For vulnerability reporting, see **[SECURITY.md](https://github.com/EgisLabs/egisai-sdk/blob/main/SECURITY.md)** — please use the disclosed channel rather than public issues for security-sensitive matters.
- The authoritative privacy and legal documents for the hosted platform are the **[Privacy Policy](https://egisai.co/privacy)** and **[Terms of Service](https://egisai.co/terms)** at `egisai.co`. This SDK ships under [Apache 2.0](https://github.com/EgisLabs/egisai-sdk/blob/main/LICENSE); use of the hosted control plane is governed by those documents.

A short summary suitable for architecture reviews:

- Governance evaluates prompts with respect to your organization’s policies before upstream invocation where applicable.
- Sensitive-content handling is architected so that raw regulated values are not sent to third-party LLMs as part of policy enforcement workflows described here.

<!-- LEGAL:certification-status:BEGIN -->
<p><strong>Certification status.</strong> The platform is being built with SOC&nbsp;2, ISO&nbsp;27001, GDPR, HIPAA, and enterprise security expectations in mind. EgisAI is <strong>not currently</strong> SOC&nbsp;2, ISO&nbsp;27001, HIPAA, FedRAMP, PCI&nbsp;DSS, or other formally certified or attested unless expressly stated on the <a class="inline" href="https://docs.egisai.co/security">Security page</a> with current supporting evidence. Customers must complete their own compliance assessment before relying on the Service for regulated workloads.</p>
<!-- LEGAL:certification-status:END -->

<!-- LEGAL:no-professional-advice:BEGIN -->
<p><strong>No professional advice.</strong> The Service provides technical controls, operational telemetry, and audit-supporting evidence. EgisAI does not provide legal, regulatory, compliance, security, audit, insurance, or risk-management advice. Customer remains responsible for determining whether its AI systems, workflows, notices, policies, controls, and records satisfy applicable laws, regulations, standards, and contractual obligations.</p>
<!-- LEGAL:no-professional-advice:END -->

---

## Troubleshooting

| Symptom | Things to check |
|---------|-----------------|
| `RuntimeError: egisai.init() requires api_key` | Set `api_key=` or `EGISAI_API_KEY`. |
| Policies never update | Network egress to your configured control plane; SSE disabled behind strict firewalls—polling still applies on an interval. |
| Calls succeed but dashboard stays empty | Confirm the SDK key matches the org you expect; verify process can reach the control plane for logging. |
| Blocked call raises unexpectedly | Review active policies in the dashboard; set `on_block="stub"` if you need non-throwing behavior. |

---

## Supported Python libraries

| Category | Library | Notes |
|----------|---------|-------|
| **Direct provider SDK** | `openai` ≥ 1.40 | Chat Completions, Responses API, sync + async, streaming, tool calls. |
| | `anthropic` ≥ 0.40 | Messages API, sync + async, streaming, tool use. |
| | `google-genai` ≥ 1.0 | `client.models.generate_content`, async, streaming, function calls. |
| | `google-generativeai` ≥ 0.8 | `GenerativeModel.generate_content`, streaming. Install via `egisai[google-legacy]`. |
| | `boto3` (AWS Bedrock) | `bedrock-runtime` Converse / ConverseStream and `bedrock-agent-runtime` `InvokeAgent`. |
| **Agent framework** | `openai-agents` | `Runner.run` — identity wrap; tool gating cascades to the OpenAI patch. |
| | `claude-agent-sdk` | `ClaudeSDKClient` / `query()` — `PreToolUse` + `PostToolUse` hooks gate tool dispatch and tool results in-process. |
| | `langchain` / `langchain-classic` | Classic `AgentExecutor.invoke` + modern `create_agent` (via LangGraph cascade). |
| | `langgraph` | `Pregel.invoke` / `.stream` and `CompiledStateGraph` — identity wrap; cascades to provider patch. |
| | `crewai` | `Agent.execute_task` — identity wrap. |
| | `autogen` | `AssistantAgent.run` — identity wrap. |
| | `agno` | `Agent.run` / `Agent.arun` — identity wrap. |
| | `strands-agents` | `Agent.__call__` — identity wrap. |
| | `smolagents` | Agent entry — identity wrap. |
| | `llama-index` | `FunctionAgent` / `ReActAgent` / `CodeActAgent` / `AgentWorkflow` — identity wrap with workflow-handler awareness. |
| | `pydantic-ai` | `Agent.run` — identity wrap. |
| | `google-adk` | ADK entry — identity wrap. |
| **HTTP fallback** | `httpx` / `requests` | Optional broad HTTP capture for libraries that bypass the official provider SDKs. Matches on known LLM provider hosts AND known model-call path tokens to avoid logging unrelated traffic. |

Minimum versions are guidance; pin in your own `requirements.txt` for reproducible builds. Only frameworks that are actually importable in your environment are activated at runtime — if you don't install a framework's package, the SDK silently skips its patch.

---

## Enforcement matrix

Different frameworks expose tool execution at different boundaries — some run the agentic loop in Python (where we can sit between the model and the tool), and a couple run it in a subprocess or on managed infrastructure. The matrix below is the honest, locked contract for what `egisai` can stop *before it happens* versus what it can only audit after the fact, per framework.

Two enforcement seams matter for SOC 2 / ISO 27001 / GDPR / HIPAA:

1. **Tool / MCP call enforcement** — block dangerous tool dispatches (`deny_tool_call`, `deny_mcp_call`, `semantic_guard`) **before** the tool runs.
2. **Tool *result* enforcement** — block / mask PII (`pii_scan`, `deny_output_regex`) in the tool's response **before** the model is shown it.

A row says **enforced** when the SDK can physically prevent the failure mode in question. It says **advisory** when the SDK observes after the fact and records the violation in the audit log but couldn't intervene.

| Framework | Tier | Tool / MCP block | Tool result PII block | How it works |
|-----------|------|------------------|----------------------|--------------|
| **OpenAI** | 1 | **enforced** | **enforced** (next call) | Output policy raises `PermissionError` before the response (with `tool_calls`) returns. Tool results round-trip Python; the next call's input phase scans them. |
| **Anthropic** | 1 | **enforced** | **enforced** (next call) | Output policy on `Messages.create` response with `tool_use` blocks. Tool result blocks in the next call's `messages` are scanned by the input phase. |
| **Google GenAI (Gemini)** | 1 | **enforced** | **enforced** (next call) | `generate_content` response with `function_call` parts is policy-gated before return; tool responses scanned on next call. |
| **Google (legacy)** | 1 | **enforced** | **enforced** (next call) | Same gate as Google GenAI; legacy `google-generativeai` package. |
| **AWS Bedrock Converse** | 1 | **enforced** | **enforced** (next call) | `Converse` / `ConverseStream` response gated; `toolUse` blocks blocked before caller dispatches; tool results scanned on next call. |
| **HTTP fallback (httpx/requests)** | 1 | **enforced** | **enforced** (next call) | Best-effort body parsing for unknown providers; the next request's payload text gets the same input-phase scan. |
| **LangChain** | 1 | **enforced** | **enforced** (next call) | Cascades to the underlying provider patch (OpenAI / Anthropic / Google). |
| **OpenAI Agents** | 2 | **enforced** | **enforced** (next call) | Identity-only wrap on `Runner.run`; cascades to inner OpenAI patch + input-phase scan on tool results. |
| **CrewAI** | 2 | **enforced** | **enforced** (next call) | Identity-only wrap on `Agent.execute_task`; cascades. |
| **AutoGen** | 2 | **enforced** | **enforced** (next call) | Identity-only wrap on `AssistantAgent.run`; cascades. |
| **LangGraph** | 2 | **enforced** | **enforced** (next call) | Identity-only wrap on `Pregel.invoke`; cascades. |
| **LlamaIndex** | 2 | **enforced** | **enforced** (next call) | Identity-only wrap on `FunctionAgent.run`; cascades. |
| **Agno** | 2 | **enforced** | **enforced** (next call) | Identity-only wrap on `Agent.run` / `Agent.arun`; cascades. |
| **smolagents** | 2 | **enforced** | **enforced** (next call) | Identity-only wrap on agent entry; cascades. |
| **Strands Agents** | 2 | **enforced** | **enforced** (next call) | Identity-only wrap on `Agent.__call__`; cascades. |
| **Pydantic AI** | 2 | **enforced** | **enforced** (next call) | Identity-only wrap on `Agent.run`; cascades. |
| **Google ADK** | 2 | **enforced** | **enforced** (next call) | Identity-only wrap on the ADK entry; cascades. |
| **Claude Agent SDK** | 3a | **enforced** (PreToolUse) | **enforced** (PostToolUse) | Subprocess agent loop. We inject `PreToolUse` AND `PostToolUse` hooks. PreToolUse gates the dispatch; PostToolUse evaluates the tool's response and substitutes via `updatedToolOutput` / `updatedMCPToolOutput` before Claude is shown the result. On older SDK versions without the hooks field, falls back to advisory mode and the audit row is honestly labelled. |
| **AWS Bedrock Agents** | 3b | **advisory** | **advisory** | Action Groups execute on AWS-managed infrastructure with no equivalent of `PostToolUse`. The patch records what happened but cannot prevent the tool dispatch OR substitute its result before the model sees it. Use the standalone `bedrock-runtime` Converse API or `claude_agent_sdk` for SOC 2 / GDPR-grade enforcement. |
| **MCP Server (inbound)** | in | **enforced** | **enforced** | Add-on (requires the `mcp_servers` entitlement). Governs inbound `tools/call` into a customer-hosted MCP server (`fastmcp` / official `mcp`). The patch evaluates input + output policies on the tool arguments and result and blocks (`block`) or masks (`sanitize`) **before** the tool handler runs and before the result returns to the caller. Dormant and zero-overhead for any org without the add-on. Unlike every row above — which govern the customer's *outbound* LLM/agent calls — this governs calls *into* the customer's server. |

What you can rely on for **every row above except Bedrock Agents**:

- A `deny_tool_call` / `deny_mcp_call` / `semantic_guard` verdict on a tool call physically stops the tool from running.
- A `pii_scan` / `deny_output_regex` / `semantic_guard` verdict on a tool result either masks the result in place (`action="sanitize"`) or refuses it (`action="block"`) before the model is shown it. For Tier 1 + Tier 2 frameworks this happens on the next round trip's input phase; for the Claude Agent SDK it happens at the `PostToolUse` hook so the model never even sees the raw bytes for one turn.
- Input-side policies (PII scan, deny_regex, deny_model, max_prompt_chars, semantic_guard on the prompt) always run before the model is called.
- Sanitization rewrites the prompt locally before it reaches the provider.
- **Aggregated OUTPUT replay** — For `claude_agent_sdk`, OUTPUT policies
  that evaluate after MCP/tool payloads have been replayed from the CLI
  stamp `verdict=block` as `enforcement_status="advisory"` on the audit
  row (text-only breaches with hooks wired remain `enforced`). The
  withhold at your code boundary via `on_block="raise"` is unchanged —
  the distinction is for SOC 2 evidence about subprocess timing.
- Audit rows distinguish `enforcement_status="enforced"` (we actually prevented the action) from `enforcement_status="advisory"` (we observed after the fact or replayed MCP payloads before the aggregated OUTPUT evaluator ran). SOC 2 / GDPR auditors can query both states.

For Bedrock Agents specifically, see [SECURITY.md](https://github.com/EgisLabs/egisai-sdk/blob/main/SECURITY.md) — the limitation is publicly documented so customers can risk-assess accordingly.

---

## Verifying PyPI artifacts (optional)

Releases are published to PyPI via automation. To verify a wheel cryptographically when verifying identity bindings published by the project:

```bash
pip download egisai==0.28.0 --no-deps
python -m sigstore verify identity \
  --cert-identity-regexp "https://github.com/EgisLabs/egisai-sdk/.+" \
  --cert-oidc-issuer "https://token.actions.githubusercontent.com" \
  egisai-0.28.0-py3-none-any.whl
```

Adjust the version to match the release you installed. A CycloneDX SBOM is attached to GitHub releases for supply-chain review.

---

## Resources

| Resource | URL |
|----------|-----|
| **Website** | [egisai.co](https://egisai.co) |
| **Documentation** | [docs.egisai.co](https://docs.egisai.co) |
| **Dashboard** | [app.egisai.co](https://app.egisai.co) |
| **PyPI** | [pypi.org/project/egisai](https://pypi.org/project/egisai) |
| **Repository & issues** | [github.com/EgisLabs/egisai-sdk](https://github.com/EgisLabs/egisai-sdk) |
| **Changelog** | [CHANGELOG.md on GitHub](https://github.com/EgisLabs/egisai-sdk/blob/main/CHANGELOG.md) |
| **Security** | [SECURITY.md on GitHub](https://github.com/EgisLabs/egisai-sdk/blob/main/SECURITY.md) |

---

## Licence

Apache License 2.0 — see the [LICENSE file in the source repository](https://github.com/EgisLabs/egisai-sdk/blob/main/LICENSE).

---

**EgisAI** — runtime governance for AI agents.
