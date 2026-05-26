# Security Policy

## Reporting a Vulnerability

EgisAI takes the security of `egisai` and the platform behind it
seriously. We're a runtime governance layer for production AI
systems, so the integrity of this SDK is our customers' integrity.

**Please report security issues privately, not in public GitHub
issues.**

- Email: **adriano@egisai.co** (monitored; replies within 48h).
- Encrypted alternatives: please mention the preferred channel in
  your initial unencrypted message and we will move to it (PGP,
  age, Signal, etc.). We don't currently publish a PGP key — if
  you require encryption end-to-end please send your public key
  in your first message and we'll send ours back.

We aim to acknowledge every report **within 48 hours** and provide
a triage update within **7 days**. We follow a **90-day responsible
disclosure window** by default, with extensions on request when a
fix is technically complex.

When you report, please include:

1. The affected version of `egisai` (`pip show egisai`).
2. A minimal reproduction (Python script or `curl` request).
3. The impact you've observed or suspect.
4. Whether the vulnerability has been disclosed to anyone else.

## Scope

In scope:

- The `egisai` Python package (the `src/egisai/` tree in this
  repository).
- The platform endpoints under `https://app.egisai.co/v1/sdk/*`
  that this SDK communicates with.
- The PyPI release artefacts (sdist + wheel) and their
  signatures.

Out of scope:

- Vulnerabilities in our customers' policy configurations
  (regex patterns, intent strings, etc.) — those are the
  customer's authoring responsibility, though we'd love a heads
  up so we can refine our policy authoring guidance.
- Issues that require a malicious operator already inside the
  customer's organisation (the threat model is operator-trusted,
  attacker-untrusted).
- Denial of service via volumetric flooding of `/v1/sdk/*`
  endpoints — these are protected by platform-side rate
  limiting; report directly to adriano@egisai.co if you find a
  bypass.

## Supply-chain integrity

Official PyPI releases of `egisai` are intended to be:

1. **Built from version-tagged sources** in this repository.
2. **Signed with sigstore** when published through OIDC trusted
   publishing (certificate and attestation metadata ship with the
   release artefacts).
3. **Published to PyPI without long-lived API tokens** where
   [trusted publishing](https://docs.pypi.org/trusted-publishers/) is
   configured.

CycloneDX SBOM files (`egisai-<version>.cdx.json`) may be attached to
GitHub releases when the release is cut from maintained automation.

To verify a wheel (identity must match the PyPI project's configured
GitHub repository for OIDC; adjust the regexp if your publisher
differs):

```bash
pip download egisai==<version> --no-deps
python -m sigstore verify identity \
  --cert-identity-regexp "https://github.com/EgisLabs/egisai-sdk/.+" \
  --cert-oidc-issuer "https://token.actions.githubusercontent.com" \
  egisai-<version>-py3-none-any.whl
```

## Security model in one paragraph

The SDK runs in your process, with your customer's API key,
calling your model providers directly. Phase 1 of the policy
engine (PII / regex / size / model allowlist) runs entirely
local — raw secrets never leave the customer's environment.
Phase 2 (LLM-judge `semantic_guard`) calls the EgisAI platform
with the **already-redacted** prompt (PII has been replaced with
typed labels by Phase 1). The platform's judge is the only
network egress on the governance critical path, and it sees
data-clean text only.

If you find a way to exfiltrate raw PII, OR to bypass a policy
that should have blocked or sanitised, that's a Critical-severity
issue and we want to know within hours.

## Tool / MCP enforcement guarantees

`egisai` distinguishes two states on every audit row:

- **`enforcement_status="enforced"`** — A policy verdict on a
  tool call PHYSICALLY PREVENTED the tool from running, OR a
  policy verdict on a tool *result* prevented the model from
  ever seeing the unredacted bytes. The action did not happen /
  the leak did not reach the model.
- **`enforcement_status="advisory"`** — A policy decided block,
  but the underlying framework's architecture meant the SDK could
  only observe after the fact. The audit row is honest about the
  gap so SOC 2 / GDPR auditors can find these via
  `WHERE verdict='block' AND enforcement_status='advisory'`.

There are THREE enforcement surfaces the SDK documents for auditors:

1. **Tool / MCP dispatch** — `deny_tool_call`, `deny_mcp_call`,
   `semantic_guard` on the call itself. Blocks dangerous
   actions (drop tables, send funds, exec arbitrary shell)
   before they run.
2. **Tool result content** — `pii_scan`, `deny_output_regex`,
   `semantic_guard` on the data the tool returned. Blocks
   leaks of PII / secrets / proprietary identifiers that
   would otherwise enter the model's context.
3. **Aggregated assistant OUTPUT (`claude_agent_sdk` only)** —
   A second evaluator runs on the concatenated assistant stream at
   ``ResultMessage``. When that evaluation replays structured
   ``tool_calls`` emitted by the CLI subprocess, a ``verdict=block``
   stamps ``enforcement_status="advisory"`` on the enclosing
   ``model_call`` row — MCP/tool bytes were already replayed before
   Python aggregated them. **Pure text-only** violations still stamp
   ``enforced`` when hooks are wired. Applications that use
   ``on_block="raise"`` continue to see ``PermissionError``; the audit
   flag distinguishes *subprocess timing truth* from *caller withhold*.

**Every framework `egisai` patches enforces surfaces **(1)** and **(2)**
EXCEPT one**: `bedrock_agent` (AWS Bedrock Agents). AWS Bedrock Agents
execute Action Groups on AWS-managed infrastructure outside the
SDK process, so we cannot intercept tool dispatch before AWS
runs it AND we cannot substitute the result before AWS feeds it
back to the model. The patch records what happened via the
trace events in the response stream and stamps audit rows as
**advisory** to honestly reflect the limit. If your application
requires hard pre-execution gating OR tool-result PII masking
for Bedrock workloads, use one of:

- The standalone `bedrock-runtime` Converse API (drive the
  agentic loop yourself in Python — tool results round-trip
  Python and the next call's input phase scans them).
- The `claude_agent_sdk` (the only subprocess-loop framework
  we patch where the SDK exposes both PreToolUse AND
  PostToolUse hooks for true enforcement).

For the full per-framework enforcement matrix, see the
"Enforcement matrix" section of [README.md](README.md).

**Claude Agent SDK:** `PreToolUse` / `PostToolUse` still satisfy rows
(1) and (2) above; item (3) is the separate aggregated OUTPUT pass
documented in the previous list.

If you find a tool call that ran despite a matching
`deny_tool_call` policy on any framework EXCEPT `bedrock_agent`,
OR a tool result that reached the model despite a matching
`pii_scan` / `deny_output_regex` policy on any framework
EXCEPT `bedrock_agent`, that's a Critical-severity bypass and
we want to know within hours.
