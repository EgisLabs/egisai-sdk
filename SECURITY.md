# Security Policy

## Reporting a Vulnerability

EgisAI takes the security of `egisai` and the platform behind it
seriously. We're a runtime governance layer for production AI
systems, so the integrity of this SDK is our customers' integrity.

**Please report security issues privately, not in public GitHub
issues.**

- Email: **security@egisai.co** (monitored; replies within 48h).
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
  limiting; report directly to security@egisai.co if you find a
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
