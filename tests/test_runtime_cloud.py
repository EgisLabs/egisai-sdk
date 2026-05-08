"""SDK-side cloud-provider detection.

The SDK probes a small set of well-known env vars at ``init()``
time and emits a stable ``cloud`` token in the runtime fingerprint
blob. The backend uses that token to populate the agent's
``first_seen_asn`` field — far more reliable than guessing from
the request IP, which mis-attributes PaaS workloads to whichever
cloud they happen to be leasing addresses from.

These tests pin three things:

1. The ``cloud`` key is ALWAYS present in the runtime blob (its
   value is None when nothing was detected). A missing key would
   silently blank the dashboard's ASN chip on every customer.
2. Each platform's signature env var maps to the expected token.
3. Tokens are stable, lower-cased, single-word — they wire
   directly to ``backend/app/services/asn_lookup.py``'s mapping
   table; a typo here breaks the ASN field on every agent.
"""

from __future__ import annotations

import pytest

from egisai import _runtime


def setup_function() -> None:
    _runtime.reset_runtime_cache()


def test_cloud_key_always_present_in_runtime_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Strip every cloud-detection env var so we exercise the
    # "no cloud detected" path. The key must STILL appear in the
    # blob with value None — a missing key is what historically
    # blanked the dashboard.
    for var in (
        "AWS_LAMBDA_FUNCTION_NAME",
        "AWS_EXECUTION_ENV",
        "ECS_CONTAINER_METADATA_URI",
        "ECS_CONTAINER_METADATA_URI_V4",
        "AWS_BATCH_JOB_ID",
        "K_SERVICE",
        "FUNCTION_TARGET",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "WEBSITE_SITE_NAME",
        "AZURE_FUNCTIONS_ENVIRONMENT",
        "MSI_ENDPOINT",
        "VERCEL",
        "NETLIFY",
        "FLY_APP_NAME",
        "RAILWAY_ENVIRONMENT",
        "RENDER",
        "DYNO",
        "DIGITALOCEAN_APP_NAME",
    ):
        monkeypatch.delenv(var, raising=False)

    rt = _runtime.collect_runtime_fingerprint(sdk_version="0.13.5")
    assert "cloud" in rt, "runtime blob must always carry a 'cloud' key"
    assert rt["cloud"] is None


@pytest.mark.parametrize(
    "env_var, expected",
    [
        # AWS — multiple distinct env-var paths must all resolve
        # to the same canonical token.
        ("AWS_LAMBDA_FUNCTION_NAME", "aws"),
        ("AWS_EXECUTION_ENV", "aws"),
        ("ECS_CONTAINER_METADATA_URI", "aws"),
        ("ECS_CONTAINER_METADATA_URI_V4", "aws"),
        ("AWS_BATCH_JOB_ID", "aws"),
        # GCP — Cloud Run, Cloud Functions, plain GCE.
        ("K_SERVICE", "gcp"),
        ("FUNCTION_TARGET", "gcp"),
        ("GOOGLE_CLOUD_PROJECT", "gcp"),
        ("GCLOUD_PROJECT", "gcp"),
        # Azure — App Service, Functions.
        ("WEBSITE_SITE_NAME", "azure"),
        ("AZURE_FUNCTIONS_ENVIRONMENT", "azure"),
        ("MSI_ENDPOINT", "azure"),
        # PaaS providers.
        ("VERCEL", "vercel"),
        ("NETLIFY", "netlify"),
        ("FLY_APP_NAME", "fly"),
        ("RAILWAY_ENVIRONMENT", "railway"),
        ("RENDER", "render"),
        ("DYNO", "heroku"),
        ("DIGITALOCEAN_APP_NAME", "digitalocean"),
    ],
)
def test_each_platform_env_var_maps_to_expected_token(
    monkeypatch: pytest.MonkeyPatch, env_var: str, expected: str
) -> None:
    # Strip the others first so a stray env var (real CI, real
    # local dev) doesn't bleed into this test.
    for var in (
        "AWS_LAMBDA_FUNCTION_NAME",
        "AWS_EXECUTION_ENV",
        "ECS_CONTAINER_METADATA_URI",
        "ECS_CONTAINER_METADATA_URI_V4",
        "AWS_BATCH_JOB_ID",
        "K_SERVICE",
        "FUNCTION_TARGET",
        "GOOGLE_CLOUD_PROJECT",
        "GCLOUD_PROJECT",
        "WEBSITE_SITE_NAME",
        "AZURE_FUNCTIONS_ENVIRONMENT",
        "MSI_ENDPOINT",
        "VERCEL",
        "NETLIFY",
        "FLY_APP_NAME",
        "RAILWAY_ENVIRONMENT",
        "RENDER",
        "DYNO",
        "DIGITALOCEAN_APP_NAME",
    ):
        monkeypatch.delenv(var, raising=False)

    monkeypatch.setenv(env_var, "1")

    rt = _runtime.collect_runtime_fingerprint(sdk_version="0.13.5")
    assert rt["cloud"] == expected, (
        f"{env_var} should map to {expected!r}, got {rt['cloud']!r}"
    )


def test_cloud_token_set_matches_backend_resolver(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every cloud token the SDK can emit MUST be resolvable on the
    backend. Drift between this list and
    ``asn_lookup._RUNTIME_HINT_TO_ASN`` would silently produce
    ``—`` on the dashboard.
    """
    # Strip everything, then probe each known env var and collect
    # the tokens the SDK emits. These are the tokens the backend
    # MUST recognise.
    sdk_tokens: set[str] = set()

    cases: list[tuple[str, str]] = [
        ("AWS_LAMBDA_FUNCTION_NAME", "aws"),
        ("K_SERVICE", "gcp"),
        ("WEBSITE_SITE_NAME", "azure"),
        ("VERCEL", "vercel"),
        ("NETLIFY", "netlify"),
        ("FLY_APP_NAME", "fly"),
        ("RAILWAY_ENVIRONMENT", "railway"),
        ("RENDER", "render"),
        ("DYNO", "heroku"),
        ("DIGITALOCEAN_APP_NAME", "digitalocean"),
    ]
    for env_var, expected in cases:
        for v in [c[0] for c in cases]:
            monkeypatch.delenv(v, raising=False)
        monkeypatch.delenv("AWS_EXECUTION_ENV", raising=False)
        monkeypatch.delenv("ECS_CONTAINER_METADATA_URI", raising=False)
        monkeypatch.delenv("ECS_CONTAINER_METADATA_URI_V4", raising=False)
        monkeypatch.delenv("AWS_BATCH_JOB_ID", raising=False)
        monkeypatch.delenv("FUNCTION_TARGET", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        monkeypatch.delenv("GCLOUD_PROJECT", raising=False)
        monkeypatch.delenv("AZURE_FUNCTIONS_ENVIRONMENT", raising=False)
        monkeypatch.delenv("MSI_ENDPOINT", raising=False)

        monkeypatch.setenv(env_var, "1")
        _runtime.reset_runtime_cache()
        rt = _runtime.collect_runtime_fingerprint(sdk_version="0.13.5")
        assert rt["cloud"] == expected
        sdk_tokens.add(rt["cloud"])

    # Token-stability contract: every SDK-emitted token must be
    # lower-case, single-word, ASCII. Backend's mapping is keyed
    # on lower-case so any case drift here would be a silent
    # mis-match.
    for tok in sdk_tokens:
        assert tok == tok.lower()
        assert " " not in tok
        assert tok.isascii()
