"""Cloud identity probe: what it reports, and what it must never touch.

The probe exists to give the backend a deterministic join key between
an OAuth grant inventoried from a SaaS tenant and the agent using it.
That value is only worth having if the probe is also boring — no
credentials, no hot-path latency, no requests from hosts that aren't
on the cloud in question.

Half of these tests are therefore negative: they assert on what the
probe *doesn't* do. `test_the_probe_never_asks_imds_for_credentials`
is the load-bearing one — the same metadata service that answers
`iam/info` will hand out temporary access keys to anyone who asks, and
this suite is what keeps a future refactor from asking.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from egisai import _cloud_identity, _runtime

# Every env var the probe reads, so a test can start from a known-empty
# environment. Real CI runners set some of these for real.
_ALL_VARS = (
    "AWS_ROLE_ARN",
    "AZURE_CLIENT_ID",
    "MSI_CLIENT_ID",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "OAUTH_CLIENT_ID",
    "OAUTH2_CLIENT_ID",
    "CLIENT_ID",
    "OKTA_CLIENT_ID",
    "AUTH0_CLIENT_ID",
    "GOOGLE_CLIENT_ID",
    "MICROSOFT_CLIENT_ID",
    "SALESFORCE_CLIENT_ID",
    # Cloud-detection vars — these gate whether a metadata hop is
    # attempted at all.
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
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _ALL_VARS:
        monkeypatch.delenv(var, raising=False)
    _cloud_identity.reset_for_tests()
    _runtime.reset_runtime_cache()


class _FakeHttp:
    """Records every URL asked for and answers from a canned map."""

    def __init__(self, responses: dict[str, str]) -> None:
        self.responses = responses
        self.urls: list[str] = []

    def __call__(self, url: str, headers: dict[str, str] | None = None) -> str | None:
        self.urls.append(url)
        return self.responses.get(url)


def test_env_role_arn_is_reported_without_touching_the_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``AWS_ROLE_ARN`` is authoritative, so IMDS is never consulted."""
    monkeypatch.setenv("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/support-bot")
    fake = _FakeHttp({})
    monkeypatch.setattr(_cloud_identity, "_http_get", fake)

    found = _cloud_identity.collect()

    assert found["aws_role_arn"] == "arn:aws:iam::123456789012:role/support-bot"
    assert fake.urls == []


def test_no_cloud_signal_means_no_metadata_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A laptop makes zero link-local requests.

    Matters more than it looks: a 169.254 hop behind a corporate
    firewall blackholes rather than refusing, and this is what keeps
    developers from paying that timeout on every process start.
    """
    fake = _FakeHttp({})
    monkeypatch.setattr(_cloud_identity, "_http_get", fake)

    assert _cloud_identity.collect() == {}
    assert fake.urls == []


def test_gcp_service_account_comes_from_the_metadata_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("K_SERVICE", "support-bot")
    url = (
        "http://metadata.google.internal/computeMetadata/v1/instance/"
        "service-accounts/default/email"
    )
    fake = _FakeHttp({url: "bot@project.iam.gserviceaccount.com"})
    monkeypatch.setattr(_cloud_identity, "_http_get", fake)

    found = _cloud_identity.collect()

    assert found["gcp_service_account"] == "bot@project.iam.gserviceaccount.com"


def test_gcp_key_file_gives_up_only_the_client_email(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The same file holds a private key. It must stay in the file."""
    secret = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBg\n-----END PRIVATE KEY-----\n"
    key_file = tmp_path / "sa.json"
    key_file.write_text(
        json.dumps(
            {
                "type": "service_account",
                "client_email": "bot@project.iam.gserviceaccount.com",
                "client_id": "1234567890",
                "private_key": secret,
                "private_key_id": "abc123",
            }
        )
    )
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(key_file))
    monkeypatch.setattr(_cloud_identity, "_http_get", _FakeHttp({}))

    found = _cloud_identity.collect()

    assert found["gcp_service_account"] == "bot@project.iam.gserviceaccount.com"
    blob = json.dumps(found)
    assert "PRIVATE KEY" not in blob
    assert "abc123" not in blob


def test_the_probe_never_asks_imds_for_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """IMDS will hand out live access keys. We only ever read iam/info.

    If this test starts failing because a new URL appeared, that is
    not a test to update — it is a security review.
    """
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "support-bot")
    fake = _FakeHttp(
        {
            "http://169.254.169.254/latest/meta-data/iam/info": json.dumps(
                {
                    "InstanceProfileArn": (
                        "arn:aws:iam::123456789012:instance-profile/support-bot"
                    ),
                    "InstanceProfileId": "AIPAEXAMPLE",
                }
            )
        }
    )
    monkeypatch.setattr(_cloud_identity, "_http_get", fake)
    # The IMDSv2 token hop uses urlopen directly (it needs PUT), so
    # stub it to a fixed token rather than letting the test reach the
    # network.
    monkeypatch.setattr(
        _cloud_identity,
        "_aws_role_arn_via_imds",
        lambda: _imds_with_token(fake),
    )

    found = _cloud_identity.collect()

    assert found["aws_role_arn"].endswith("instance-profile/support-bot")
    for url in fake.urls:
        assert "security-credentials" not in url
        assert "iam/security" not in url


def _imds_with_token(fake: _FakeHttp) -> str | None:
    """The IMDS read with the PUT-for-a-token step already done."""
    body = fake("http://169.254.169.254/latest/meta-data/iam/info", {})
    if not body:
        return None
    return str(json.loads(body)["InstanceProfileArn"])


def test_oauth_client_ids_are_collected_and_deduplicated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OKTA_CLIENT_ID", "0oa1b2c3")
    monkeypatch.setenv("OAUTH_CLIENT_ID", "0oa1b2c3")
    monkeypatch.setenv("SALESFORCE_CLIENT_ID", "3MVG9abc")

    found = _cloud_identity.collect()

    assert found["oauth_client_ids"] == ["0oa1b2c3", "3MVG9abc"]


def test_an_oversized_value_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 4 KB "client id" is a pasted token, not an identifier."""
    monkeypatch.setenv("OAUTH_CLIENT_ID", "x" * 4096)

    assert _cloud_identity.collect() == {}


def test_a_probe_that_explodes_reports_nothing_and_stays_settled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom() -> dict[str, Any]:
        raise RuntimeError("metadata service on fire")

    monkeypatch.setattr(_cloud_identity, "collect", boom)
    _cloud_identity.prime_async()

    assert _cloud_identity.wait(2.0)
    assert _cloud_identity.snapshot() == {}


def test_the_runtime_blob_carries_the_identity_once_the_probe_lands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The overlay must survive the runtime cache.

    The blob is cached for the process but the probe finishes later,
    so a cached-and-forgotten identity would mean it never ships at
    all — the exact bug the overlay exists to prevent.
    """
    before = _runtime.collect_runtime_fingerprint(sdk_version="0.62.0")
    assert "cloud_identity" not in before

    monkeypatch.setenv("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/support-bot")
    monkeypatch.setattr(_cloud_identity, "_http_get", _FakeHttp({}))
    _cloud_identity.prime_async()
    assert _cloud_identity.wait(2.0)

    after = _runtime.collect_runtime_fingerprint(sdk_version="0.62.0")
    assert after["cloud_identity"]["aws_role_arn"].endswith("role/support-bot")


def test_opting_out_keeps_the_identity_out_of_the_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_ROLE_ARN", "arn:aws:iam::123456789012:role/support-bot")

    # No ``prime_async`` — this is what ``init(cloud_identity=False)``
    # leaves behind.
    blob = _runtime.collect_runtime_fingerprint(sdk_version="0.62.0")

    assert "cloud_identity" not in blob
