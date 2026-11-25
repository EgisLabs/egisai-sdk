"""Custom (self-hosted) model hosts are audited on the raw-HTTP fallback.

Case 4 of the custom-models feature: a customer registers a private /
self-hosted model endpoint (vLLM, TGI, Ollama, …) that the SDK has no
dedicated adapter for, and calls it over plain ``httpx`` / ``requests``.
The backend ships the org's registered hostnames on the policy feed
(``custom_model_hosts`` in ``/v1/sdk/policies``); the SDK adds them to
its model-call host allow-list so the call shows up on the dashboard as
an audited (``verdict="allow"``) Network-layer event — never blocked,
fail-open.

These tests pin:

* the predicate (``_looks_like_model_call``) honours the custom-host set
  with the same host+path discipline as the built-in public hosts;
* the policy handshake end-to-end plumbs ``custom_model_hosts`` into the
  predicate via ``refresh_now``;
* a 304 leaves the audited set untouched;
* an empty feed clears the set without touching the built-in hosts.
"""

from __future__ import annotations

import pytest

from egisai._patches import http as http_patch
from egisai._patches.http import (
    _looks_like_model_call,
    set_custom_model_hosts,
)


@pytest.fixture(autouse=True)
def _reset_custom_hosts():
    """Every test starts and ends with an empty custom-host set."""
    set_custom_model_hosts([])
    yield
    set_custom_model_hosts([])


def test_custom_host_with_inference_path_is_matched() -> None:
    set_custom_model_hosts(["llm.internal.acme.corp"])
    assert _looks_like_model_call(
        "https://llm.internal.acme.corp/v1/chat/completions"
    )
    # Ollama / TGI style paths are covered too.
    assert _looks_like_model_call(
        "http://llm.internal.acme.corp:11434/api/generate"
    )


def test_custom_host_without_inference_path_is_ignored() -> None:
    """Health checks / metrics on the same host are not model calls."""
    set_custom_model_hosts(["llm.internal.acme.corp"])
    assert not _looks_like_model_call(
        "https://llm.internal.acme.corp/healthz"
    )
    assert not _looks_like_model_call(
        "https://llm.internal.acme.corp/metrics"
    )


def test_unregistered_host_stays_ignored() -> None:
    set_custom_model_hosts(["llm.internal.acme.corp"])
    # A different private host the operator never registered.
    assert not _looks_like_model_call(
        "https://other.internal.acme.corp/v1/chat/completions"
    )


def test_builtin_public_hosts_still_match_with_custom_set_empty() -> None:
    set_custom_model_hosts([])
    assert _looks_like_model_call("https://api.openai.com/v1/chat/completions")


def test_host_matching_is_case_insensitive() -> None:
    set_custom_model_hosts(["LLM.Internal.ACME.Corp"])
    assert _looks_like_model_call(
        "https://llm.internal.acme.corp/v1/chat/completions"
    )


def test_set_custom_model_hosts_tolerates_none() -> None:
    set_custom_model_hosts(["llm.internal.acme.corp"])
    set_custom_model_hosts(None)
    assert not _looks_like_model_call(
        "https://llm.internal.acme.corp/v1/chat/completions"
    )


def test_policy_feed_plumbs_custom_hosts_end_to_end(fake_backend) -> None:
    """The handshake's ``custom_model_hosts`` reach the predicate."""
    import egisai
    from egisai import _policy_cache

    fake_backend.set_rules(
        [], etag='"v1"', custom_model_hosts=["llm.internal.acme.corp"]
    )

    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )

    assert http_patch._custom_hosts_snapshot() == frozenset(
        {"llm.internal.acme.corp"}
    )
    assert _looks_like_model_call(
        "https://llm.internal.acme.corp/v1/chat/completions"
    )

    # A new snapshot that drops the host clears the audited set.
    fake_backend.set_rules([], etag='"v2"', custom_model_hosts=[])
    assert _policy_cache.refresh_now() is True
    assert http_patch._custom_hosts_snapshot() == frozenset()


def test_304_leaves_custom_hosts_untouched(fake_backend) -> None:
    """A 304 (nothing changed) must not clear the audited host set."""
    import egisai
    from egisai import _policy_cache

    fake_backend.set_rules(
        [], etag='"v1"', custom_model_hosts=["llm.internal.acme.corp"]
    )
    egisai.init(
        api_key="egis_live_x",
        app="a",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )
    assert http_patch._custom_hosts_snapshot() == frozenset(
        {"llm.internal.acme.corp"}
    )

    # ETag unchanged ⇒ backend answers 304 ⇒ cache (and audited
    # host set) stays exactly as-is.
    assert _policy_cache.refresh_now() is False
    assert http_patch._custom_hosts_snapshot() == frozenset(
        {"llm.internal.acme.corp"}
    )
