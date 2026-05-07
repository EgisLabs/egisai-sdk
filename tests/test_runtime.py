"""Runtime fingerprint collection and caching.

The fingerprint shipped to the platform powers the agent's
Provenance card. We check two things:

* The dict has every key the dashboard renders — a silent rename
  here blanks out the Provenance card on every customer's
  dashboard with no error surfaced.
* The collection is process-cached. Walking
  ``importlib.metadata`` for four framework names + reading
  ``/proc/1/cgroup`` on every per-prompt agent registration
  would burn measurable CPU; the values can't change inside a
  process so we collect once.
"""

from __future__ import annotations

from egisai import _runtime


def setup_function() -> None:
    _runtime.reset_runtime_cache()


def test_fingerprint_has_every_key_dashboard_reads() -> None:
    rt = _runtime.collect_runtime_fingerprint(sdk_version="1.2.3")
    # These are the exact keys the dashboard's AgentIdentityModal
    # Provenance card reads. A rename here breaks the UI silently.
    for key in (
        "sdk_version",
        "python",
        "implementation",
        "os",
        "platform",
        "machine",
        "container",
        "serverless",
        "frameworks",
    ):
        assert key in rt, f"missing {key!r} in {rt!r}"
    assert rt["sdk_version"] == "1.2.3"
    assert isinstance(rt["frameworks"], dict)
    # ``container`` is a clean bool — never None — so the dashboard
    # can render a yes/no badge.
    assert isinstance(rt["container"], bool)


def test_fingerprint_is_process_cached() -> None:
    a = _runtime.collect_runtime_fingerprint(sdk_version="1.0.0")
    b = _runtime.collect_runtime_fingerprint(sdk_version="1.0.0")
    assert a == b
    # Cache MUST return defensive copies so a caller mutating the
    # dict can't poison subsequent reads.
    a["python"] = "MUTATED"
    c = _runtime.collect_runtime_fingerprint(sdk_version="1.0.0")
    assert c["python"] != "MUTATED"


def test_cache_invalidates_when_sdk_version_changes() -> None:
    # SDK version is the only field a redeploy can change inside a
    # running test; bumping it must trigger a fresh collect.
    a = _runtime.collect_runtime_fingerprint(sdk_version="1.0.0")
    b = _runtime.collect_runtime_fingerprint(sdk_version="2.0.0")
    assert a["sdk_version"] == "1.0.0"
    assert b["sdk_version"] == "2.0.0"
