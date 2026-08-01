"""Container CPU allowance, not the host's core count.

Getting this wrong is not a rounding error: a one-vCPU Cloud Run
container on an eight-core host reports eight cores, which sized the
gateway's governance pool at 32 threads and turned a five-second NER
pass into a minute-long queue under concurrent load.
"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest

from egisai.policy import _cpu


@pytest.fixture(autouse=True)
def _fresh():
    _cpu.reset_cache_for_test()
    yield
    _cpu.reset_cache_for_test()


def _fake_fs(monkeypatch: pytest.MonkeyPatch, files: dict[str, str]) -> None:
    """Route only the cgroup paths to canned contents."""
    real_open = builtins.open

    def fake_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        key = str(path)
        if key in files:
            import io

            return io.StringIO(files[key])
        if key in (_cpu._CGROUP_V2_MAX, _cpu._CGROUP_V1_QUOTA, _cpu._CGROUP_V1_PERIOD):
            raise FileNotFoundError(key)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)


def test_cgroup_v2_quota_wins_over_host_cores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_fs(monkeypatch, {_cpu._CGROUP_V2_MAX: "400000 100000\n"})
    monkeypatch.setattr("os.cpu_count", lambda: 64)
    assert _cpu.available_cpus() == 4


def test_cgroup_v2_single_vcpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact Cloud Run default that caused the incident."""
    _fake_fs(monkeypatch, {_cpu._CGROUP_V2_MAX: "100000 100000\n"})
    monkeypatch.setattr("os.cpu_count", lambda: 8)
    assert _cpu.available_cpus() == 1


def test_cgroup_v2_unlimited_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_fs(monkeypatch, {_cpu._CGROUP_V2_MAX: "max 100000\n"})
    monkeypatch.setattr("os.cpu_count", lambda: 6)
    monkeypatch.setattr(_cpu, "_affinity_cpus", lambda: None)
    assert _cpu.available_cpus() == 6


def test_cgroup_v1_quota(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_fs(
        monkeypatch,
        {_cpu._CGROUP_V1_QUOTA: "200000\n", _cpu._CGROUP_V1_PERIOD: "100000\n"},
    )
    monkeypatch.setattr("os.cpu_count", lambda: 32)
    assert _cpu.available_cpus() == 2


def test_cgroup_v1_unlimited_quota_falls_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_fs(
        monkeypatch,
        {_cpu._CGROUP_V1_QUOTA: "-1\n", _cpu._CGROUP_V1_PERIOD: "100000\n"},
    )
    monkeypatch.setattr("os.cpu_count", lambda: 3)
    monkeypatch.setattr(_cpu, "_affinity_cpus", lambda: None)
    assert _cpu.available_cpus() == 3


def test_fractional_quota_floors_at_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 0.5-vCPU container must report 1, never 0."""
    _fake_fs(monkeypatch, {_cpu._CGROUP_V2_MAX: "50000 100000\n"})
    assert _cpu.available_cpus() == 1


def test_malformed_cgroup_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The SDK runs on macOS and Windows — this must not explode."""
    _fake_fs(monkeypatch, {_cpu._CGROUP_V2_MAX: "garbage\n"})
    monkeypatch.setattr("os.cpu_count", lambda: 2)
    monkeypatch.setattr(_cpu, "_affinity_cpus", lambda: None)
    assert _cpu.available_cpus() == 2


def test_result_is_memoized(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resolution sits behind pool construction; don't re-read /sys."""
    reads = {"n": 0}
    real_open = builtins.open

    def counting_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if str(path) == _cpu._CGROUP_V2_MAX:
            reads["n"] += 1
            import io

            return io.StringIO("200000 100000\n")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)
    first = _cpu.available_cpus()
    for _ in range(20):
        assert _cpu.available_cpus() == first
    assert reads["n"] == 1


def test_always_at_least_one_on_this_machine() -> None:
    """Unmocked sanity check on whatever platform CI runs."""
    assert _cpu.available_cpus() >= 1
    assert isinstance(_cpu.available_cpus(), int)


def test_real_cgroup_file_is_parsed_if_present() -> None:
    """If this box has cgroup v2, the parse must agree with the file."""
    path = Path(_cpu._CGROUP_V2_MAX)
    if not path.exists():
        pytest.skip("no cgroup v2 on this platform")
    quota_raw, _, period_raw = path.read_text().strip().partition(" ")
    if quota_raw == "max":
        pytest.skip("cgroup v2 present but unconstrained")
    expected = max(1, int(int(quota_raw) / int(period_raw)))
    assert _cpu.available_cpus() == expected
