"""How many CPUs this process may actually use.

``os.cpu_count()`` reports the cores visible on the *host*, which is
the wrong number inside a container. Cloud Run, Kubernetes, and Docker
all constrain CPU with cgroup quotas rather than by hiding cores, so a
container limited to one vCPU on an eight-core machine still sees
``os.cpu_count() == 8``.

Sizing a thread pool off that number is actively harmful for CPU-bound
work: eight threads' worth of NER dispatched onto one vCPU doesn't run
eight times faster, it runs at the same speed with eight times the
context-switching and eight times the peak memory. The gateway hit
exactly this — a governance pool sized ``cpus * 4`` put up to 32
threads on a single-vCPU Cloud Run container.

Resolution order, first hit wins:

1. **cgroup v2** — ``/sys/fs/cgroup/cpu.max`` holds ``"<quota> <period>"``
   (or ``"max <period>"`` when unconstrained). The ratio is the CPU
   allowance.
2. **cgroup v1** — ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us``, with a
   quota of ``-1`` meaning unconstrained.
3. **CPU affinity** — ``sched_getaffinity`` respects taskset-style
   pinning, which quotas don't cover.
4. **``os.cpu_count()``** — the host-wide fallback.

Every step is defensive: this runs inside customer processes via the
SDK, on platforms where ``/sys/fs/cgroup`` may be absent (macOS,
Windows), unreadable, or formatted unexpectedly. Any problem falls
through to the next source, and the final answer is always at least 1.
"""

from __future__ import annotations

import os

__all__ = ["available_cpus", "reset_cache_for_test"]

_CGROUP_V2_MAX = "/sys/fs/cgroup/cpu.max"
_CGROUP_V1_QUOTA = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
_CGROUP_V1_PERIOD = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"

# Resolved once — the limit cannot change during the process's life,
# and this sits behind pool construction on the request path.
_cached: int | None = None


def _read_int(path: str) -> int | None:
    try:
        with open(path, encoding="utf-8") as fh:
            return int(fh.read().strip())
    except (OSError, ValueError):
        return None


def _cgroup_v2_cpus() -> float | None:
    try:
        with open(_CGROUP_V2_MAX, encoding="utf-8") as fh:
            quota_raw, _, period_raw = fh.read().strip().partition(" ")
    except OSError:
        return None
    if quota_raw == "max":
        return None
    try:
        quota = int(quota_raw)
        period = int(period_raw)
    except ValueError:
        return None
    if quota <= 0 or period <= 0:
        return None
    return quota / period


def _cgroup_v1_cpus() -> float | None:
    quota = _read_int(_CGROUP_V1_QUOTA)
    period = _read_int(_CGROUP_V1_PERIOD)
    if quota is None or period is None:
        return None
    if quota <= 0 or period <= 0:
        return None
    return quota / period


def _affinity_cpus() -> int | None:
    getaffinity = getattr(os, "sched_getaffinity", None)
    if getaffinity is None:
        return None
    try:
        return len(getaffinity(0))
    except OSError:
        return None


def available_cpus() -> int:
    """CPUs this process may actually use — never less than 1.

    A fractional quota rounds **down** to whole cores but floors at 1,
    so a 0.5-vCPU container reports 1 rather than 0. Rounding up would
    reintroduce the oversubscription this function exists to prevent.
    """
    global _cached
    if _cached is not None:
        return _cached

    quota = _cgroup_v2_cpus()
    if quota is None:
        quota = _cgroup_v1_cpus()

    if quota is not None:
        resolved = max(1, int(quota))
    else:
        resolved = _affinity_cpus() or os.cpu_count() or 1

    _cached = max(1, resolved)
    return _cached


def reset_cache_for_test() -> None:
    """Forget the memoized value so a test can patch the sources."""
    global _cached
    _cached = None
