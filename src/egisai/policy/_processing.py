"""Policy processing-time reporting.

``policy_latency_ms`` is processing only — local CPU plus the judge
model's prompt_time + completion_time. SDK and gateway stamp the
same number. There is no opt-out.

The definition below is the stamp-site contract. Do not paraphrase
it at the call sites; import ``PROCESSING_MS_DEFINITION``.
"""

from __future__ import annotations

import contextvars
from collections.abc import Sequence
from typing import Any

# Critical-path processing time of every rule that ran on this step: local CPU plus the judge model's prompt_time + completion_time. Excludes SDK ↔ gateway transit, gateway ↔ Cerebras transit, TLS connect, Cerebras queue_time, and human-approval wait. Compose: sum sequential phases (input, tools, output, later steps). max parallel judge calls in one wave. Count each judge_group once.
PROCESSING_MS_DEFINITION = (
    "Critical-path processing time of every rule that ran on this step: "
    "local CPU plus the judge model's prompt_time + completion_time. "
    "Excludes SDK ↔ gateway transit, gateway ↔ Cerebras transit, TLS "
    "connect, Cerebras queue_time, and human-approval wait. Compose: sum "
    "sequential phases (input, tools, output, later steps). max parallel "
    "judge calls in one wave. Count each judge_group once."
)

# Sentinel: no check() recorded a value on this task. Distinct from
# ``None`` (clocks missing — omit, contribute 0) and ``0.0`` (cache
# hit / no compute).
UNSET: object = object()

_last: contextvars.ContextVar[Any] = contextvars.ContextVar(
    "egisai_processing_ms", default=UNSET
)


def record(ms: float | None) -> None:
    """Publish the last network check's processing ms onto this task.

    ``None`` means clocks were missing — omit, never fall back to the
    HTTP wait. ``0.0`` is a cache hit. A float is prompt+completion
    (semantic) or local smart-tier CPU (injection).
    """
    _last.set(ms)


def take() -> Any:
    """Pop the last ``record()``. Returns ``UNSET`` when none ran."""
    value = _last.get()
    _last.set(UNSET)
    return value


def record_from_payload(data: Any, *, cache_hit: bool = False) -> None:
    """SDK clients: honour ``processing_ms`` from the judge / injection JSON.

    Missing field → omit (``None``), never the HTTP wait.
    """
    if cache_hit:
        record(0.0)
        return
    raw = data.get("processing_ms") if isinstance(data, dict) else None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        record(float(raw))
        return
    record(None)


def max_wave(
    items: Sequence[tuple[str | None, float | None]],
) -> float | None:
    """One parallel wave: max of members, each ``judge_group`` once.

    ``None`` ms omits (contributes 0). Empty ``items`` contributes
    nothing (returns ``None``) so a walk with no judge calls does
    not inject a zero wave.
    """
    if not items:
        return None
    seen: set[str] = set()
    values: list[float] = []
    for group, ms in items:
        if group:
            if group in seen:
                continue
            seen.add(group)
        values.append(0.0 if ms is None else float(ms))
    if not values:
        return None
    return max(values)


def rule_ms(wall: float, taken: Any) -> float:
    """Per-rule ``PolicyTiming.ms``.

    Unset (local rule, or judge never called) keeps the wall — that
    wall *is* local CPU. A recorded ``None`` stamps 0 (omit). A
    recorded float is used as-is.
    """
    if taken is UNSET:
        return wall
    if taken is None:
        return 0.0
    return round(float(taken), 3)


def latency_ms(decision: Any, wall_ms: int) -> int:
    """Stamp ``policy_latency_ms``.

    Critical-path processing time of every rule that ran on this step: local CPU plus the judge model's prompt_time + completion_time. Excludes SDK ↔ gateway transit, gateway ↔ Cerebras transit, TLS connect, Cerebras queue_time, and human-approval wait. Compose: sum sequential phases (input, tools, output, later steps). max parallel judge calls in one wave. Count each judge_group once.
    """
    processing = getattr(decision, "processing_ms", None)
    if processing is not None:
        return max(0, int(round(float(processing))))
    return max(0, int(wall_ms))
