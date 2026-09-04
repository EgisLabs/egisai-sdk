"""Allow-only MiniLM first stage in front of the semantic_guard judge.

Never returns a block. Missing model, missing patterns, short text,
non-English, circuit breaker, or any exception → escalate to the
LLM judge. Blocks still come only from the judge.

Engine switch (``EGISAI_SEMANTIC_ENGINE``):

- unset / ``judge`` — today's path (default)
- ``cascade`` — skip the judge when the local score is below threshold
- ``local`` — test-only; never escalate (warn once)

Shadow (``EGISAI_SEMANTIC_SHADOW=1``) runs the local scorer AND the
judge, logs both, and always returns the judge's verdict.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from egisai.policy._onnx_embed import Embedder, try_load

LOGGER = logging.getLogger("egisai.semantic_local")

_DEFAULT_THRESHOLD = 0.35
_DEFAULT_MAX_MS = 50.0
_DEFAULT_MIN_CHARS = 80
_WINDOW = 32

_UNSET: object = object()
_embedder: Embedder | None | object = _UNSET
_embed_cache: dict[str, tuple[float, ...]] = {}
_cache_lock = threading.Lock()
_times: deque[float] = deque(maxlen=_WINDOW)
_times_lock = threading.Lock()
_tripped = False
_local_mode_warned = False
_load_ms: float | None = None


@dataclass(frozen=True)
class LocalObs:
    policy_id: str | None
    name: str
    local_score: float | None
    local_verdict: str  # allow | escalate  (never block)
    local_ms: float
    reason: str


@dataclass(frozen=True)
class JudgePrefilter:
    """MiniLM gate for ``POST /v1/sdk/judge`` / ``judge_sync``.

    ``skip_llm`` is True only when cascade/local would drop every
    pattern group. Shadow never skips. Missing groups fail open.
    """

    skip_llm: bool
    observations: tuple[LocalObs, ...]
    shadow: bool


def engine_mode() -> str:
    raw = (os.getenv("EGISAI_SEMANTIC_ENGINE") or "judge").strip().lower()
    if raw in ("cascade", "local", "judge"):
        return raw
    return "judge"


def shadow_enabled() -> bool:
    flag = (os.getenv("EGISAI_SEMANTIC_SHADOW") or "").strip()
    if flag not in ("1", "true", "True", "on", "ON"):
        return False
    raw = (os.getenv("EGISAI_SEMANTIC_SHADOW_SAMPLE_RATE") or "1.0").strip()
    try:
        rate = float(raw)
    except ValueError:
        rate = 1.0
    if rate <= 0:
        return False
    if rate >= 1:
        return True
    return random.random() < rate


def local_threshold() -> float:
    raw = (os.getenv("EGISAI_SEMANTIC_LOCAL_THRESHOLD") or "").strip()
    try:
        v = float(raw) if raw else _DEFAULT_THRESHOLD
        return v if 0.0 <= v <= 1.0 else _DEFAULT_THRESHOLD
    except ValueError:
        return _DEFAULT_THRESHOLD


def _max_ms() -> float:
    raw = (os.getenv("EGISAI_SEMANTIC_LOCAL_MAX_MS") or "").strip()
    try:
        v = float(raw) if raw else _DEFAULT_MAX_MS
        return v if v > 0 else _DEFAULT_MAX_MS
    except ValueError:
        return _DEFAULT_MAX_MS


def _min_chars() -> int:
    raw = (os.getenv("EGISAI_SEMANTIC_LANG_MIN_CHARS") or "").strip()
    try:
        v = int(raw) if raw else _DEFAULT_MIN_CHARS
        return v if v > 0 else _DEFAULT_MIN_CHARS
    except ValueError:
        return _DEFAULT_MIN_CHARS


def set_embedder(embedder: Embedder | None) -> None:
    """Tests inject a fake; ``None`` forces the missing-model path."""
    global _embedder
    _embedder = embedder


def reset_for_tests() -> None:
    global _embedder, _tripped, _local_mode_warned, _load_ms
    _embedder = _UNSET
    _embed_cache.clear()
    _times.clear()
    _tripped = False
    _local_mode_warned = False
    _load_ms = None


def prime_embedder() -> float | None:
    """Eager-load the ONNX session. Returns load milliseconds, or None."""
    started = time.monotonic()
    emb = _get_embedder()
    if emb is None:
        return None
    global _load_ms
    _load_ms = round((time.monotonic() - started) * 1000.0, 3)
    return _load_ms


def _get_embedder() -> Embedder | None:
    global _embedder
    if _embedder is not _UNSET:
        return _embedder  # type: ignore[return-value]
    _embedder = try_load()
    return _embedder  # type: ignore[return-value]


def _warn_local_mode() -> None:
    global _local_mode_warned
    if _local_mode_warned:
        return
    _local_mode_warned = True
    LOGGER.warning(
        "EGISAI_SEMANTIC_ENGINE=local is test-only; the local gate "
        "never blocks and never calls the judge"
    )


def looks_english(text: str) -> bool | None:
    """Cheap Latin-letter ratio. ``None`` = unknown (fail open to judge)."""
    try:
        letters = [c for c in text if c.isalpha()]
        if not letters:
            return None
        latin = 0
        for c in letters:
            o = ord(c)
            if (65 <= o <= 90) or (97 <= o <= 122) or (0x00C0 <= o <= 0x024F):
                latin += 1
        return (latin / len(letters)) >= 0.85
    except Exception:  # noqa: BLE001
        return None


def _circuit_tripped() -> bool:
    with _times_lock:
        return _tripped


def _record_ms(ms: float) -> None:
    global _tripped
    ceiling = _max_ms()
    with _times_lock:
        _times.append(ms)
        if len(_times) < 8:
            return
        ordered = sorted(_times)
        idx = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
        p95 = ordered[idx]
        was = _tripped
        _tripped = p95 > ceiling
        if _tripped and not was:
            LOGGER.warning(
                "egisai.semantic_local circuit open p95_ms=%.1f ceiling=%.1f",
                p95,
                ceiling,
            )
        elif was and not _tripped:
            LOGGER.info("egisai.semantic_local circuit closed p95_ms=%.1f", p95)


def _cache_key(text: str) -> str:
    # Prefix-stable: the judge window is a suffix of the transcript,
    # so hashing the exact string we embed keeps repeats cheap without
    # tying the key to later characters that fall out of the window.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_texts(texts: Sequence[str]) -> dict[str, tuple[float, ...]] | None:
    embedder = _get_embedder()
    if embedder is None:
        return None
    missing: list[str] = []
    out: dict[str, tuple[float, ...]] = {}
    with _cache_lock:
        for t in texts:
            key = _cache_key(t)
            hit = _embed_cache.get(key)
            if hit is None:
                missing.append(t)
            else:
                out[t] = hit
    if missing:
        vectors = embedder.embed(missing)
        if vectors is None or len(vectors) != len(missing):
            return None
        with _cache_lock:
            for text, vec in zip(missing, vectors, strict=True):
                stored = tuple(float(x) for x in vec)
                _embed_cache[_cache_key(text)] = stored
                out[text] = stored
    return out


def _cos(a: Sequence[float], b: Sequence[float]) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return float(sum(a[i] * b[i] for i in range(n)))


def _score_against(
    text_vec: Sequence[float],
    detect: list[Sequence[float]],
    exclude: list[Sequence[float]],
) -> float:
    max_d = max((_cos(text_vec, d) for d in detect), default=0.0)
    max_e = max((_cos(text_vec, e) for e in exclude), default=0.0)
    return max_d - max_e


def _patterns_of(policy: Any) -> tuple[list[str], list[str]]:
    raw = getattr(policy, "semantic_patterns", ()) or ()
    detect: list[str] = []
    exclude: list[str] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = str(item.get("kind") or "").strip().lower()
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        if kind == "detect":
            detect.append(text)
        elif kind == "exclude":
            exclude.append(text)
    return detect, exclude


def _targets_of(policy: Any) -> list[str]:
    cfg = getattr(policy, "config", None) or {}
    raw = cfg.get("targets") if isinstance(cfg, dict) else None
    if isinstance(raw, list) and raw:
        return [str(t) for t in raw if isinstance(t, str)]
    return ["text"]


def score_policy(
    policy: Any,
    *,
    text: str,
    tool_texts: Sequence[str],
    embeddings: dict[str, tuple[float, ...]],
    started: float,
) -> LocalObs:
    name = str(getattr(policy, "name", "") or "")
    pid = getattr(policy, "id", None)
    pid_s = str(pid) if pid else None
    detect_txt, exclude_txt = _patterns_of(policy)

    def elapsed() -> float:
        return round((time.monotonic() - started) * 1000.0, 3)

    if not detect_txt:
        return LocalObs(pid_s, name, None, "escalate", elapsed(), "no_patterns")
    targets = _targets_of(policy)
    scores: list[float] = []
    to_score: list[str] = []
    if "text" in targets and text:
        to_score.append(text)
    if "tool_calls" in targets:
        to_score.extend(tool_texts)
    if not to_score:
        return LocalObs(pid_s, name, None, "escalate", elapsed(), "empty")
    min_chars = _min_chars()
    for candidate in to_score:
        if len(candidate) < min_chars:
            return LocalObs(pid_s, name, None, "escalate", elapsed(), "short")
        lang = looks_english(candidate)
        if lang is not True:
            return LocalObs(pid_s, name, None, "escalate", elapsed(), "lang")
    detect_vecs: list[Sequence[float]] = []
    exclude_vecs: list[Sequence[float]] = []
    for t in detect_txt:
        vec = embeddings.get(t)
        if vec is None:
            return LocalObs(pid_s, name, None, "escalate", elapsed(), "no_model")
        detect_vecs.append(vec)
    for t in exclude_txt:
        vec = embeddings.get(t)
        if vec is None:
            return LocalObs(pid_s, name, None, "escalate", elapsed(), "no_model")
        exclude_vecs.append(vec)
    for candidate in to_score:
        vec = embeddings.get(candidate)
        if vec is None:
            return LocalObs(pid_s, name, None, "escalate", elapsed(), "no_model")
        scores.append(_score_against(vec, detect_vecs, exclude_vecs))
    # Most suspicious surface wins; skip only when EVERY surface is quiet.
    score = max(scores) if scores else 0.0
    if any(s >= local_threshold() for s in scores):
        return LocalObs(pid_s, name, score, "escalate", elapsed(), "threshold")
    return LocalObs(pid_s, name, score, "allow", elapsed(), "threshold")


def score_policies(
    policies: Sequence[Any],
    *,
    text: str,
    tool_texts: Sequence[str],
) -> list[LocalObs]:
    started = time.monotonic()
    if _circuit_tripped():
        elapsed = round((time.monotonic() - started) * 1000.0, 3)
        return [
            LocalObs(
                str(getattr(p, "id", "") or "") or None,
                str(getattr(p, "name", "") or ""),
                None,
                "escalate",
                elapsed,
                "circuit",
            )
            for p in policies
        ]
    needed: list[str] = []
    if text:
        needed.append(text)
    needed.extend(tool_texts)
    for p in policies:
        d, e = _patterns_of(p)
        needed.extend(d)
        needed.extend(e)
    unique = list(dict.fromkeys(t for t in needed if t))
    embeddings = _embed_texts(unique)
    elapsed = round((time.monotonic() - started) * 1000.0, 3)
    _record_ms(elapsed)
    if embeddings is None:
        return [
            LocalObs(
                str(getattr(p, "id", "") or "") or None,
                str(getattr(p, "name", "") or ""),
                None,
                "escalate",
                elapsed,
                "no_model",
            )
            for p in policies
        ]
    return [
        score_policy(
            p,
            text=text,
            tool_texts=tool_texts,
            embeddings=embeddings,
            started=started,
        )
        for p in policies
    ]


def prefilter_judge_text(
    text: str,
    *,
    pattern_groups: Sequence[Sequence[dict[str, str]]] = (),
) -> JudgePrefilter:
    """Score each pattern group against ``text`` for the HTTP judge.

    Skip the LLM only when every non-empty group local-allows and
    the engine is ``cascade`` (or ``local``) with shadow off. Empty
    groups, missing model, short/lang/circuit, or any exception
    fail open to the LLM. ``shadow`` is sampled once so the caller
    can log after the judge without rolling the dice again.
    """
    shadow = shadow_enabled()
    empty = JudgePrefilter(False, (), shadow)
    try:
        groups = [tuple(g) for g in pattern_groups if g]
        if not groups:
            return empty
        mode = engine_mode()
        if mode == "judge" and not shadow:
            return empty
        shims = [
            SimpleNamespace(
                id=None,
                name=f"judge-{i}",
                config={},
                semantic_patterns=g,
            )
            for i, g in enumerate(groups)
        ]
        observations = tuple(
            score_policies(shims, text=text, tool_texts=[])
        )
        if shadow:
            return JudgePrefilter(False, observations, True)
        if mode == "local":
            _warn_local_mode()
            return JudgePrefilter(True, observations, False)
        if mode == "cascade":
            skip = bool(observations) and all(
                o.local_verdict == "allow" for o in observations
            )
            return JudgePrefilter(skip, observations, False)
        return JudgePrefilter(False, observations, False)
    except Exception:  # noqa: BLE001
        LOGGER.debug("prefilter_judge_text failed open", exc_info=True)
        return JudgePrefilter(False, (), shadow)


def should_skip_judge(
    policy: Any,
    *,
    text: str,
    tool_texts: Sequence[str],
) -> bool:
    """True when cascade/local may drop this policy from the judge."""
    mode = engine_mode()
    if mode == "judge" or shadow_enabled():
        return False
    if mode == "local":
        _warn_local_mode()
        return True
    obs = score_policies([policy], text=text, tool_texts=tool_texts)
    return bool(obs) and obs[0].local_verdict == "allow"


def filter_escalations(
    policies: list[Any],
    *,
    text: str,
    tool_texts: Sequence[str],
) -> tuple[list[Any], list[LocalObs]]:
    """Drop locally-allowed policies when cascade is on.

    Shadow mode scores but keeps every policy in the judge set.
    ``local`` mode drops all (never escalate). ``judge`` keeps all.
    """
    mode = engine_mode()
    shadow = shadow_enabled()
    need_scores = shadow or mode in ("cascade", "local")
    if not need_scores:
        return policies, []
    observations = score_policies(policies, text=text, tool_texts=tool_texts)
    if shadow or mode == "judge":
        return policies, observations
    if mode == "local":
        _warn_local_mode()
        return [], observations
    kept = [
        p
        for p, obs in zip(policies, observations, strict=True)
        if obs.local_verdict != "allow"
    ]
    return kept, observations


def emit_shadow(
    observations: Sequence[LocalObs],
    *,
    matched_names: set[str],
    hook: str,
    text_sha256: str,
    semantic_in_scope: int,
    judge_ms: float,
) -> None:
    if not observations:
        return
    policies_payload = []
    for obs in observations:
        judge_verdict = "block" if obs.name in matched_names else "allow"
        policies_payload.append(
            {
                "policy_id": obs.policy_id,
                "name": obs.name,
                "hook_type": hook or "",
                "local_score": obs.local_score,
                "local_verdict": obs.local_verdict,
                "judge_verdict": judge_verdict,
                "local_ms": obs.local_ms,
                "reason": obs.reason,
            }
        )
    joint = all(o.local_verdict == "allow" for o in observations)
    payload = {
        "hook_type": hook or "",
        "semantic_in_scope": semantic_in_scope,
        "judge_ms": judge_ms,
        "text_sha256": text_sha256,
        "joint_local_allow": joint,
        "would_skip_judge": joint,
        "policies": policies_payload,
    }
    LOGGER.info("egisai.semantic_shadow %s", json.dumps(payload, default=str))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
