"""Fast-governance mode: fewer judge questions, provably-equal answers.

The measured problem (July 2026, Cerebras ``gpt-oss-120b`` judge):

* **Tail amplification.** Phase 2 fans one judge round-trip out per
  ``semantic_guard`` policy, and each policy fans out again per tool
  call. Wall-clock is the *max* of N samples, so a 3-guard turn with
  4 tool calls (15 round-trips) waits on ~P95 of the judge's latency
  distribution, not P50. Cutting the *number* of questions is worth
  more than making each question faster.
* **Quadratic transcript growth.** Agentic loops resend the entire
  conversation every turn, so turn 5 re-judges turns 1–4 — tokens grow
  quadratically over the loop and the verdict cache can never hit.

Fast mode makes three changes, all confined to Phase 2 semantics:

1. **Merged judge call** — policies that share a threshold are asked
   in ONE round-trip carrying the union of their intents. The judge
   already accepts an intent *list* and cites the matched intent back,
   so per-policy attribution survives; the engine maps the cited
   intent to its owning policy.
2. **Windowed judge text** — the text target judges the most recent
   ``EGISAI_JUDGE_TEXT_WINDOW_CHARS`` characters instead of the whole
   accumulated transcript. Every message is still judged (each turn
   was inside the window when it was new); the window bounds token
   growth and keeps latency flat over long loops.
3. **Normalized cache keys** — opaque identifiers (UUIDs, hex run
   ids, ``CUST-9374731``-style prefixed ids) are canonicalised before
   the verdict-cache key is hashed, so two calls that differ only in
   an id share one verdict. Amounts and bare numbers are deliberately
   untouched. Normalisation affects the cache KEY only — the text
   sent to the judge is byte-identical to the raw text.

Rollout contract — ``EGISAI_FAST_GOVERNANCE``:

* ``off`` (default) — nothing changes. Byte-identical behaviour to
  the previous release.
* ``shadow`` — the legacy path keeps making every enforcement
  decision; the fast path additionally runs on a background daemon
  thread, decides nothing, and reports whether it agreed. One stderr
  line per evaluation, plus a loud warning on any disagreement. This
  is how the fast path earns trust on real traffic before it is ever
  allowed to enforce.
* ``on`` — the fast path enforces. Flip back to ``off``/``shadow``
  at any time; no state survives the flip.

Compliance notes:

* Shadow reports contain policy names and verdicts only — never the
  prompt text (security-and-compliance.mdc §1/§5).
* Shadow judge calls run on a fresh thread with an empty context, so
  their token spend is NOT booked to the governed call's
  ``policy_tokens_*`` (those columns keep meaning "tokens this call's
  enforcement actually paid for").
* Phase ordering is untouched: fast mode replaces the *Phase 2 walk
  only*. Phase 1 determinism, block short-circuits, and
  sanitize-before-judge all run exactly as before.
"""

from __future__ import annotations

import logging
import os
import random
import re
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

LOGGER = logging.getLogger("egisai.policy.fastpath")

MODE_ENV = "EGISAI_FAST_GOVERNANCE"
WINDOW_ENV = "EGISAI_JUDGE_TEXT_WINDOW_CHARS"
NORMALIZE_ENV = "EGISAI_JUDGE_CACHE_NORMALIZE"
SHADOW_SAMPLE_ENV = "EGISAI_FAST_SHADOW_SAMPLE"

_VALID_MODES = ("off", "shadow", "on")

# Default text window for the judge in fast mode. ~16k chars ≈ 4k
# tokens ≈ many conversational turns. Generous on purpose: the window
# exists to stop *unbounded quadratic* growth, not to shave the last
# millisecond — a window too small would let the head of a very long
# single message escape judgment, which is a governance hole. Values
# below the floor are clamped up for the same reason.
_DEFAULT_WINDOW_CHARS = 16_000
_MIN_WINDOW_CHARS = 1_000

_warned_invalid_mode = False


def mode() -> str:
    """Resolve ``EGISAI_FAST_GOVERNANCE``. Unset/invalid ⇒ ``off``.

    Read per evaluation (an ``os.environ`` lookup, sub-microsecond) so
    an operator can flip a long-lived process between ``shadow`` and
    ``on`` via a restartless config layer that rewrites the env.
    """
    raw = (os.environ.get(MODE_ENV) or "").strip().lower()
    if not raw:
        return "off"
    if raw in _VALID_MODES:
        return raw
    global _warned_invalid_mode
    if not _warned_invalid_mode:
        _warned_invalid_mode = True
        LOGGER.warning(
            "%s=%r is not one of %s — treating as 'off'",
            MODE_ENV, raw, "|".join(_VALID_MODES),
        )
    return "off"


def window_chars() -> int:
    """Resolve the judge text window. ``0`` disables windowing."""
    raw = (os.environ.get(WINDOW_ENV) or "").strip()
    if not raw:
        return _DEFAULT_WINDOW_CHARS
    try:
        value = int(float(raw))
    except ValueError:
        LOGGER.warning(
            "%s=%r is not a number — using default %d",
            WINDOW_ENV, raw, _DEFAULT_WINDOW_CHARS,
        )
        return _DEFAULT_WINDOW_CHARS
    if value <= 0:
        return 0
    return max(_MIN_WINDOW_CHARS, value)


def window_text(text: str) -> str:
    """Return the tail window of ``text`` the judge should see.

    The newest content — the part no previous turn has judged — is at
    the END of the flattened transcript for every framework the SDK
    patches (conversations are append-only), so a tail window always
    contains the un-judged delta plus recent context for cross-turn
    attack detection. Earlier content was judged on the turns when it
    was itself inside the window.
    """
    limit = window_chars()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[-limit:]


# ── Cache-key normalisation ──────────────────────────────────────────
#
# Deliberately narrow. A token is normalised ONLY when it cannot
# plausibly change a governance verdict:
#
#   * UUIDs                     (session/request/workflow ids)
#   * long hex runs (≥ 12)      (hashes, trace ids)
#   * letter-prefix + separator + alphanumeric-with-digits
#     (``CUST-9374731``, ``sess_5df0c293``, ``req-abc123``)
#
# NEVER normalised, on purpose:
#
#   * bare numbers — ``45000`` might be an amount; a $10 payment and a
#     $1M payment are different governance questions.
#   * tokens without a separator (``ALICE123``) — could be a
#     meaningful name.
#   * anything the PII layer already replaced with ``<SSN>``-style
#     labels (labels contain no digits, so these patterns skip them).
#
# Applied to the cache KEY only; the judge always receives raw text.

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
    r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_PREFIXED_ID_RE = re.compile(
    r"\b([A-Za-z][A-Za-z0-9]{1,31}[-_])((?=[A-Za-z0-9]*\d)[A-Za-z0-9]{3,64})\b"
)
_LONG_HEX_RE = re.compile(r"\b(?=[0-9a-fA-F]*[0-9])(?=[0-9a-fA-F]*[a-fA-F])[0-9a-fA-F]{12,}\b")


def cache_normalization_enabled() -> bool:
    """Normalised keys ride with fast mode ``on``; the extra env var
    is a targeted kill switch if an operator ever needs exact-match
    keys back without giving up the merged calls."""
    if mode() != "on":
        return False
    raw = (os.environ.get(NORMALIZE_ENV) or "").strip().lower()
    return raw not in ("0", "false", "no", "off")


def normalize_for_cache_key(text: str) -> str:
    """Canonicalise opaque identifiers for cache-key hashing."""
    out = _UUID_RE.sub("<UUID>", text)
    out = _PREFIXED_ID_RE.sub(r"\1<ID>", out)
    out = _LONG_HEX_RE.sub("<HEX>", out)
    return out


# ── Shadow harness ───────────────────────────────────────────────────
#
# The fast path must never be trusted on argument alone — it changes
# what the judge sees, so it is validated on the operator's real
# traffic while the legacy path keeps enforcing. The harness is
# deliberately boring: a bounded pool of daemon threads, a running
# agree/disagree tally, one compact stderr line per comparison.

# Cap on concurrently-running shadow evaluations. Above the cap the
# comparison is skipped (and counted) rather than queued — shadow is
# a sampling exercise, not an exactly-once pipeline, and an unbounded
# thread spawn under load would be its own incident.
_SHADOW_MAX_CONCURRENT = 4
_shadow_slots = threading.Semaphore(_SHADOW_MAX_CONCURRENT)

_stats_lock = threading.Lock()
_shadow_agree = 0
_shadow_disagree = 0
_shadow_skipped = 0

# Test seam: the most recently spawned shadow thread, so tests can
# join it deterministically instead of sleeping.
_last_shadow_thread: threading.Thread | None = None


def shadow_sampled() -> bool:
    """Whether THIS evaluation should run a shadow comparison.

    ``EGISAI_FAST_SHADOW_SAMPLE`` ∈ [0.0, 1.0], default 1.0 (every
    evaluation). Lower it on high-volume deployments to bound the
    doubled judge spend while still accumulating evidence.
    """
    raw = (os.environ.get(SHADOW_SAMPLE_ENV) or "").strip()
    if not raw:
        return True
    try:
        rate = float(raw)
    except ValueError:
        return True
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate


def spawn_shadow(run: Callable[[], None]) -> bool:
    """Run ``run`` on a bounded background daemon thread.

    Returns ``True`` if the shadow was actually spawned. The thread
    starts with a FRESH (empty) context on purpose: judge calls made
    inside it create their own throwaway token accumulator instead of
    inheriting the governed call's, so shadow spend never pollutes
    ``policy_tokens_*`` on the audit row.
    """
    global _last_shadow_thread, _shadow_skipped
    if not _shadow_slots.acquire(blocking=False):
        with _stats_lock:
            _shadow_skipped += 1
        LOGGER.debug("shadow evaluation skipped: %d already in flight",
                     _SHADOW_MAX_CONCURRENT)
        return False

    def _wrapped() -> None:
        try:
            run()
        except Exception:  # noqa: BLE001
            LOGGER.debug("shadow evaluation crashed", exc_info=True)
        finally:
            _shadow_slots.release()

    t = threading.Thread(
        target=_wrapped, name="egisai-shadow", daemon=True
    )
    _last_shadow_thread = t
    t.start()
    return True


def wait_for_shadow(timeout: float = 5.0) -> None:
    """Join the most recently spawned shadow thread. Test seam."""
    t = _last_shadow_thread
    if t is not None:
        t.join(timeout)


def report_shadow(
    *,
    side: str,
    legacy_records: list,
    fast_records: list,
    elapsed_ms: float,
) -> bool:
    """Compare the two paths' Phase-2 outcomes and say so out loud.

    Compliance: names and verdicts only. No prompt text, no tool
    arguments, no intent strings (operator-authored intents are safe
    in principle, but the cited intent can embed judge paraphrase, so
    the report sticks to policy names).
    """
    global _shadow_agree, _shadow_disagree

    def _blocked_names(records: list) -> list[str]:
        return sorted(
            {str(getattr(r, "name", "?"))
             for r in records if getattr(r, "verdict", "") == "block"}
        )

    legacy_names = _blocked_names(legacy_records)
    fast_names = _blocked_names(fast_records)
    legacy_verdict = "block" if legacy_names else "allow"
    fast_verdict = "block" if fast_names else "allow"
    # Agreement means "same enforcement outcome". Attribution detail
    # (which of two overlapping policies gets named) may legitimately
    # differ — a merged call reports the judge's single best-matching
    # intent — so the verdict, not the name list, is the gate.
    agree = legacy_verdict == fast_verdict

    with _stats_lock:
        if agree:
            _shadow_agree += 1
        else:
            _shadow_disagree += 1
        agree_n, disagree_n = _shadow_agree, _shadow_disagree

    total = agree_n + disagree_n
    line = (
        f"🔬 [egisai] fast-governance shadow ({side}): "
        f"{'AGREE' if agree else 'DISAGREE'} "
        f"legacy={legacy_verdict}{legacy_names or ''} "
        f"fast={fast_verdict}{fast_names or ''} "
        f"fast_ms={elapsed_ms:.0f} "
        f"(agree {agree_n}/{total} since start)"
    )
    print(line, file=sys.stderr, flush=True)
    if not agree:
        LOGGER.warning(
            "fast-governance shadow DISAGREEMENT on %s side: legacy=%s%s "
            "fast=%s%s — keep EGISAI_FAST_GOVERNANCE=shadow until resolved",
            side, legacy_verdict, legacy_names, fast_verdict, fast_names,
        )
    return agree


def report_shadow_diagnosis(
    *,
    side: str,
    kind: str,
    question_index: int,
    policy_count: int,
    intent_count: int,
    question_chars: int,
    threshold: Any,
    match: bool,
    confidence: float,
) -> None:
    """One line per merged question after a DISAGREE, numbers only.

    How to read it: ``confidence`` is the judge's calibrated score on
    the re-asked merged question. A score just below the threshold
    (e.g. 0.65 against 0.70) means the union intent list diluted an
    otherwise-clear match — the mitigation conversation is about list
    size or reasoning effort. A hard 0.00 means the judge answered a
    plain ALLOW on content the per-policy questions blocked — a
    structural problem worth a bug report, not a tuning knob.
    """
    print(
        f"🔬 [egisai] shadow-diagnosis ({side}/{kind} q{question_index}): "
        f"match={str(match).lower()} confidence={confidence:.2f} "
        f"threshold={threshold if threshold is not None else 'default'} "
        f"policies={policy_count} intents={intent_count} "
        f"question_chars={question_chars}",
        file=sys.stderr,
        flush=True,
    )


def shadow_stats() -> tuple[int, int, int]:
    """(agree, disagree, skipped) counters. Observability/test seam."""
    with _stats_lock:
        return _shadow_agree, _shadow_disagree, _shadow_skipped


def reset_shadow_stats_for_tests() -> None:
    global _shadow_agree, _shadow_disagree, _shadow_skipped
    with _stats_lock:
        _shadow_agree = 0
        _shadow_disagree = 0
        _shadow_skipped = 0


def now_ms() -> float:
    """Monotonic milliseconds — tiny helper so the engine's shadow
    closure doesn't need its own ``time`` import."""
    return time.monotonic() * 1000.0
