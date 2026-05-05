"""Defensive regex compilation + matching for operator-supplied patterns.

Two-layer protection against catastrophic backtracking (ReDoS):

1. ``safe_compile`` — rejects patterns that are too long, contain
   nested-quantifier shapes, or contain long runs of optional /
   star quantifiers responsible for most real-world ReDoS reports.
2. ``safe_search`` — runs the match on a background daemon thread
   with a wall-clock timeout (default 50 ms). On timeout the
   function returns ``None`` ("no match") so the customer's call is
   unblocked, and a warning is logged. This layer protects against
   any blocking call that *releases the GIL* — sleeps, network
   reads, etc. Python's built-in ``re`` engine does **not** release
   the GIL during matching, so Layer 2 cannot interrupt a runaway
   ``re`` evaluation; that's why Layer 1 is intentionally strict.
"""

from __future__ import annotations

import logging
import re
import threading

LOGGER = logging.getLogger("egisai.policy.regex_safe")


_MAX_PATTERN_LEN = 1024
_MAX_INPUT_LEN = 64 * 1024

# Quantified group followed by another quantifier — the canonical
# nested-quantifier shape behind most catastrophic-backtracking reports.
_NESTED_QUANT_RE = re.compile(
    r"\)[\*\+\?]?\s*[\*\+\{]"
    r"|"
    r"[\)\]]\?\?\s*[\*\+]"
    r"|"
    r"\([^)]*?[\*\+\{]\)\s*[\*\+\{]"
)

_RUNAWAY_QUANT_MIN = 5
_RUNAWAY_QUANT_WINDOW = 25


def _has_runaway_optional_chain(pattern: str) -> bool:
    """Detect a chain of 5+ ``?`` / ``*`` quantifiers in a short window.

    Catches the runaway shape ``a?a?a?a?a?aaaaa`` which has no group
    at the top level but still triggers exponential backtracking when
    each optional and the trailing mandatory atom can match the same
    character. We deliberately ignore ``?`` that follows an opening
    paren (``(?:``, ``(?P<name>``, ``(?=``, …) — those are group
    syntax, not quantifiers — and we skip escaped characters so a
    pattern containing ``\\?`` literal question marks isn't penalised.
    """
    positions: list[int] = []
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "\\" and i + 1 < n:
            i += 2
            continue
        if c == "(" and i + 1 < n and pattern[i + 1] == "?":
            i += 2
            continue
        if c in "?*":
            positions.append(i)
        i += 1

    if len(positions) < _RUNAWAY_QUANT_MIN:
        return False
    for j in range(len(positions) - (_RUNAWAY_QUANT_MIN - 1)):
        window = positions[j + _RUNAWAY_QUANT_MIN - 1] - positions[j]
        if window <= _RUNAWAY_QUANT_WINDOW:
            return True
    return False


DEFAULT_REGEX_TIMEOUT_S = 0.05


class UnsafePatternError(ValueError):
    """``safe_compile`` rejected the pattern."""


def safe_compile(pattern: str, flags: int = 0) -> re.Pattern[str]:
    """Compile ``pattern`` with safety guards.

    Raises ``UnsafePatternError`` if the pattern fails validation,
    or ``re.error`` for syntax errors.
    """
    if not isinstance(pattern, str):
        raise UnsafePatternError(
            f"regex pattern must be a string, got {type(pattern).__name__}"
        )
    if len(pattern) > _MAX_PATTERN_LEN:
        raise UnsafePatternError(
            f"regex pattern exceeds {_MAX_PATTERN_LEN} chars; refusing to compile"
        )
    if _NESTED_QUANT_RE.search(pattern):
        raise UnsafePatternError(
            "regex pattern contains a nested quantifier shape "
            "(``(...)+`` followed by another quantifier) that is a "
            "common cause of catastrophic backtracking; refusing to "
            "compile. Rewrite the pattern to use possessive quantifiers "
            "or atomic groups, or split it into multiple patterns."
        )
    if _has_runaway_optional_chain(pattern):
        raise UnsafePatternError(
            "regex pattern contains 5+ optional / star quantifiers in "
            "close succession (e.g. ``a?a?a?a?a?aaaaa``) — that shape "
            "causes exponential backtracking when the optional atoms "
            "and the mandatory tail can match the same character. "
            "Rewrite the pattern to use atomic groups, anchored "
            "alternation, or possessive quantifiers."
        )
    return re.compile(pattern, flags)


def safe_search(
    pattern: str | re.Pattern[str],
    text: str,
    flags: int = 0,
    *,
    timeout_s: float = DEFAULT_REGEX_TIMEOUT_S,
) -> re.Match[str] | None:
    """Drop-in for ``re.search`` with input-length and wall-clock guards.

    Returns ``None`` on timeout, compile failure, or any worker-thread
    exception.
    """
    if not isinstance(text, str):
        return None
    if len(text) > _MAX_INPUT_LEN:
        text = text[:_MAX_INPUT_LEN]

    try:
        compiled = (
            pattern
            if isinstance(pattern, re.Pattern)
            else safe_compile(pattern, flags)
        )
    except UnsafePatternError as exc:
        LOGGER.warning("safe_search: rejected unsafe pattern (%s)", exc)
        return None
    except re.error as exc:
        LOGGER.warning("safe_search: invalid regex syntax (%s)", exc)
        return None

    result_box: list[re.Match[str] | None] = [None]
    exc_box: list[BaseException | None] = [None]

    def _runner() -> None:
        try:
            result_box[0] = compiled.search(text)
        except BaseException as e:  # noqa: BLE001 — propagate via box
            exc_box[0] = e

    worker = threading.Thread(target=_runner, name="egisai-regex-safe", daemon=True)
    worker.start()
    worker.join(timeout=timeout_s)
    if worker.is_alive():
        LOGGER.warning(
            "safe_search: regex exceeded %.0f ms timeout; treating as no match. "
            "Rewrite the pattern to eliminate catastrophic backtracking.",
            timeout_s * 1000,
        )
        return None
    if exc_box[0] is not None:
        LOGGER.warning(
            "safe_search: regex match raised %s; treating as no match",
            exc_box[0].__class__.__name__,
        )
        return None
    return result_box[0]
