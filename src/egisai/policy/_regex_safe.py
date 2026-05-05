"""Defensive regex compilation + matching for operator-supplied patterns.

Two-layer protection against catastrophic backtracking (ReDoS):

1. ``safe_compile`` — rejects patterns that are too long or contain
   nested-quantifier shapes responsible for most real-world ReDoS
   reports.
2. ``safe_search`` — runs the match on a background daemon thread
   with a wall-clock timeout (default 50 ms); on timeout, returns
   "no match" so the customer's request is unblocked.
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Optional

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
    return re.compile(pattern, flags)


def safe_search(
    pattern: str | re.Pattern[str],
    text: str,
    flags: int = 0,
    *,
    timeout_s: float = DEFAULT_REGEX_TIMEOUT_S,
) -> Optional[re.Match[str]]:
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

    result_box: list[Optional[re.Match[str]]] = [None]
    exc_box: list[Optional[BaseException]] = [None]

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
