"""Background loader for the Presidio analyzer + spaCy NER model.

Why this is its own module:

* Loading the spaCy ``en_core_web_lg`` NER model takes 1–3 s (and on a
  fresh install we additionally download ~750 MB the first time the
  SDK runs). Doing that synchronously inside ``egisai.init()`` would
  break the SDK's "instant first call" contract from
  ``sdk-design-philosophy.mdc``.
* The hot path (every model call the customer makes) needs a fast,
  thread-safe accessor that returns the analyzer if it's ready and
  ``None`` otherwise. ``None`` triggers the regex fallback in
  :mod:`egisai.policy.pii`, so PII protection is **never** off — only
  the NER-driven entities (names, locations, GDPR special-category
  text) are temporarily unavailable until the model is warm.
* Fail-open semantics. If the customer is in a sealed environment
  with no internet, ``spacy.cli.download`` will fail; we surface a
  single warning to stderr and continue running with the regex
  fallback. The user's ``client.messages.create(...)`` is never
  blocked by our model setup.

Lifetime:

* ``prime_analyzer_async()`` is called once from ``egisai.init()``;
  subsequent calls are no-ops (idempotent).
* The first call ``spawn``s a daemon thread that does the slow work
  off the user's call path.
* When the thread finishes (success or failure), it stamps
  ``_state`` so subsequent ``try_get_analyzer()`` calls can return
  immediately without locking.
"""
from __future__ import annotations

import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from presidio_analyzer import AnalyzerEngine

LOGGER = logging.getLogger("egisai.pii")

# Pin to a model size that produces frontier-quality NER for English.
# ``en_core_web_lg`` is 750 MB on disk and is the default for every
# customer per the runtime PII upgrade. The size is intentional: the
# user explicitly chose "super strong by default from day one."
_SPACY_MODEL_NAME = "en_core_web_lg"


# ── Module-level state ─────────────────────────────────────────────


@dataclass
class _AnalyzerState:
    """The single source of truth for warm-up status.

    Only ``_lock`` and ``_state`` are mutated after import. The
    daemon thread that does the slow work writes through the lock;
    the hot path reads without the lock (worst case it sees a
    one-instruction-stale view, which is benign because the analyzer
    is only ever written once per process).
    """

    # Presidio analyzer instance once loaded (and not failed).
    analyzer: AnalyzerEngine | None = None
    # ``True`` while the background thread is doing the heavy lift.
    loading: bool = False
    # Set to ``True`` after the thread terminates, regardless of
    # outcome. Hot path checks this to know "we already tried; don't
    # ask again until the process restarts".
    settled: bool = False
    # Captured exception for diagnostics; ``None`` on success.
    error: BaseException | None = None
    # ``True`` once we've kicked off a load (idempotency guard).
    primed: bool = False
    # Signalled exactly once when ``_load_in_background`` returns
    # (success OR failure). Lets callers ``wait_for_warm()`` without
    # busy-polling. Defined here rather than as a module-level
    # singleton so ``reset_for_tests()`` rebuilds it cleanly with
    # the rest of the state.
    settle_event: threading.Event = field(default_factory=threading.Event)


_state = _AnalyzerState()
_lock = threading.Lock()


# ── Public API ──────────────────────────────────────────────────────


def prime_analyzer_async(*, quiet: bool = False) -> None:
    """Start loading the analyzer in a daemon thread, idempotently.

    Called once from ``egisai.init()``. Returns immediately. The hot
    path checks ``try_get_analyzer()`` on every PII scan and falls
    back to the regex chain whenever it returns ``None``.

    ``quiet`` mirrors the same flag on ``egisai.init()`` — when set,
    we don't print the friendly "downloading PII model" notice on
    first run, so containerized / pipelines stay silent.
    """
    with _lock:
        if _state.primed:
            return
        _state.primed = True
        _state.loading = True

    thread = threading.Thread(
        target=_load_in_background,
        kwargs={"quiet": quiet},
        name="egisai-pii-loader",
        daemon=True,
    )
    thread.start()


def try_get_analyzer() -> AnalyzerEngine | None:
    """Return the analyzer if warm, ``None`` if still loading or failed.

    Hot-path safe: a single attribute read without acquiring the
    lock. The slot is only ever assigned-once (None → AnalyzerEngine
    instance) so a stale read is safe.
    """
    return _state.analyzer


def is_settled() -> bool:
    """``True`` once the background thread has finished (success or fail)."""
    return _state.settled


def is_loading() -> bool:
    """``True`` while the background thread is still working."""
    return _state.loading and not _state.settled


def last_error() -> BaseException | None:
    """The exception that ended the load thread, if any. ``None`` on success."""
    return _state.error


def wait_for_warm(timeout_secs: float) -> bool:
    """Block up to ``timeout_secs`` for the analyzer to settle.

    Returns ``True`` if the analyzer is ready (and ``try_get_analyzer()``
    will return a live instance), ``False`` on timeout OR if the load
    thread settled with an error. The False case is "the regex fallback
    will run" — callers should treat that as expected, not as a bug.

    Cheap, event-based: the daemon thread sets a ``threading.Event``
    in its ``finally`` block, so this call wakes up the instant the
    load completes rather than busy-polling. ``timeout_secs <= 0``
    is a non-blocking probe (returns immediately).

    Designed to be called from the SDK's policy entry points exactly
    once per process. Higher-level code (the input-phase gate in
    ``egisai._evaluator``) is responsible for the one-shot policy:
    waiting *here* is always safe, but waiting on every call defeats
    the point of the daemon-loader pattern.
    """
    # Fast path — already warm or already failed. Avoids touching
    # the Event at all on the steady-state hot path.
    if _state.settled:
        return _state.analyzer is not None
    if timeout_secs <= 0:
        return False
    # ``wait()`` returns True if the flag is set within the timeout,
    # False on timeout. Either way we re-read the analyzer slot to
    # report the true outcome (load could have failed AFTER the
    # event fired).
    _state.settle_event.wait(timeout_secs)
    return _state.analyzer is not None


def reset_for_tests() -> None:
    """Wipe loader state so tests can drive a fresh load.

    Intended for the SDK test suite only — production callers should
    rely on ``prime_analyzer_async`` being idempotent.
    """
    global _state
    _state = _AnalyzerState()


# ── Implementation ──────────────────────────────────────────────────


def _load_in_background(*, quiet: bool) -> None:
    """Body of the daemon thread. Best-effort, fail-open."""
    try:
        analyzer = _build_analyzer(quiet=quiet)
        with _lock:
            _state.analyzer = analyzer
            _state.error = None
    except BaseException as exc:  # noqa: BLE001 - intentionally broad; fail-open
        LOGGER.warning(
            "[egisai] PII NER analyzer failed to load (%s: %s) — "
            "falling back to regex+checksum detection. "
            "Names / locations / GDPR special-category text will not "
            "be flagged until this is fixed.",
            exc.__class__.__name__,
            exc,
        )
        # Surface the warning on the operator's dashboard via a
        # fire-and-forget telemetry POST. The send itself is
        # bulletproof (any failure is swallowed silently inside
        # ``post_startup_warning``), but we still wrap the call in
        # a defensive try/except so even a future regression in
        # the import path can't break the daemon thread's
        # fail-open contract. See ``_backend.post_startup_warning``
        # for the privacy / reliability contract this call honors.
        try:
            from egisai._backend import post_startup_warning

            post_startup_warning("pii_ner_loader_failed", exc)
        except Exception:  # noqa: BLE001 - belt-and-suspenders, fail-open
            pass
        with _lock:
            _state.error = exc
    finally:
        with _lock:
            _state.loading = False
            _state.settled = True
        # Signal AFTER the lock release so any thread parked in
        # ``wait_for_warm`` wakes up to a fully-published state.
        # ``set()`` is idempotent — even if the thread somehow ran
        # twice (it can't; the ``primed`` guard prevents it), the
        # extra signal is a no-op.
        _state.settle_event.set()


def _build_analyzer(*, quiet: bool) -> AnalyzerEngine:
    """Construct a Presidio analyzer with our custom recognizers.

    Performs three steps:
      1. ensure ``en_core_web_lg`` is installed (download if missing);
      2. instantiate Presidio's ``AnalyzerEngine`` configured for that model;
      3. register our four custom Egis recognizers on the analyzer's registry.

    Each step's failure is fatal for the loader (the daemon thread
    catches and swallows). The hot path then keeps using the regex
    fallback.
    """
    _ensure_spacy_model_present(quiet=quiet)

    # Imports are scoped here so the cost (~hundreds of ms of pyc
    # loading) is paid in the daemon thread, not on ``egisai.init()``.
    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    from egisai.policy._pii_recognizers import register_custom_recognizers

    nlp_configuration: dict[str, Any] = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": _SPACY_MODEL_NAME}],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()

    analyzer = AnalyzerEngine(
        nlp_engine=nlp_engine,
        supported_languages=["en"],
    )

    register_custom_recognizers(analyzer.registry)

    # No success line here: the SDK's main ``✓ [egisai] active …``
    # banner already confirms the SDK is alive, and the PII engine
    # is an implementation detail the operator doesn't need to see
    # on every process start. We still surface failures (a warning
    # from ``_load_in_background`` when the daemon thread can't load
    # the analyzer) so misconfigurations remain visible.
    return analyzer


def _ensure_spacy_model_present(*, quiet: bool) -> None:
    """Check for ``en_core_web_lg``; download it if missing.

    spaCy ships models as standalone wheels on Explosion's GitHub
    releases. ``spacy.cli.download`` runs ``pip install`` under the
    hood — the same mechanism every spaCy production deployment
    uses. We invoke it once at startup if the model isn't installed
    so customers don't have to remember an extra step after
    ``pip install egisai``.

    Raises if the download fails — caller logs and falls back.
    """
    import spacy

    if spacy.util.is_package(_SPACY_MODEL_NAME):
        return

    if not quiet:
        # Loud + friendly: this only happens on a fresh install, and
        # the user is going to wait 30–90 s for a 750 MB download.
        # Telling them what's happening is way better than a silent
        # delay that looks like a hang.
        print(
            "⚠ [egisai] downloading PII NER model (one-time, ~750 MB) — "
            f"{_SPACY_MODEL_NAME}. Until it finishes, name / location "
            "detection is unavailable; checksum-validated detectors "
            "(SSN, credit card, IBAN, passport, …) keep running.",
            file=sys.stderr,
            flush=True,
        )

    # ``spacy.cli.download`` exits with a non-zero status on failure
    # rather than raising; capture and translate to an exception so
    # the daemon thread's outer ``try`` can swallow it consistently.
    try:
        from spacy.cli.download import download as spacy_download

        spacy_download(_SPACY_MODEL_NAME, False, False)
    except SystemExit as exc:  # pip install failed inside spacy.cli
        raise RuntimeError(
            f"spaCy model {_SPACY_MODEL_NAME!r} could not be downloaded "
            f"(pip exit code {exc.code}). The SDK will keep running "
            "with regex+checksum detection only."
        ) from exc

    # Sanity check: confirm spaCy now sees it. This guards against
    # network races (e.g. partial downloads) that complete without
    # raising but leave the package half-installed.
    if not spacy.util.is_package(_SPACY_MODEL_NAME):
        raise RuntimeError(
            f"spaCy model {_SPACY_MODEL_NAME!r} reports as missing "
            "after download claimed success."
        )
