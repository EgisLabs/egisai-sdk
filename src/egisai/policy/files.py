"""File / attachment governance — verdict only, never sanitize bytes.

Unlike text governance (which can mask PII in place), file governance is
**report-only**: a file is either ``allow`` or ``block`` with a human
reason. We never rewrite file contents.

Two tiers, in order (mirrors the deterministic-first contract in
security-and-compliance.mdc rule 2):

1. **Deterministic, local, pure-Python** — format allow/block lists and a
   max-size cap. Runs everywhere (SDK pip-install, gateway, egress) with
   no network, no dependency, no bytes required (filename + size are
   enough). This is what lets the browser extension block ``.exe`` the
   instant it's attached.

2. **Content inspection** — OCR + parse + PII/safety scan of the file
   bytes. This is heavy and lives in the standalone file worker, reached
   through an installed :class:`RemoteFileEngine` (the control plane
   installs it; the bare SDK has none). When no engine is installed the
   content tier is skipped — the deterministic tier still enforces format
   and size. When an engine IS installed and it errors, the call
   **fails closed** (block) by default: the whole point is to never let
   an un-scanned file through.

Public API (stable):

* :class:`Attachment`  — one file seen on a call (metadata + optional bytes)
* :class:`FileFinding` — one reason a file was blocked
* :func:`evaluate`     — run both tiers against a policy config
* :func:`normalize_format` — filename/mime → canonical lowercase extension
* :func:`set_remote_engine` / :func:`get_remote_engine`
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Protocol

LOGGER = logging.getLogger("egisai.policy.files")


# ── Public dataclasses ──────────────────────────────────────────────


@dataclass(frozen=True)
class Attachment:
    """One file/attachment carried on a governed call.

    ``fmt`` is the canonical lowercase extension with no leading dot
    (``"pdf"``, ``"png"``, ``"docx"``). ``content`` is the raw bytes when
    the control point has them in hand (SDK payload, gateway body);
    ``None`` when only metadata is available (e.g. the browser extension's
    first-pass, where bytes are uploaded separately). ``size_bytes`` is 0
    when unknown.
    """

    name: str
    mime: str
    fmt: str
    size_bytes: int = 0
    content: bytes | None = None

    @property
    def effective_size(self) -> int:
        """Best-known size: the declared size, else the byte length."""
        if self.size_bytes:
            return self.size_bytes
        return len(self.content) if self.content is not None else 0


@dataclass(frozen=True)
class FileFinding:
    """One reason a file was blocked.

    ``name`` / ``fmt`` identify the file; ``reason_code`` is a stable
    machine code; ``message`` is the operator-facing sentence. Never
    carries raw file content (security-and-compliance.mdc rule 1).
    """

    name: str
    fmt: str
    reason_code: str
    message: str


# ── Installable content-inspection engine ───────────────────────────


class RemoteFileEngine(Protocol):
    """Structural type for the heavy content-inspection backend.

    Implemented by the control plane's file-worker client. Given a list
    of attachments that carry bytes, returns one :class:`FileFinding` per
    file that should be blocked (empty list ⇒ all clean). MUST raise on
    any transport/protocol error so :func:`evaluate` can fail closed.
    """

    def scan(
        self,
        attachments: list[Attachment],
        *,
        max_size_bytes: int | None = None,
    ) -> list[FileFinding]: ...


_remote_engine: RemoteFileEngine | None = None


def set_remote_engine(engine: RemoteFileEngine | None) -> None:
    """Install (or clear, with ``None``) the content-inspection engine.

    Call once at process start-up. Not part of the public pip-install
    contract — this is for a trusted caller (our control plane) that owns
    the file worker. ``None`` restores deterministic-only behavior.
    """
    global _remote_engine
    _remote_engine = engine
    if engine is not None:
        LOGGER.info(
            "[egisai] remote file engine installed: %r", type(engine).__name__
        )
    else:
        LOGGER.info("[egisai] remote file engine cleared — format/size only.")


def get_remote_engine() -> RemoteFileEngine | None:
    """Return the installed content-inspection engine, or ``None``."""
    return _remote_engine


# ── Helpers ─────────────────────────────────────────────────────────

# A small MIME → extension map for the common cases where a filename has
# no extension but the content-type is known (browser uploads, API image
# parts). Deliberately tiny — the filename extension is the primary
# signal; this is only a fallback.
_MIME_TO_EXT: dict[str, str] = {
    "application/pdf": "pdf",
    "image/png": "png",
    "image/jpeg": "jpg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
    "image/tiff": "tiff",
    "image/bmp": "bmp",
    "text/plain": "txt",
    "text/csv": "csv",
    "text/markdown": "md",
    "application/zip": "zip",
    "application/x-zip-compressed": "zip",
    "application/msword": "doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",  # noqa: E501
    "application/vnd.ms-excel": "xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",  # noqa: E501
    "application/vnd.ms-powerpoint": "ppt",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",  # noqa: E501
    "application/json": "json",
    "application/x-msdownload": "exe",
    "application/octet-stream": "",
}


def normalize_format(name: str | None, mime: str | None = None) -> str:
    """Return the canonical lowercase extension with no leading dot.

    Prefers the filename extension; falls back to the MIME map. Returns
    ``""`` when neither yields a recognizable format.
    """
    if name:
        ext = os.path.splitext(name)[1]
        if ext:
            return ext[1:].lower().strip()
    if mime:
        return _MIME_TO_EXT.get(mime.split(";")[0].strip().lower(), "")
    return ""


def _norm_set(raw: object) -> set[str]:
    """Normalize a config list of extensions to a lowercase, no-dot set."""
    if not isinstance(raw, list):
        return set()
    out: set[str] = set()
    for item in raw:
        if not isinstance(item, str):
            continue
        cleaned = item.strip().lstrip(".").lower()
        if cleaned:
            out.add(cleaned)
    return out


# ── The evaluator ───────────────────────────────────────────────────


def evaluate(
    attachments: list[Attachment] | tuple[Attachment, ...],
    config: dict[str, object],
) -> FileFinding | None:
    """Run file governance against one policy config. First block wins.

    ``config`` keys (all optional):

    * ``blocked_formats``  — extensions always refused (``["exe","zip"]``)
    * ``allowed_formats``  — if set, ONLY these are permitted; anything
      else is refused
    * ``max_size_bytes``   — per-file size cap
    * ``inspect_formats``  — extensions sent for content inspection; empty
      ⇒ inspect every file that carries bytes
    * ``on_error``         — ``"block"`` (default, fail closed) or
      ``"allow"`` when the content engine errors
    * ``message``          — override the operator-facing block message

    Returns the first :class:`FileFinding` that blocks, or ``None`` when
    every file is allowed.
    """
    if not attachments:
        return None

    blocked = _norm_set(config.get("blocked_formats"))
    allowed = _norm_set(config.get("allowed_formats"))
    inspect = _norm_set(config.get("inspect_formats"))
    max_size_raw = config.get("max_size_bytes")
    max_size = max_size_raw if isinstance(max_size_raw, int) and max_size_raw > 0 else None
    custom_message = config.get("message")
    custom_message = custom_message if isinstance(custom_message, str) else None

    # ── Tier 1: deterministic format + size (pure, no network) ──
    for att in attachments:
        fmt = att.fmt or normalize_format(att.name, att.mime)
        label = att.name or (f"a .{fmt} file" if fmt else "a file")
        if blocked and fmt in blocked:
            return FileFinding(
                name=att.name,
                fmt=fmt,
                reason_code="file_format_blocked",
                message=custom_message
                or f"Blocked: the file type .{fmt} is not permitted ({label}).",
            )
        if allowed and fmt not in allowed:
            return FileFinding(
                name=att.name,
                fmt=fmt,
                reason_code="file_format_not_allowed",
                message=custom_message
                or (
                    f"Blocked: {label} is a .{fmt or 'unknown'} file, which is "
                    "not on the allowed list."
                ),
            )
        if max_size is not None and att.effective_size > max_size:
            mb = att.effective_size / (1024 * 1024)
            return FileFinding(
                name=att.name,
                fmt=fmt,
                reason_code="file_too_large",
                message=custom_message
                or f"Blocked: {label} is too large ({mb:.1f} MB).",
            )

    # ── Tier 2: content inspection (OCR + parse + scan) via engine ──
    engine = _remote_engine
    if engine is None:
        return None

    to_inspect = [
        att
        for att in attachments
        if att.content is not None
        and (not inspect or (att.fmt or normalize_format(att.name, att.mime)) in inspect)
    ]
    if not to_inspect:
        return None

    try:
        findings = engine.scan(to_inspect, max_size_bytes=max_size)
    except Exception as exc:  # noqa: BLE001 — fail closed on files by default
        on_error = config.get("on_error", "block")
        if on_error == "allow":
            LOGGER.warning(
                "[egisai] file content scan unavailable; failing OPEN per "
                "policy config: %s",
                exc,
            )
            return None
        LOGGER.warning(
            "[egisai] file content scan unavailable; failing CLOSED (block): %s",
            exc,
        )
        first = to_inspect[0]
        return FileFinding(
            name=first.name,
            fmt=first.fmt,
            reason_code="file_scan_unavailable",
            message=custom_message
            or "Blocked: the file could not be verified as safe.",
        )

    if findings:
        f = findings[0]
        if custom_message:
            return FileFinding(
                name=f.name,
                fmt=f.fmt,
                reason_code=f.reason_code,
                message=custom_message,
            )
        return f
    return None
