"""The ``file_scan`` policy kind: verdict-only file/attachment governance.

Covers the three layers that make up the feature in the SDK:

1. ``egisai.policy.files`` — the two-tier evaluator (deterministic
   format/size, then content inspection via an installed engine).
2. ``egisai._evaluator.extract_payload_attachments`` — pulling
   attachments out of OpenAI / Anthropic / Gemini multimodal payloads.
3. ``engine.evaluate_policies`` — the ``file_scan`` kind wired into
   Phase 1 and returning a block verdict.
"""
from __future__ import annotations

import base64

import pytest

from egisai._evaluator import extract_payload_attachments
from egisai.policy import files as file_scanner
from egisai.policy.engine import (
    PolicyContext,
    PolicyRule,
    evaluate_policies,
)
from egisai.policy.files import Attachment, FileFinding


def _rule(config: dict) -> PolicyRule:
    return PolicyRule(
        id="1",
        name="file-governance",
        type="file_scan",
        tenant=None,
        config=config,
    )


def _ctx(attachments: tuple) -> PolicyContext:
    return PolicyContext(
        tenant="t",
        model="gpt-4o",
        prompt_text="here is a file",
        prompt_chars=14,
        stream=False,
        attachments=attachments,
    )


@pytest.fixture(autouse=True)
def _clear_engine():
    """Every test starts with no content-inspection engine installed."""
    file_scanner.set_remote_engine(None)
    yield
    file_scanner.set_remote_engine(None)


# ── normalize_format ────────────────────────────────────────────────


def test_normalize_format_from_filename():
    assert file_scanner.normalize_format("report.PDF") == "pdf"
    assert file_scanner.normalize_format("archive.tar.gz") == "gz"
    assert file_scanner.normalize_format("noext") == ""


def test_normalize_format_from_mime_fallback():
    assert file_scanner.normalize_format("", "image/png") == "png"
    assert file_scanner.normalize_format(None, "application/pdf") == "pdf"


# ── Tier 1: deterministic format / size ─────────────────────────────


def test_blocked_format_is_refused():
    att = Attachment(name="virus.exe", mime="", fmt="exe", size_bytes=10)
    finding = file_scanner.evaluate([att], {"blocked_formats": ["exe", "zip"]})
    assert finding is not None
    assert finding.reason_code == "file_format_blocked"


def test_allowed_list_refuses_everything_else():
    att = Attachment(name="data.csv", mime="", fmt="csv", size_bytes=10)
    finding = file_scanner.evaluate([att], {"allowed_formats": ["pdf", "png"]})
    assert finding is not None
    assert finding.reason_code == "file_format_not_allowed"


def test_allowed_list_permits_listed():
    att = Attachment(name="ok.pdf", mime="", fmt="pdf", size_bytes=10)
    assert file_scanner.evaluate([att], {"allowed_formats": ["pdf"]}) is None


def test_size_cap_blocks_large_file():
    att = Attachment(name="big.pdf", mime="", fmt="pdf", size_bytes=5_000_000)
    finding = file_scanner.evaluate([att], {"max_size_bytes": 1_000_000})
    assert finding is not None
    assert finding.reason_code == "file_too_large"


def test_no_attachments_is_allow():
    assert file_scanner.evaluate([], {"blocked_formats": ["exe"]}) is None


def test_custom_message_overrides():
    att = Attachment(name="x.exe", mime="", fmt="exe")
    finding = file_scanner.evaluate(
        [att], {"blocked_formats": ["exe"], "message": "No executables."}
    )
    assert finding is not None
    assert finding.message == "No executables."


# ── Tier 2: content inspection via installed engine ─────────────────


class _FakeEngine:
    """Flags any attachment whose bytes contain ``b'SSN'``."""

    def scan(self, attachments, *, max_size_bytes=None):
        out = []
        for a in attachments:
            if a.content and b"SSN" in a.content:
                out.append(
                    FileFinding(
                        name=a.name,
                        fmt=a.fmt,
                        reason_code="pii_in_file",
                        message=f"Blocked: {a.name} contains an SSN.",
                    )
                )
        return out


def test_content_engine_blocks_on_finding():
    file_scanner.set_remote_engine(_FakeEngine())
    att = Attachment(name="form.pdf", mime="", fmt="pdf", content=b"name SSN 123")
    finding = file_scanner.evaluate([att], {})
    assert finding is not None
    assert finding.reason_code == "pii_in_file"


def test_content_engine_allows_clean_file():
    file_scanner.set_remote_engine(_FakeEngine())
    att = Attachment(name="clean.pdf", mime="", fmt="pdf", content=b"hello world")
    assert file_scanner.evaluate([att], {}) is None


def test_no_engine_skips_content_inspection():
    # No engine installed: format/size enforced, content tier skipped
    # (fail-open on availability for content, per plan).
    att = Attachment(name="form.pdf", mime="", fmt="pdf", content=b"SSN 123")
    assert file_scanner.evaluate([att], {}) is None


def test_inspect_formats_scopes_content_scan():
    file_scanner.set_remote_engine(_FakeEngine())
    att = Attachment(name="note.txt", mime="", fmt="txt", content=b"SSN 123")
    # txt not in inspect_formats → not sent to engine
    assert file_scanner.evaluate([att], {"inspect_formats": ["pdf"]}) is None


class _BrokenEngine:
    def scan(self, attachments, *, max_size_bytes=None):
        raise RuntimeError("worker down")


def test_engine_error_fails_closed_by_default():
    file_scanner.set_remote_engine(_BrokenEngine())
    att = Attachment(name="form.pdf", mime="", fmt="pdf", content=b"data")
    finding = file_scanner.evaluate([att], {})
    assert finding is not None
    assert finding.reason_code == "file_scan_unavailable"


def test_engine_error_can_fail_open_when_configured():
    file_scanner.set_remote_engine(_BrokenEngine())
    att = Attachment(name="form.pdf", mime="", fmt="pdf", content=b"data")
    assert file_scanner.evaluate([att], {"on_error": "allow"}) is None


# ── extract_payload_attachments ─────────────────────────────────────


def _data_url(mime: str, data: bytes) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def test_extract_openai_chat_image():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is this"},
                    {
                        "type": "image_url",
                        "image_url": {"url": _data_url("image/png", b"PNGDATA")},
                    },
                ],
            }
        ]
    }
    atts = extract_payload_attachments(payload)
    assert len(atts) == 1
    assert atts[0].fmt == "png"
    assert atts[0].content == b"PNGDATA"


def test_extract_openai_responses_input_file():
    payload = {
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": "report.pdf",
                        "file_data": _data_url("application/pdf", b"%PDF-1.4"),
                    }
                ],
            }
        ]
    }
    atts = extract_payload_attachments(payload)
    assert len(atts) == 1
    assert atts[0].fmt == "pdf"
    assert atts[0].name == "report.pdf"


def test_extract_anthropic_image():
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64.b64encode(b"JPEG").decode(),
                        },
                    }
                ],
            }
        ]
    }
    atts = extract_payload_attachments(payload)
    assert len(atts) == 1
    assert atts[0].fmt == "jpg"
    assert atts[0].content == b"JPEG"


def test_extract_gemini_inline_data():
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": base64.b64encode(b"G").decode(),
                        }
                    }
                ]
            }
        ]
    }
    atts = extract_payload_attachments(payload)
    assert len(atts) == 1
    assert atts[0].fmt == "png"


def test_extract_no_attachments_for_text_only():
    payload = {"messages": [{"role": "user", "content": "just text"}]}
    assert extract_payload_attachments(payload) == []


# ── engine.evaluate_policies wiring ─────────────────────────────────


def test_evaluate_policies_blocks_on_blocked_format():
    rule = _rule({"blocked_formats": ["exe"]})
    ctx = _ctx((Attachment(name="x.exe", mime="", fmt="exe"),))
    decision = evaluate_policies([rule], ctx)
    assert decision.verdict == "block"
    assert decision.reason_code == "file_format_blocked"


def test_evaluate_policies_allows_clean_format():
    rule = _rule({"blocked_formats": ["exe"]})
    ctx = _ctx((Attachment(name="ok.pdf", mime="", fmt="pdf"),))
    decision = evaluate_policies([rule], ctx)
    assert decision.verdict == "allow"


def test_evaluate_policies_content_scan_blocks():
    file_scanner.set_remote_engine(_FakeEngine())
    rule = _rule({})
    ctx = _ctx((Attachment(name="f.pdf", mime="", fmt="pdf", content=b"SSN"),))
    decision = evaluate_policies([rule], ctx)
    assert decision.verdict == "block"
    assert decision.reason_code == "pii_in_file"
