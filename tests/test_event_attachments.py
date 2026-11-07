"""``build_event`` stamps file metadata — format + size, never content.

The dashboard's "Files uploaded" display reads the ``attachments`` key
on the audit event. These tests pin that it carries metadata only (no
filename, no bytes) and never breaks the call path on a bad payload.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest

import egisai
from egisai._events import build_event


@pytest.fixture
def initialised(fake_backend: Any) -> None:
    egisai.init(
        api_key="egis_live_x",
        app="attach-test",
        env="t",
        base_url="http://fake",
        enable_sse=False,
    )


def _png_data_url() -> str:
    # 1x1 transparent PNG, enough bytes to be a real decode.
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
    )
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def test_text_only_payload_has_no_attachments_key(initialised: None) -> None:
    ev = build_event(
        source="openai",
        target="chat",
        payload={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert "attachments" not in ev


def test_image_part_lands_as_format_and_size(initialised: None) -> None:
    ev = build_event(
        source="openai",
        target="chat",
        payload={
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {
                            "type": "image_url",
                            "image_url": {"url": _png_data_url()},
                        },
                    ],
                }
            ]
        },
    )
    atts = ev.get("attachments")
    assert isinstance(atts, list) and len(atts) == 1
    entry = atts[0]
    # Metadata only — exactly these two keys, nothing else.
    assert set(entry.keys()) == {"format", "size_bytes"}
    assert entry["format"] == "png"
    assert entry["size_bytes"] > 0


def test_a_bad_payload_shape_never_raises(initialised: None) -> None:
    # A non-dict payload must fail open — no attachments, no crash.
    ev = build_event(source="openai", target="chat", payload="not a dict")
    assert "attachments" not in ev
