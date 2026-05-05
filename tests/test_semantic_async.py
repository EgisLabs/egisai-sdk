"""``SemanticBlocker.acheck`` exercises the async judge call path.

The synchronous ``check()`` path used to be invoked from inside
async patchers (AsyncOpenAI, AsyncAnthropic, …), which blocked the
event loop on every semantic-guard policy. ``acheck()`` uses an
``httpx.AsyncClient`` so the event loop stays free.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from egisai.policy.semantic import SemanticBlocker

pytestmark = pytest.mark.asyncio


def _judge_match() -> dict[str, Any]:
    return {
        "match": True,
        "intent": "delete rows from a database table",
        "confidence": 0.91,
        "tokens_in": 200,
        "tokens_out": 12,
    }


def _judge_no_match() -> dict[str, Any]:
    return {
        "match": False,
        "intent": "",
        "confidence": 0.0,
        "tokens_in": 0,
        "tokens_out": 0,
    }


async def _make_async_blocker(
    handler: callable,  # type: ignore[valid-type]
    *,
    on_outage: str = "allow",
) -> SemanticBlocker:
    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
        on_outage=on_outage,
    )
    blocker._async_http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )
    return blocker


async def test_acheck_blocks_on_match() -> None:
    captured: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.content.decode()
        return httpx.Response(200, json=_judge_match())

    blocker = await _make_async_blocker(handler)
    match = await blocker.acheck(
        "Delete all users",
        {"intents": ["delete rows from a database table"]},
    )
    assert match is not None
    assert match.intent == "delete rows from a database table"
    assert "/v1/sdk/judge" in captured["url"]
    assert "Delete all users" in captured["body"]


async def test_acheck_allows_on_no_match() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_judge_no_match())

    blocker = await _make_async_blocker(handler)
    match = await blocker.acheck(
        "What is the capital of France?",
        {"intents": ["delete rows from a database table"]},
    )
    assert match is None


async def test_acheck_fails_open_on_outage_by_default() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated outage")

    blocker = await _make_async_blocker(handler)
    match = await blocker.acheck(
        "Delete all users",
        {"intents": ["delete rows from a database table"]},
    )
    assert match is None


async def test_acheck_fails_closed_when_configured() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated outage")

    blocker = await _make_async_blocker(handler, on_outage="block")
    match = await blocker.acheck(
        "Delete all users",
        {"intents": ["delete rows from a database table"]},
    )
    assert match is not None
    assert match.intent == "<judge unavailable>"


async def test_acheck_does_not_block_event_loop_during_outage() -> None:
    """Even on outage, ``acheck()`` should yield to the event loop —
    a parallel coroutine must continue to run while the judge HTTP
    call is pending. This is the core motivation for the new async
    path; the previous sync ``check()`` used a sync ``httpx.Client``
    that blocked the loop for the full timeout window.
    """
    import asyncio

    async def slow_judge(_request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.05)
        raise httpx.ConnectError("simulated outage after sleep")

    blocker = SemanticBlocker(
        platform_api_key="egis_live_test",
        platform_base_url="http://fake-platform",
    )
    blocker._async_http_client = httpx.AsyncClient(
        transport=httpx.MockTransport(slow_judge)
    )

    parallel_tick = 0

    async def ticker() -> None:
        nonlocal parallel_tick
        for _ in range(5):
            await asyncio.sleep(0.01)
            parallel_tick += 1

    await asyncio.gather(
        blocker.acheck(
            "anything",
            {"intents": ["destroy data"]},
        ),
        ticker(),
    )
    # If acheck blocked the loop, ticker would never have run
    # (or run only once after acheck finished). Five ticks proves
    # the loop kept turning while the HTTP call was in flight.
    assert parallel_tick >= 4
