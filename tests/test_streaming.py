from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from mai.core.streaming import (
    StreamEvent,
    gemini_delta,
    gemini_usage,
    openai_delta,
    openai_usage,
    parse_sse_data,
)


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("data: {}", "{}"),
        ("data:{}", "{}"),
        ("data: [DONE]", "[DONE]"),
        ("", None),
        ("   ", None),
        (": keep-alive", None),
        ("event: message", None),
    ],
)
def test_parse_sse_data(line: str, expected: str | None) -> None:
    assert parse_sse_data(line) == expected


def test_openai_delta_reads_content() -> None:
    chunk = '{"choices":[{"delta":{"content":"안녕"}}]}'
    assert openai_delta(chunk) == "안녕"


def test_openai_delta_handles_role_only_chunk() -> None:
    assert openai_delta('{"choices":[{"delta":{"role":"assistant"}}]}') == ""


def test_openai_delta_falls_back_to_message() -> None:
    chunk = '{"choices":[{"message":{"content":"full"}}]}'
    assert openai_delta(chunk) == "full"


@pytest.mark.parametrize(
    "chunk",
    ["[DONE]", "", "not json", "{}", '{"choices":[]}', '{"choices":[null]}'],
)
def test_openai_delta_survives_junk(chunk: str) -> None:
    assert openai_delta(chunk) == ""


def test_openai_usage_extracted() -> None:
    assert openai_usage('{"usage":{"total_tokens":12}}') == {"total_tokens": 12}
    assert openai_usage("[DONE]") is None
    assert openai_usage('{"usage":null}') is None


def test_gemini_delta_joins_parts() -> None:
    chunk = '{"candidates":[{"content":{"parts":[{"text":"가"},{"text":"나"}]}}]}'
    assert gemini_delta(chunk) == "가나"


@pytest.mark.parametrize(
    "chunk",
    ["", "not json", "{}", '{"candidates":[]}', '{"candidates":[{"content":{}}]}'],
)
def test_gemini_delta_survives_junk(chunk: str) -> None:
    assert gemini_delta(chunk) == ""


def test_gemini_usage_extracted() -> None:
    chunk = '{"usageMetadata":{"totalTokenCount":9}}'
    assert gemini_usage(chunk) == {"totalTokenCount": 9}
    assert gemini_usage("{}") is None


def test_stream_event_defaults() -> None:
    event = StreamEvent(kind="delta", text="hi")
    assert event.provider == ""
    assert event.usage is None


def sse(*chunks: str) -> bytes:
    return "".join(f"data: {chunk}\n\n" for chunk in chunks).encode()


SSE_OK = sse(
    json.dumps({"choices": [{"delta": {"content": "안"}}]}),
    json.dumps({"choices": [{"delta": {"content": "녕"}}]}),
    json.dumps({"usage": {"total_tokens": 5}}),
    "[DONE]",
)


def _patch_router(router, monkeypatch):
    monkeypatch.setattr(router.catalog, "resolve", AsyncMock(return_value="m"))
    monkeypatch.setattr(router.health, "cooling", AsyncMock(return_value={}))
    monkeypatch.setattr(router.health, "mark_success", AsyncMock())
    monkeypatch.setattr(router.health, "mark_failure", AsyncMock(return_value=600))


@pytest.mark.asyncio
async def test_astream_yields_deltas_then_done(monkeypatch, respx_mock):
    from mai.core import router as router_module

    monkeypatch.setattr(
        router_module.MultiProviderRouter,
        "configured",
        lambda self, name: name == "groq",
    )
    monkeypatch.setattr(
        router_module.MultiProviderRouter, "_key", lambda self, name: "k"
    )
    respx_mock.post(router_module.OPENAI_ENDPOINTS["groq"]).respond(
        200, content=SSE_OK, headers={"Content-Type": "text/event-stream"}
    )

    router = router_module.MultiProviderRouter()
    _patch_router(router, monkeypatch)

    events = [event async for event in router.astream("hi", provider="groq")]

    assert [e.text for e in events if e.kind == "delta"] == ["안", "녕"]
    assert events[-1].kind == "done"
    assert events[-1].usage == {"total_tokens": 5}


@pytest.mark.asyncio
async def test_astream_falls_back_before_first_token(monkeypatch, respx_mock):
    from mai.core import router as router_module

    monkeypatch.setattr(
        router_module.MultiProviderRouter,
        "configured",
        lambda self, name: name in {"groq", "cerebras"},
    )
    monkeypatch.setattr(
        router_module.MultiProviderRouter, "_key", lambda self, name: "k"
    )
    respx_mock.post(router_module.OPENAI_ENDPOINTS["groq"]).respond(500)
    respx_mock.post(router_module.OPENAI_ENDPOINTS["cerebras"]).respond(
        200, content=SSE_OK, headers={"Content-Type": "text/event-stream"}
    )

    router = router_module.MultiProviderRouter()
    _patch_router(router, monkeypatch)

    events = [event async for event in router.astream("hi")]

    kinds = [e.kind for e in events]
    assert kinds[0] == "fallback"
    assert events[0].provider == "groq"
    assert [e.text for e in events if e.kind == "delta"] == ["안", "녕"]
    assert events[-1].provider == "cerebras"
