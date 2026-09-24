from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Literal

EventKind = Literal["delta", "fallback", "done"]


@dataclass(frozen=True, slots=True)
class StreamEvent:
    kind: EventKind
    text: str = ""
    provider: str = ""
    model: str = ""
    detail: str = ""
    usage: dict[str, Any] | None = field(default=None)


def parse_sse_data(line: str) -> str | None:
    """Return the payload of an SSE data line, or None if it is not one."""
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if not stripped.startswith("data:"):
        return None
    return stripped[5:].strip()


def openai_delta(chunk: str) -> str:
    """Extract incremental text from one OpenAI-compatible SSE payload."""
    if not chunk or chunk == "[DONE]":
        return ""
    try:
        payload = json.loads(chunk)
    except json.JSONDecodeError:
        return ""

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""

    first = choices[0]
    if not isinstance(first, dict):
        return ""

    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content

    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content

    return ""


def openai_usage(chunk: str) -> dict[str, Any] | None:
    if not chunk or chunk == "[DONE]":
        return None
    try:
        payload = json.loads(chunk)
    except json.JSONDecodeError:
        return None
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else None


def gemini_delta(chunk: str) -> str:
    """Extract incremental text from one Gemini streaming payload."""
    if not chunk:
        return ""
    try:
        payload = json.loads(chunk)
    except json.JSONDecodeError:
        return ""

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return ""

    content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
    if not isinstance(content, dict):
        return ""

    parts = content.get("parts")
    if not isinstance(parts, list):
        return ""

    pieces = [
        part["text"]
        for part in parts
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]
    return "".join(pieces)


def gemini_usage(chunk: str) -> dict[str, Any] | None:
    if not chunk:
        return None
    try:
        payload = json.loads(chunk)
    except json.JSONDecodeError:
        return None
    usage = payload.get("usageMetadata")
    return usage if isinstance(usage, dict) else None


async def iter_sse_lines(lines: AsyncIterator[str]) -> AsyncIterator[str]:
    """Yield SSE data payloads from an async line iterator."""
    async for line in lines:
        payload = parse_sse_data(line)
        if payload is not None:
            yield payload
