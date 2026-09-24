import json

import httpx
import pytest
import respx

from mai.core.health import HealthStore
from mai.core.policy import Approval, Policy
from mai.core.router import MODELS, MultiProviderRouter, RouterError
from mai.core.tools import ToolExecutor


def _router(tmp_path):
    router = MultiProviderRouter(health=HealthStore(tmp_path / "health.db"))
    router.configured = lambda provider: True
    router._key = lambda provider: "test-key"

    async def resolve(client, provider, configured):
        return configured

    router.catalog.resolve = resolve
    return router


@pytest.mark.asyncio
async def test_openai_tool_call_round_trip(tmp_path) -> None:
    router = _router(tmp_path)
    target = tmp_path / "note.txt"
    target.write_text("hello", encoding="utf-8")
    executor = ToolExecutor(Policy(tmp_path, Approval.READ_ONLY))
    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(endpoint).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call-1",
                                            "type": "function",
                                            "function": {
                                                "name": "fs_read",
                                                "arguments": '{"path":"note.txt"}',
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    },
                ),
                httpx.Response(
                    200,
                    json={
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "hello",
                                }
                            }
                        ]
                    },
                ),
            ]
        )

        result = await router.ask(
            "read the note",
            provider="groq",
            tool_executor=executor,
            use_health=False,
        )

    assert result.text == "hello"
    assert route.call_count == 2
    second_messages = route.calls[1].request.content
    assert second_messages is not None
    second_body = json.loads(second_messages)
    assert second_body["tools"]
    assert second_body["tool_choice"] == "auto"
    assert second_body["messages"][-2]["role"] == "assistant"
    assert second_body["messages"][-2]["tool_calls"][0]["id"] == "call-1"
    assert second_body["messages"][-1]["tool_call_id"] == "call-1"
    assert json.loads(second_body["messages"][-1]["content"])["content"] == "hello"


@pytest.mark.asyncio
async def test_gemini_function_call_round_trip(tmp_path) -> None:
    router = _router(tmp_path)
    (tmp_path / "note.txt").write_text("hello", encoding="utf-8")
    executor = ToolExecutor(Policy(tmp_path, Approval.READ_ONLY))
    endpoint = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"models/{MODELS['gemini']}:generateContent"
    )

    with respx.mock(assert_all_called=True) as mock:
        route = mock.post(endpoint).mock(
            side_effect=[
                httpx.Response(
                    200,
                    json={
                        "candidates": [
                            {
                                "content": {
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "name": "fs_read",
                                                "args": {"path": "note.txt"},
                                            }
                                        }
                                    ]
                                }
                            }
                        ]
                    },
                ),
                httpx.Response(
                    200,
                    json={"candidates": [{"content": {"parts": [{"text": "done"}]}}]},
                ),
            ]
        )

        result = await router.ask(
            "read the note",
            provider="gemini",
            tool_executor=executor,
            use_health=False,
        )

    assert result.text == "done"
    assert route.call_count == 2
    second_body = json.loads(route.calls[1].request.content)
    response_part = second_body["contents"][-1]["parts"][0]
    assert response_part["functionResponse"]["name"] == "fs_read"
    assert response_part["functionResponse"]["response"]["result"]["content"] == "hello"


@pytest.mark.asyncio
async def test_tool_round_limit_stops_repeated_calls(tmp_path) -> None:
    router = _router(tmp_path)
    executor = ToolExecutor(Policy(tmp_path, Approval.READ_ONLY))
    endpoint = "https://api.groq.com/openai/v1/chat/completions"
    response = httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-loop",
                                "type": "function",
                                "function": {
                                    "name": "fs_list",
                                    "arguments": '{"path":"."}',
                                },
                            }
                        ],
                    }
                }
            ]
        },
    )

    with respx.mock(assert_all_called=False) as mock:
        route = mock.post(endpoint).mock(return_value=response)
        with pytest.raises(RouterError) as error:
            await router.ask(
                "loop",
                provider="groq",
                tool_executor=executor,
                max_tool_rounds=1,
                use_health=False,
            )

    assert route.call_count == 2
    assert error.value.attempts[-1].kind == "tool_limit"
