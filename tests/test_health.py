import httpx
import pytest

from mai.core.health import HealthStore
from mai.core.router import EmptyResponse, MultiProviderRouter


def _status_error(code: int, headers: dict[str, str] | None = None):
    request = httpx.Request("POST", "https://example.test/v1/chat")
    response = httpx.Response(code, headers=headers or {}, request=request)
    return httpx.HTTPStatusError("boom", request=request, response=response)


def test_classify_known_statuses() -> None:
    assert MultiProviderRouter.classify(_status_error(402))[0] == "credit"
    assert MultiProviderRouter.classify(_status_error(429))[0] == "rate"
    assert MultiProviderRouter.classify(_status_error(404))[0] == "model"
    assert MultiProviderRouter.classify(_status_error(503))[0] == "server"
    assert MultiProviderRouter.classify(EmptyResponse("x"))[0] == "empty"


def test_retry_after_seconds() -> None:
    exc = _status_error(429, {"retry-after": "42"})
    assert MultiProviderRouter.retry_after(exc) == pytest.approx(42.0)


@pytest.mark.asyncio
async def test_cooldown_lifecycle(tmp_path) -> None:
    store = HealthStore(tmp_path / "health.db")

    seconds = await store.mark_failure("groq", "credit", "no credit")
    assert seconds == 24 * 60 * 60

    cooling = await store.cooling(["gemini", "groq"])
    assert "groq" in cooling
    assert "gemini" not in cooling

    await store.mark_success("groq")
    assert await store.cooling(["groq"]) == {}


@pytest.mark.asyncio
async def test_retry_after_overrides_default(tmp_path) -> None:
    store = HealthStore(tmp_path / "health.db")
    seconds = await store.mark_failure("groq", "rate", "limited", retry_after=5)
    assert seconds == 5


@pytest.mark.asyncio
async def test_reset_clears_cooldown(tmp_path) -> None:
    store = HealthStore(tmp_path / "health.db")
    await store.mark_failure("nvidia", "credit", "no credit")

    assert await store.reset("nvidia") == 1
    assert await store.cooling(["nvidia"]) == {}
