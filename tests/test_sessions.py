import pytest

from mai.core.sessions import SessionStore


@pytest.mark.asyncio
async def test_session_roundtrip(tmp_path) -> None:
    store = SessionStore(tmp_path / "s.db")
    session_id = await store.create()

    await store.append(session_id, "user", "안녕")
    await store.append(session_id, "assistant", "반가워", "groq", "gpt-oss-20b")

    history = await store.messages(session_id)
    assert [m.role for m in history] == ["user", "assistant"]
    assert history[1].provider == "groq"


@pytest.mark.asyncio
async def test_title_from_first_user_message(tmp_path) -> None:
    store = SessionStore(tmp_path / "s.db")
    session_id = await store.create()
    await store.append(session_id, "user", "첫 질문입니다")

    rows = await store.sessions()
    assert rows[0].title == "첫 질문입니다"
    assert rows[0].message_count == 1


@pytest.mark.asyncio
async def test_limit_returns_tail(tmp_path) -> None:
    store = SessionStore(tmp_path / "s.db")
    session_id = await store.create()

    for index in range(5):
        await store.append(session_id, "user", f"m{index}")

    history = await store.messages(session_id, limit=2)
    assert [m.content for m in history] == ["m3", "m4"]


@pytest.mark.asyncio
async def test_delete_removes_session(tmp_path) -> None:
    store = SessionStore(tmp_path / "s.db")
    session_id = await store.create()
    await store.append(session_id, "user", "hi")

    assert await store.delete(session_id) is True
    assert await store.messages(session_id) == []
    assert await store.delete(session_id) is False
