import time
import uuid
from dataclasses import dataclass

import aiosqlite

from mai.core.health import database_path


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str
    provider: str
    model: str
    created_at: float


@dataclass(frozen=True, slots=True)
class SessionRow:
    session_id: str
    title: str
    message_count: int
    created_at: float
    updated_at: float


class SessionStore:
    def __init__(self, path=None) -> None:
        self.path = path or database_path()
        self._ready = False

    async def init(self) -> None:
        if self._ready:
            return

        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    title      TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role       TEXT NOT NULL,
                    content    TEXT NOT NULL,
                    provider   TEXT NOT NULL DEFAULT '',
                    model      TEXT NOT NULL DEFAULT '',
                    created_at REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session "
                "ON messages (session_id, id)"
            )
            await db.commit()

        self._ready = True

    async def create(self, title: str = "") -> str:
        await self.init()
        session_id = uuid.uuid4().hex[:12]
        now = time.time()

        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO sessions (session_id, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, title.strip(), now, now),
            )
            await db.commit()

        return session_id

    async def append(
        self,
        session_id: str,
        role: str,
        content: str,
        provider: str = "",
        model: str = "",
    ) -> None:
        await self.init()
        now = time.time()

        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO messages
                    (session_id, role, content, provider, model, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, content, provider, model, now),
            )
            await db.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            # 제목이 비어 있으면 첫 사용자 메시지로 채운다
            if role == "user":
                await db.execute(
                    """
                    UPDATE sessions SET title = ?
                    WHERE session_id = ? AND title = ''
                    """,
                    (content.strip()[:60], session_id),
                )
            await db.commit()

    async def messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> list[Message]:
        await self.init()
        query = (
            "SELECT role, content, provider, model, created_at "
            "FROM messages WHERE session_id = ? ORDER BY id"
        )
        params: tuple = (session_id,)

        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()

        history = [
            Message(
                role=row["role"],
                content=row["content"],
                provider=row["provider"],
                model=row["model"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

        return history[-limit:] if limit else history

    async def sessions(self, limit: int = 20) -> list[SessionRow]:
        await self.init()

        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT s.session_id, s.title, s.created_at, s.updated_at,
                       COUNT(m.id) AS message_count
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.session_id
                GROUP BY s.session_id
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = await cursor.fetchall()

        return [
            SessionRow(
                session_id=row["session_id"],
                title=row["title"] or "(untitled)",
                message_count=row["message_count"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    async def latest(self) -> str | None:
        rows = await self.sessions(limit=1)
        return rows[0].session_id if rows else None

    async def delete(self, session_id: str) -> bool:
        await self.init()

        async with aiosqlite.connect(self.path) as db:
            await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            cursor = await db.execute(
                "DELETE FROM sessions WHERE session_id = ?", (session_id,)
            )
            await db.commit()
            return bool(cursor.rowcount)
