import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
from platformdirs import user_data_dir

# 실패 유형별 기본 휴면 시간(초)
COOLDOWN_SECONDS: dict[str, int] = {
    "credit": 24 * 60 * 60,  # 402: 무료 크레딧 소진
    "auth": 60 * 60,  # 401/403: 키 문제
    "model": 30 * 60,  # 404: 모델 ID 불일치
    "rate": 90,  # 429: 속도 제한 (Retry-After 우선)
    "server": 5 * 60,  # 5xx
    "network": 60,  # 타임아웃/네트워크
    "empty": 10 * 60,  # 본문이 비어서 온 경우
    "request": 15 * 60,  # 400
    "unknown": 5 * 60,
}

MAX_COOLDOWN = 24 * 60 * 60


def database_path() -> Path:
    directory = Path(user_data_dir("mai"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "mai.db"


@dataclass(frozen=True, slots=True)
class HealthRow:
    provider: str
    cooldown_until: float
    last_kind: str
    last_detail: str
    success_count: int
    failure_count: int
    updated_at: float

    @property
    def remaining(self) -> int:
        return max(0, round(self.cooldown_until - time.time()))

    @property
    def available(self) -> bool:
        return self.remaining == 0


class HealthStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or database_path()
        self._ready = False

    async def init(self) -> None:
        if self._ready:
            return

        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS provider_health (
                    provider       TEXT    PRIMARY KEY,
                    cooldown_until REAL    NOT NULL DEFAULT 0,
                    last_kind      TEXT    NOT NULL DEFAULT '',
                    last_detail    TEXT    NOT NULL DEFAULT '',
                    success_count  INTEGER NOT NULL DEFAULT 0,
                    failure_count  INTEGER NOT NULL DEFAULT 0,
                    updated_at     REAL    NOT NULL DEFAULT 0
                )
                """
            )
            await db.commit()

        self._ready = True

    async def mark_success(self, provider: str) -> None:
        await self.init()
        now = time.time()

        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO provider_health (
                    provider, cooldown_until, last_kind, last_detail,
                    success_count, failure_count, updated_at
                )
                VALUES (?, 0, 'ok', 'response received', 1, 0, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    cooldown_until = 0,
                    last_kind      = 'ok',
                    last_detail    = 'response received',
                    success_count  = success_count + 1,
                    updated_at     = excluded.updated_at
                """,
                (provider, now),
            )
            await db.commit()

    async def mark_failure(
        self,
        provider: str,
        kind: str,
        detail: str,
        retry_after: float | None = None,
    ) -> int:
        await self.init()

        base = COOLDOWN_SECONDS.get(kind, COOLDOWN_SECONDS["unknown"])
        seconds = int(min(max(retry_after or base, 1), MAX_COOLDOWN))
        now = time.time()
        until = now + seconds

        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO provider_health (
                    provider, cooldown_until, last_kind, last_detail,
                    success_count, failure_count, updated_at
                )
                VALUES (?, ?, ?, ?, 0, 1, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    cooldown_until = excluded.cooldown_until,
                    last_kind      = excluded.last_kind,
                    last_detail    = excluded.last_detail,
                    failure_count  = failure_count + 1,
                    updated_at     = excluded.updated_at
                """,
                (provider, until, kind, detail, now),
            )
            await db.commit()

        return seconds

    async def cooling(self, providers: Sequence[str]) -> dict[str, int]:
        """휴면 중인 공급자와 남은 초를 반환한다."""
        await self.init()

        if not providers:
            return {}

        placeholders = ",".join("?" for _ in providers)
        now = time.time()

        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                f"""
                SELECT provider, cooldown_until
                FROM provider_health
                WHERE provider IN ({placeholders})
                  AND cooldown_until > ?
                """,
                (*providers, now),
            )
            rows = await cursor.fetchall()

        return {
            row["provider"]: max(1, round(row["cooldown_until"] - now)) for row in rows
        }

    async def rows(self) -> list[HealthRow]:
        await self.init()

        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM provider_health ORDER BY provider")
            records = await cursor.fetchall()

        return [
            HealthRow(
                provider=row["provider"],
                cooldown_until=row["cooldown_until"],
                last_kind=row["last_kind"],
                last_detail=row["last_detail"],
                success_count=row["success_count"],
                failure_count=row["failure_count"],
                updated_at=row["updated_at"],
            )
            for row in records
        ]

    async def reset(self, provider: str | None = None) -> int:
        await self.init()

        async with aiosqlite.connect(self.path) as db:
            if provider is None:
                cursor = await db.execute(
                    "UPDATE provider_health SET cooldown_until = 0"
                )
            else:
                cursor = await db.execute(
                    "UPDATE provider_health SET cooldown_until = 0 WHERE provider = ?",
                    (provider,),
                )
            await db.commit()
            return cursor.rowcount or 0
