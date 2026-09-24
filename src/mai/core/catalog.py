import json
import time
from collections.abc import Sequence
from typing import Any
from urllib.parse import quote

import aiosqlite
import httpx

from mai.core.health import database_path

CACHE_TTL_SECONDS = 24 * 60 * 60

PREFERENCES: dict[str, tuple[str, ...]] = {
    "gemini": ("flash-lite-latest", "flash-lite", "flash-latest"),
    "cerebras": ("llama-3.1-8b", "llama3.1-8b", "8b", "instruct"),
    "groq": ("gpt-oss-20b", "8b-instant", "instruct"),
    "mistral": ("mistral-small-latest", "mistral-small", "ministral"),
    "nvidia": ("gemma-3-4b-it", "gemma-3-12b-it", "gemma-4-31b-it"),
    "openrouter": (":free",),
    "cloudflare": ("llama-3.2-3b-instruct", "llama-3.1-8b-instruct-fp8", "instruct"),
    "huggingface": ("gemma-2-2b-it", "-it", "instruct"),
}

LIST_URLS: dict[str, str] = {
    "cerebras": "https://api.cerebras.ai/v1/models",
    "groq": "https://api.groq.com/openai/v1/models",
    "mistral": "https://api.mistral.ai/v1/models",
    "nvidia": "https://integrate.api.nvidia.com/v1/models",
    "openrouter": "https://openrouter.ai/api/v1/models",
    "huggingface": "https://router.huggingface.co/v1/models",
}


def extract_ids(provider: str, payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return []

    items: list[Any] = []
    for key in ("data", "models", "result"):
        value = payload.get(key)
        if isinstance(value, list):
            items = value
            break
        if isinstance(value, dict) and isinstance(value.get("data"), list):
            items = value["data"]
            break

    ids: list[str] = []
    for item in items:
        if isinstance(item, str):
            ids.append(item)
            continue
        if not isinstance(item, dict):
            continue

        if provider == "cloudflare":
            raw = item.get("name") or item.get("id") or item.get("model")
        else:
            raw = item.get("id") or item.get("name") or item.get("model")
        if not isinstance(raw, str):
            continue
        if provider == "gemini" and raw.startswith("models/"):
            raw = raw[len("models/") :]

        ids.append(raw)

    return ids


def pick_model(
    provider: str,
    configured: str,
    models: Sequence[str],
) -> str:
    if not models or configured in models:
        return configured

    for hint in PREFERENCES.get(provider, ()):
        for candidate in models:
            if hint in candidate:
                return candidate

    return models[0]


class ModelCatalog:
    """공급자별 모델 목록을 캐싱하고 실제 사용할 ID를 고른다."""

    def __init__(self, router: Any, ttl: int = CACHE_TTL_SECONDS) -> None:
        self.router = router
        self.ttl = ttl
        self.path = database_path()
        self._ready = False
        self._memory: dict[str, list[str]] = {}

    async def init(self) -> None:
        if self._ready:
            return

        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS model_cache (
                    provider   TEXT PRIMARY KEY,
                    models     TEXT NOT NULL,
                    fetched_at REAL NOT NULL
                )
                """
            )
            await db.commit()

        self._ready = True

    async def _read_cache(self, provider: str) -> list[str] | None:
        await self.init()

        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT models, fetched_at FROM model_cache WHERE provider = ?",
                (provider,),
            )
            row = await cursor.fetchone()

        if row is None or time.time() - row["fetched_at"] > self.ttl:
            return None

        try:
            models = json.loads(row["models"])
        except json.JSONDecodeError:
            return None

        return models if isinstance(models, list) else None

    async def _write_cache(
        self,
        provider: str,
        models: Sequence[str],
    ) -> None:
        await self.init()

        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """
                INSERT INTO model_cache (provider, models, fetched_at)
                VALUES (?, ?, ?)
                ON CONFLICT(provider) DO UPDATE SET
                    models     = excluded.models,
                    fetched_at = excluded.fetched_at
                """,
                (provider, json.dumps(list(models)), time.time()),
            )
            await db.commit()

    async def _fetch(
        self,
        client: httpx.AsyncClient,
        provider: str,
    ) -> list[str]:
        key = self.router._key(provider)

        if provider == "gemini":
            response = await client.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                headers={"x-goog-api-key": key, "Accept": "application/json"},
                params={"pageSize": 200},
            )
        elif provider == "cloudflare":
            account = quote(self.router._credential(provider, "account_id"), safe="")
            response = await client.get(
                "https://api.cloudflare.com/client/v4/accounts/"
                f"{account}/ai/models/search",
                headers=self.router._headers(key),
                params={"per_page": 200, "task": "Text Generation"},
            )
        else:
            response = await client.get(
                LIST_URLS[provider], headers=self.router._headers(key)
            )

        response.raise_for_status()
        return extract_ids(provider, response.json())

    async def available(
        self,
        client: httpx.AsyncClient,
        provider: str,
        refresh: bool = False,
    ) -> list[str]:
        if not refresh:
            if provider in self._memory:
                return self._memory[provider]

            cached = await self._read_cache(provider)
            if cached is not None:
                self._memory[provider] = cached
                return cached

        try:
            models = await self._fetch(client, provider)
        except (httpx.HTTPError, KeyError, ValueError, RuntimeError):
            return []

        if models:
            await self._write_cache(provider, models)
            self._memory[provider] = models

        return models

    async def resolve(
        self,
        client: httpx.AsyncClient,
        provider: str,
        configured: str,
    ) -> str:
        models = await self.available(client, provider)
        return pick_model(provider, configured, models)

    async def clear(self) -> None:
        await self.init()
        self._memory.clear()

        async with aiosqlite.connect(self.path) as db:
            await db.execute("DELETE FROM model_cache")
            await db.commit()
