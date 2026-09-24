import asyncio
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import httpx

from mai.core.providers import PROVIDERS
from mai.core.secrets import SecretStore, SecretStoreError


@dataclass(frozen=True, slots=True)
class CheckResult:
    provider: str
    status: str
    http_status: int | None
    detail: str
    latency_ms: int


store = SecretStore()


def _bearer(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "mai/0.1.0",
    }


def _model_count(payload: Any) -> int | None:
    if not isinstance(payload, dict):
        return None

    for key in ("models", "data", "result"):
        value = payload.get(key)

        if isinstance(value, list):
            return len(value)

        if isinstance(value, dict):
            nested = value.get("data")
            if isinstance(nested, list):
                return len(nested)

    return None


def _safe_error(response: httpx.Response) -> str:
    status = response.status_code

    if status == 401:
        return "invalid or mismatched credential"
    if status == 403:
        return "credential recognized, but permission/access denied"
    if status == 402:
        return "credential recognized, but billing/credits blocked"
    if status == 429:
        return "credential recognized, but rate limit reached"
    if status >= 500:
        return "provider server error"

    return f"unexpected HTTP response: {status}"


async def _request_for_provider(
    client: httpx.AsyncClient,
    provider_name: str,
) -> httpx.Response:
    provider = PROVIDERS[provider_name]
    credentials: dict[str, str] = {}

    for field in provider.credentials:
        value = store.get(provider.name, field.name)
        if value is None:
            raise SecretStoreError(f"Missing credential: {provider.name}:{field.name}")
        credentials[field.name] = value

    if provider_name == "gemini":
        return await client.get(
            "https://generativelanguage.googleapis.com/v1beta/models",
            headers={
                "x-goog-api-key": credentials["api_key"],
                "Accept": "application/json",
                "User-Agent": "mai/0.1.0",
            },
        )

    if provider_name == "cerebras":
        return await client.get(
            "https://api.cerebras.ai/v1/models",
            headers=_bearer(credentials["api_key"]),
        )

    if provider_name == "openrouter":
        return await client.get(
            "https://openrouter.ai/api/v1/key",
            headers=_bearer(credentials["api_key"]),
        )

    if provider_name == "groq":
        return await client.get(
            "https://api.groq.com/openai/v1/models",
            headers=_bearer(credentials["api_key"]),
        )

    if provider_name == "nvidia":
        return await client.get(
            "https://integrate.api.nvidia.com/v1/models",
            headers=_bearer(credentials["api_key"]),
        )

    if provider_name == "mistral":
        return await client.get(
            "https://api.mistral.ai/v1/models",
            headers=_bearer(credentials["api_key"]),
        )

    if provider_name == "cloudflare":
        account_id = quote(credentials["account_id"], safe="")
        url = (
            "https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/models/search"
        )
        return await client.get(
            url,
            headers=_bearer(credentials["api_token"]),
            params={"per_page": 1},
        )

    if provider_name == "huggingface":
        return await client.get(
            "https://huggingface.co/api/whoami-v2",
            headers=_bearer(credentials["token"]),
        )

    raise ValueError(f"Unsupported provider: {provider_name}")


async def check_provider(
    client: httpx.AsyncClient,
    provider_name: str,
) -> CheckResult:
    started = time.perf_counter()

    try:
        response = await _request_for_provider(client, provider_name)
        latency_ms = int((time.perf_counter() - started) * 1000)

        if response.status_code == 200:
            try:
                payload = response.json()
                count = _model_count(payload)
            except ValueError:
                count = None

            detail = "credential accepted"
            if count is not None:
                detail = f"credential accepted; {count} model(s) visible"

            return CheckResult(
                provider=provider_name,
                status="ok",
                http_status=200,
                detail=detail,
                latency_ms=latency_ms,
            )

        if response.status_code in {402, 429}:
            status = "limited"
        elif response.status_code == 403:
            status = "denied"
        else:
            status = "failed"

        return CheckResult(
            provider=provider_name,
            status=status,
            http_status=response.status_code,
            detail=_safe_error(response),
            latency_ms=latency_ms,
        )

    except SecretStoreError as exc:
        return CheckResult(
            provider=provider_name,
            status="failed",
            http_status=None,
            detail=str(exc),
            latency_ms=0,
        )
    except httpx.TimeoutException:
        return CheckResult(
            provider=provider_name,
            status="failed",
            http_status=None,
            detail="request timed out",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    except httpx.HTTPError as exc:
        return CheckResult(
            provider=provider_name,
            status="failed",
            http_status=None,
            detail=f"network error: {type(exc).__name__}",
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


async def check_providers(
    provider_names: list[str],
    timeout: float,
) -> list[CheckResult]:
    limits = httpx.Limits(
        max_connections=len(provider_names),
        max_keepalive_connections=len(provider_names),
    )

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        limits=limits,
    ) as client:
        return await asyncio.gather(
            *(check_provider(client, provider_name) for provider_name in provider_names)
        )
