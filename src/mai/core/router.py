import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

import httpx

from mai.core.catalog import ModelCatalog
from mai.core.health import HealthStore
from mai.core.secrets import SecretStore
from mai.core.streaming import (
    StreamEvent,
    gemini_delta,
    gemini_usage,
    openai_delta,
    openai_usage,
    parse_sse_data,
)
from mai.core.tools import TOOL_SCHEMAS, ToolDispatcher, ToolResult

DEFAULT_ROUTE: tuple[str, ...] = (
    "groq",
    "cerebras",
    "mistral",
    "nvidia",
    "cloudflare",
    "openrouter",
    "gemini",
    "huggingface",
)

MODELS: dict[str, str] = {
    "gemini": "gemini-flash-lite-latest",
    "cerebras": "qwen-3.8-27b",
    "groq": "openai/gpt-oss-20b",
    "mistral": "mistral-small-latest",
    "nvidia": "meta/llama-3.1-8b-instruct",
    "openrouter": "google/gemma-4-26b-a4b-it:free",
    "cloudflare": "@cf/meta/llama-3.1-8b-instruct",
    "huggingface": "google/gemma-2-2b-it",
}

# 기본은 api_key, 다른 이름을 쓰는 공급자만 예외 처리
TIMEOUT_MULTIPLIER: dict[str, float] = {
    "gemini": 3.0,
    "cloudflare": 1.5,
    "huggingface": 2.0,
}

KEY_FIELD: dict[str, str] = {
    "cloudflare": "api_token",
    "huggingface": "token",
}

OPENAI_ENDPOINTS: dict[str, str] = {
    "cerebras": "https://api.cerebras.ai/v1/chat/completions",
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "nvidia": "https://integrate.api.nvidia.com/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
    "huggingface": "https://router.huggingface.co/v1/chat/completions",
}


class EmptyResponse(RuntimeError):
    """공급자가 200을 주고도 본문이 비어 있는 경우."""


class MissingCredential(RuntimeError):
    pass


class ToolRoundLimit(RuntimeError):
    """Raised when a model keeps requesting tools without producing text."""


@dataclass(frozen=True, slots=True)
class Attempt:
    provider: str
    model: str
    status: str
    detail: str
    kind: str = ""


@dataclass(frozen=True, slots=True)
class ChatResult:
    text: str
    provider: str
    model: str
    attempts: tuple[Attempt, ...]
    usage: dict[str, Any] | None = None


class RouterError(RuntimeError):
    def __init__(self, attempts: list[Attempt]) -> None:
        self.attempts = tuple(attempts)
        details = "; ".join(f"{item.provider}: {item.detail}" for item in attempts)
        super().__init__(f"Every provider failed. {details}")


class MultiProviderRouter:
    def __init__(self, health: HealthStore | None = None) -> None:
        self.secrets = SecretStore()
        self.health = health or HealthStore()
        self.catalog = ModelCatalog(self)

    # ---------- credentials ----------

    def _credential(self, provider: str, field: str) -> str:
        value = self.secrets.get(provider, field)
        if not value:
            raise MissingCredential(f"{provider}:{field} is not configured")
        return value

    def _key(self, provider: str) -> str:
        return self._credential(provider, KEY_FIELD.get(provider, "api_key"))

    def configured(self, provider: str) -> bool:
        try:
            self._key(provider)
            if provider == "cloudflare":
                self._credential(provider, "account_id")
        except MissingCredential:
            return False
        return True

    @staticmethod
    def _headers(key: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "mai/0.1.0",
        }

    # ---------- response parsing ----------

    @staticmethod
    def _openai_text(payload: Any) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            return ""

        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            return "".join(
                item.get("text", "") for item in content if isinstance(item, dict)
            ).strip()
        return ""

    @staticmethod
    def _gemini_text(payload: Any) -> str:
        try:
            parts = payload["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError):
            return ""
        return "".join(
            part.get("text", "") for part in parts if isinstance(part, dict)
        ).strip()

    @staticmethod
    def _tool_result_json(result: ToolResult) -> str:
        return json.dumps(
            {
                "ok": result.ok,
                "content": result.content,
                "truncated": result.truncated,
                "detail": result.detail,
            },
            ensure_ascii=False,
        )

    @staticmethod
    def _openai_message(payload: Any) -> dict[str, Any]:
        try:
            message = payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EmptyResponse("openai-compatible response had no message") from exc
        if not isinstance(message, dict):
            raise EmptyResponse("openai-compatible response had invalid message")
        return message

    @staticmethod
    def _openai_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
        calls = message.get("tool_calls", [])
        if not isinstance(calls, list):
            return []
        return [call for call in calls if isinstance(call, dict)]

    @staticmethod
    def _gemini_schema(value: Any) -> Any:
        """Convert an OpenAI JSON schema to Gemini's supported subset."""
        unsupported = {
            "$schema",
            "additionalProperties",
            "default",
            "examples",
            "title",
            "strict",
        }

        if isinstance(value, dict):
            return {
                key: MultiProviderRouter._gemini_schema(item)
                for key, item in value.items()
                if key not in unsupported
            }

        if isinstance(value, list):
            return [MultiProviderRouter._gemini_schema(item) for item in value]

        return value

    @classmethod
    def _gemini_declarations(
        cls,
        schemas: tuple[dict[str, Any], ...] = TOOL_SCHEMAS,
    ) -> list[dict[str, Any]]:
        declarations: list[dict[str, Any]] = []

        for item in schemas:
            if not isinstance(item, dict):
                continue

            function = item.get("function")
            if not isinstance(function, dict):
                continue

            declaration = {
                key: cls._gemini_schema(function[key])
                for key in ("name", "description", "parameters")
                if key in function
            }
            declarations.append(declaration)

        return declarations

    async def _call_openai_with_tools(
        self,
        client: httpx.AsyncClient,
        provider: str,
        prompt: str,
        max_tokens: int,
        model: str,
        tool_executor: ToolDispatcher,
        max_tool_rounds: int,
    ) -> tuple[str, dict[str, Any] | None]:
        key = self._key(provider)
        if provider == "cloudflare":
            account = quote(self._credential(provider, "account_id"), safe="")
            url = (
                "https://api.cloudflare.com/client/v4/accounts/"
                f"{account}/ai/v1/chat/completions"
            )
        else:
            url = OPENAI_ENDPOINTS[provider]

        headers = self._headers(key)
        if provider == "openrouter":
            headers["HTTP-Referer"] = "http://localhost"
            headers["X-Title"] = "MAI"

        token_field = "max_completion_tokens" if provider == "groq" else "max_tokens"
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": prompt},
        ]

        for round_index in range(max_tool_rounds + 1):
            final_round = round_index == max_tool_rounds
            if final_round:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool-use limit reached. Do not call any more tools. "
                            "Using only the tool results already provided, give "
                            "the best possible final answer now. Clearly mention "
                            "anything that could not be verified."
                        ),
                    }
                )

            body: dict[str, Any] = {
                "model": model,
                "messages": messages,
                token_field: max_tokens,
            }
            if not final_round:
                body["tools"] = list(tool_executor.schemas)
                body["tool_choice"] = "auto"

            response = await client.post(url, headers=headers, json=body)
            response.raise_for_status()
            payload = response.json()
            message = self._openai_message(payload)
            calls = self._openai_tool_calls(message)
            if not calls:
                text = self._openai_text(payload)
                if not text:
                    raise EmptyResponse(f"{provider} returned no text")
                return text, payload.get("usage")

            if round_index >= max_tool_rounds:
                raise ToolRoundLimit(f"maximum tool rounds exceeded for {provider}")

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content"),
                "tool_calls": calls,
            }
            messages.append(assistant_message)
            for call in calls:
                function = call.get("function")
                if not isinstance(function, dict):
                    function = {}
                name = function.get("name")
                arguments = function.get("arguments", "{}")
                result = await tool_executor.dispatch(
                    name if isinstance(name, str) else "",
                    arguments if isinstance(arguments, (str, dict)) else "",
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": str(call.get("id", "")),
                        "content": self._tool_result_json(result),
                    }
                )

        raise ToolRoundLimit(f"maximum tool rounds exceeded for {provider}")

    async def _call_gemini_with_tools(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        max_tokens: int,
        model: str,
        tool_executor: ToolDispatcher,
        max_tool_rounds: int,
    ) -> tuple[str, dict[str, Any] | None]:
        key = self._key("gemini")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent"
        )
        contents: list[dict[str, Any]] = [
            {"role": "user", "parts": [{"text": prompt}]},
        ]
        text_parts: list[str] = []

        for round_index in range(max_tool_rounds + 1):
            final_round = round_index == max_tool_rounds
            if final_round:
                contents.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": (
                                    "Tool-use limit reached. Do not call any "
                                    "more functions. Using only the function "
                                    "results already provided, give the best "
                                    "possible final answer now. Clearly mention "
                                    "anything that could not be verified."
                                )
                            }
                        ],
                    }
                )

            request_body: dict[str, Any] = {
                "contents": contents,
                "generationConfig": {"maxOutputTokens": max_tokens},
            }
            if not final_round:
                request_body["tools"] = [
                    {
                        "functionDeclarations": self._gemini_declarations(
                            tool_executor.schemas
                        )
                    }
                ]
                request_body["toolConfig"] = {"functionCallingConfig": {"mode": "AUTO"}}

            response = await client.post(
                url,
                headers={
                    "x-goog-api-key": key,
                    "Content-Type": "application/json",
                    "User-Agent": "mai/0.1.0",
                },
                json=request_body,
            )
            response.raise_for_status()
            payload = response.json()
            try:
                parts = payload["candidates"][0]["content"]["parts"]
            except (KeyError, IndexError, TypeError) as exc:
                raise EmptyResponse("gemini response had no parts") from exc
            if not isinstance(parts, list):
                raise EmptyResponse("gemini response had invalid parts")

            function_calls = [
                part["functionCall"]
                for part in parts
                if isinstance(part, dict) and isinstance(part.get("functionCall"), dict)
            ]
            for part in parts:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])

            if not function_calls:
                text = "".join(text_parts).strip()
                if not text:
                    raise EmptyResponse("gemini returned no text")
                return text, payload.get("usageMetadata")

            if round_index >= max_tool_rounds:
                raise ToolRoundLimit("maximum tool rounds exceeded for gemini")

            contents.append({"role": "model", "parts": parts})
            response_parts: list[dict[str, Any]] = []
            for call in function_calls:
                name = call.get("name")
                args = call.get("args", {})
                result = await tool_executor.dispatch(
                    name if isinstance(name, str) else "",
                    args if isinstance(args, dict) else "",
                )
                response_parts.append(
                    {
                        "functionResponse": {
                            "name": name if isinstance(name, str) else "",
                            "response": {
                                "result": json.loads(self._tool_result_json(result)),
                            },
                        }
                    }
                )
            contents.append({"role": "user", "parts": response_parts})

        raise ToolRoundLimit("maximum tool rounds exceeded for gemini")

    # ---------- provider calls ----------

    async def _call_gemini(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        max_tokens: int,
        model: str,
    ) -> tuple[str, dict[str, Any] | None]:
        key = self._key("gemini")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:generateContent"
        )

        response = await client.post(
            url,
            headers={
                "x-goog-api-key": key,
                "Content-Type": "application/json",
                "User-Agent": "mai/0.1.0",
            },
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens},
            },
        )
        response.raise_for_status()
        payload = response.json()
        text = self._gemini_text(payload)

        if not text:
            raise EmptyResponse("gemini returned no text")

        return text, payload.get("usageMetadata")

    async def _call_openai_compatible(
        self,
        client: httpx.AsyncClient,
        provider: str,
        prompt: str,
        max_tokens: int,
        model: str,
    ) -> tuple[str, dict[str, Any] | None]:
        key = self._key(provider)

        if provider == "cloudflare":
            account = quote(self._credential(provider, "account_id"), safe="")
            url = (
                "https://api.cloudflare.com/client/v4/accounts/"
                f"{account}/ai/v1/chat/completions"
            )
        else:
            url = OPENAI_ENDPOINTS[provider]

        headers = self._headers(key)
        if provider == "openrouter":
            headers["HTTP-Referer"] = "http://localhost"
            headers["X-Title"] = "MAI"

        token_field = "max_completion_tokens" if provider == "groq" else "max_tokens"
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            token_field: max_tokens,
        }

        response = await client.post(url, headers=headers, json=body)
        response.raise_for_status()
        payload = response.json()
        text = self._openai_text(payload)

        if not text:
            raise EmptyResponse(f"{provider} returned no text")

        return text, payload.get("usage")

    async def _stream_openai_compatible(
        self,
        client: httpx.AsyncClient,
        provider: str,
        prompt: str,
        max_tokens: int,
        model: str,
    ) -> AsyncIterator[StreamEvent]:
        key = self._key(provider)

        if provider == "cloudflare":
            account = quote(self._credential(provider, "account_id"), safe="")
            url = (
                "https://api.cloudflare.com/client/v4/accounts/"
                f"{account}/ai/v1/chat/completions"
            )
        else:
            url = OPENAI_ENDPOINTS[provider]

        headers = self._headers(key)
        if provider == "openrouter":
            headers["HTTP-Referer"] = "http://localhost"
            headers["X-Title"] = "MAI"

        token_field = "max_completion_tokens" if provider == "groq" else "max_tokens"
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            token_field: max_tokens,
            "stream": True,
        }

        received = False
        usage: dict[str, Any] | None = None

        async with client.stream("POST", url, headers=headers, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                chunk = parse_sse_data(line)
                if chunk is None:
                    continue
                if chunk == "[DONE]":
                    break
                usage = openai_usage(chunk) or usage
                piece = openai_delta(chunk)
                if piece:
                    received = True
                    yield StreamEvent(
                        kind="delta", text=piece, provider=provider, model=model
                    )

        if not received:
            raise EmptyResponse(f"{provider} returned no text")

        yield StreamEvent(kind="done", provider=provider, model=model, usage=usage)

    async def _stream_gemini(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        max_tokens: int,
        model: str,
    ) -> AsyncIterator[StreamEvent]:
        key = self._key("gemini")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"models/{model}:streamGenerateContent?alt=sse"
        )

        received = False
        usage: dict[str, Any] | None = None

        async with client.stream(
            "POST",
            url,
            headers={
                "x-goog-api-key": key,
                "Content-Type": "application/json",
                "User-Agent": "mai/0.1.0",
            },
            json={
                "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens},
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                chunk = parse_sse_data(line)
                if chunk is None:
                    continue
                usage = gemini_usage(chunk) or usage
                piece = gemini_delta(chunk)
                if piece:
                    received = True
                    yield StreamEvent(
                        kind="delta", text=piece, provider="gemini", model=model
                    )

        if not received:
            raise EmptyResponse("gemini returned no text")

        yield StreamEvent(kind="done", provider="gemini", model=model, usage=usage)

    def _stream(
        self,
        client: httpx.AsyncClient,
        provider: str,
        prompt: str,
        max_tokens: int,
        model: str,
    ) -> AsyncIterator[StreamEvent]:
        if provider == "gemini":
            return self._stream_gemini(client, prompt, max_tokens, model)
        return self._stream_openai_compatible(
            client, provider, prompt, max_tokens, model
        )

    async def _call(
        self,
        client: httpx.AsyncClient,
        provider: str,
        prompt: str,
        max_tokens: int,
        model: str,
    ) -> tuple[str, dict[str, Any] | None]:
        if provider == "gemini":
            return await self._call_gemini(client, prompt, max_tokens, model)
        return await self._call_openai_compatible(
            client, provider, prompt, max_tokens, model
        )

    # ---------- failure classification ----------

    @staticmethod
    def classify(exc: Exception) -> tuple[str, str]:
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            table: dict[int, tuple[str, str]] = {
                400: ("request", "request or model rejected"),
                401: ("auth", "authentication failed"),
                402: ("credit", "free credit unavailable"),
                403: ("auth", "access denied"),
                404: ("model", "model unavailable"),
                408: ("network", "provider timeout"),
                429: ("rate", "free-tier limit reached"),
            }
            if status in table:
                return table[status]
            if status >= 500:
                return "server", f"provider server error (HTTP {status})"
            return "request", f"unexpected HTTP {status}"

        if isinstance(exc, EmptyResponse):
            return "empty", "response body was empty"
        if isinstance(exc, ToolRoundLimit):
            return "tool_limit", str(exc)
        if isinstance(exc, httpx.TimeoutException):
            return "network", "request timed out"
        if isinstance(exc, httpx.NetworkError):
            return "network", "network error"

        return "unknown", str(exc) or type(exc).__name__

    @staticmethod
    def retry_after(exc: Exception) -> float | None:
        if not isinstance(exc, httpx.HTTPStatusError):
            return None

        raw = exc.response.headers.get("retry-after")
        if not raw:
            return None

        try:
            return max(0.0, float(raw))
        except ValueError:
            pass

        try:
            moment = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None

        if moment is None:
            return None

        return max(0.0, moment.timestamp() - time.time())

    # ---------- main entry ----------

    async def ask(
        self,
        prompt: str,
        provider: str = "auto",
        max_tokens: int = 1024,
        timeout: float = 60.0,
        use_health: bool = True,
        tool_executor: ToolDispatcher | None = None,
        max_tool_rounds: int = 6,
    ) -> ChatResult:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least 1")

        if provider == "auto":
            route = (
                ("groq", "gemini", "openrouter")
                if tool_executor is not None
                else DEFAULT_ROUTE
            )
        elif provider in MODELS:
            route = (provider,)
        else:
            available = ", ".join(("auto", *MODELS))
            raise ValueError(f"Unknown provider: {provider}. Available: {available}")

        explicit = len(route) == 1
        attempts: list[Attempt] = []
        candidates: list[str] = []

        cooling: dict[str, int] = {}
        if use_health and not explicit:
            cooling = await self.health.cooling(route)

        for name in route:
            if not self.configured(name):
                attempts.append(
                    Attempt(
                        provider=name,
                        model=MODELS[name],
                        status="skipped",
                        detail="credential not configured",
                        kind="missing",
                    )
                )
                continue

            if name in cooling:
                attempts.append(
                    Attempt(
                        provider=name,
                        model=MODELS[name],
                        status="skipped",
                        detail=f"cooling down for {cooling[name]}s",
                        kind="cooldown",
                    )
                )
                continue

            candidates.append(name)

        if not candidates:
            raise RouterError(attempts)

        budget = timeout * max(TIMEOUT_MULTIPLIER.get(name, 1.0) for name in candidates)

        async with httpx.AsyncClient(timeout=budget, follow_redirects=True) as client:
            for name in candidates:
                model = await self.catalog.resolve(client, name, MODELS[name])
                try:
                    if tool_executor is None:
                        text, usage = await self._call(
                            client, name, prompt, max_tokens, model
                        )
                    elif name == "gemini":
                        text, usage = await self._call_gemini_with_tools(
                            client,
                            prompt,
                            max_tokens,
                            model,
                            tool_executor,
                            max_tool_rounds,
                        )
                    else:
                        text, usage = await self._call_openai_with_tools(
                            client,
                            name,
                            prompt,
                            max_tokens,
                            model,
                            tool_executor,
                            max_tool_rounds,
                        )
                except (
                    httpx.HTTPError,
                    EmptyResponse,
                    MissingCredential,
                    RuntimeError,
                    ValueError,
                ) as exc:
                    kind, detail = self.classify(exc)

                    if use_health:
                        seconds = await self.health.mark_failure(
                            name, kind, detail, self.retry_after(exc)
                        )
                        detail = f"{detail} (cooling {seconds}s)"

                    attempts.append(
                        Attempt(
                            provider=name,
                            model=model,
                            status="failed",
                            detail=detail,
                            kind=kind,
                        )
                    )
                    continue

                if use_health:
                    await self.health.mark_success(name)

                attempts.append(
                    Attempt(
                        provider=name,
                        model=model,
                        status="success",
                        detail="response received",
                        kind="ok",
                    )
                )
                return ChatResult(
                    text=text,
                    provider=name,
                    model=model,
                    attempts=tuple(attempts),
                    usage=usage,
                )

        raise RouterError(attempts)

    async def compare(
        self,
        prompt: str,
        providers: Sequence[str] | None = None,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ) -> list[ChatResult | Attempt]:
        """Ask several providers in parallel and return every outcome."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        names = (
            list(providers)
            if providers
            else [name for name in DEFAULT_ROUTE if self.configured(name)]
        )
        unknown = [name for name in names if name not in MODELS]
        if unknown:
            raise ValueError(f"Unknown provider(s): {', '.join(unknown)}")
        if not names:
            raise ValueError("No configured providers to compare.")

        budget = timeout * max(TIMEOUT_MULTIPLIER.get(name, 1.0) for name in names)

        async def one(client: httpx.AsyncClient, name: str) -> ChatResult | Attempt:
            if not self.configured(name):
                return Attempt(
                    provider=name,
                    model=MODELS[name],
                    status="skipped",
                    detail="credential not configured",
                    kind="missing",
                )

            started = time.perf_counter()
            try:
                model = await self.catalog.resolve(client, name, MODELS[name])
                text_out, usage = await self._call(
                    client, name, prompt, max_tokens, model
                )
            except (
                httpx.HTTPError,
                EmptyResponse,
                MissingCredential,
                RuntimeError,
                ValueError,
            ) as exc:
                kind, detail = self.classify(exc)
                elapsed = time.perf_counter() - started
                return Attempt(
                    provider=name,
                    model=MODELS[name],
                    status="failed",
                    detail=f"{detail} ({elapsed:.1f}s)",
                    kind=kind,
                )

            elapsed = time.perf_counter() - started
            return ChatResult(
                text=text_out,
                provider=name,
                model=model,
                usage=usage,
                attempts=[
                    Attempt(
                        provider=name,
                        model=model,
                        status="ok",
                        detail=f"{elapsed:.1f}s",
                        kind="ok",
                    )
                ],
            )

        async with httpx.AsyncClient(timeout=budget, follow_redirects=True) as client:
            return list(await asyncio.gather(*(one(client, name) for name in names)))

    async def astream(
        self,
        prompt: str,
        provider: str = "auto",
        max_tokens: int = 1024,
        timeout: float = 60.0,
        use_health: bool = True,
    ) -> AsyncIterator[StreamEvent]:
        """Stream a reply, falling back only before the first token arrives."""
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Prompt cannot be empty.")

        if provider == "auto":
            route = DEFAULT_ROUTE
        elif provider in MODELS:
            route = (provider,)
        else:
            available = ", ".join(("auto", *MODELS))
            raise ValueError(f"Unknown provider: {provider}. Available: {available}")

        explicit = len(route) == 1
        attempts: list[Attempt] = []
        candidates: list[str] = []

        cooling: dict[str, int] = {}
        if use_health and not explicit:
            cooling = await self.health.cooling(route)

        for name in route:
            if not self.configured(name):
                attempts.append(
                    Attempt(
                        provider=name,
                        model=MODELS[name],
                        status="skipped",
                        detail="credential not configured",
                        kind="missing",
                    )
                )
                continue
            if name in cooling:
                attempts.append(
                    Attempt(
                        provider=name,
                        model=MODELS[name],
                        status="skipped",
                        detail=f"cooling down for {cooling[name]}s",
                        kind="cooldown",
                    )
                )
                continue
            candidates.append(name)

        if not candidates:
            raise RouterError(attempts)

        budget = timeout * max(TIMEOUT_MULTIPLIER.get(name, 1.0) for name in candidates)

        async with httpx.AsyncClient(timeout=budget, follow_redirects=True) as client:
            for name in candidates:
                model = await self.catalog.resolve(client, name, MODELS[name])
                started = False
                try:
                    async for event in self._stream(
                        client, name, prompt, max_tokens, model
                    ):
                        if event.kind == "delta":
                            started = True
                        yield event
                except (
                    httpx.HTTPError,
                    EmptyResponse,
                    MissingCredential,
                    RuntimeError,
                    ValueError,
                ) as exc:
                    kind, detail = self.classify(exc)
                    if use_health:
                        seconds = await self.health.mark_failure(
                            name, kind, detail, self.retry_after(exc)
                        )
                        detail = f"{detail} (cooling {seconds}s)"

                    attempts.append(
                        Attempt(
                            provider=name,
                            model=model,
                            status="failed",
                            detail=detail,
                            kind=kind,
                        )
                    )

                    if started:
                        yield StreamEvent(
                            kind="done",
                            provider=name,
                            model=model,
                            detail=f"stream interrupted: {detail}",
                        )
                        return

                    yield StreamEvent(
                        kind="fallback", provider=name, model=model, detail=detail
                    )
                    continue

                if use_health:
                    await self.health.mark_success(name)
                return

        raise RouterError(attempts)
