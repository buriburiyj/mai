from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlparse

from mcp import Client, StdioServerParameters
from platformdirs import user_config_dir

from mai.core.tools import ToolDispatcher, ToolResult

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class MCPHostError(RuntimeError):
    """Raised for invalid MCP host configuration or connection failures."""


@dataclass(frozen=True, slots=True)
class MCPServerConfig:
    name: str
    transport: str
    command: str | None = None
    args: tuple[str, ...] = ()
    url: str | None = None
    enabled: bool = True

    def validate(self) -> None:
        if not _NAME_PATTERN.fullmatch(self.name):
            raise MCPHostError(
                "server name must contain lowercase letters, numbers, "
                "and single hyphens only"
            )

        if self.transport not in {"stdio", "http"}:
            raise MCPHostError("transport must be 'stdio' or 'http'")

        if self.transport == "stdio":
            if not self.command or not self.command.strip():
                raise MCPHostError("stdio servers require a command")
            if self.url is not None:
                raise MCPHostError("stdio servers cannot define a URL")
            values = (self.command, *self.args)
            if any("\x00" in value for value in values):
                raise MCPHostError("stdio command contains a null byte")
            return

        if not self.url:
            raise MCPHostError("HTTP servers require a URL")
        if self.command is not None or self.args:
            raise MCPHostError("HTTP servers cannot define a command")

        parsed = urlparse(self.url)
        hostname = (parsed.hostname or "").lower()
        local = hostname in {"localhost", "127.0.0.1", "::1"}

        if parsed.scheme != "https" and not (parsed.scheme == "http" and local):
            raise MCPHostError(
                "remote MCP servers must use HTTPS; plain HTTP is allowed "
                "only for localhost"
            )
        if not parsed.netloc:
            raise MCPHostError("invalid MCP server URL")
        if parsed.username or parsed.password:
            raise MCPHostError("credentials must not be embedded in MCP URLs")


@dataclass(frozen=True, slots=True)
class MCPProbeResult:
    name: str
    server_name: str
    protocol_version: str
    tools: tuple[str, ...]
    resources: tuple[str, ...]
    prompts: tuple[str, ...]


class MCPRegistry:
    """Persistent external MCP server configuration."""

    def __init__(self, path: Path | None = None) -> None:
        default = Path(user_config_dir("mai")) / "mcp-servers.json"
        self.path = (path or default).expanduser()

    def _read(self) -> dict[str, MCPServerConfig]:
        if not self.path.exists():
            return {}

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MCPHostError(f"unable to read MCP configuration: {exc}") from exc

        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise MCPHostError("unsupported MCP configuration format")

        raw_servers = payload.get("servers")
        if not isinstance(raw_servers, list):
            raise MCPHostError("MCP servers must be a list")

        servers: dict[str, MCPServerConfig] = {}
        for raw in raw_servers:
            if not isinstance(raw, dict):
                raise MCPHostError("invalid MCP server entry")
            try:
                config = MCPServerConfig(
                    name=raw["name"],
                    transport=raw["transport"],
                    command=raw.get("command"),
                    args=tuple(raw.get("args", ())),
                    url=raw.get("url"),
                    enabled=bool(raw.get("enabled", True)),
                )
            except (KeyError, TypeError) as exc:
                raise MCPHostError("invalid MCP server entry") from exc
            config.validate()
            if config.name in servers:
                raise MCPHostError(f"duplicate MCP server name: {config.name}")
            servers[config.name] = config

        return servers

    def list(self) -> list[MCPServerConfig]:
        return sorted(self._read().values(), key=lambda item: item.name)

    def get(self, name: str) -> MCPServerConfig:
        servers = self._read()
        try:
            return servers[name]
        except KeyError as exc:
            raise MCPHostError(f"MCP server not found: {name}") from exc

    def add(
        self,
        config: MCPServerConfig,
        *,
        replace: bool = False,
    ) -> None:
        config.validate()
        servers = self._read()

        if config.name in servers and not replace:
            raise MCPHostError(f"MCP server already exists: {config.name}")

        servers[config.name] = config
        self._write(servers.values())

    def remove(self, name: str) -> bool:
        servers = self._read()
        if name not in servers:
            return False
        del servers[name]
        self._write(servers.values())
        return True

    def _write(self, servers: object) -> None:
        values = list(servers)
        payload = {
            "version": 1,
            "servers": [
                {
                    **asdict(item),
                    "args": list(item.args),
                }
                for item in sorted(values, key=lambda item: item.name)
            ],
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass

        temporary = self.path.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            temporary.replace(self.path)
        except OSError as exc:
            raise MCPHostError(f"unable to write MCP configuration: {exc}") from exc


def _client_target(config: MCPServerConfig) -> str | StdioServerParameters:
    config.validate()

    if config.transport == "http":
        if config.url is None:
            raise MCPHostError("HTTP server URL is missing")
        return config.url

    if config.command is None:
        raise MCPHostError("stdio server command is missing")

    return StdioServerParameters(
        command=config.command,
        args=list(config.args),
    )


async def _all_tools(client: Client) -> list[object]:
    items: list[object] = []
    cursor: str | None = None

    while True:
        page = await client.list_tools(cursor=cursor)
        items.extend(page.tools)
        if page.next_cursor is None:
            return items
        cursor = page.next_cursor


async def probe_server(
    config: MCPServerConfig,
    timeout: float = 30.0,
) -> MCPProbeResult:
    """Connect to a configured MCP server and inspect capabilities."""
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    target = _client_target(config)

    async with asyncio.timeout(timeout):
        async with Client(target) as client:
            tools = await _all_tools(client)

            resources: list[object] = []
            if getattr(client.server_capabilities, "resources", None):
                page = await client.list_resources()
                resources.extend(page.resources)

            prompts: list[object] = []
            if getattr(client.server_capabilities, "prompts", None):
                page = await client.list_prompts()
                prompts.extend(page.prompts)

            info = client.server_info
            server_name = info.name if info is not None else config.name

            return MCPProbeResult(
                name=config.name,
                server_name=server_name,
                protocol_version=str(client.protocol_version),
                tools=tuple(str(getattr(tool, "name", "")) for tool in tools),
                resources=tuple(
                    str(getattr(resource, "uri", "")) for resource in resources
                ),
                prompts=tuple(str(getattr(prompt, "name", "")) for prompt in prompts),
            )


MCPApprovalCallback = Callable[[str, str], Awaitable[bool]]
MCPEventCallback = Callable[[str, str], Awaitable[None]]


def _external_tool_name(server_name: str, tool_name: str) -> str:
    """Create a portable, collision-resistant model function name."""
    raw = f"mcp_{server_name}_{tool_name}"
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", raw)
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")

    if len(cleaned) <= 64:
        return cleaned

    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
    return f"{cleaned[:53]}_{digest}"


class MCPToolExecutor:
    """Expose one connected external MCP server as model-callable tools."""

    def __init__(
        self,
        config: MCPServerConfig,
        approval_callback: MCPApprovalCallback | None = None,
        event_callback: MCPEventCallback | None = None,
        allowed_tools: tuple[str, ...] | None = None,
    ) -> None:
        self.config = config
        self.approval_callback = approval_callback
        self.event_callback = event_callback
        self.allowed_tools = (
            frozenset(allowed_tools) if allowed_tools is not None else None
        )
        self._stack: AsyncExitStack | None = None
        self._client: Client | None = None
        self._schemas: tuple[dict[str, Any], ...] = ()
        self._tool_names: dict[str, str] = {}

    @property
    def schemas(self) -> tuple[dict[str, Any], ...]:
        return self._schemas

    async def __aenter__(self) -> Self:
        if not self.config.enabled:
            raise MCPHostError(f"MCP server is disabled: {self.config.name}")

        stack = AsyncExitStack()
        try:
            client = await stack.enter_async_context(
                Client(_client_target(self.config))
            )
            tools = await _all_tools(client)
        except BaseException:
            await stack.aclose()
            raise

        schemas: list[dict[str, Any]] = []
        names: dict[str, str] = {}

        for tool in tools:
            original_name = str(getattr(tool, "name", ""))
            if not original_name:
                continue
            if (
                self.allowed_tools is not None
                and original_name not in self.allowed_tools
            ):
                continue

            public_name = _external_tool_name(
                self.config.name,
                original_name,
            )
            if public_name in names:
                await stack.aclose()
                raise MCPHostError(f"MCP tool-name collision: {public_name}")

            input_schema = getattr(tool, "input_schema", None)
            if not isinstance(input_schema, dict):
                input_schema = {
                    "type": "object",
                    "properties": {},
                }

            description = str(
                getattr(tool, "description", "") or f"External MCP tool {original_name}"
            )

            names[public_name] = original_name
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": public_name,
                        "description": (
                            f"[MCP server: {self.config.name}] {description}"
                        ),
                        "parameters": input_schema,
                    },
                }
            )

        self._stack = stack
        self._client = client
        self._schemas = tuple(schemas)
        self._tool_names = names
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._client = None
        self._schemas = ()
        self._tool_names = {}

    async def _emit(self, event: str, detail: str) -> None:
        if self.event_callback is not None:
            await self.event_callback(event, detail)

    async def dispatch(
        self,
        name: str,
        arguments: str | dict[str, Any],
    ) -> ToolResult:
        original_name = self._tool_names.get(name)
        if original_name is None:
            return ToolResult(
                False,
                "",
                detail=f"unknown external MCP tool: {name}",
            )

        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError as exc:
                return ToolResult(
                    False,
                    "",
                    detail=f"invalid JSON arguments: {exc}",
                )
        else:
            parsed = arguments

        if not isinstance(parsed, dict):
            return ToolResult(
                False,
                "",
                detail="MCP tool arguments must be a JSON object",
            )

        approval_detail = json.dumps(
            {
                "server": self.config.name,
                "tool": original_name,
                "arguments": parsed,
            },
            ensure_ascii=False,
        )

        if self.approval_callback is None:
            return ToolResult(
                False,
                "",
                detail="external MCP tool requires explicit approval",
            )

        if not await self.approval_callback("mcp", approval_detail):
            await self._emit("denied", approval_detail)
            return ToolResult(
                False,
                "",
                detail="external MCP tool denied by user",
            )

        if self._client is None:
            return ToolResult(
                False,
                "",
                detail="MCP server is not connected",
            )

        await self._emit("started", approval_detail)

        try:
            result = await self._client.call_tool(
                original_name,
                parsed,
            )
        except Exception as exc:  # noqa: BLE001
            await self._emit("completed", f"{approval_detail} ok=false")
            return ToolResult(
                False,
                "",
                detail=f"MCP tool failed: {type(exc).__name__}: {exc}",
            )

        if result.structured_content is not None:
            content = json.dumps(
                result.structured_content,
                ensure_ascii=False,
            )
        else:
            blocks: list[str] = []
            for block in result.content:
                value = getattr(block, "text", None)
                if isinstance(value, str):
                    blocks.append(value)
            content = "\n".join(blocks)

        raw = content.encode("utf-8")
        truncated = len(raw) > 32 * 1024
        if truncated:
            content = raw[: 32 * 1024].decode(
                "utf-8",
                errors="ignore",
            )

        ok = not bool(result.is_error)
        await self._emit(
            "completed",
            f"{approval_detail} ok={str(ok).lower()} "
            f"truncated={str(truncated).lower()}",
        )
        return ToolResult(
            ok,
            content,
            truncated=truncated,
            detail="" if ok else "external MCP tool returned an error",
        )


class CombinedToolExecutor:
    """Combine multiple tool sources behind one router interface."""

    def __init__(self, *executors: ToolDispatcher) -> None:
        self.executors = executors
        schemas: list[dict[str, Any]] = []
        owners: dict[str, ToolDispatcher] = {}

        for executor in executors:
            for schema in executor.schemas:
                function = schema.get("function")
                if not isinstance(function, dict):
                    raise MCPHostError("invalid tool schema")
                name = function.get("name")
                if not isinstance(name, str) or not name:
                    raise MCPHostError("tool schema has no name")
                if name in owners:
                    raise MCPHostError(f"duplicate tool name: {name}")
                owners[name] = executor
                schemas.append(schema)

        self._schemas = tuple(schemas)
        self._owners = owners

    @property
    def schemas(self) -> tuple[dict[str, Any], ...]:
        return self._schemas

    async def dispatch(
        self,
        name: str,
        arguments: str | dict[str, Any],
    ) -> ToolResult:
        executor = self._owners.get(name)
        if executor is None:
            return ToolResult(False, "", detail=f"unknown tool: {name}")
        return await executor.dispatch(name, arguments)
