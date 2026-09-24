from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from mai.core.policy import Policy, PolicyError

ApprovalCallback = Callable[[str, str], Awaitable[bool]]
ToolEventCallback = Callable[[str, str], Awaitable[None]]


class ToolDispatcher(Protocol):
    """Interface accepted by the multi-provider router."""

    @property
    def schemas(self) -> tuple[dict[str, Any], ...]:
        """Return OpenAI-compatible function schemas."""
        ...

    async def dispatch(
        self,
        name: str,
        arguments: str | dict[str, Any],
    ) -> ToolResult:
        """Execute a named tool."""
        ...


@dataclass(frozen=True, slots=True)
class ToolResult:
    ok: bool
    content: str
    truncated: bool = False
    detail: str = ""


TOOL_SCHEMAS: tuple[dict[str, Any], ...] = (
    {
        "type": "function",
        "function": {
            "name": "fs_read",
            "description": "Read a UTF-8 text file inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative file path.",
                    },
                    "max_bytes": {"type": "integer", "minimum": 1},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fs_list",
            "description": "List files and directories inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace-relative directory path.",
                    },
                    "depth": {"type": "integer", "minimum": 0, "maximum": 20},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "shell_run",
            "description": "Run an approved command inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {
                    "argv": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Workspace-relative working directory.",
                    },
                },
                "required": ["argv"],
                "additionalProperties": False,
            },
        },
    },
)


class ToolExecutor:
    @property
    def schemas(self) -> tuple[dict[str, Any], ...]:
        """Return only built-in tools allowed for this request."""
        if self.allowed_tools is None:
            return TOOL_SCHEMAS
        return tuple(
            schema
            for schema in TOOL_SCHEMAS
            if schema["function"]["name"] in self.allowed_tools
        )

    def __init__(
        self,
        policy: Policy,
        approval_callback: ApprovalCallback | None = None,
        event_callback: ToolEventCallback | None = None,
        allowed_tools: tuple[str, ...] | None = None,
    ) -> None:
        self.policy = policy
        self.approval_callback = approval_callback
        self.event_callback = event_callback
        self.allowed_tools = (
            frozenset(allowed_tools) if allowed_tools is not None else None
        )

    async def _emit(self, event: str, detail: str) -> None:
        if self.event_callback is None:
            return
        try:
            await self.event_callback(event, detail)
        except (OSError, RuntimeError, TypeError, ValueError):
            return

    async def _approve(self, kind: str, description: str) -> bool:
        try:
            requires_approval = self.policy.requires_approval(kind)
        except PolicyError:
            return False

        if not requires_approval:
            return True
        if self.approval_callback is None:
            return False
        return await self.approval_callback(kind, description)

    async def fs_read(
        self,
        path: str | Path,
        max_bytes: int | None = None,
    ) -> ToolResult:
        await self._emit("started", f"fs_read: {path}")
        try:
            target = self.policy.resolve(path)
            limit = max_bytes or self.policy.max_read_bytes
            if limit < 1:
                return ToolResult(False, "", detail="max_bytes must be positive")
            if not target.is_file():
                return ToolResult(False, "", detail=f"not a file: {target}")

            data = target.read_bytes()
            if b"\x00" in data[:4096]:
                return ToolResult(False, "", detail="binary files are not supported")
            truncated = len(data) > limit
            content = data[:limit].decode("utf-8", errors="replace")
            result = ToolResult(True, content, truncated=truncated)
            await self._emit(
                "completed",
                f"fs_read: {path} ok=true truncated={truncated}",
            )
            return result
        except (PolicyError, OSError, UnicodeError) as exc:
            result = ToolResult(False, "", detail=str(exc))
            await self._emit("denied", f"fs_read: {path} failed")
            return result

    async def fs_list(self, path: str | Path, depth: int = 1) -> ToolResult:
        await self._emit("started", f"fs_list: {path}")
        if depth < 0:
            result = ToolResult(False, "", detail="depth must not be negative")
            await self._emit("denied", f"fs_list: {path} failed")
            return result
        try:
            root = self.policy.resolve(path)
            if not root.is_dir():
                result = ToolResult(False, "", detail=f"not a directory: {root}")
                await self._emit("denied", f"fs_list: {path} failed")
                return result
            entries: list[str] = []
            await self._list_directory(root, root, depth, entries)
            content = "\n".join(sorted(entries))
            raw = content.encode("utf-8")
            truncated = len(raw) > self.policy.max_output_bytes
            if truncated:
                content = raw[: self.policy.max_output_bytes].decode(
                    "utf-8", errors="ignore"
                )
            result = ToolResult(True, content, truncated=truncated)
            await self._emit(
                "completed",
                f"fs_list: {path} ok=true truncated={truncated}",
            )
            return result
        except (PolicyError, OSError) as exc:
            result = ToolResult(False, "", detail=str(exc))
            await self._emit("denied", f"fs_list: {path} failed")
            return result

    async def _list_directory(
        self,
        root: Path,
        current: Path,
        depth: int,
        entries: list[str],
    ) -> None:
        for entry in current.iterdir():
            resolved = self.policy.resolve(entry)
            entries.append(str(resolved.relative_to(root)))
            if resolved.is_dir() and depth > 0:
                await self._list_directory(root, resolved, depth - 1, entries)

    async def shell_run(
        self,
        argv: Sequence[str],
        cwd: str | Path | None = None,
    ) -> ToolResult:
        command = [str(item) for item in argv]
        try:
            self.policy.check_command(command)
            working_directory = self.policy.resolve(cwd or ".")
            if not working_directory.is_dir():
                return ToolResult(
                    False, "", detail=f"not a directory: {working_directory}"
                )

            description = json.dumps(
                {"argv": command, "cwd": str(working_directory)},
                ensure_ascii=False,
            )
            if not await self._approve("command", description):
                await self._emit("denied", description)
                return ToolResult(False, "", detail="operation denied by user")
            await self._emit("started", description)

            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=working_directory,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.policy.command_timeout
                )
            except TimeoutError:
                process.kill()
                await process.communicate()
                result = ToolResult(False, "", detail="command timed out")
                await self._emit("completed", f"{description} ok=false")
                return result

            output = stdout.decode("utf-8", errors="replace")
            error = stderr.decode("utf-8", errors="replace")
            content = "\n".join(item for item in (output, error) if item)
            suffix = f"exit_code={process.returncode}"
            content = f"{content.rstrip()}\n{suffix}" if content else suffix
            raw = content.encode("utf-8")
            truncated = len(raw) > self.policy.max_output_bytes
            if truncated:
                content = raw[: self.policy.max_output_bytes].decode(
                    "utf-8", errors="ignore"
                )
            result = ToolResult(
                process.returncode == 0,
                content,
                truncated=truncated,
                detail="" if process.returncode == 0 else "command failed",
            )
            await self._emit(
                "completed",
                f"{description} ok={result.ok} truncated={result.truncated}",
            )
            return result
        except (PolicyError, OSError, ValueError) as exc:
            await self._emit("denied", "shell_run failed")
            return ToolResult(False, "", detail=str(exc))

    async def dispatch(
        self,
        name: str,
        arguments: str | dict[str, Any],
    ) -> ToolResult:
        if name not in {"fs_read", "fs_list", "shell_run"}:
            return ToolResult(False, "", detail=f"unknown tool: {name}")
        if self.allowed_tools is not None and name not in self.allowed_tools:
            return ToolResult(
                False,
                "",
                detail=f"tool not allowed by active skill: {name}",
            )

        if isinstance(arguments, str):
            try:
                parsed: Any = json.loads(arguments)
            except (TypeError, json.JSONDecodeError) as exc:
                return ToolResult(False, "", detail=f"invalid JSON arguments: {exc}")
        else:
            parsed = arguments

        if not isinstance(parsed, dict):
            return ToolResult(False, "", detail="tool arguments must be a JSON object")

        if name == "fs_read":
            path = parsed.get("path")
            max_bytes = parsed.get("max_bytes")
            if not isinstance(path, str):
                return ToolResult(False, "", detail="fs_read.path must be a string")
            if max_bytes is not None and (
                isinstance(max_bytes, bool) or not isinstance(max_bytes, int)
            ):
                return ToolResult(
                    False, "", detail="fs_read.max_bytes must be an integer"
                )
            return await self.fs_read(path, max_bytes)

        if name == "fs_list":
            path = parsed.get("path")
            depth = parsed.get("depth")
            if not isinstance(path, str):
                return ToolResult(False, "", detail="fs_list.path must be a string")
            if depth is not None and (
                isinstance(depth, bool) or not isinstance(depth, int)
            ):
                return ToolResult(False, "", detail="fs_list.depth must be an integer")
            return await self.fs_list(path, depth if depth is not None else 1)

        argv = parsed.get("argv")
        cwd = parsed.get("cwd")
        if (
            not isinstance(argv, list)
            or not argv
            or not all(isinstance(item, str) for item in argv)
        ):
            return ToolResult(
                False, "", detail="shell_run.argv must be a non-empty string array"
            )
        if cwd is not None and not isinstance(cwd, str):
            return ToolResult(False, "", detail="shell_run.cwd must be a string")
        return await self.shell_run(argv, cwd)
