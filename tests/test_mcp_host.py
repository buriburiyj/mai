from pathlib import Path

import pytest

from mai.core.mcp_host import (
    MCPHostError,
    MCPRegistry,
    MCPServerConfig,
)


def test_registry_round_trip(tmp_path: Path) -> None:
    registry = MCPRegistry(tmp_path / "mcp.json")
    config = MCPServerConfig(
        name="local-files",
        transport="stdio",
        command="uv",
        args=("run", "server.py"),
    )

    registry.add(config)

    assert registry.get("local-files") == config
    assert registry.list() == [config]


def test_registry_rejects_duplicate(tmp_path: Path) -> None:
    registry = MCPRegistry(tmp_path / "mcp.json")
    config = MCPServerConfig(
        name="demo",
        transport="stdio",
        command="python",
    )
    registry.add(config)

    with pytest.raises(MCPHostError, match="already exists"):
        registry.add(config)


def test_registry_remove(tmp_path: Path) -> None:
    registry = MCPRegistry(tmp_path / "mcp.json")
    registry.add(
        MCPServerConfig(
            name="demo",
            transport="stdio",
            command="python",
        )
    )

    assert registry.remove("demo") is True
    assert registry.remove("demo") is False
    assert registry.list() == []


def test_allows_https_remote_server() -> None:
    MCPServerConfig(
        name="remote",
        transport="http",
        url="https://example.com/mcp",
    ).validate()


def test_allows_local_plain_http() -> None:
    MCPServerConfig(
        name="local",
        transport="http",
        url="http://localhost:8000/mcp",
    ).validate()


def test_rejects_remote_plain_http() -> None:
    config = MCPServerConfig(
        name="unsafe",
        transport="http",
        url="http://example.com/mcp",
    )

    with pytest.raises(MCPHostError, match="must use HTTPS"):
        config.validate()


def test_rejects_credentials_in_url() -> None:
    config = MCPServerConfig(
        name="unsafe",
        transport="http",
        url="https://user:password@example.com/mcp",
    )

    with pytest.raises(MCPHostError, match="must not be embedded"):
        config.validate()


def test_rejects_stdio_url_mix() -> None:
    config = MCPServerConfig(
        name="mixed",
        transport="stdio",
        command="python",
        url="https://example.com/mcp",
    )

    with pytest.raises(MCPHostError, match="cannot define a URL"):
        config.validate()


def test_external_tool_name_is_portable() -> None:
    from mai.core.mcp_host import _external_tool_name

    name = _external_tool_name("my-server", "read.file")

    assert name == "mcp_my_server_read_file"
    assert len(name) <= 64


@pytest.mark.asyncio
async def test_combined_executor_dispatches_to_owner() -> None:
    from typing import Any

    from mai.core.mcp_host import CombinedToolExecutor
    from mai.core.tools import ToolResult

    class FakeExecutor:
        @property
        def schemas(self) -> tuple[dict[str, Any], ...]:
            return (
                {
                    "type": "function",
                    "function": {
                        "name": "fake_tool",
                        "description": "Test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                },
            )

        async def dispatch(
            self,
            name: str,
            arguments: str | dict[str, Any],
        ) -> ToolResult:
            return ToolResult(name == "fake_tool", "called")

    executor = CombinedToolExecutor(FakeExecutor())
    result = await executor.dispatch("fake_tool", {})

    assert result.ok is True
    assert result.content == "called"


@pytest.mark.asyncio
async def test_combined_executor_rejects_unknown_tool() -> None:
    from mai.core.mcp_host import CombinedToolExecutor

    executor = CombinedToolExecutor()
    result = await executor.dispatch("missing", {})

    assert result.ok is False
    assert "unknown tool" in result.detail
