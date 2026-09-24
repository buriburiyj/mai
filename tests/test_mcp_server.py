from pathlib import Path

import pytest

from mai.core.mcp_server import MCPWorkspace, create_mcp_server


@pytest.mark.asyncio
async def test_mcp_workspace_reads_inside_root(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hello", encoding="utf-8")
    workspace = MCPWorkspace(tmp_path)

    result = await workspace.read_file("hello.txt")

    assert result["ok"] is True
    assert result["content"] == "hello"


@pytest.mark.asyncio
async def test_mcp_workspace_blocks_parent_traversal(tmp_path: Path) -> None:
    workspace = MCPWorkspace(tmp_path)

    result = await workspace.read_file("../outside.txt")

    assert result["ok"] is False
    assert result["detail"]


@pytest.mark.asyncio
async def test_mcp_workspace_lists_files(tmp_path: Path) -> None:
    (tmp_path / "one.txt").write_text("1", encoding="utf-8")
    workspace = MCPWorkspace(tmp_path)

    result = await workspace.list_files(".", depth=0)

    assert result["ok"] is True
    assert "one.txt" in result["content"]


def test_creates_mcp_server(tmp_path: Path) -> None:
    server = create_mcp_server(tmp_path)

    assert server is not None
