from pathlib import Path

import pytest

from mai.core.policy import Approval, Policy
from mai.core.tools import ToolExecutor


@pytest.fixture
def policy(tmp_path: Path) -> Policy:
    return Policy(
        tmp_path,
        Approval.ASK,
        max_read_bytes=8,
        max_output_bytes=32,
        command_timeout=0.1,
    )


@pytest.mark.asyncio
async def test_fs_read_reads_and_truncates(policy: Policy, tmp_path: Path) -> None:
    (tmp_path / "note.txt").write_text("123456789", encoding="utf-8")
    result = await ToolExecutor(policy).fs_read("note.txt")
    assert result.ok is True
    assert result.content == "12345678"
    assert result.truncated is True


@pytest.mark.asyncio
async def test_fs_read_rejects_workspace_escape(policy: Policy) -> None:
    result = await ToolExecutor(policy).fs_read("../outside.txt")
    assert result.ok is False
    assert "escapes workspace" in result.detail


@pytest.mark.asyncio
async def test_shell_run_rejects_disallowed_command(policy: Policy) -> None:
    result = await ToolExecutor(policy).shell_run(["rm", "-rf", "file.txt"])
    assert result.ok is False
    assert "command not allowed" in result.detail


@pytest.mark.asyncio
async def test_shell_run_times_out(policy: Policy, tmp_path: Path) -> None:
    script = tmp_path / "sleep.py"
    script.write_text("import time\ntime.sleep(1)\n", encoding="utf-8")
    auto_policy = Policy(
        policy.root,
        Approval.AUTO,
        max_read_bytes=policy.max_read_bytes,
        max_output_bytes=policy.max_output_bytes,
        command_timeout=policy.command_timeout,
    )
    result = await ToolExecutor(auto_policy).shell_run(["python3", str(script)])
    assert result.ok is False
    assert result.detail == "command timed out"


@pytest.mark.asyncio
async def test_shell_run_respects_approval(policy: Policy, tmp_path: Path) -> None:
    script = tmp_path / "print_ok.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    async def deny(kind: str, description: str) -> bool:
        assert kind == "command"
        assert "python3" in description
        return False

    result = await ToolExecutor(policy, approval_callback=deny).shell_run(
        ["python3", str(script)]
    )
    assert result.ok is False
    assert result.detail == "operation denied by user"


@pytest.mark.asyncio
async def test_shell_run_executes_after_approval(
    policy: Policy,
    tmp_path: Path,
) -> None:
    script = tmp_path / "print_ok.py"
    script.write_text("print('ok')\n", encoding="utf-8")

    async def approve(kind: str, description: str) -> bool:
        return kind == "command"

    result = await ToolExecutor(policy, approval_callback=approve).shell_run(
        ["python3", str(script)]
    )
    assert result.ok is True
    assert "ok" in result.content
    assert "exit_code=0" in result.content


@pytest.mark.asyncio
async def test_read_only_policy_blocks_commands(tmp_path: Path) -> None:
    policy = Policy(tmp_path, Approval.READ_ONLY)
    result = await ToolExecutor(policy).shell_run(["pwd"])
    assert result.ok is False
    assert "disabled in read-only mode" in result.detail


@pytest.mark.asyncio
async def test_fs_list_lists_nested_files(policy: Policy, tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("b", encoding="utf-8")
    result = await ToolExecutor(policy).fs_list(".", depth=2)
    assert result.ok is True
    assert "a.txt" in result.content
    assert "nested/b.txt" in result.content


@pytest.mark.asyncio
async def test_dispatch_rejects_unknown_tool(policy: Policy) -> None:
    result = await ToolExecutor(policy).dispatch("exec", "{}")
    assert result.ok is False
    assert result.detail == "unknown tool: exec"


@pytest.mark.asyncio
async def test_dispatch_rejects_invalid_arguments(policy: Policy) -> None:
    result = await ToolExecutor(policy).dispatch("fs_read", "not-json")
    assert result.ok is False
    assert result.detail.startswith("invalid JSON arguments:")


@pytest.mark.asyncio
async def test_dispatch_rejects_wrong_argument_types(policy: Policy) -> None:
    result = await ToolExecutor(policy).dispatch(
        "shell_run",
        {"argv": ["python3", 1]},
    )
    assert result.ok is False
    assert "string array" in result.detail


def test_allowed_tools_filter_exposed_schemas(tmp_path: Path) -> None:
    from mai.core.policy import Approval, Policy
    from mai.core.tools import ToolExecutor

    executor = ToolExecutor(
        Policy.load(root=tmp_path, approval=Approval.READ_ONLY),
        allowed_tools=("fs_list",),
    )

    names = {schema["function"]["name"] for schema in executor.schemas}

    assert names == {"fs_list"}
