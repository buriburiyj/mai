from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from mai.core.policy import Approval, Policy
from mai.core.skills import SkillCatalog, SkillError
from mai.core.tools import ToolExecutor, ToolResult


def _serialize_result(result: ToolResult) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "content": result.content,
        "truncated": result.truncated,
        "detail": result.detail,
    }


class MCPWorkspace:
    """Safe MCP adapter around MAI's existing ToolExecutor."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        policy = Policy.load(root=self.root, approval=Approval.READ_ONLY)
        self.executor = ToolExecutor(policy)
        self.skills = SkillCatalog.default(self.root)

    async def read_file(
        self,
        path: str,
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        result = await self.executor.fs_read(path, max_bytes)
        return _serialize_result(result)

    async def list_files(
        self,
        path: str = ".",
        depth: int = 1,
    ) -> dict[str, Any]:
        result = await self.executor.fs_list(path, depth)
        return _serialize_result(result)

    def skill_index(self) -> str:
        skills, errors = self.skills.validate_all()
        payload = {
            "skills": [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "allowed_tools": list(skill.allowed_tools),
                }
                for skill in skills
            ],
            "errors": errors,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def skill_text(self, name: str) -> str:
        skill = self.skills.load(name)
        return (
            f"# {skill.metadata.name}\n\n"
            f"{skill.metadata.description}\n\n"
            f"{skill.instructions}"
        )


def create_mcp_server(root: Path) -> MCPServer:
    workspace = MCPWorkspace(root)

    server = MCPServer(
        name="mai",
        title="MAI Workspace Tools",
        description=("Read-only workspace tools and Agent Skills provided by MAI."),
        instructions=(
            "All file paths are restricted to the configured workspace. "
            "Command execution is not exposed through this MCP server."
        ),
        version="0.1.0",
    )

    @server.tool(
        name="fs_read",
        description="Read a UTF-8 text file inside the MAI workspace.",
    )
    async def fs_read(
        path: str,
        max_bytes: int | None = None,
    ) -> dict[str, Any]:
        return await workspace.read_file(path, max_bytes)

    @server.tool(
        name="fs_list",
        description="List files and directories inside the MAI workspace.",
    )
    async def fs_list(
        path: str = ".",
        depth: int = 1,
    ) -> dict[str, Any]:
        return await workspace.list_files(path, depth)

    @server.resource(
        "mai://workspace",
        name="workspace",
        description="Absolute path of the active MAI workspace.",
        mime_type="text/plain",
    )
    def workspace_resource() -> str:
        return str(workspace.root)

    @server.resource(
        "mai://skills",
        name="skills",
        description="Available Agent Skills and their descriptions.",
        mime_type="application/json",
    )
    def skills_resource() -> str:
        return workspace.skill_index()

    @server.resource(
        "mai://skills/{name}",
        name="skill",
        description="Load the complete instructions for one Agent Skill.",
        mime_type="text/markdown",
    )
    def skill_resource(name: str) -> str:
        try:
            return workspace.skill_text(name)
        except SkillError as exc:
            raise ValueError(str(exc)) from exc

    @server.prompt(
        name="use-skill",
        description="Apply one MAI Agent Skill to a user request.",
    )
    def use_skill(name: str, request: str) -> str:
        try:
            return workspace.skills.apply(name, request)
        except SkillError as exc:
            raise ValueError(str(exc)) from exc

    return server
