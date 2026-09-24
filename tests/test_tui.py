import pytest
from rich.markdown import Markdown
from rich.panel import Panel

from mai.tui import MaiChatApp, MCPApprovalModal


@pytest.mark.asyncio
async def test_mcp_approval_modal_css_parses() -> None:
    app = MaiChatApp()

    async with app.run_test() as pilot:
        await app.push_screen(
            MCPApprovalModal(
                "mai-local",
                "fs_list",
                {"path": "src/mai", "depth": 1},
            )
        )
        await pilot.pause()

        assert isinstance(app.screen, MCPApprovalModal)


@pytest.mark.asyncio
async def test_tui_uses_slash_commands_for_skills() -> None:
    app = MaiChatApp()

    async with app.run_test():
        assert not list(app.query("#skill"))
        assert not list(app.query("#mcp-server"))

        assert app._handle_slash_command("/skill auto") is True
        assert app.skill_mode == "auto"


@pytest.mark.asyncio
async def test_tui_slash_tools_command_changes_state() -> None:
    app = MaiChatApp()

    async with app.run_test():
        assert app.local_tools_enabled is True

        app._handle_slash_command("/tools off")

        assert app.local_tools_enabled is False


@pytest.mark.asyncio
async def test_unknown_slash_command_stays_local() -> None:
    app = MaiChatApp()

    async with app.run_test():
        assert app._handle_slash_command("/does-not-exist") is True
        assert "Unknown command" in app.query_one("#chat-log").text


@pytest.mark.asyncio
async def test_tui_uses_colored_selectable_markdown() -> None:
    app = MaiChatApp()

    async with app.run_test() as pilot:
        log = app.query_one("#chat-log")

        log.write(
            Panel(
                Markdown("## Important\n\n- highlighted item"),
                title="MAI",
            )
        )

        assert log.language == "markdown"
        assert log.theme == "mai-chat"
        assert "## MAI" in log.text
        assert "## Important" in log.text
        assert "╭" not in log.text

        await pilot.pause()
        assert len(log.query("MarkdownH2")) >= 2


@pytest.mark.asyncio
async def test_tui_tokens_command_changes_output_limit() -> None:
    app = MaiChatApp()

    async with app.run_test():
        assert app.max_tokens == 4096

        assert app._handle_slash_command("/tokens 8192") is True

        assert app.max_tokens == 8192
        assert "Maximum output tokens: 8192" in app.query_one("#chat-log").text


@pytest.mark.asyncio
async def test_tui_help_renders_as_markdown_table() -> None:
    app = MaiChatApp()

    async with app.run_test() as pilot:
        assert app._handle_slash_command("/help") is True

        await pilot.pause()

        log = app.query_one("#chat-log")
        assert "| 명령어 | 설명 |" in log.text
        assert "`/provider <name>`" in log.text
        assert "`/tokens [512-8192]`" in log.text
        assert len(log.query("MarkdownTable")) >= 1


@pytest.mark.asyncio
async def test_extended_slash_commands() -> None:
    app = MaiChatApp()

    async with app.run_test():
        assert app._handle_slash_command("/help mcp") is True
        assert app._handle_slash_command("/mcp status") is True
        assert app._handle_slash_command("/tools list") is True
        assert app._handle_slash_command("/about") is True
        assert app._handle_slash_command("/version") is True
        assert app._handle_slash_command("/pwd") is True
        assert app._handle_slash_command("/context") is True
        assert app._handle_slash_command("/statsu") is True

        transcript = app.query_one("#chat-log").text
        assert "Help · mcp" in transcript
        assert "MCP status" in transcript
        assert "Local tools" in transcript
        assert "About MAI" in transcript
        assert "Working directory" in transcript
        assert "혹시 다음 명령을 찾으셨나요?" in transcript
