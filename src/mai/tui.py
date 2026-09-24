import asyncio
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from io import StringIO
from pathlib import Path
from typing import ClassVar

import aiosqlite
from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.text import Text
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widgets import Button, Footer, Input, Select, Static
from textual.widgets import Markdown as TextualMarkdown

from mai.core.commands import (
    command_suggestions,
    parse_command,
    render_help,
)
from mai.core.mcp_host import (
    CombinedToolExecutor,
    MCPHostError,
    MCPRegistry,
    MCPToolExecutor,
)
from mai.core.policy import Approval, Policy
from mai.core.router import MultiProviderRouter, RouterError
from mai.core.sessions import SessionStore
from mai.core.skills import SkillCatalog, SkillError
from mai.core.tools import ToolExecutor

PROVIDER_OPTIONS = [
    ("Automatic fallback", "auto"),
    ("Google Gemini", "gemini"),
    ("Groq", "groq"),
    ("OpenRouter", "openrouter"),
    ("Cerebras", "cerebras"),
]


class ApprovalModal(ModalScreen[bool]):
    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("y", "allow", "Allow"),
        ("enter", "allow", "Allow"),
        ("n", "deny", "Deny"),
        ("escape", "deny", "Deny"),
    ]

    CSS = """
    ApprovalModal {
        align: center middle;
    }

    #approval-box {
        width: 78;
        height: auto;
        padding: 1 2;
        background: #151e33;
        border: thick #f0a04b;
    }

    #approval-buttons {
        height: 3;
        margin-top: 1;
        align: right middle;
    }

    #approval-buttons Button {
        margin-left: 1;
    }
    """

    def __init__(self, argv: list[str], cwd: str) -> None:
        super().__init__()
        self.argv = argv
        self.cwd = cwd

    def compose(self) -> ComposeResult:
        command = json.dumps(self.argv, ensure_ascii=False)
        with Vertical(id="approval-box"):
            yield Static("Command approval required", id="approval-title")
            yield Static(f"argv: {command}")
            yield Static(f"cwd: {self.cwd}")
            with Horizontal(id="approval-buttons"):
                yield Button("Allow", id="allow", variant="success")
                yield Button("Deny", id="deny", variant="error")

    def on_mount(self) -> None:
        self.query_one("#deny", Button).focus()

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)

    def on_key(self, event: Key) -> None:
        if event.key in {"y", "enter"}:
            event.stop()
            self.action_allow()
        elif event.key in {"n", "escape"}:
            event.stop()
            self.action_deny()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "allow")


class MCPApprovalModal(ModalScreen[bool]):
    """Require explicit approval for an external MCP tool call."""

    CSS = """
    MCPApprovalModal {
        align: center middle;
        background: #000000b3;
    }

    #mcp-approval-box {
        width: 78;
        height: auto;
        max-height: 80%;
        padding: 1 2;
        background: #11182b;
        border: tall #ffb454;
    }

    #mcp-approval-title {
        height: 2;
        color: #ffcf70;
        text-style: bold;
    }

    #mcp-approval-details {
        height: auto;
        max-height: 18;
        padding: 1;
        background: #0b1020;
        color: #e8ecf4;
        overflow-y: auto;
    }

    #mcp-approval-buttons {
        height: 3;
        align-horizontal: right;
        margin-top: 1;
    }

    #mcp-allow {
        margin-right: 1;
    }
    """

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("y", "allow", "Allow"),
        ("n", "deny", "Deny"),
        ("escape", "deny", "Deny"),
        ("enter", "deny", "Deny"),
    ]

    def __init__(
        self,
        server: str,
        tool: str,
        arguments: dict[str, object],
    ) -> None:
        super().__init__()
        self.server = server
        self.tool = tool
        self.arguments = arguments

    def compose(self) -> ComposeResult:
        details = json.dumps(
            {
                "server": self.server,
                "tool": self.tool,
                "arguments": self.arguments,
            },
            ensure_ascii=False,
            indent=2,
        )

        with Vertical(id="mcp-approval-box"):
            yield Static(
                "External MCP tool approval required",
                id="mcp-approval-title",
            )
            yield Static(details, id="mcp-approval-details")
            with Horizontal(id="mcp-approval-buttons"):
                yield Button(
                    "Allow (Y)",
                    id="mcp-allow",
                    variant="success",
                )
                yield Button(
                    "Deny (N / Enter)",
                    id="mcp-deny",
                    variant="error",
                )

    def on_mount(self) -> None:
        self.query_one("#mcp-deny", Button).focus()

    @on(Button.Pressed)
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mcp-allow":
            self.dismiss(True)
        else:
            self.dismiss(False)

    def action_allow(self) -> None:
        self.dismiss(True)

    def action_deny(self) -> None:
        self.dismiss(False)


class SelectableChatLog(TextualMarkdown):
    """Rendered Markdown chat log with text selection support."""

    # Compatibility properties retained for existing callers and tests.
    language = "markdown"
    theme = "mai-chat"

    def __init__(self, *args, **kwargs) -> None:
        # Accept previous RichLog/TextArea-compatible options.
        for option in (
            "highlight",
            "markup",
            "wrap",
            "auto_scroll",
            "read_only",
            "soft_wrap",
            "show_line_numbers",
            "show_cursor",
            "highlight_cursor_line",
            "compact",
            "language",
            "theme",
        ):
            kwargs.pop(option, None)

        super().__init__("", *args, **kwargs)

    @property
    def text(self) -> str:
        """Return the complete Markdown transcript."""
        return self.source

    @property
    def selected_text(self) -> str:
        """Return the current Textual screen selection."""
        if not self.is_mounted:
            return ""
        return self.screen.get_selected_text() or ""

    @staticmethod
    def _plain(value: object) -> str:
        if isinstance(value, Text):
            return value.plain
        return str(value)

    @classmethod
    def _to_markdown(cls, content: object) -> str:
        """Convert chat renderables to Markdown source."""
        if isinstance(content, Panel):
            title = cls._plain(content.title).strip() if content.title else ""
            renderable = content.renderable

            if isinstance(renderable, RichMarkdown):
                body = renderable.markup
            elif isinstance(renderable, Text):
                body = renderable.plain
            else:
                body = str(renderable)

            if title:
                return f"## {title}\n\n{body}"
            return body

        if isinstance(content, RichMarkdown):
            return content.markup

        if isinstance(content, Text):
            plain = content.plain

            if plain.startswith("Tool "):
                label, separator, detail = plain.partition(":")
                if separator:
                    return f"> **{label}:**{detail}"

            if plain.startswith("Skill activated:"):
                return f"> **{plain}**"

            return plain

        if isinstance(content, str):
            return content

        buffer = StringIO()
        console = Console(
            file=buffer,
            width=120,
            force_terminal=False,
            color_system=None,
            soft_wrap=True,
        )
        console.print(content)
        return buffer.getvalue().rstrip()

    def write(self, content, **_kwargs) -> None:
        """Append and render a Markdown chat message."""
        rendered = self._to_markdown(content).strip()
        if not rendered:
            return

        separator = "\n\n---\n\n" if self.source else ""
        self.update(f"{self.source}{separator}{rendered}")
        self.call_after_refresh(self.scroll_end, animate=False)

    def clear(self) -> None:
        """Clear the rendered transcript."""
        self.update("")


class MaiChatApp(App[None]):
    CSS_SELECTION = """
    #chat-log .text-area--selection {
        background: #3b82f6;
        color: #ffffff;
        text-style: bold;
    }
    """

    TITLE = "MAI"
    SUB_TITLE = "Multi-provider AI"

    CSS = """
    Screen {
        background: #0b1020;
        color: #e8ecf4;
    }

    #topbar {
        height: 5;
        padding: 1 2;
        background: #11182b;
        border-bottom: solid #273657;
    }

    #brand {
        width: 1fr;
        content-align: left middle;
        color: #75d7ff;
        text-style: bold;
    }

    #provider {
        width: 25;
        margin-right: 1;
    }



    #status {
        width: 34;
        content-align: right middle;
        color: #8b9bb4;
    }

    #chat-log {
        height: 1fr;
        padding: 1 3;
        background: #0b1020;
        scrollbar-color: #375178;
        scrollbar-background: #11182b;
    }

    #stream {
        height: auto;
        max-height: 14;
        padding: 1 3;
        background: #0e1526;
        border-top: solid #273657;
        color: #d7dee9;
        overflow-y: auto;
        display: none;
    }

    #stream.active {
        display: block;
    }

    #composer {
        height: 5;
        padding: 1 2;
        background: #11182b;
        border-top: solid #273657;
    }

    #message {
        width: 1fr;
        border: tall #375178;
        background: #151e33;
    }

    #message:focus {
        border: tall #5ac8fa;
    }

    Footer {
        background: #0d1425;
        color: #8898b5;
    }

    #chat-log .text-area--selection {
        background: #3b82f6;
        color: #ffffff;
        text-style: bold;
    }


    /* Rendered chat Markdown */
    #chat-log {
        height: 1fr;
        overflow-y: auto;
        scrollbar-color: #f59e0b;
        scrollbar-color-hover: #ffb454;
        scrollbar-color-active: #ff9f43;
    }

    #chat-log MarkdownBlock {
        color: #d7dee9;
        background: transparent;
    }

    #chat-log MarkdownH1,
    #chat-log MarkdownH2,
    #chat-log MarkdownH3,
    #chat-log MarkdownH4,
    #chat-log MarkdownH5,
    #chat-log MarkdownH6 {
        color: #ff9f43;
        text-style: bold;
        margin-top: 1;
        margin-bottom: 1;
    }

    #chat-log MarkdownH1 {
        border-bottom: solid #d97706;
    }

    #chat-log MarkdownH2 {
        border-bottom: tall #8a4f12;
    }

    #chat-log MarkdownHorizontalRule {
        color: #d97706;
        margin: 1 0;
    }

    #chat-log MarkdownBlockQuote {
        color: #ffcf99;
        background: #171522;
        border-left: thick #ff9f43;
        padding-left: 1;
        margin: 1 0;
    }

    #chat-log MarkdownFence {
        color: #a5e8ff;
        background: #101827;
        border: round #334155;
        padding: 1;
        margin: 1 0;
    }

    #chat-log MarkdownTable {
        color: #d7dee9;
        background: #101827;
        border: round #d97706;
        margin: 1 0;
    }

"""

    BINDINGS: ClassVar[list[tuple[str, str, str]]] = [
        ("ctrl+l", "clear_chat", "Clear"),
        ("ctrl+x", "cancel_request", "Cancel"),
        ("ctrl+c", "quit", "Quit"),
        ("escape", "focus_input", "Input"),
    ]

    def __init__(
        self,
        provider: str = "auto",
        max_tokens: int = 4096,
        history_turns: int = 10,
        session_id: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__()
        self.router = MultiProviderRouter()
        self.provider = provider
        self.max_tokens = max_tokens
        self.history_turns = history_turns
        self.timeout = timeout
        self.history: list[tuple[str, str]] = []
        self.sessions = SessionStore()
        self.session_id = session_id
        self._persist_warned = False
        self._stream_buffer = ""
        self._stream_timer = None
        self._request_worker = None
        self.skill_mode = "auto"
        self.active_skill_name: str | None = None
        self.skill_catalog = SkillCatalog.default(Path.cwd())
        self.local_tools_enabled = True
        self.mcp_server = "off"
        self.mcp_registry = MCPRegistry()
        self.tool_executor = ToolExecutor(
            Policy.load(root=Path.cwd(), approval=Approval.ASK),
            approval_callback=self._approve_tool,
            event_callback=self._tool_event,
        )

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static(
                "◆ MAI  ·  Free Multi-AI Agent",
                id="brand",
            )
            yield Select(
                PROVIDER_OPTIONS,
                value=self.provider,
                allow_blank=False,
                id="provider",
            )
            yield Static("Ready · /help", id="status")

        yield SelectableChatLog(
            id="chat-log",
            wrap=True,
            highlight=True,
            markup=False,
        )

        yield Static("", id="stream")

        with Horizontal(id="composer"):
            yield Input(
                placeholder="메시지를 입력하거나 /help · /tokens 8192",
                suggester=SuggestFromList(
                    command_suggestions(),
                    case_sensitive=False,
                ),
                id="message",
            )

        yield Footer()

    def on_mount(self) -> None:
        log = self.query_one("#chat-log", SelectableChatLog)
        log.write(
            Panel(
                Text(
                    "여러 무료 AI를 자동으로 선택하는 MAI입니다.\n"
                    "질문을 입력하거나 상단에서 공급자를 선택하세요.",
                    style="bright_white",
                ),
                title="Welcome",
                title_align="left",
                border_style="bright_blue",
                padding=(1, 2),
            )
        )
        self.query_one("#message", Input).focus()

    def action_clear_chat(self) -> None:
        if self._request_worker is not None:
            self._request_worker.cancel()
        self.history.clear()
        self.session_id = None
        log = self.query_one("#chat-log", SelectableChatLog)
        log.clear()
        log.write(
            Panel(
                "Conversation history cleared.",
                border_style="dim",
            )
        )

    def action_focus_input(self) -> None:
        self.query_one("#message", Input).focus()

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id != "provider":
            return

        self.provider = str(event.value)
        self.query_one("#status", Static).update(f"Provider: {self.provider}")

    async def _ensure_session(self) -> str:
        if self.session_id is None:
            self.session_id = await self.sessions.create()
        return self.session_id

    async def _persist(
        self,
        role: str,
        content: str,
        provider: str = "",
        model: str = "",
    ) -> None:
        try:
            session_id = await self._ensure_session()
            await self.sessions.append(session_id, role, content, provider, model)
        except (aiosqlite.Error, OSError) as exc:
            # 저장 실패가 대화를 막지는 않는다. 한 번만 알려 준다.
            if not self._persist_warned:
                self._persist_warned = True
                self.notify(
                    f"Session saving disabled: {exc}",
                    severity="warning",
                )

    async def load_session(self, session_id: str) -> None:
        """저장된 세션을 불러와 history를 복원한다."""
        messages = await self.sessions.messages(session_id)
        self.session_id = session_id
        self.history = [
            (m.role, m.content) for m in messages if m.role in {"user", "assistant"}
        ]

    async def _approve_tool(self, kind: str, description: str) -> bool:
        if kind == "mcp":
            try:
                payload = json.loads(description)
                server = payload["server"]
                tool = payload["tool"]
                arguments = payload["arguments"]
                if not isinstance(server, str) or not isinstance(tool, str):
                    return False
                if not isinstance(arguments, dict):
                    return False
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                return False

            try:
                result = await self.push_screen_wait(
                    MCPApprovalModal(server, tool, arguments)
                )
            except RuntimeError:
                return False
            return result is True

        if kind != "command":
            return False

        try:
            payload = json.loads(description)
            argv = payload["argv"]
            cwd = payload["cwd"]
            if not isinstance(argv, list) or not all(
                isinstance(item, str) for item in argv
            ):
                return False
            if not isinstance(cwd, str):
                return False
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False

        try:
            result = await self.push_screen_wait(ApprovalModal(argv, cwd))
        except RuntimeError:
            return False
        return result is True

    async def _tool_event(self, event: str, detail: str) -> None:
        display = detail
        if detail.startswith("{"):
            try:
                # MCP completion details may append status fields after JSON.
                payload, _json_end = json.JSONDecoder().raw_decode(detail)
                if "server" in payload and "tool" in payload:
                    display = (
                        f"MCP {payload['server']} · {payload['tool']} · "
                        f"{json.dumps(payload['arguments'], ensure_ascii=False)}"
                    )
                else:
                    display = (
                        f"argv={json.dumps(payload['argv'], ensure_ascii=False)} "
                        f"cwd={payload['cwd']}"
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                display = "tool details unavailable"

        labels = {
            "started": "Tool requested",
            "completed": "Tool completed",
            "denied": "Tool denied",
        }
        self.query_one("#chat-log", SelectableChatLog).write(
            Text(f"{labels.get(event, 'Tool update')}: {display}", style="dim")
        )
        await self._persist(
            "tool",
            f"event={event}; {self._tool_summary(detail)}",
        )

    @staticmethod
    def _tool_summary(detail: str) -> str:
        if detail.startswith("{"):
            try:
                payload = json.loads(detail)
                if "server" in payload and "tool" in payload:
                    return (
                        f"tool=mcp; server={payload['server']}; "
                        f"name={payload['tool']}; "
                        f"arguments={json.dumps(payload['arguments'], ensure_ascii=False)}"
                    )
                return (
                    "tool=shell_run; "
                    f"argv={json.dumps(payload['argv'], ensure_ascii=False)}; "
                    f"cwd={payload['cwd']}"
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return "tool=unknown; details=unavailable"
        return f"target={detail}"

    def _conversation_prompt(self, user_input: str) -> str:
        recent = self.history[-(self.history_turns * 2) :]

        lines = [
            "Continue this conversation naturally.",
            "Answer the user's newest message.",
            "",
        ]

        for role, content in recent:
            label = "User" if role == "user" else "Assistant"
            lines.append(f"{label}: {content}")

        lines.append(f"User: {user_input}")
        lines.append("Assistant:")
        return "\n".join(lines)

    def _write_command_result(
        self,
        content: str,
        *,
        title: str = "Command",
        error: bool = False,
    ) -> None:
        self.query_one("#chat-log", SelectableChatLog).write(
            Panel(
                content,
                title=title,
                border_style="red" if error else "cyan",
                padding=(0, 1),
            )
        )

    def _command_help(self, arguments: list[str]) -> str:
        command = arguments[0] if arguments else None
        return render_help(command)

    def _command_status(self) -> str:
        return (
            "| 설정 | 값 |\n"
            "| :--- | :--- |\n"
            f"| Provider | `{self.provider}` |\n"
            f"| 최대 출력 토큰 | `{self.max_tokens}` |\n"
            f"| Skill mode | `{self.skill_mode}` |\n"
            f"| Active Skill | "
            f"`{self.active_skill_name or 'none'}` |\n"
            f"| MCP server | `{self.mcp_server}` |\n"
            f"| Local tools | "
            f"`{'on' if self.local_tools_enabled else 'off'}` |\n"
            f"| Session | `{self.session_id or 'new'}` |"
        )

    @staticmethod
    def _command_about() -> str:
        return (
            "MAI는 여러 무료 AI Provider와 Agent Skill, "
            "로컬 도구 및 MCP 서버를 하나의 TUI에서 사용하는 "
            "멀티 AI 에이전트입니다.\n\n"
            "- Provider 자동 선택\n"
            "- Agent Skill 자동 선택\n"
            "- 로컬 및 MCP 도구 승인 실행\n"
            "- Markdown 대화 렌더링\n"
            "- 세션 저장"
        )

    @staticmethod
    def _command_version() -> str:
        def package_version(name: str) -> str:
            try:
                return version(name)
            except PackageNotFoundError:
                return "unknown"

        return (
            "| 구성 요소 | 버전 |\n"
            "| :--- | :--- |\n"
            f"| MAI | `{package_version('mai')}` |\n"
            f"| Python | `{platform.python_version()}` |\n"
            f"| Textual | `{package_version('textual')}` |\n"
            f"| Rich | `{package_version('rich')}` |"
        )

    def _command_context(self) -> str:
        message_count = len(self.history)
        characters = sum(len(content) for _role, content in self.history)
        estimated_tokens = max(0, characters // 4)

        return (
            "| 항목 | 값 |\n"
            "| :--- | ---: |\n"
            f"| 저장된 메시지 | {message_count} |\n"
            f"| 문자 수 | {characters:,} |\n"
            f"| 예상 컨텍스트 토큰 | 약 {estimated_tokens:,} |\n"
            f"| 유지할 이전 turn | {self.history_turns} |\n"
            f"| 최대 출력 토큰 | {self.max_tokens:,} |"
        )

    def _command_doctor(self) -> str:
        skills, skill_errors = self.skill_catalog.validate_all()

        try:
            servers = self.mcp_registry.list()
            mcp_error = ""
        except MCPHostError as exc:
            servers = []
            mcp_error = str(exc)

        rows = [
            "| 검사 | 상태 | 상세 |",
            "| :--- | :---: | :--- |",
            (f"| Provider | ✅ | 현재 `{self.provider}` |"),
            (
                f"| Agent Skills | "
                f"{'✅' if not skill_errors else '⚠️'} | "
                f"{len(skills)}개 발견 |"
            ),
            (
                f"| MCP 설정 | "
                f"{'❌' if mcp_error else '✅'} | "
                f"{mcp_error or f'{len(servers)}개 서버 등록'} |"
            ),
            (
                f"| 로컬 도구 | "
                f"{'✅' if self.local_tools_enabled else '⏸️'} | "
                f"{'활성화' if self.local_tools_enabled else '비활성화'} |"
            ),
            (
                f"| 작업 디렉터리 | "
                f"{'✅' if Path.cwd().is_dir() else '❌'} | "
                f"`{Path.cwd()}` |"
            ),
        ]

        if skill_errors:
            rows.extend(
                (
                    "",
                    "### Skill 경고",
                    "",
                    *(f"- {error}" for error in skill_errors),
                )
            )

        return "\n".join(rows)

    def _handle_cancel_command(self) -> None:
        worker = self._request_worker

        if worker is None:
            self._write_command_result(
                "실행 중인 요청이 없습니다.",
                title="Cancel",
            )
            return

        worker.cancel()
        self._request_worker = None

        message = self.query_one("#message", Input)
        message.disabled = False
        message.focus()

        self.query_one("#status", Static).update("Cancelled · /help")
        self._write_command_result(
            "현재 요청을 취소했습니다.",
            title="Cancel",
        )

    def _handle_provider_command(self, arguments: list[str]) -> None:
        names = {str(value) for _label, value in PROVIDER_OPTIONS}

        if not arguments:
            self._write_command_result(
                f"Current provider: {self.provider}",
                title="Provider",
            )
            return

        requested = arguments[0].lower()

        if requested == "list":
            labels = {str(value): str(label) for label, value in PROVIDER_OPTIONS}
            rows = [
                "| Provider | 이름 | 활성 |",
                "| :--- | :--- | :---: |",
            ]
            for name in sorted(names):
                rows.append(
                    f"| `{name}` | {labels.get(name, name)} | "
                    f"{'✅' if name == self.provider else ''} |"
                )
            self._write_command_result(
                "\n".join(rows),
                title="Providers",
            )
            return

        if requested == "use":
            if len(arguments) < 2:
                self._write_command_result(
                    "Usage: `/provider use <name>`",
                    title="Provider error",
                    error=True,
                )
                return
            requested = arguments[1].lower()

        if requested not in names:
            self._write_command_result(
                f"Unknown provider: {requested}\nUse /provider list.",
                title="Provider error",
                error=True,
            )
            return

        self.provider = requested
        self.query_one("#provider", Select).value = requested
        self.query_one("#status", Static).update(f"Provider: {requested}")
        self._write_command_result(
            f"Provider selected: {requested}",
            title="Provider",
        )

    def _handle_skill_command(self, arguments: list[str]) -> None:
        if not arguments:
            self._write_command_result(
                f"Skill mode: {self.skill_mode}\n"
                f"Active Skill: {self.active_skill_name or 'none'}",
                title="Skill",
            )
            return

        action = arguments[0].lower()

        if action == "reload":
            self.skill_catalog = SkillCatalog.default(Path.cwd())
            skills, errors = self.skill_catalog.validate_all()
            self._write_command_result(
                f"Skill catalog reloaded.\n\n"
                f"- Skills: **{len(skills)}**\n"
                f"- Errors: **{len(errors)}**",
                title="Agent Skills",
                error=bool(errors),
            )
            return

        if action == "list":
            skills = self.skill_catalog.list()
            if skills:
                rows = [
                    "| Skill | 설명 | 허용 도구 |",
                    "| :--- | :--- | :--- |",
                ]
                for skill in skills:
                    allowed = ", ".join(skill.allowed_tools) or "none"
                    rows.append(
                        f"| `{skill.name}` | {skill.description} | `{allowed}` |"
                    )
                content = "\n".join(rows)
            else:
                content = "등록된 Agent Skill이 없습니다."

            self._write_command_result(
                content,
                title="Agent Skills",
            )
            return

        if action == "show":
            if len(arguments) < 2:
                self._write_command_result(
                    "Usage: /skill show <name>",
                    title="Skill error",
                    error=True,
                )
                return

            try:
                skill = self.skill_catalog.load(arguments[1])
            except SkillError as exc:
                self._write_command_result(
                    str(exc),
                    title="Skill error",
                    error=True,
                )
                return

            allowed = ", ".join(skill.metadata.allowed_tools) or "none"
            self._write_command_result(
                f"{skill.metadata.description}\n\n"
                f"Allowed tools: {allowed}\n\n"
                f"{skill.instructions}",
                title=f"Skill · {skill.metadata.name}",
            )
            return

        if action == "use":
            if len(arguments) < 2:
                self._write_command_result(
                    "Usage: `/skill use <name>`",
                    title="Skill error",
                    error=True,
                )
                return
            action = arguments[1].lower()

        if action in {"auto", "off"}:
            self.skill_mode = action
            self.active_skill_name = None
            self.query_one("#status", Static).update(f"Skill: {action}")
            self._write_command_result(
                f"Skill mode selected: {action}",
                title="Skill",
            )
            return

        try:
            metadata = self.skill_catalog.get_metadata(action)
        except SkillError as exc:
            self._write_command_result(
                f"{exc}\nUse /skill list.",
                title="Skill error",
                error=True,
            )
            return

        self.skill_mode = metadata.name
        self.active_skill_name = None
        self.query_one("#status", Static).update(f"Skill: {metadata.name}")
        self._write_command_result(
            f"Skill selected: {metadata.name}\n"
            f"Allowed tools: "
            f"{', '.join(metadata.allowed_tools) or 'none'}",
            title="Skill",
        )

    def _handle_mcp_command(self, arguments: list[str]) -> None:
        try:
            all_servers = self.mcp_registry.list()
            servers = [server for server in all_servers if server.enabled]
        except MCPHostError as exc:
            self._write_command_result(
                str(exc),
                title="MCP error",
                error=True,
            )
            return

        if not arguments:
            arguments = ["status"]

        action = arguments[0].lower()

        if action == "status":
            active = next(
                (server for server in all_servers if server.name == self.mcp_server),
                None,
            )
            self._write_command_result(
                "| 항목 | 값 |\n"
                "| :--- | :--- |\n"
                f"| 선택된 서버 | `{self.mcp_server}` |\n"
                f"| 활성화 상태 | "
                f"`{'off' if active is None else 'on'}` |\n"
                f"| 등록된 서버 | `{len(all_servers)}` |\n"
                f"| 사용 가능한 서버 | `{len(servers)}` |",
                title="MCP status",
            )
            return

        if action == "list":
            if not all_servers:
                self._write_command_result(
                    "등록된 MCP 서버가 없습니다.",
                    title="MCP servers",
                )
                return

            rows = [
                "| 서버 | Transport | 설정 | 선택 |",
                "| :--- | :--- | :---: | :---: |",
            ]
            for server in all_servers:
                rows.append(
                    f"| `{server.name}` | `{server.transport}` | "
                    f"{'✅' if server.enabled else '⏸️'} | "
                    f"{'✅' if server.name == self.mcp_server else ''} |"
                )

            self._write_command_result(
                "\n".join(rows),
                title="MCP servers",
            )
            return

        if action == "use":
            if len(arguments) < 2:
                self._write_command_result(
                    "Usage: `/mcp use <server>`",
                    title="MCP error",
                    error=True,
                )
                return
            requested = arguments[1]
        else:
            requested = arguments[0]

        if requested.lower() == "off":
            self.mcp_server = "off"
            self.query_one("#status", Static).update("MCP: off")
            self._write_command_result(
                "MCP를 비활성화했습니다.",
                title="MCP",
            )
            return

        names = {server.name for server in servers}
        if requested not in names:
            self._write_command_result(
                f"알 수 없거나 비활성화된 MCP 서버입니다: "
                f"`{requested}`\n\n"
                "등록된 서버는 `/mcp list`로 확인하세요.",
                title="MCP error",
                error=True,
            )
            return

        self.mcp_server = requested
        self.query_one("#status", Static).update(f"MCP: {requested}")
        self._write_command_result(
            f"MCP 서버를 선택했습니다: `{requested}`\n\n"
            "> 외부 도구 호출은 계속 사용자 승인이 필요합니다.",
            title="MCP",
        )

    def _handle_tools_command(self, arguments: list[str]) -> None:
        if not arguments:
            state = "on" if self.local_tools_enabled else "off"
            self._write_command_result(
                "| 항목 | 값 |\n"
                "| :--- | :--- |\n"
                f"| 로컬 도구 | `{state}` |\n"
                f"| 등록된 도구 | `{len(self.tool_executor.schemas)}` |",
                title="Tools",
            )
            return

        requested = arguments[0].lower()

        if requested in {"list", "desc"}:
            rows = [
                "| 도구 | 설명 | 상태 |",
                "| :--- | :--- | :---: |",
            ]

            for schema in self.tool_executor.schemas:
                function = schema.get("function", {})
                name = function.get("name", "unknown")
                description = function.get(
                    "description",
                    "",
                )

                if requested == "list":
                    description = "로컬 도구"

                rows.append(
                    f"| `{name}` | {description} | "
                    f"{'✅' if self.local_tools_enabled else '⏸️'} |"
                )

            self._write_command_result(
                "\n".join(rows),
                title="Local tools",
            )
            return

        if requested not in {"on", "off"}:
            self._write_command_result(
                "Usage: `/tools list|desc|on|off`",
                title="Tools error",
                error=True,
            )
            return

        self.local_tools_enabled = requested == "on"
        self._write_command_result(
            f"Local tools: `{requested}`",
            title="Tools",
        )

    def _handle_tokens_command(self, arguments: list[str]) -> None:
        if not arguments:
            self._write_command_result(
                f"Maximum output tokens: {self.max_tokens}",
                title="Output limit",
            )
            return

        if len(arguments) != 1:
            self._write_command_result(
                "Usage: /tokens <512-8192>",
                title="Output limit error",
                error=True,
            )
            return

        try:
            requested = int(arguments[0])
        except ValueError:
            requested = 0

        if not 512 <= requested <= 8192:
            self._write_command_result(
                "Token limit must be between 512 and 8192.",
                title="Output limit error",
                error=True,
            )
            return

        self.max_tokens = requested
        self.query_one("#status", Static).update(f"Tokens: {requested}")
        self._write_command_result(
            f"Maximum output tokens: {requested}",
            title="Output limit",
        )

    def _copy_chat(self) -> None:
        transcript = self.query_one(
            "#chat-log",
            SelectableChatLog,
        ).text

        if not transcript:
            self._write_command_result(
                "There is no chat text to copy.",
                title="Clipboard",
                error=True,
            )
            return

        try:
            subprocess.run(
                ["pbcopy"],
                input=transcript,
                text=True,
                check=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            self.copy_to_clipboard(transcript)

        self.notify(
            f"Copied {len(transcript)} characters",
            title="Clipboard",
            timeout=2,
        )

    def _handle_slash_command(self, command: str) -> bool:
        """Handle a slash command without contacting a provider."""
        parsed = parse_command(command)
        if parsed is None:
            return False

        raw_name = parsed.raw_name
        arguments = parsed.arguments
        spec = parsed.spec

        if spec is None:
            suggestions = parsed.suggestions
            suggestion_text = (
                "\n\n혹시 다음 명령을 찾으셨나요?\n\n"
                + "\n".join(f"- `{suggestion}`" for suggestion in suggestions)
                if suggestions
                else ""
            )
            self._write_command_result(
                f"Unknown command / 알 수 없는 명령입니다: "
                f"`/{raw_name}`"
                f"{suggestion_text}\n\n"
                "전체 명령은 `/help`에서 확인하세요.",
                title="Command error",
                error=True,
            )
            return True

        name = spec.name

        if name == "exit":
            self.exit()
        elif name == "help":
            self._write_command_result(
                self._command_help(arguments),
                title=(f"Help · {arguments[0]}" if arguments else "MAI commands"),
            )
        elif name == "status":
            self._write_command_result(
                self._command_status(),
                title="MAI status",
            )
        elif name == "about":
            self._write_command_result(
                self._command_about(),
                title="About MAI",
            )
        elif name == "version":
            self._write_command_result(
                self._command_version(),
                title="Version",
            )
        elif name == "doctor":
            self._write_command_result(
                self._command_doctor(),
                title="MAI doctor",
            )
        elif name == "pwd":
            self._write_command_result(
                f"`{Path.cwd()}`",
                title="Working directory",
            )
        elif name == "context":
            self._write_command_result(
                self._command_context(),
                title="Context",
            )
        elif name == "provider":
            self._handle_provider_command(arguments)
        elif name == "skill":
            self._handle_skill_command(arguments)
        elif name == "mcp":
            self._handle_mcp_command(arguments)
        elif name == "tools":
            self._handle_tools_command(arguments)
        elif name == "tokens":
            self._handle_tokens_command(arguments)
        elif name == "cancel":
            self._handle_cancel_command()
        elif name == "clear":
            self.action_clear_chat()
            self.query_one("#status", Static).update("New conversation · /help")
        elif name == "copy":
            self._copy_chat()

        return True

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.disabled:
            return
        user_input = event.value.strip()

        if not user_input:
            return

        if user_input.startswith("/"):
            event.input.clear()
            self._handle_slash_command(user_input)
            return

        log = self.query_one("#chat-log", SelectableChatLog)
        status = self.query_one("#status", Static)

        log.write(
            Panel(
                RichMarkdown(user_input),
                title="You",
                title_align="right",
                border_style="green",
                padding=(0, 1),
            )
        )

        event.input.clear()
        event.input.disabled = True
        status.update("● Thinking…")
        if self._streamable(user_input):
            self._request_worker = self._run_stream_request(user_input)
        else:
            self._request_worker = self._run_request(user_input)

    async def _ask_with_active_tools(
        self,
        prompt: str,
        allowed_tools: tuple[str, ...] | None = None,
    ):
        local_executor = None
        if self.local_tools_enabled:
            local_executor = ToolExecutor(
                self.tool_executor.policy,
                approval_callback=self._approve_tool,
                event_callback=self._tool_event,
                allowed_tools=allowed_tools,
            )

        if self.mcp_server == "off":
            return await self.router.ask(
                prompt=prompt,
                provider=self.provider,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                tool_executor=local_executor,
                max_tool_rounds=6,
            )

        config = self.mcp_registry.get(self.mcp_server)
        async with MCPToolExecutor(
            config,
            approval_callback=self._approve_tool,
            event_callback=self._tool_event,
            allowed_tools=allowed_tools,
        ) as external_executor:
            active_executor = (
                CombinedToolExecutor(
                    local_executor,
                    external_executor,
                )
                if local_executor is not None
                else external_executor
            )
            return await self.router.ask(
                prompt=prompt,
                provider=self.provider,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                tool_executor=active_executor,
                max_tool_rounds=6,
            )

    def _streamable(self, user_input: str) -> bool:
        """도구·MCP가 꺼져 있고 매칭되는 스킬도 없을 때만 스트리밍한다."""
        if self.local_tools_enabled or self.mcp_server != "off":
            return False
        try:
            return self.skill_catalog.resolve(self.skill_mode, user_input) is None
        except SkillError:
            return False

    def _flush_stream(self) -> None:
        panel = self.query_one("#stream", Static)
        buffer = self._stream_buffer
        panel.update(buffer[-4000:] + " ▌" if buffer else " ▌")
        panel.scroll_end(animate=False)

    def _stream_begin(self) -> None:
        panel = self.query_one("#stream", Static)
        self._stream_buffer = ""
        panel.add_class("active")
        panel.update(" ▌")
        self._stream_timer = self.set_interval(0.08, self._flush_stream)

    def _stream_end(self) -> None:
        if self._stream_timer is not None:
            self._stream_timer.stop()
            self._stream_timer = None
        panel = self.query_one("#stream", Static)
        panel.remove_class("active")
        panel.update("")

    @work(exclusive=True, group="request")
    async def _run_stream_request(self, user_input: str) -> None:
        log = self.query_one("#chat-log", SelectableChatLog)
        status = self.query_one("#status", Static)
        input_widget = self.query_one("#message", Input)
        provider = self.provider
        model = ""

        try:
            await self._persist("user", user_input)
            self.active_skill_name = None
            prompt = self._conversation_prompt(user_input)
            self._stream_begin()

            async for event in self.router.astream(
                prompt=prompt,
                provider=self.provider,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            ):
                if event.kind == "delta":
                    self._stream_buffer += event.text
                    provider = event.provider
                    model = event.model
                elif event.kind == "fallback":
                    status.update(f"● {event.provider} failed · trying next")
                else:
                    provider = event.provider or provider
                    model = event.model or model
        except asyncio.CancelledError:
            self._stream_end()
            await self._finish_stream(user_input, provider, model, cancelled=True)
            status.update("Cancelled")
            raise
        except RouterError as exc:
            self._stream_end()
            details = "\n".join(
                f"• {item.provider}: {item.detail}" for item in exc.attempts
            )
            log.write(Panel(details, title="Request failed", border_style="red"))
            status.update("Error")
        except (MCPHostError, SkillError, ValueError) as exc:
            self._stream_end()
            log.write(Panel(str(exc), title="Invalid request", border_style="red"))
            status.update("Error")
        except Exception as exc:  # noqa: BLE001
            self._stream_end()
            log.write(
                Panel(
                    (
                        f"{type(exc).__name__}: request failed safely. "
                        "No credentials were printed."
                    ),
                    title="Internal error",
                    border_style="red",
                )
            )
            status.update("Error")
        else:
            self._stream_end()
            await self._finish_stream(user_input, provider, model)
            status.update(f"Ready · {provider}")
        finally:
            input_widget.disabled = False
            input_widget.focus()

    async def _finish_stream(
        self,
        user_input: str,
        provider: str,
        model: str,
        cancelled: bool = False,
    ) -> None:
        answer = self._stream_buffer.strip()
        self._stream_buffer = ""
        if not answer:
            return

        self.history.append(("user", user_input))
        self.history.append(("assistant", answer))
        await self._persist("assistant", answer, provider, model)

        suffix = " · cancelled" if cancelled else ""
        self.query_one("#chat-log", SelectableChatLog).write(
            Panel(
                RichMarkdown(answer),
                title=f"MAI · {provider} · {model}{suffix}",
                title_align="left",
                border_style="bright_blue",
                padding=(0, 1),
            )
        )

    @work(exclusive=True, group="request")
    async def _run_request(self, user_input: str) -> None:
        log = self.query_one("#chat-log", SelectableChatLog)
        status = self.query_one("#status", Static)
        input_widget = self.query_one("#message", Input)
        try:
            await self._persist("user", user_input)

            skill = self.skill_catalog.resolve(
                self.skill_mode,
                user_input,
            )
            prompt = self._conversation_prompt(user_input)
            allowed_tools = None

            if skill is not None:
                self.active_skill_name = skill.metadata.name
                allowed_tools = skill.metadata.allowed_tools
                prompt = self.skill_catalog.render(skill, prompt)
                log.write(
                    Text(
                        f"Skill activated: {skill.metadata.name}",
                        style="dim cyan",
                    )
                )
                await self._persist(
                    "tool",
                    f"skill={skill.metadata.name}",
                )
                status.update(f"Skill: {skill.metadata.name} · Thinking")
            else:
                self.active_skill_name = None

            result = await self._ask_with_active_tools(
                prompt,
                allowed_tools,
            )
        except asyncio.CancelledError:
            status.update("Cancelled")
            raise
        except RouterError as exc:
            details = "\n".join(
                f"• {item.provider}: {item.detail}" for item in exc.attempts
            )
            log.write(Panel(details, title="Request failed", border_style="red"))
            status.update("Error")
        except (MCPHostError, SkillError, ValueError) as exc:
            log.write(
                Panel(
                    str(exc),
                    title="Invalid request",
                    border_style="red",
                )
            )
            status.update("Error")
        except Exception as exc:  # noqa: BLE001
            log.write(
                Panel(
                    (
                        f"{type(exc).__name__}: request failed safely. "
                        "No credentials were printed."
                    ),
                    title="Internal error",
                    border_style="red",
                )
            )
            status.update("Error")
        else:
            self.history.append(("user", user_input))
            self.history.append(("assistant", result.text))
            await self._persist("assistant", result.text, result.provider, result.model)

            log.write(
                Panel(
                    RichMarkdown(result.text),
                    title=f"MAI · {result.provider} · {result.model}",
                    title_align="left",
                    border_style="bright_blue",
                    padding=(0, 1),
                )
            )
            skill_status = (
                f" · {self.active_skill_name}"
                if self.active_skill_name is not None
                else ""
            )
            status.update(f"Ready · {result.provider}{skill_status}")
        finally:
            input_widget.disabled = False
            input_widget.focus()

    def action_cancel_request(self) -> None:
        if self._request_worker is not None:
            self._request_worker.cancel()

    @on(events.MouseUp)
    def _auto_copy_on_mouse_up(self, _event: events.MouseUp) -> None:
        """Copy selected TUI text immediately after mouse drag."""
        self.call_after_refresh(self._copy_selected_text)

    def _copy_selected_text(self) -> None:
        selected = self.query_one("#chat-log", SelectableChatLog).selected_text
        if not selected:
            return

        if selected == getattr(self, "_last_copied_selection", None):
            return

        try:
            subprocess.run(
                ["pbcopy"],
                input=selected,
                text=True,
                check=True,
                timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            # Fallback for terminals supporting Textual/OSC52 clipboard copy.
            self.copy_to_clipboard(selected)

        self._last_copied_selection = selected
        self.notify(
            f"Copied {len(selected)} characters",
            title="Clipboard",
            timeout=2,
        )
