import asyncio
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Annotated

import keyring
import typer
from platformdirs import user_config_dir, user_data_dir
from rich.console import Console
from rich.table import Table

from mai.core.mcp_host import (
    CombinedToolExecutor,
    MCPHostError,
    MCPRegistry,
    MCPServerConfig,
    MCPToolExecutor,
    probe_server,
)
from mai.core.mcp_server import create_mcp_server
from mai.core.policy import Approval, Policy
from mai.core.provider_checks import check_providers
from mai.core.providers import PROVIDERS, get_provider
from mai.core.router import MultiProviderRouter, RouterError
from mai.core.secrets import SecretStore, SecretStoreError
from mai.core.skills import SkillCatalog, SkillError
from mai.core.tools import ToolExecutor
from mai.tui import MaiChatApp

console = Console()
store = SecretStore()

app = typer.Typer(
    name="mai",
    help="Free multi-provider AI agent for CLI, MCP, Skills, and the web.",
    no_args_is_help=True,
)

secrets_app = typer.Typer(
    help="Manage credentials in the macOS Keychain.",
    no_args_is_help=True,
)

providers_app = typer.Typer(
    help="Inspect and manage AI providers.",
    no_args_is_help=True,
)

app.add_typer(secrets_app, name="secrets")
app.add_typer(providers_app, name="providers")


skills_app = typer.Typer(
    help="Discover, inspect, and validate Agent Skills.",
    no_args_is_help=True,
)

mcp_app = typer.Typer(
    help="Run and manage Model Context Protocol integrations.",
    no_args_is_help=True,
)

app.add_typer(skills_app, name="skills")
app.add_typer(mcp_app, name="mcp")


@app.command("version")
def version_info() -> None:
    """Show the installed MAI version."""
    console.print(f"MAI {version('mai')}")


@app.command("ask")
def ask_command(
    prompt: str = typer.Argument(
        ...,
        help="Question or task for MAI.",
    ),
    provider: str = typer.Option(
        "auto",
        "--provider",
        "-p",
        help="auto, gemini, cerebras, groq, or openrouter.",
    ),
    max_tokens: int = typer.Option(
        1024,
        "--max-tokens",
        min=1,
        max=8192,
        help="Maximum output tokens.",
    ),
    timeout: float = typer.Option(
        60.0,
        "--timeout",
        min=1.0,
        help="Timeout for each provider.",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Return structured JSON.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show provider attempts and fallback details.",
    ),
    skill: str | None = typer.Option(
        None,
        "--skill",
        help="Activate an Agent Skill by name, or use 'auto'.",
    ),
    mcp_server: str | None = typer.Option(
        None,
        "--mcp",
        help="Use tools from one configured external MCP server.",
    ),
    approve_mcp_tools: bool = typer.Option(
        False,
        "--approve-mcp-tools",
        help=("Explicitly allow the selected MCP server's tools for this one request."),
    ),
    tools: bool = typer.Option(
        False,
        "--tools/--no-tools",
        help="Allow workspace file tools; shell commands remain denied.",
    ),
) -> None:
    """Ask a question using automatic free-provider fallback."""
    active_skill = None
    if skill is not None:
        try:
            catalog = SkillCatalog.default(Path.cwd())
            active_skill = catalog.resolve(skill, prompt)
            if active_skill is not None:
                prompt = catalog.render(active_skill, prompt)
        except SkillError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc

    allowed_tools = (
        active_skill.metadata.allowed_tools if active_skill is not None else None
    )

    router = MultiProviderRouter()
    local_executor = None
    if tools:
        policy = Policy.load(root=Path.cwd(), approval=Approval.READ_ONLY)
        local_executor = ToolExecutor(
            policy,
            allowed_tools=allowed_tools,
        )

    if mcp_server is not None and not approve_mcp_tools:
        console.print(
            "[red]External MCP tools require --approve-mcp-tools for "
            "this request.[/red]"
        )
        raise typer.Exit(2)

    async def run_request():
        if mcp_server is None:
            return await router.ask(
                prompt=prompt,
                provider=provider.lower(),
                max_tokens=max_tokens,
                timeout=timeout,
                tool_executor=local_executor,
                max_tool_rounds=6,
            )

        config = MCPRegistry().get(mcp_server)

        async def approve_external(
            _kind: str,
            _description: str,
        ) -> bool:
            return approve_mcp_tools

        async with MCPToolExecutor(
            config,
            approval_callback=approve_external,
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
            return await router.ask(
                prompt=prompt,
                provider=provider.lower(),
                max_tokens=max_tokens,
                timeout=timeout,
                tool_executor=active_executor,
                max_tool_rounds=6,
            )

    try:
        with console.status("[bold cyan]Choosing a provider and generating..."):
            result = asyncio.run(run_request())
    except (MCPHostError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc
    except RouterError as exc:
        console.print("[red]Every configured provider failed.[/red]")

        for attempt in exc.attempts:
            console.print(
                f"  [yellow]{attempt.provider}[/yellow] "
                f"({attempt.model}): {attempt.detail}"
            )

        raise typer.Exit(1) from exc

    if json_output:
        payload = {
            "text": result.text,
            "provider": result.provider,
            "model": result.model,
            "usage": result.usage,
            "attempts": [
                {
                    "provider": item.provider,
                    "model": item.model,
                    "status": item.status,
                    "detail": item.detail,
                }
                for item in result.attempts
            ],
        }
        typer.echo(json.dumps(payload, ensure_ascii=False))
        return

    console.print(f"[dim]Provider: {result.provider} | Model: {result.model}[/dim]")
    console.print()
    console.print(result.text)

    if verbose:
        console.print()
        console.print("[bold]Routing attempts[/bold]")

        for attempt in result.attempts:
            color = "green" if attempt.status == "success" else "yellow"
            console.print(
                f"  [{color}]{attempt.provider}[/{color}] "
                f"{attempt.model}: {attempt.detail}"
            )


@app.command()
def doctor() -> None:
    """Check the local MAI environment."""
    config_path = Path(user_config_dir("mai"))
    data_path = Path(user_data_dir("mai"))

    config_path.mkdir(parents=True, exist_ok=True)
    data_path.mkdir(parents=True, exist_ok=True)

    table = Table(title="MAI environment")
    table.add_column("Item")
    table.add_column("Value")

    table.add_row("Python", sys.version.split()[0])
    table.add_row("Architecture", platform.machine())
    table.add_row("macOS", platform.mac_ver()[0])
    table.add_row("Keyring", type(keyring.get_keyring()).__name__)
    table.add_row("Config", str(config_path))
    table.add_row("Data", str(data_path))

    console.print(table)


@providers_app.command("list")
def list_providers() -> None:
    """List supported providers."""
    table = Table(title="Supported providers")
    table.add_column("ID")
    table.add_column("Provider")
    table.add_column("Credentials")

    for provider in PROVIDERS.values():
        fields = ", ".join(field.label for field in provider.credentials)
        table.add_row(provider.name, provider.display_name, fields)

    console.print(table)


@providers_app.command("status")
def provider_status() -> None:
    """Show credential configuration status."""
    table = Table(title="Provider status")
    table.add_column("Provider")
    table.add_column("Credentials")
    table.add_column("Status")

    try:
        for provider in PROVIDERS.values():
            status = store.status(provider)

            if status.complete:
                label = "[green]configured[/green]"
            elif status.partial:
                label = "[yellow]partial[/yellow]"
            else:
                label = "[dim]not configured[/dim]"

            count = f"{status.configured}/{status.required}"
            table.add_row(provider.display_name, count, label)
    except SecretStoreError as exc:
        console.print(f"[red]Keychain error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(table)


@secrets_app.command("set")
def set_secret(
    provider_name: str = typer.Argument(
        ...,
        help="Provider ID, such as gemini, groq, cloudflare, or hf.",
    ),
) -> None:
    """Store provider credentials in the macOS Keychain."""
    try:
        provider = get_provider(provider_name)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    console.print(f"[bold]{provider.display_name}[/bold]")
    console.print("Credentials will be stored in the macOS Keychain.")

    try:
        for field in provider.credentials:
            value = typer.prompt(
                field.label,
                hide_input=field.sensitive,
                confirmation_prompt=field.sensitive,
            )
            store.set(provider.name, field.name, value)
    except (SecretStoreError, typer.Abort) as exc:
        console.print(f"[red]Credential setup failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[green]Saved credentials for {provider.display_name}.[/green]")


@secrets_app.command("delete")
def delete_secret(
    provider_name: str = typer.Argument(..., help="Provider ID."),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Delete without an additional confirmation.",
    ),
) -> None:
    """Delete provider credentials from the macOS Keychain."""
    try:
        provider = get_provider(provider_name)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    if not yes:
        confirmed = typer.confirm(
            f"Delete all credentials for {provider.display_name}?"
        )
        if not confirmed:
            raise typer.Abort()

    deleted = 0

    try:
        for field in provider.credentials:
            deleted += int(store.delete(provider.name, field.name))
    except SecretStoreError as exc:
        console.print(f"[red]Keychain error:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print(f"[green]Deleted {deleted} credential item(s).[/green]")


@providers_app.command("test")
def test_providers(
    provider_name: str | None = typer.Argument(
        None,
        help="Optional provider ID. Omit to test every provider.",
    ),
    timeout: float = typer.Option(
        15.0,
        "--timeout",
        min=1.0,
        help="Request timeout in seconds.",
    ),
) -> None:
    """Validate credentials without running model inference."""
    if provider_name is None:
        names = list(PROVIDERS)
    else:
        try:
            names = [get_provider(provider_name).name]
        except ValueError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc

    console.print("[dim]Checking official provider APIs without inference...[/dim]")

    results = asyncio.run(check_providers(names, timeout))

    table = Table(title="Provider connection test")
    table.add_column("Provider")
    table.add_column("Result")
    table.add_column("HTTP")
    table.add_column("Latency")
    table.add_column("Details")

    failures = 0

    for result in results:
        provider = PROVIDERS[result.provider]

        if result.status == "ok":
            status_label = "[green]PASS[/green]"
        elif result.status == "limited":
            status_label = "[yellow]LIMITED[/yellow]"
        elif result.status == "denied":
            status_label = "[yellow]DENIED[/yellow]"
            failures += 1
        else:
            status_label = "[red]FAIL[/red]"
            failures += 1

        http_status = str(result.http_status) if result.http_status is not None else "-"

        table.add_row(
            provider.display_name,
            status_label,
            http_status,
            f"{result.latency_ms} ms",
            result.detail,
        )

    console.print(table)

    if failures:
        raise typer.Exit(1)


@app.command("chat")
def chat_command(
    provider: str = typer.Option(
        "auto",
        "--provider",
        "-p",
        help="Initial provider.",
    ),
    max_tokens: int = typer.Option(
        1024,
        "--max-tokens",
        min=1,
        max=8192,
        help="Maximum output tokens per response.",
    ),
    history_turns: int = typer.Option(
        10,
        "--history-turns",
        min=1,
        max=50,
        help="Recent conversation turns to retain.",
    ),
    timeout: float = typer.Option(
        60.0,
        "--timeout",
        min=1.0,
        help="Timeout for each provider.",
    ),
    resume: str = typer.Option(
        "",
        "--resume",
        "-r",
        help="Resume a session id, or 'last' for the most recent session.",
    ),
) -> None:
    """Open the full-screen MAI chat interface."""
    allowed = {
        "auto",
        "gemini",
        "groq",
        "openrouter",
        "cerebras",
    }

    selected = provider.lower()

    if selected not in allowed:
        console.print(
            "[red]Available providers: auto, gemini, groq, openrouter, cerebras[/red]"
        )
        raise typer.Exit(2)

    import asyncio as _asyncio

    from mai.core.sessions import SessionStore

    session_id: str | None = None
    preloaded: list[tuple[str, str]] = []

    if resume:
        store = SessionStore()

        async def _load() -> tuple[str | None, list[tuple[str, str]]]:
            target = await store.latest() if resume == "last" else resume

            if target is None:
                return None, []

            messages = await store.messages(target)
            if not messages:
                return None, []

            return target, [(m.role, m.content) for m in messages]

        session_id, preloaded = _asyncio.run(_load())

        if session_id is None:
            console.print(
                f"[yellow]No session found for '{resume}'. Starting a new one.[/yellow]"
            )
        else:
            console.print(
                f"[dim]Resuming session {session_id} ({len(preloaded)} messages).[/dim]"
            )

    chat_app = MaiChatApp(
        provider=selected,
        max_tokens=max_tokens,
        history_turns=history_turns,
        timeout=timeout,
        session_id=session_id,
    )
    chat_app.history = preloaded
    chat_app.run()


health_app = typer.Typer(
    help="Inspect and reset provider cooldowns.",
    no_args_is_help=True,
)

app.add_typer(health_app, name="health")


@health_app.command("show")
def health_show() -> None:
    """Show provider health and remaining cooldowns."""
    import asyncio as _asyncio

    from mai.core.health import HealthStore

    rows = _asyncio.run(HealthStore().rows())

    if not rows:
        console.print("[dim]No provider history yet. Run 'mai ask' first.[/dim]")
        return

    table = Table(title="Provider health")
    table.add_column("Provider")
    table.add_column("State")
    table.add_column("Cooldown")
    table.add_column("OK", justify="right")
    table.add_column("Fail", justify="right")
    table.add_column("Last result")

    for row in rows:
        if row.available:
            state = "[green]ready[/green]"
            cooldown = "-"
        else:
            state = "[yellow]cooling[/yellow]"
            cooldown = f"{row.remaining}s"

        table.add_row(
            row.provider,
            state,
            cooldown,
            str(row.success_count),
            str(row.failure_count),
            f"{row.last_kind}: {row.last_detail}",
        )

    console.print(table)


@health_app.command("reset")
def health_reset(
    provider_name: str = typer.Argument(
        "all",
        help="Provider ID, or 'all' to clear every cooldown.",
    ),
) -> None:
    """Clear cooldowns so providers are retried immediately."""
    import asyncio as _asyncio

    from mai.core.health import HealthStore

    target = None if provider_name.lower() == "all" else provider_name.lower()
    changed = _asyncio.run(HealthStore().reset(target))

    console.print(f"[green]Cleared cooldown for {changed} provider(s).[/green]")


sessions_app = typer.Typer(
    help="Browse and manage saved conversations.",
    no_args_is_help=True,
)

app.add_typer(sessions_app, name="sessions")


@sessions_app.command("list")
def sessions_list(
    limit: int = typer.Option(20, "--limit", min=1, help="Rows to display."),
) -> None:
    """List saved sessions, newest first."""
    import asyncio as _asyncio
    from datetime import datetime

    from mai.core.sessions import SessionStore

    rows = _asyncio.run(SessionStore().sessions(limit=limit))

    if not rows:
        console.print("[dim]No saved sessions yet. Run 'mai chat' first.[/dim]")
        return

    table = Table(title="Saved sessions")
    table.add_column("ID")
    table.add_column("Title")
    table.add_column("Msgs", justify="right")
    table.add_column("Updated")

    for row in rows:
        stamp = datetime.fromtimestamp(
            row.updated_at, tz=datetime.now().astimezone().tzinfo
        ).strftime("%m-%d %H:%M")
        table.add_row(row.session_id, row.title, str(row.message_count), stamp)

    console.print(table)


@sessions_app.command("show")
def sessions_show(
    session_id: str = typer.Argument(..., help="Session ID from 'sessions list'."),
) -> None:
    """Print every message in a session."""
    import asyncio as _asyncio

    from mai.core.sessions import SessionStore

    history = _asyncio.run(SessionStore().messages(session_id))

    if not history:
        console.print(f"[yellow]No messages for session {session_id}.[/yellow]")
        raise typer.Exit(1)

    for message in history:
        if message.role == "user":
            console.print("\n[bold green]You[/bold green]")
        else:
            label = message.provider or "assistant"
            console.print(
                f"\n[bold cyan]{label}[/bold cyan] [dim]{message.model}[/dim]"
            )
        console.print(message.content)


@sessions_app.command("delete")
def sessions_delete(
    session_id: str = typer.Argument(..., help="Session ID to remove."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation."),
) -> None:
    """Delete a session and its messages."""
    import asyncio as _asyncio

    from mai.core.sessions import SessionStore

    if not yes and not typer.confirm(f"Delete session {session_id}?"):
        raise typer.Abort()

    removed = _asyncio.run(SessionStore().delete(session_id))

    if removed:
        console.print(f"[green]Deleted session {session_id}.[/green]")
    else:
        console.print(f"[yellow]Session {session_id} not found.[/yellow]")
        raise typer.Exit(1)


@skills_app.command("list")
def skills_list() -> None:
    """List valid Agent Skills available to the current workspace."""
    catalog = SkillCatalog.default(Path.cwd())
    skills, errors = catalog.validate_all()

    table = Table(title="Agent Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Allowed tools")

    for skill in skills:
        table.add_row(
            skill.name,
            skill.description,
            ", ".join(skill.allowed_tools) or "none",
        )

    console.print(table)

    for error in errors:
        console.print(f"[yellow]Warning:[/yellow] {error}")


@skills_app.command("show")
def skills_show(
    name: str = typer.Argument(..., help="Skill name."),
) -> None:
    """Show the complete instructions for one Agent Skill."""
    try:
        skill = SkillCatalog.default(Path.cwd()).load(name)
    except SkillError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[bold cyan]{skill.metadata.name}[/bold cyan]")
    console.print(skill.metadata.description)
    console.print()
    console.print(skill.instructions)


@skills_app.command("validate")
def skills_validate() -> None:
    """Validate all discovered SKILL.md files."""
    skills, errors = SkillCatalog.default(Path.cwd()).validate_all()

    for skill in skills:
        console.print(f"[green]PASS[/green] {skill.name}")

    for error in errors:
        console.print(f"[red]FAIL[/red] {error}")

    if errors:
        raise typer.Exit(1)

    console.print(f"[green]{len(skills)} skill(s) valid.[/green]")


@mcp_app.command("serve")
def mcp_serve(
    root: Annotated[
        Path | None,
        typer.Option(
            "--root",
            help=(
                "Workspace directory exposed through read-only MCP tools. "
                "Defaults to the current directory."
            ),
            exists=True,
            file_okay=False,
            dir_okay=True,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Run MAI's read-only MCP server over stdio."""
    workspace_root = (root or Path.cwd()).expanduser().resolve()
    server = create_mcp_server(workspace_root)
    server.run("stdio")


@mcp_app.command("list")
def mcp_list() -> None:
    """List configured external MCP servers."""
    try:
        servers = MCPRegistry().list()
    except MCPHostError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not servers:
        console.print("[dim]No external MCP servers configured.[/dim]")
        return

    table = Table(title="External MCP servers")
    table.add_column("Name", style="cyan")
    table.add_column("Transport")
    table.add_column("Target")
    table.add_column("Enabled")

    for server in servers:
        if server.transport == "stdio":
            target = " ".join((server.command or "", *server.args))
        else:
            target = server.url or ""
        table.add_row(
            server.name,
            server.transport,
            target,
            "yes" if server.enabled else "no",
        )

    console.print(table)


@mcp_app.command("add")
def mcp_add(
    name: str = typer.Argument(
        ...,
        help="Unique lowercase server name.",
    ),
    url: Annotated[
        str | None,
        typer.Option(
            "--url",
            help="Streamable HTTP endpoint. Remote endpoints require HTTPS.",
        ),
    ] = None,
    command: Annotated[
        str | None,
        typer.Option(
            "--command",
            help="Executable for a local stdio MCP server.",
        ),
    ] = None,
    arg: Annotated[
        list[str] | None,
        typer.Option(
            "--arg",
            help="One stdio argument; repeat for multiple arguments.",
        ),
    ] = None,
    replace: Annotated[
        bool,
        typer.Option(
            "--replace",
            help="Replace an existing entry with the same name.",
        ),
    ] = False,
) -> None:
    """Register an external stdio or Streamable HTTP MCP server."""
    if (url is None) == (command is None):
        console.print("[red]Provide exactly one of --url or --command.[/red]")
        raise typer.Exit(2)

    config = MCPServerConfig(
        name=name,
        transport="http" if url is not None else "stdio",
        url=url,
        command=command,
        args=tuple(arg or ()),
    )

    try:
        MCPRegistry().add(config, replace=replace)
    except MCPHostError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    console.print(f"[green]Registered MCP server {name}.[/green]")


@mcp_app.command("remove")
def mcp_remove(
    name: str = typer.Argument(..., help="Configured server name."),
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation."),
    ] = False,
) -> None:
    """Remove an external MCP server configuration."""
    if not yes and not typer.confirm(f"Remove MCP server {name}?"):
        raise typer.Abort()

    try:
        removed = MCPRegistry().remove(name)
    except MCPHostError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    if not removed:
        console.print(f"[yellow]MCP server not found: {name}[/yellow]")
        raise typer.Exit(1)

    console.print(f"[green]Removed MCP server {name}.[/green]")


@mcp_app.command("test")
def mcp_test(
    name: str = typer.Argument(..., help="Configured server name."),
    timeout: Annotated[
        float,
        typer.Option(
            "--timeout",
            min=1.0,
            help="Total connection and capability-discovery timeout.",
        ),
    ] = 30.0,
) -> None:
    """Connect to an external MCP server and inspect its capabilities."""
    registry = MCPRegistry()

    try:
        config = registry.get(name)
        result = asyncio.run(probe_server(config, timeout=timeout))
    except Exception as exc:
        console.print(
            f"[red]MCP connection failed:[/red] {type(exc).__name__}: {str(exc)[:500]}"
        )
        raise typer.Exit(1) from exc

    table = Table(title=f"MCP connection: {result.name}")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Server", result.server_name)
    table.add_row("Protocol", result.protocol_version)
    table.add_row("Tools", ", ".join(result.tools) or "none")
    table.add_row("Resources", ", ".join(result.resources) or "none")
    table.add_row("Prompts", ", ".join(result.prompts) or "none")
    console.print(table)
