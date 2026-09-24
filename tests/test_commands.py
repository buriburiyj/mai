from mai.core.commands import (
    COMMAND_SPECS,
    command_suggestions,
    parse_command,
    render_help,
    resolve_command,
    suggest_commands,
)


def test_resolves_command_aliases() -> None:
    assert resolve_command("exit").name == "exit"
    assert resolve_command("quit").name == "exit"
    assert resolve_command("q").name == "exit"
    assert resolve_command("providers").name == "provider"
    assert resolve_command("skills").name == "skill"


def test_help_is_generated_from_registry() -> None:
    help_text = render_help()

    assert "| 분류 | 명령어 | 설명 |" in help_text
    assert r"`/mcp [list\|status\|use <server>\|off\|<server>]`" in help_text
    assert "`/help [command]`" in help_text


def test_detailed_help() -> None:
    help_text = render_help("mcp")

    assert "### `/mcp" in help_text
    assert "MCP 서버" in help_text


def test_unknown_help_suggests_similar_command() -> None:
    help_text = render_help("mc")

    assert "알 수 없는 명령" in help_text
    assert "/mcp" in help_text


def test_command_suggestions_include_common_commands() -> None:
    suggestions = command_suggestions()

    assert "/help" in suggestions
    assert "/mcp list" in suggestions
    assert "/exit" in suggestions
    assert "/quit" in suggestions
    assert "/q" in suggestions


def test_typo_suggestions() -> None:
    assert "/status" in suggest_commands("statsu")


def test_parse_command_rejects_plain_text():
    assert parse_command("안녕하세요") is None
    assert parse_command("   ") is None


def test_parse_command_splits_name_and_arguments():
    parsed = parse_command("/provider use groq")
    assert parsed is not None
    assert parsed.spec is not None
    assert parsed.spec.name == "provider"
    assert parsed.arguments == ["use", "groq"]


def test_parse_command_resolves_alias_with_default_argument():
    parsed = parse_command("/models")
    assert parsed is not None
    assert parsed.spec is not None
    assert parsed.spec.name == "provider"
    assert parsed.arguments == ["list"]


def test_parse_command_keeps_explicit_alias_arguments():
    parsed = parse_command("/skills reload")
    assert parsed is not None
    assert parsed.arguments == ["reload"]


def test_parse_command_is_case_insensitive():
    parsed = parse_command("/HELP")
    assert parsed is not None
    assert parsed.spec is not None
    assert parsed.spec.name == "help"


def test_parse_command_suggests_close_match():
    parsed = parse_command("/porvider")
    assert parsed is not None
    assert parsed.spec is None
    assert parsed.suggestions


def test_help_table_lists_every_command() -> None:
    table = render_help()
    for spec in COMMAND_SPECS:
        assert f"/{spec.name}" in table
        assert spec.category in table


def test_help_table_has_no_html() -> None:
    table = render_help()
    assert "<br>" not in table
    assert "&nbsp;" not in table


def test_help_table_escapes_pipes_in_usage() -> None:
    table = render_help()
    for spec in COMMAND_SPECS:
        if "|" in spec.usage:
            assert spec.usage.replace("|", "\\|") in table


def test_help_detail_keeps_raw_pipes() -> None:
    for spec in COMMAND_SPECS:
        if "|" in spec.usage:
            assert spec.usage in render_help(spec.name)
