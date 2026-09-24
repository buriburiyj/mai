from pathlib import Path

import pytest

from mai.core.skills import SkillCatalog, SkillError


def write_skill(root: Path, name: str, description: str = "Test skill.") -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        f"description: {description}\n"
        "allowed-tools: fs_list fs_read\n"
        "---\n"
        "# Instructions\n\n"
        "Inspect the requested files safely.\n",
        encoding="utf-8",
    )


def test_discovers_metadata_without_loading_body(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    write_skill(root, "project-review")

    catalog = SkillCatalog((root,))
    skills = catalog.list()

    assert [skill.name for skill in skills] == ["project-review"]
    assert skills[0].allowed_tools == ("fs_list", "fs_read")


def test_loads_skill_instructions(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    write_skill(root, "project-review")

    skill = SkillCatalog((root,)).load("project-review")

    assert skill.metadata.name == "project-review"
    assert "Inspect the requested files safely." in skill.instructions


def test_rejects_directory_name_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    directory = root / "wrong-directory"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nname: another-name\ndescription: Invalid mismatch.\n---\nInstructions.\n",
        encoding="utf-8",
    )

    _, errors = SkillCatalog((root,)).validate_all()

    assert errors
    assert "must match directory" in errors[0]


def test_rejects_invalid_name(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    directory = root / "Bad_Name"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nname: Bad_Name\ndescription: Invalid name.\n---\nInstructions.\n",
        encoding="utf-8",
    )

    with pytest.raises(SkillError, match="lowercase"):
        SkillCatalog((root,)).metadata_from_path(directory / "SKILL.md")


def test_applies_skill_to_request(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    write_skill(root, "project-review")

    prompt = SkillCatalog((root,)).apply(
        "project-review",
        "Explain this project.",
    )

    assert "[Activated Agent Skill: project-review]" in prompt
    assert "[User request]" in prompt
    assert "Explain this project." in prompt


def test_auto_selects_skill_from_metadata_triggers(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    write_skill(
        root,
        "project-review",
        "Inspect project structure and architecture.",
    )

    selected = SkillCatalog((root,)).select("Explain the project architecture.")

    assert selected is not None
    assert selected.name == "project-review"


def test_auto_selection_returns_none_for_unrelated_request(
    tmp_path: Path,
) -> None:
    root = tmp_path / "skills"
    write_skill(
        root,
        "project-review",
        "Inspect project structure and architecture.",
    )

    selected = SkillCatalog((root,)).select("What is the weather tomorrow?")

    assert selected is None
