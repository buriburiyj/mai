from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from platformdirs import user_config_dir

_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_FRONTMATTER_BYTES = 64 * 1024


class SkillError(ValueError):
    """Raised when an Agent Skill is missing or invalid."""


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    name: str
    description: str
    root: Path
    license: str | None = None
    compatibility: str | None = None
    metadata: dict[str, str] | None = None
    allowed_tools: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Skill:
    metadata: SkillMetadata
    instructions: str


class SkillCatalog:
    """Discover and load Agent Skills with progressive disclosure."""

    def __init__(self, roots: tuple[Path, ...]) -> None:
        self.roots = tuple(root.expanduser().resolve() for root in roots)

    @classmethod
    def default(cls, workspace: Path | None = None) -> SkillCatalog:
        workspace_root = (workspace or Path.cwd()).expanduser().resolve()
        user_root = Path(user_config_dir("mai")) / "skills"
        return cls((workspace_root / "skills", user_root))

    @staticmethod
    def _frontmatter(path: Path) -> dict[str, Any]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                first = handle.readline()
                if first.strip() != "---":
                    raise SkillError(f"{path}: missing YAML frontmatter")

                lines: list[str] = []
                size = 0
                for line in handle:
                    if line.strip() == "---":
                        break
                    size += len(line.encode("utf-8"))
                    if size > _MAX_FRONTMATTER_BYTES:
                        raise SkillError(f"{path}: frontmatter is too large")
                    lines.append(line)
                else:
                    raise SkillError(f"{path}: unclosed YAML frontmatter")
        except OSError as exc:
            raise SkillError(f"{path}: {exc}") from exc

        try:
            data = yaml.safe_load("".join(lines))
        except yaml.YAMLError as exc:
            raise SkillError(f"{path}: invalid YAML: {exc}") from exc

        if not isinstance(data, dict):
            raise SkillError(f"{path}: frontmatter must be a mapping")
        return data

    @staticmethod
    def _validate_optional_string(
        data: dict[str, Any],
        key: str,
        maximum: int,
        path: Path,
    ) -> str | None:
        value = data.get(key)
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise SkillError(f"{path}: {key} must be a non-empty string")
        if len(value) > maximum:
            raise SkillError(f"{path}: {key} exceeds {maximum} characters")
        return value.strip()

    @classmethod
    def metadata_from_path(cls, path: Path) -> SkillMetadata:
        path = path.expanduser().resolve()
        data = cls._frontmatter(path)

        name = data.get("name")
        description = data.get("description")

        if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
            raise SkillError(
                f"{path}: name must use lowercase letters, numbers, and hyphens"
            )
        if len(name) > 64:
            raise SkillError(f"{path}: name exceeds 64 characters")
        if name != path.parent.name:
            raise SkillError(
                f"{path}: skill name must match directory {path.parent.name!r}"
            )

        if not isinstance(description, str) or not description.strip():
            raise SkillError(f"{path}: description is required")
        if len(description) > 1024:
            raise SkillError(f"{path}: description exceeds 1024 characters")

        license_name = cls._validate_optional_string(data, "license", 500, path)
        compatibility = cls._validate_optional_string(data, "compatibility", 500, path)

        raw_metadata = data.get("metadata")
        parsed_metadata: dict[str, str] | None = None
        if raw_metadata is not None:
            if not isinstance(raw_metadata, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in raw_metadata.items()
            ):
                raise SkillError(f"{path}: metadata must map strings to strings")
            parsed_metadata = dict(raw_metadata)

        raw_allowed = data.get("allowed-tools")
        allowed_tools: tuple[str, ...] = ()
        if raw_allowed is not None:
            if not isinstance(raw_allowed, str):
                raise SkillError(
                    f"{path}: allowed-tools must be a space-separated string"
                )
            allowed_tools = tuple(raw_allowed.split())

        return SkillMetadata(
            name=name,
            description=description.strip(),
            root=path.parent,
            license=license_name,
            compatibility=compatibility,
            metadata=parsed_metadata,
            allowed_tools=allowed_tools,
        )

    def validate_all(
        self,
    ) -> tuple[list[SkillMetadata], list[str]]:
        skills: list[SkillMetadata] = []
        errors: list[str] = []
        names: set[str] = set()

        for root in self.roots:
            if not root.is_dir():
                continue
            for path in sorted(root.glob("*/SKILL.md")):
                try:
                    metadata = self.metadata_from_path(path)
                except SkillError as exc:
                    errors.append(str(exc))
                    continue

                if metadata.name in names:
                    errors.append(f"{path}: duplicate skill name {metadata.name!r}")
                    continue
                names.add(metadata.name)
                skills.append(metadata)

        return skills, errors

    def list(self) -> list[SkillMetadata]:
        skills, _ = self.validate_all()
        return skills

    def get_metadata(self, name: str) -> SkillMetadata:
        if not _NAME_PATTERN.fullmatch(name):
            raise SkillError(f"invalid skill name: {name!r}")

        for root in self.roots:
            path = root / name / "SKILL.md"
            if path.is_file():
                return self.metadata_from_path(path)

        raise SkillError(f"skill not found: {name}")

    def load(self, name: str) -> Skill:
        metadata = self.get_metadata(name)
        path = metadata.root / "SKILL.md"

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SkillError(f"{path}: {exc}") from exc

        lines = text.splitlines()
        closing_index = next(
            (
                index
                for index, line in enumerate(lines[1:], start=1)
                if line.strip() == "---"
            ),
            None,
        )
        if closing_index is None:
            raise SkillError(f"{path}: unclosed YAML frontmatter")

        instructions = "\n".join(lines[closing_index + 1 :]).strip()
        if not instructions:
            raise SkillError(f"{path}: skill instructions are empty")

        return Skill(metadata=metadata, instructions=instructions)

    @staticmethod
    def _selection_terms(value: str) -> set[str]:
        """Return normalized English and Korean terms for skill matching."""
        return {
            term
            for term in re.findall(r"[a-z0-9가-힣]+", value.lower())
            if len(term) >= 2
        }

    def select(self, request: str) -> SkillMetadata | None:
        """Select the most relevant skill using metadata only."""
        request_terms = self._selection_terms(request)
        if not request_terms:
            return None

        best: SkillMetadata | None = None
        best_score = 0

        for metadata in self.list():
            trigger_text = ""
            if metadata.metadata is not None:
                trigger_text = metadata.metadata.get("auto-triggers", "")

            candidate_terms = self._selection_terms(
                " ".join(
                    (
                        metadata.name.replace("-", " "),
                        metadata.description,
                        trigger_text,
                    )
                )
            )
            overlap = request_terms & candidate_terms
            score = sum(len(term) for term in overlap)

            normalized_name = metadata.name.replace("-", " ")
            if normalized_name in request.lower():
                score += 100

            if score > best_score:
                best = metadata
                best_score = score

        return best if best_score > 0 else None

    def resolve(self, selection: str, request: str) -> Skill | None:
        """Resolve off, auto, or an explicit skill name."""
        if selection == "off":
            return None
        if selection == "auto":
            metadata = self.select(request)
            return self.load(metadata.name) if metadata is not None else None
        return self.load(selection)

    @staticmethod
    def render(skill: Skill, request: str) -> str:
        """Apply an already loaded skill to a user request."""
        return (
            f"[Activated Agent Skill: {skill.metadata.name}]\n"
            f"{skill.instructions}\n\n"
            "[User request]\n"
            f"{request}"
        )

    def apply(self, name: str, request: str) -> str:
        return self.render(self.load(name), request)
