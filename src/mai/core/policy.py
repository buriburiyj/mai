from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

MAX_READ_BYTES = 256 * 1024
MAX_OUTPUT_BYTES = 32 * 1024
COMMAND_TIMEOUT = 30.0

DENY_NAMES: frozenset[str] = frozenset(
    {
        ".env",
        ".env.local",
        ".netrc",
        ".npmrc",
        ".pypirc",
        ".git-credentials",
        "credentials",
        "id_rsa",
        "id_ed25519",
    }
)

DENY_DIRS: frozenset[str] = frozenset(
    {".ssh", ".aws", ".gnupg", ".kube", ".docker", "Library"}
)

ALLOWED_COMMANDS: frozenset[str] = frozenset(
    {
        "awk",
        "cat",
        "find",
        "git",
        "grep",
        "head",
        "ls",
        "node",
        "npm",
        "pwd",
        "pytest",
        "python",
        "python3",
        "ruff",
        "sed",
        "sort",
        "tail",
        "uv",
        "wc",
        "which",
    }
)

SHELL_METACHARACTERS = frozenset({"|", "&", ";", ">", "<", "`", "$", "(", ")"})


class Approval(str, Enum):
    READ_ONLY = "read-only"
    ASK = "ask"
    AUTO = "auto"


class PolicyError(RuntimeError):
    """Raised when a tool call violates the workspace policy."""


@dataclass(frozen=True)
class Policy:
    root: Path
    approval: Approval = Approval.ASK
    max_read_bytes: int = MAX_READ_BYTES
    max_output_bytes: int = MAX_OUTPUT_BYTES
    command_timeout: float = COMMAND_TIMEOUT

    @classmethod
    def load(
        cls,
        root: str | Path | None = None,
        approval: str | Approval | None = None,
    ) -> Policy:
        raw_root = root or os.environ.get("MAI_WORKSPACE") or Path.cwd()
        resolved = Path(raw_root).expanduser().resolve()
        if not resolved.is_dir():
            raise PolicyError(f"workspace is not a directory: {resolved}")

        raw_mode = approval or os.environ.get("MAI_APPROVAL") or Approval.ASK
        try:
            mode = Approval(raw_mode)
        except ValueError as exc:
            allowed = ", ".join(item.value for item in Approval)
            raise PolicyError(
                f"unknown approval mode: {raw_mode} (expected {allowed})"
            ) from exc

        return cls(root=resolved, approval=mode)

    def resolve(self, raw: str | Path) -> Path:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate

        resolved = candidate.resolve()
        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise PolicyError(f"path escapes workspace: {resolved}")

        relative = resolved.relative_to(self.root)
        for part in relative.parts:
            if part in DENY_DIRS:
                raise PolicyError(f"blocked directory: {part}")
        if resolved.name in DENY_NAMES:
            raise PolicyError(f"blocked file: {resolved.name}")

        return resolved

    def check_command(self, argv: Sequence[str]) -> None:
        if self.approval is Approval.READ_ONLY:
            raise PolicyError("command execution disabled in read-only mode")
        if not argv:
            raise PolicyError("empty command")

        program = Path(argv[0]).name
        if program not in ALLOWED_COMMANDS:
            allowed = ", ".join(sorted(ALLOWED_COMMANDS))
            raise PolicyError(f"command not allowed: {program} (allowed: {allowed})")

        for token in argv[1:]:
            bad = SHELL_METACHARACTERS.intersection(token)
            if bad:
                found = "".join(sorted(bad))
                raise PolicyError(f"shell metacharacters not allowed: {found}")

    def requires_approval(self, kind: str) -> bool:
        if kind == "read":
            return False
        if self.approval is Approval.READ_ONLY:
            raise PolicyError(f"{kind} operations disabled in read-only mode")
        return self.approval is Approval.ASK
