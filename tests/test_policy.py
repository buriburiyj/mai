from __future__ import annotations

import pytest

from mai.core.policy import Approval, Policy, PolicyError


@pytest.fixture()
def policy(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("print('hi')\n")
    return Policy.load(root=tmp_path)


def test_resolve_relative_path(policy):
    assert policy.resolve("src/main.py").name == "main.py"


def test_resolve_rejects_escape(policy):
    with pytest.raises(PolicyError, match="escapes workspace"):
        policy.resolve("../../etc/passwd")


def test_resolve_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("nope\n")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "link.txt").symlink_to(outside / "secret.txt")

    with pytest.raises(PolicyError, match="escapes workspace"):
        Policy.load(root=workspace).resolve("link.txt")


def test_resolve_blocks_secret_names(policy):
    with pytest.raises(PolicyError, match="blocked file"):
        policy.resolve(".env")


def test_check_command_allows_listed_program(policy):
    policy.check_command(["git", "status"])


def test_check_command_rejects_unlisted_program(policy):
    with pytest.raises(PolicyError, match="not allowed"):
        policy.check_command(["curl", "https://example.com"])


def test_check_command_rejects_metacharacters(policy):
    with pytest.raises(PolicyError, match="metacharacters"):
        policy.check_command(["git", "status", "&&", "rm"])


def test_read_only_blocks_commands(tmp_path):
    strict = Policy.load(root=tmp_path, approval=Approval.READ_ONLY)
    with pytest.raises(PolicyError, match="read-only"):
        strict.check_command(["git", "status"])


def test_requires_approval_by_mode(tmp_path):
    assert Policy.load(root=tmp_path, approval="ask").requires_approval("exec") is True
    assert (
        Policy.load(root=tmp_path, approval="auto").requires_approval("exec") is False
    )
    assert Policy.load(root=tmp_path, approval="ask").requires_approval("read") is False
