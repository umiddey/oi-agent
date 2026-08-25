"""Tests for jailed repo tools."""

import pytest

from oi_agent.agent.tools import (
    TOOL_SCHEMAS_ANTHROPIC,
    TOOL_SCHEMAS_OPENAI,
    ToolError,
    execute_tool,
    grep,
    read_file,
)


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "mod.py").write_text(
        "def charge_tenant(amount):\n    return amount * 2\n")
    return tmp_path


def test_read_file_ok(repo):
    text = read_file(repo, "src/mod.py")
    assert "charge_tenant" in text


def test_read_file_blocks_traversal(repo):
    with pytest.raises(ToolError):
        read_file(repo, "../outside.py")


def test_read_file_blocks_absolute(repo):
    with pytest.raises(ToolError):
        read_file(repo, "/etc/passwd")


def test_grep_finds_symbol(repo):
    hits = grep(repo, "charge_tenant")
    assert any("mod.py" in h for h in hits)


def test_grep_is_case_insensitive(tmp_path):
    """Lowercased terms must match CamelCase/acronym code (git & fallback)."""
    (tmp_path / "auth.py").write_text("class OAuthClient:\n    pass\n")
    hits = grep(tmp_path, "oauth")
    assert any("OAuthClient" in h for h in hits)


@pytest.mark.parametrize("sensitive", [
    ".env", ".env.local", ".git/config", "deploy.key", "certs/server.pem",
    "config/.secret",
])
def test_read_file_denies_sensitive_paths(repo, sensitive):
    """Secrets/metadata inside the repo must be refused, not read."""
    target = repo / sensitive
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("SECRET=x")
    with pytest.raises(ToolError):
        read_file(repo, sensitive)


def test_tool_schemas_describe_same_tools():
    """OpenAI and Anthropic schema variants must cover the same two tools."""
    assert [s["function"]["name"] for s in TOOL_SCHEMAS_OPENAI] == [
        "grep_repo", "read_file"
    ]
    assert [s["name"] for s in TOOL_SCHEMAS_ANTHROPIC] == [
        "grep_repo", "read_file"
    ]
    openai_grep = TOOL_SCHEMAS_OPENAI[0]["function"]["parameters"]
    assert openai_grep == TOOL_SCHEMAS_ANTHROPIC[0]["input_schema"]


def test_execute_tool_grep_repo_happy_path(repo):
    hits = execute_tool(repo, "grep_repo", {"patterns": ["charge_tenant"]})
    assert "src/mod.py" in hits
    assert "charge_tenant" in hits

@pytest.mark.parametrize("patterns", [
    "charge_tenant|missing_symbol",
    """["']?charge_tenant["']?|missing_symbol"]""",
])
def test_execute_tool_recovers_single_string_pattern(repo, patterns):
    """Provider-emitted single regex strings run as one grep pattern."""
    hits = execute_tool(repo, "grep_repo", {"patterns": patterns})
    assert "src/mod.py" in hits
    assert "charge_tenant" in hits


def test_execute_tool_read_file_happy_path(repo):
    text = execute_tool(repo, "read_file", {"path": "src/mod.py"})
    assert "charge_tenant" in text


def test_execute_tool_read_file_truncates_large_files(repo):
    """read_file must truncate, never refuse: the model self-corrects on
    partial evidence but a hard error would stall the loop."""
    (repo / "big.py").write_text("x = 0\n" * 20_000)
    out = execute_tool(
        repo, "read_file", {"path": "big.py", "max_bytes": 100}
    )
    assert "[... file evidence truncated ...]" in out


def test_execute_tool_output_capped_at_40k_chars(repo):
    (repo / "big.txt").write_text("y" * 60_000)
    out = execute_tool(
        repo, "read_file", {"path": "big.txt", "max_bytes": 50000}
    )
    assert len(out) <= 40_000
    assert out.endswith("[... tool output truncated ...]")


def test_execute_tool_containment_errors_returned_not_raised(repo):
    result = execute_tool(repo, "read_file", {"path": "../outside.py"})
    assert result.startswith("ERROR:")


def test_execute_tool_sensitive_path_denied(repo):
    (repo / ".env").write_text("SECRET=x")
    result = execute_tool(repo, "read_file", {"path": ".env"})
    assert result.startswith("ERROR:")


def test_execute_tool_unknown_tool(repo):
    result = execute_tool(repo, "delete_everything", {})
    assert result.startswith("ERROR:") and "unknown tool" in result


@pytest.mark.parametrize("arguments", [
    {},
    {"patterns": []},
    {"patterns": ""},
    {"patterns": 42},
    {"patterns": [42]},
    {"patterns": ["ok"] * 13},
])
def test_execute_tool_rejects_bad_arguments(repo, arguments):
    """Structural argument mistakes must come back as error text so the
    model can correct its next call instead of the audit crashing."""
    result = execute_tool(repo, "grep_repo", arguments)
    assert result.startswith("ERROR:")
