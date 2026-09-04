"""Tests for the native OpenCode OI agent and fail-closed permissions."""

from pathlib import Path

from oi_agent.opencode.policy import build_agent_config, build_permission_policy
from oi_agent.opencode.prompt import build_prompt


def test_policy_is_read_only_and_repo_scoped(tmp_path):
    """Only read/glob/list tools are allowed, grep and write tools are denied."""
    root = (tmp_path / "repo").resolve()
    policy = build_permission_policy(root)
    assert policy["*"] == "deny"
    for tool in ("read", "glob", "list"):
        assert policy[tool]["*"] == "allow"
    assert policy["read"][".env"] == "deny"
    assert policy["read"][".git/**"] == "deny"
    assert policy["read"]["**/*credentials*"] == "deny"
    assert policy["read"]["**/*secret*"] == "deny"
    assert policy["read"]["**/*token*"] == "deny"
    assert policy["external_directory"] == "deny"
    assert policy["lsp"] == "allow"
    assert policy["grep"] == "deny"
    assert policy["bash"] == "deny"
    assert policy["edit"] == "deny"
    assert policy["write"] == "deny"


def test_native_agent_has_steps_model_and_inline_prompt(tmp_path):
    """Agent configuration is native OpenCode JSON, not a linked skill."""
    prompt = build_prompt("brief and precise")
    config = build_agent_config(
        "provider/model", 30, Path(tmp_path), prompt
    )
    agent = config["agent"]["oi"]
    assert agent["mode"] == "primary"
    assert agent["model"] == "provider/model"
    assert agent["steps"] == 30
    assert agent["permission"]["*"] == "deny"
    assert "brief and precise" in agent["prompt"]
    assert "never override safety" in agent["prompt"].lower()
