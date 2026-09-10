"""Tests for the opt-in docs-only write mode (.md/.txt + folder gate)."""

import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from oi_agent.config import Config, WatchTarget, load_config, save_config
from oi_agent.opencode.policy import (
    build_agent_config,
    build_permission_policy,
    normalize_write_dirs,
)
from oi_agent.opencode.prompt import build_prompt
from oi_agent.opencode.runner import _collect_doc_attachments


def _binary() -> str:
    """Return a portable executable accepted by config validation."""
    return shutil.which("true") or "/bin/true"


def _valid_config() -> Config:
    """Build a valid isolated test configuration."""
    cfg = Config(opencode_binary=_binary(), opencode_model="provider/model")
    cfg.watches = [WatchTarget(channel_id=111, repo_path="/srv/repo")]
    return cfg


def _git_repo(tmp_path: Path) -> Path:
    """Create a minimal committed git fixture."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "code.py").write_text("x = 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    return repo


# --- policy ---------------------------------------------------------------


def test_docs_only_allows_md_txt_and_denies_code(tmp_path):
    """Docs-only grants edit/write for .md/.txt; code and tools stay denied."""
    policy = build_permission_policy(tmp_path, write_mode="docs-only")
    assert policy["*"] == "deny"
    for tool in ("edit", "write"):
        rules = policy[tool]
        assert rules["*"] == "deny"
        assert rules["**/*.md"] == "allow"
        assert rules["**/*.txt"] == "allow"
        assert rules[".env"] == "deny"
        assert rules[".git/**"] == "deny"
    assert policy["bash"] == "deny"
    assert policy["patch"] == "deny"
    assert policy["read"]["*"] == "allow"


def test_docs_only_folder_gate_narrows_scope(tmp_path):
    """With write_dirs set, .md outside the scope is not allowed."""
    policy = build_permission_policy(
        tmp_path, write_mode="docs-only", write_dirs=["docs/plans"]
    )
    rules = policy["edit"]
    assert rules["*"] == "deny"
    assert rules["docs/plans/**/*.md"] == "allow"
    assert rules["docs/plans/**/*.txt"] == "allow"
    assert "**/*.md" not in rules
    assert "**/*.txt" not in rules


def test_unknown_write_mode_fails_closed_to_read_only(tmp_path):
    """Anything but docs-only keeps edit/write denied."""
    policy = build_permission_policy(tmp_path, write_mode="bogus")
    assert policy["edit"] == "deny"
    assert policy["write"] == "deny"


def test_normalize_write_dirs_drops_escapes():
    """Traversal, absolute, and empty scopes never widen the allow set."""
    assert normalize_write_dirs(["docs/plans/", "docs/memory"]) == ["docs/plans", "docs/memory"]
    assert normalize_write_dirs(["../evil", "/abs", ".", "", None]) == []  # type: ignore[list-item]
    assert normalize_write_dirs(None) == []


def test_agent_config_carries_docs_only_description(tmp_path):
    """Agent description reflects the granted mode."""
    prompt = build_prompt("voice")
    ro = build_agent_config("p/m", 10, tmp_path, prompt)["agent"]["oi"]
    docs = build_agent_config("p/m", 10, tmp_path, prompt, write_mode="docs-only")["agent"]["oi"]
    assert "Read-only" in ro["description"]
    assert "docs-only" in docs["description"]
    assert docs["permission"]["edit"]["**/*.md"] == "allow"


# --- prompt ---------------------------------------------------------------


def test_prompt_read_only_keeps_create_ban():
    """Default prompt still forbids all file creation."""
    prompt = build_prompt("voice")
    assert "Never edit, delete, create" in prompt
    assert "ONLY *.md and *.txt" not in prompt


def test_prompt_read_only_names_the_enable_command():
    """Read-only refusal points at the exact opt-in command."""
    prompt = build_prompt("voice")
    assert "disabled by configuration" in prompt
    assert "oi config set write_mode docs-only" in prompt


def test_prompt_docs_only_allows_docs_with_scope():
    """Docs-only prompt grants .md/.txt and names the folder gate."""
    prompt = build_prompt("voice", write_mode="docs-only", write_dirs=["docs/plans"])
    assert "ONLY *.md and *.txt" in prompt
    assert "docs/plans" in prompt
    assert "no code, no configs" in prompt


def test_prompt_docs_only_forbids_false_inability_claims():
    """Docs-only prompt bans 'no write tool / read-only' confabulation."""
    prompt = build_prompt("voice", write_mode="docs-only")
    assert "never claim you have no write tool" in prompt
    assert "WRITE THE FILE with the tools" in prompt
    assert "Never tell the user to create, paste, or commit the file" in prompt
    assert "never claim you have no write tool" not in build_prompt("voice")


# --- config + CLI ---------------------------------------------------------


def test_write_mode_roundtrip_and_validation(tmp_path):
    """write_mode/write_dirs survive save/load; bad values rejected."""
    cfg = _valid_config()
    cfg.write_mode = "docs-only"
    cfg.write_dirs = ["docs/plans", "docs/memory"]
    path = tmp_path / "config.toml"
    save_config(cfg, path)
    back = load_config(path)
    assert back.write_mode == "docs-only"
    assert back.write_dirs == ["docs/plans", "docs/memory"]

    bad = _valid_config()
    bad.write_mode = "full-write"
    with pytest.raises(ValueError, match="write_mode"):
        save_config(bad, tmp_path / "bad.toml")

    evil = _valid_config()
    evil.write_dirs = ["../escape"]
    with pytest.raises(ValueError, match="write_dirs"):
        save_config(evil, tmp_path / "evil.toml")


def test_config_set_write_mode_and_dirs_cli_roundtrip(tmp_path):
    """oi config set accepts write_mode and comma-separated write_dirs."""
    from oi_agent.cli import app

    runner = CliRunner()
    path = tmp_path / "config.toml"
    save_config(_valid_config(), path)

    result = runner.invoke(app, ["config", "set", "write_mode", "docs-only", "--config", str(path)])
    assert result.exit_code == 0, result.output
    assert load_config(path).write_mode == "docs-only"

    result = runner.invoke(
        app, ["config", "set", "write_dirs", "docs/plans,docs/memory", "--config", str(path)]
    )
    assert result.exit_code == 0, result.output
    assert load_config(path).write_dirs == ["docs/plans", "docs/memory"]

    shown = runner.invoke(app, ["config", "show", "--config", str(path)])
    assert shown.exit_code == 0, shown.output
    assert "write_mode = docs-only" in shown.output

    bad = runner.invoke(app, ["config", "set", "write_mode", "yolo", "--config", str(path)])
    assert bad.exit_code == 1
    assert load_config(path).write_mode == "docs-only"


# --- runner attachment collection -----------------------------------------


def test_collects_fresh_md_but_ignores_code_and_stale(tmp_path):
    """Only freshly touched .md/.txt qualify; code and old files ignored."""
    repo = _git_repo(tmp_path)
    since = time.time()
    fresh = repo / "plan.md"
    fresh.write_text("# plan\n", encoding="utf-8")
    (repo / "notes.txt").write_text("hi\n", encoding="utf-8")
    (repo / "evil.py").write_text("print(1)\n", encoding="utf-8")
    stale = repo / "old.md"
    stale.write_text("stale\n", encoding="utf-8")
    old_mtime = since - 3600
    import os

    os.utime(stale, (old_mtime, old_mtime))

    found = _collect_doc_attachments([repo], write_dirs=[], since_epoch=since)
    names = [Path(p).name for p in found]
    assert "plan.md" in names
    assert "notes.txt" in names
    assert "evil.py" not in names
    assert "old.md" not in names


def test_collect_respects_folder_gate(tmp_path):
    """Scoped mode only picks docs inside the allowed directories."""
    repo = _git_repo(tmp_path)
    since = time.time()
    scoped_dir = repo / "docs" / "plans"
    scoped_dir.mkdir(parents=True)
    (scoped_dir / "plan.md").write_text("# scoped\n", encoding="utf-8")
    (repo / "root.md").write_text("# root\n", encoding="utf-8")

    found = _collect_doc_attachments([repo], write_dirs=["docs/plans"], since_epoch=since)
    names = [Path(p).name for p in found]
    assert "plan.md" in names
    assert "root.md" not in names


# --- poster file upload ----------------------------------------------------


@pytest.mark.asyncio
async def test_deliver_file_uploads_md_and_rejects_code(tmp_path, monkeypatch):
    """deliver_file POSTs multipart for .md and refuses other types."""
    from oi_agent.poster import Poster
    from oi_agent.store import Store

    cfg = Config()
    cfg.watches = [WatchTarget(channel_id=111, repo_path="/repo", post_hourly_cap=1)]
    store = Store(tmp_path / "state.db")
    captured: dict = {}
    doc = tmp_path / "plan.md"
    doc.write_text("# plan\n", encoding="utf-8")

    class FakeClient:
        """Async HTTP client recording multipart upload arguments."""

        def __init__(self, *args, **kwargs):
            """Accept the httpx client constructor arguments."""

        async def __aenter__(self):
            """Enter fake client."""
            return self

        async def __aexit__(self, *args):
            """Exit fake client."""
            return False

        async def post(self, url, headers, data, files):
            """Capture multipart payload and return created message ID."""
            captured["url"] = url
            captured["data"] = data
            captured["files"] = files
            return SimpleNamespace(status_code=200, text="", json=lambda: {"id": "4242"})

    monkeypatch.setattr("oi_agent.poster.httpx.AsyncClient", FakeClient)
    poster = Poster(cfg, store, "token")

    result = await poster.deliver_file(111, 111, str(doc), reply_to_message_id=777)
    assert result.ok is True
    assert result.discord_message_id == 4242
    assert captured["url"] == "https://discord.com/api/v10/channels/111/messages"
    assert "payload_json" in captured["data"]
    assert "files[0]" in captured["files"]
    # File upload is best-effort: it must not consume the hourly text cap.
    assert store.posts_in_last_hour(111) == 0

    bad = tmp_path / "evil.py"
    bad.write_text("print(1)\n", encoding="utf-8")
    denied = await poster.deliver_file(111, 111, str(bad))
    assert denied.ok is False
    assert denied.error_class == "file_type_denied"
