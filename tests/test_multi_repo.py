"""Unit and integration tests for Many-to-Many (M:N) multi-repo and guild-wide watches.

Covers prompt consistency, secret isolation across read/glob/list, thread-parent
guild filter enforcement, strict schema validation, per-repository dirty provenance,
guild reconciliation with active/archived threads, guild message admission & delivery,
and interactive CLI management.
"""

from __future__ import annotations

import asyncio
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import discord
import pytest
from typer.testing import CliRunner

from oi_agent.cli import app
from oi_agent.config import Config, WatchTarget, load_config, save_config, validate_config
from oi_agent.opencode.policy import build_permission_policy
from oi_agent.opencode.prompt import build_prompt
from oi_agent.opencode.runner import OpenCodeRunner
from oi_agent.poster import DeliveryResult
from oi_agent.store import Store
from oi_agent.watch.discord_client import OIWatcher

# ==============================================================================
# Helper Mock Classes
# ==============================================================================

class FakeAuthor:
    """Minimal Discord author object."""

    def __init__(self, user_id: int, name: str = "alice", bot: bool = False) -> None:
        self.id = user_id
        self.display_name = name
        self.bot = bot


class FakeMention:
    """Minimal Discord mention object."""

    def __init__(self, user_id: int) -> None:
        self.id = user_id


class FakeMessage:
    """Minimal Discord message object."""

    def __init__(
        self,
        message_id: int,
        channel: FakeChannel,
        author: FakeAuthor,
        content: str,
        mentions: list[FakeMention] | None = None,
    ) -> None:
        self.id = message_id
        self.channel = channel
        self.author = author
        self.content = content
        self.mentions = list(mentions or [])
        self.attachments: list[Any] = []
        self.embeds: list[Any] = []
class FakeChannel:
    """Fake Discord channel or thread with message history and archived threads."""

    def __init__(
        self,
        channel_id: int,
        history: list[FakeMessage] | None = None,
        parent_id: int | None = None,
        guild_id: int | None = None,
        archived_threads: list[FakeChannel] | None = None,
    ) -> None:
        self.id = channel_id
        self.parent_id = parent_id
        self.guild_id = guild_id
        self.type = 0
        self._history = list(history or [])
        self._archived_threads = list(archived_threads or [])

    async def history(
        self, limit: int = 50, after: discord.Object | None = None, oldest_first: bool = False
    ):
        msgs = list(self._history)
        if after is not None:
            after_id = after.id if hasattr(after, "id") else int(after)
            msgs = [m for m in msgs if m.id > after_id]
        if oldest_first:
            msgs = sorted(msgs, key=lambda m: m.id)
        else:
            msgs = sorted(msgs, key=lambda m: m.id, reverse=True)
        for msg in msgs[:limit]:
            yield msg

    async def fetch_message(self, message_id: int) -> FakeMessage:
        for m in self._history:
            if m.id == message_id:
                return m
        raise discord.NotFound(MagicMock(), "message not found")

    async def archived_threads(self, limit: int = 50) -> list[FakeChannel]:
        return self._archived_threads[:limit]


class FakeGuild:
    """Fake Discord guild with channels and active threads."""

    def __init__(
        self,
        guild_id: int = 999,
        channels: list[FakeChannel] | None = None,
        threads: list[FakeChannel] | None = None,
    ) -> None:
        self.id = guild_id
        self.channels = list(channels or [])
        self.threads = list(threads or [])

    def active_threads(self) -> list[FakeChannel]:
        return self.threads


def _create_mock_opencode(tmp_path: Path, mutate_repo: Path | None = None) -> Path:
    """Create a real executable script simulating OpenCode runner."""
    script_path = tmp_path / "mock_opencode"
    mutate_snippet = ""
    if mutate_repo:
        mutate_snippet = f'with open("{mutate_repo}/dirty.txt", "w") as f: f.write("dirty\\n")'

    body = f"""#!{sys.executable}
import json, sys
{mutate_snippet}
part = {{"id": "p2", "messageID": "msg-1", "sessionID": "sess-multi",
        "type": "text", "text": "[[OI_FINAL_ANSWER]]\\nMulti-repo audit completed"}}
print(json.dumps({{"type": "step_start", "timestamp": 0, "sessionID": "sess-multi",
                  "part": {{"id": "p1", "messageID": "msg-1",
                           "sessionID": "sess-multi", "type": "step-start"}}}}))
print(json.dumps({{"type": "text", "timestamp": 0, "sessionID": "sess-multi", "part": part}}))
print(json.dumps({{"type": "step_finish", "timestamp": 0, "sessionID": "sess-multi",
                  "part": {{"id": "p3", "messageID": "msg-1",
                           "sessionID": "sess-multi", "type": "step-finish",
                           "reason": "stop"}}}}))
"""
    script_path.write_text(body, encoding="utf-8")
    script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)
    return script_path


# ==============================================================================
# 1. Configuration & Target Definition Tests
# ==============================================================================

def test_watch_target_properties_single_repo():
    """Verify legacy single repo target properties."""
    target = WatchTarget(channel_id=123, repo_path="/repo/one")
    assert target.repo_paths == ["/repo/one"]
    assert target.primary_repo == "/repo/one"
    assert target.is_guild_watch is False
    assert target.matches_channel(123) is True
    assert target.matches_channel(456) is False


def test_watch_target_properties_multi_repo():
    """Verify multi-repo target properties."""
    target = WatchTarget(channel_id=123, repos=["/repo/one", "/repo/two"])
    assert target.repo_paths == ["/repo/one", "/repo/two"]
    assert target.primary_repo == "/repo/one"
    assert target.is_guild_watch is False
    assert target.matches_channel(123) is True


def test_watch_target_guild_wide_matching_and_filters():
    """Verify guild-wide watch matching with ignore and allow channel lists."""
    guild_target = WatchTarget(
        guild_id=999,
        repo_path="/repo/one",
        ignored_channels=[100, 101],
    )
    assert guild_target.is_guild_watch is True
    assert guild_target.matches_channel(50, guild_id=999) is True
    assert guild_target.matches_channel(100, guild_id=999) is False  # Ignored
    assert guild_target.matches_channel(101, guild_id=999) is False  # Ignored
    assert guild_target.matches_channel(50, guild_id=888) is False  # Different guild

    guild_target_allow = WatchTarget(
        guild_id=999,
        repo_path="/repo/one",
        allowed_channels=[200, 201],
    )
    assert guild_target_allow.matches_channel(200, guild_id=999) is True
    assert guild_target_allow.matches_channel(201, guild_id=999) is True
    assert guild_target_allow.matches_channel(202, guild_id=999) is False  # Not in allowlist


def test_strict_type_validation_and_mutual_exclusivity(tmp_path):
    """Verify malformed list types and ambiguous channel/guild targets raise ValueError."""
    # 1. repos must be a list
    bad_toml_1 = tmp_path / "bad1.toml"
    bad_toml_1.write_text("""
discord_token_env = "T"
db_path = "/tmp/db"
personality = "P"
opencode_binary = "/bin/true"
opencode_model = "prov/mod"
[[watch]]
channel_id = 123
repos = "not-a-list"
""", encoding="utf-8")
    with pytest.raises(ValueError, match="'repos' must be an array of strings"):
        load_config(bad_toml_1)

    # 2. ignored_channels must be a list of integers
    bad_toml_2 = tmp_path / "bad2.toml"
    bad_toml_2.write_text("""
discord_token_env = "T"
db_path = "/tmp/db"
personality = "P"
opencode_binary = "/bin/true"
opencode_model = "prov/mod"
[[watch]]
guild_id = 123
repo_path = "/tmp/repo"
ignored_channels = "not-a-list"
""", encoding="utf-8")
    with pytest.raises(ValueError, match="'ignored_channels' must be an array of integers"):
        load_config(bad_toml_2)

    # 3. Exactly one of channel_id or guild_id
    bad_target_both = WatchTarget(channel_id=100, guild_id=200, repo_path="/tmp/repo")
    with pytest.raises(ValueError, match="cannot specify both channel_id and guild_id"):
        validate_config(Config(opencode_binary="/bin/true", watches=[bad_target_both]))

    bad_target_neither = WatchTarget(channel_id=0, guild_id=0, repo_path="/tmp/repo")
    with pytest.raises(ValueError, match="must specify exactly one positive channel_id or guild_id"):
        validate_config(Config(opencode_binary="/bin/true", watches=[bad_target_neither]))


def test_validate_config_directly_constructed_targets_hardening():
    """Verify validate_config strictly validates types, non-empty paths, and positive filter IDs."""
    # Negative channel ID
    with pytest.raises(ValueError, match="channel_id must be a non-negative integer"):
        validate_config(Config(opencode_binary="/bin/true", watches=[WatchTarget(channel_id=-1, repo_path="/tmp")]))

    # Negative guild ID
    with pytest.raises(ValueError, match="guild_id must be a non-negative integer"):
        validate_config(Config(opencode_binary="/bin/true", watches=[WatchTarget(guild_id=-1, repo_path="/tmp")]))

    # Negative ignored channel ID
    with pytest.raises(ValueError, match="ignored_channels item must be a positive integer"):
        validate_config(Config(
            opencode_binary="/bin/true",
            watches=[WatchTarget(guild_id=100, repo_path="/tmp", ignored_channels=[-5])],
        ))

    # Zero allowed channel ID
    with pytest.raises(ValueError, match="allowed_channels item must be a positive integer"):
        validate_config(Config(
            opencode_binary="/bin/true",
            watches=[WatchTarget(guild_id=100, repo_path="/tmp", allowed_channels=[0])],
        ))

    # Empty repos strings
    with pytest.raises(ValueError, match="repos item must be a non-empty string"):
        validate_config(Config(
            opencode_binary="/bin/true",
            watches=[WatchTarget(channel_id=100, repos=["/tmp", "  "])],
        ))

    # Invalid post_hourly_cap
    with pytest.raises(ValueError, match="post_hourly_cap must be an integer >= 1"):
        validate_config(Config(
            opencode_binary="/bin/true",
            watches=[WatchTarget(channel_id=100, repo_path="/tmp", post_hourly_cap=0)],
        ))

    # Invalid bot_user_id
    with pytest.raises(ValueError, match="bot_user_id must be a non-negative integer"):
        validate_config(Config(
            opencode_binary="/bin/true",
            watches=[WatchTarget(channel_id=100, repo_path="/tmp", bot_user_id=-1)],
        ))


# ==============================================================================
# 2. Multi-Root Permission Policy & Secret Isolation
# ==============================================================================

def test_secret_denial_covers_read_glob_and_list_across_all_roots():
    """Verify secret deny patterns are strictly enforced across read, glob, list for primary & auxiliary roots."""
    primary = Path("/home/user/work/repo-main")
    aux = Path("/home/user/work/repo-aux")

    policy = build_permission_policy(primary, additional_repo_roots=[aux])

    # Check read, glob, and list all have secret blocking
    for tool in ("read", "glob", "list"):
        tool_rules = policy[tool]
        assert isinstance(tool_rules, dict)
        assert tool_rules["*"] == "allow"
        assert tool_rules[".git/**"] == "deny"
        assert tool_rules[".env"] == "deny"
        assert tool_rules["**/.env"] == "deny"
        assert tool_rules["**/.env.*"] == "deny"
        assert tool_rules["**/*secret*"] == "deny"

        # Auxiliary absolute paths must also be protected
        aux_str = str(aux.resolve())
        assert tool_rules[f"{aux_str}/.git/**"] == "deny"
        assert tool_rules[f"{aux_str}/.env"] == "deny"
        assert tool_rules[f"{aux_str}/**/.env"] == "deny"

    # Mutations must be denied
    assert policy["edit"] == "deny"
    assert policy["write"] == "deny"
    assert policy["bash"] == "deny"
    assert policy["mcp"] == "deny"

    # External directory must grant access to auxiliary root
    ext = policy["external_directory"]
    assert isinstance(ext, dict)
    assert ext["*"] == "deny"
    assert ext[str(aux.resolve())] == "allow"
    assert ext[f"{aux.resolve()}/**"] == "allow"


# ==============================================================================
# 3. Prompt Consistency (No Multi-Repo Contradiction)
# ==============================================================================

def test_prompt_consistency_no_singular_contradiction():
    """Verify prompt explicitly permits authorized watched repositories without singular restrictions."""
    prompt = build_prompt(
        personality="Senior tone",
        primary_repo="/work/primary",
        additional_repos=["/work/secondary"],
    )

    # Must contain multi-repo guidance
    assert "MULTIPLE REPOSITORY ACCESS:" in prompt
    assert "Primary workspace (--dir): /work/primary" in prompt
    assert "- /work/secondary" in prompt

    # Must NOT contain singular contradictory restrictions
    assert "Inspect only the watched repository" not in prompt
    assert "outside the repo." not in prompt

    # Must contain authorized plural permissions
    assert "explicitly authorized watched repositories" in prompt
    assert "outside the authorized repository roots" in prompt


# ==============================================================================
# 4. Gateway Thread-Parent Guild Filter & Owner ID Resolution
# ==============================================================================

def test_thread_parent_channel_ignore_filter_enforcement(tmp_path):
    """Verify threads under an ignored parent channel are rejected and owner ID is parent."""
    # Guild watch on Guild 999 with channel 500 ignored
    guild_target = WatchTarget(
        guild_id=999,
        repo_path=str(tmp_path / "repo"),
        ignored_channels=[500],
    )
    cfg = Config(watches=[guild_target])
    store = Store(tmp_path / "test.db")
    watcher = OIWatcher(cfg, store)

    # Thread 601 has parent 500 (ignored)
    ignored_thread = FakeChannel(601, parent_id=500, guild_id=999)
    target, owner = watcher._resolve_target(601, ignored_thread)
    assert target is None
    assert owner is None

    # Thread 701 has parent 200 (allowed)
    allowed_thread = FakeChannel(701, parent_id=200, guild_id=999)
    target_ok, owner_ok = watcher._resolve_target(701, allowed_thread)
    assert target_ok is guild_target
    assert owner_ok == 200  # Owner must be parent channel ID!


# ==============================================================================
# 5. End-to-End Guild Target Reconciliation (Active & Archived Threads)
# ==============================================================================

@pytest.mark.asyncio
async def test_guild_reconciliation_discovery_and_active_archived_threads(tmp_path):
    """Verify _reconcile_targets auto-discovers guild channels, active threads, and archived threads."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    guild_target = WatchTarget(
        guild_id=999,
        repo_path=str(repo),
        ignored_channels=[500],
    )
    cfg = Config(watches=[guild_target], opencode_binary="/bin/true")
    store = Store(tmp_path / "test.db")
    watcher = OIWatcher(cfg, store)

    # Build fake guild structure:
    # - Channel 101 (active): has message 1001, archived thread 701 (with message 7001)
    # - Channel 500 (ignored): has message 5001
    # - Thread 601 (active under 101): has message 6001
    # - Thread 602 (active under 500 - ignored parent): has message 6002
    ch_archived_701 = FakeChannel(701, parent_id=101, guild_id=999)
    ch_archived_701._history = [FakeMessage(7001, ch_archived_701, FakeAuthor(10), "archived msg")]

    ch_101 = FakeChannel(101, guild_id=999, archived_threads=[ch_archived_701])
    ch_101._history = [FakeMessage(1001, ch_101, FakeAuthor(10), "channel 101 msg")]

    ch_ignored_500 = FakeChannel(500, guild_id=999)
    ch_ignored_500._history = [FakeMessage(5001, ch_ignored_500, FakeAuthor(10), "ignored msg")]

    thread_601 = FakeChannel(601, parent_id=101, guild_id=999)
    thread_601._history = [FakeMessage(6001, thread_601, FakeAuthor(10), "thread 601 msg")]

    thread_ignored_602 = FakeChannel(602, parent_id=500, guild_id=999)
    thread_ignored_602._history = [FakeMessage(6002, thread_ignored_602, FakeAuthor(10), "ignored thread msg")]

    fake_guild = FakeGuild(
        guild_id=999,
        channels=[ch_101, ch_ignored_500],
        threads=[thread_601, thread_ignored_602],
    )

    watcher._test_guilds = [fake_guild]  # type: ignore[attr-defined]
    watcher._channel_cache = {
        101: ch_101, 500: ch_ignored_500, 601: thread_601, 602: thread_ignored_602, 701: ch_archived_701,
    }

    await watcher._reconcile_targets()

    # Verify cursors updated in SQLite store for valid channels & threads:
    assert store.get_scope_cursor("101") == 1001
    assert store.get_scope_cursor("601") == 6001
    assert store.get_scope_cursor("701") == 7001

    # Verify ignored channel and thread under ignored parent were NOT reconciled:
    assert store.get_scope_cursor("500") is None
    assert store.get_scope_cursor("602") is None


# ==============================================================================
# 6. Guild Watch Admission, Debounce, and Delivery End-to-End Flow
# ==============================================================================

@pytest.mark.asyncio
async def test_guild_watch_admission_debounce_and_delivery_flow(tmp_path, monkeypatch):
    """Verify a message sent in a guild thread is admitted, debounced, audited, and posted."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    monkeypatch.setattr("oi_agent.watch.discord_client.BURST_QUIET_SECONDS", 0.05)
    repo1 = tmp_path / "repo1"
    repo1.mkdir()
    (repo1 / "code.py").write_text("print('hello')", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo1), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo1), "config", "user.name", "T"], check=True)
    subprocess.run(["git", "-C", str(repo1), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(repo1), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo1), "commit", "-qm", "init1"], check=True)

    mock_opencode = _create_mock_opencode(tmp_path)

    guild_target = WatchTarget(
        guild_id=999,
        repos=[str(repo1)],
        bot_user_id=99,
        ignored_channels=[500],
    )
    cfg = Config(opencode_binary=str(mock_opencode), opencode_model="prov/model", watches=[guild_target])
    store = Store(tmp_path / "test.db")
    watcher = OIWatcher(cfg, store)

    thread_channel = FakeChannel(601, parent_id=101, guild_id=999)
    watcher._channel_cache[601] = thread_channel
    watcher._channel_cache[101] = FakeChannel(101, guild_id=999)

    msg = FakeMessage(
        message_id=8888,
        channel=thread_channel,
        author=FakeAuthor(1234, name="bob"),
        content="<@99> how does this work in repo1?",
        mentions=[FakeMention(99)],
    )
    thread_channel._history.append(msg)

    # Mock poster to capture delivery
    delivered_chunks: list[dict[str, Any]] = []

    async def mock_deliver_chunk(
        _self: Any,
        channel_id: int,
        target_channel_id: int,
        chunk: str,
        nonce: str,
        reply_to_message_id: int | None = None,
        _client: Any = None,
    ) -> DeliveryResult:
        delivered_chunks.append({
            "channel_id": channel_id,
            "target_channel_id": target_channel_id,
            "body": chunk,
            "nonce": nonce,
            "reply_to": reply_to_message_id,
        })
        return DeliveryResult(ok=True, discord_message_id=99001)

    monkeypatch.setattr("oi_agent.watch.discord_client.Poster.deliver_chunk", mock_deliver_chunk)

    # 1. Ingest event through on_message gateway hook
    await watcher.on_message(msg)

    # 2. Verify job admitted in store
    stats = store.get_queue_stats()
    assert stats["pending_jobs"] == 1

    # 3. Wait for debounce and processing
    for _ in range(40):
        if delivered_chunks:
            break
        await asyncio.sleep(0.05)
    # 4. Verify delivered through poster and marked done in store
    assert len(delivered_chunks) == 1
    assert delivered_chunks[0]["channel_id"] == 601
    assert "Multi-repo audit completed" in delivered_chunks[0]["body"]
    assert delivered_chunks[0]["reply_to"] == 8888

    stats_after = store.get_queue_stats()
    assert stats_after["pending_jobs"] == 0
    assert stats_after["done_jobs"] == 1
    assert stats_after["done_batches"] == 1

    await watcher.close()


# ==============================================================================
# 7. Per-Repository Dirty Provenance Tracking (Real Subprocess Execution)
# ==============================================================================

@pytest.mark.asyncio
async def test_multi_repo_per_repository_dirty_provenance(tmp_path, monkeypatch):
    """Verify -dirty suffix is applied exclusively to the specific repository that changed."""
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))

    # Create two git repos
    repo1 = tmp_path / "repo1"
    repo1.mkdir()
    (repo1 / "file1.txt").write_text("v1", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo1), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo1), "config", "user.name", "T"], check=True)
    subprocess.run(["git", "-C", str(repo1), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(repo1), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo1), "commit", "-qm", "init1"], check=True)

    repo2 = tmp_path / "repo2"
    repo2.mkdir()
    (repo2 / "file2.txt").write_text("v2", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo2), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo2), "config", "user.name", "T"], check=True)
    subprocess.run(["git", "-C", str(repo2), "config", "user.email", "t@e.com"], check=True)
    subprocess.run(["git", "-C", str(repo2), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo2), "commit", "-qm", "init2"], check=True)

    mock_binary = _create_mock_opencode(tmp_path, mutate_repo=repo1)

    target = WatchTarget(channel_id=100, repos=[str(repo1), str(repo2)])
    cfg = Config(opencode_binary=str(mock_binary), opencode_model="prov/model")
    runner = OpenCodeRunner(cfg)

    reply = await runner.run(target, "author", "question", "")
    assert reply.ok is True
    # repo1 was mutated by the mock runner, repo2 remained clean
    assert "repo1@" in reply.sha
    assert "repo2@" in reply.sha
    assert "repo1@" in reply.sha and "-dirty" in reply.sha.split("repo1@")[1].split(",")[0]
    assert "-dirty" not in reply.sha.split("repo2@")[1]


# ==============================================================================
# 8. CLI Management & Init Wizard Tests
# ==============================================================================

def test_cli_watch_add_remove_and_show_multi_repo(tmp_path):
    """Verify CLI watch-add with --guild-id, --repos, --ignored-channels, and watch-remove."""
    cfg_file = tmp_path / "test_config.toml"
    repo1 = tmp_path / "r1"
    repo1.mkdir()
    subprocess.run(["git", "-C", str(repo1), "init", "-q"], check=True)
    repo2 = tmp_path / "r2"
    repo2.mkdir()
    subprocess.run(["git", "-C", str(repo2), "init", "-q"], check=True)

    save_config(Config(opencode_binary="/bin/true", watches=[WatchTarget(111, str(repo1))]), path=cfg_file)

    runner = CliRunner()

    # 1. Add guild watch with multi-repos and ignored channels
    res_add = runner.invoke(app, [
        "config", "watch-add",
        "--guild-id", "999",
        "--repos", f"{repo1},{repo2}",
        "--ignored-channels", "400,401",
        "--config", str(cfg_file),
    ])
    assert res_add.exit_code == 0
    assert "watching guild 999" in res_add.stdout

    # 2. Verify config show output
    res_show = runner.invoke(app, ["config", "show", "--config", str(cfg_file)])
    assert res_show.exit_code == 0
    assert "guild 999" in res_show.stdout
    assert "ignored=[400, 401]" in res_show.stdout

    # 3. Remove guild watch
    res_rm = runner.invoke(app, [
        "config", "watch-remove", "999", "--config", str(cfg_file)
    ])
    assert res_rm.exit_code == 0
    assert "removed watch 999" in res_rm.stdout

    loaded = load_config(cfg_file)
    assert len([w for w in loaded.watches if w.guild_id == 999]) == 0


def test_cli_init_guild_and_multi_repo(tmp_path, monkeypatch):
    """Verify oi init wizard accepts guild watches, ignored channels, and comma-separated repos."""
    cfg_file = tmp_path / "init_config.toml"
    secrets_file = tmp_path / "secrets.env"
    repo1 = tmp_path / "repo1"
    repo1.mkdir()
    subprocess.run(["git", "-C", str(repo1), "init", "-q"], check=True)
    repo2 = tmp_path / "repo2"
    repo2.mkdir()
    subprocess.run(["git", "-C", str(repo2), "init", "-q"], check=True)

    monkeypatch.setattr("getpass.getpass", lambda *args: "test-bot-token")
    monkeypatch.setattr("oi_agent.cli._ensure_opencode", lambda model, consent: "openai/gpt-4o-mini")

    # Inputs sequence:
    # 1. Env var name: default (enter)
    # 2. Personality: default (enter)
    # 3. Watch target type: "g" (guild)
    # 4. Discord guild id: "987654"
    # 5. Ignored channels: "500,501"
    # 6. Allowed channels: "" (enter)
    # 7. Repo paths: f"{repo1},{repo2}"
    # 8. bot user id: "12345"
    # 9. posts per hour cap: "5"
    # 10. Watch target type: "q" (done)
    inputs = [
        "",                     # env var name
        "",                     # personality
        "g",                    # target type
        "987654",               # guild id
        "500,501",              # ignored channels
        "",                     # allowed channels
        f"{repo1},{repo2}",     # repos
        "12345",                # bot user id
        "5",                    # cap
        "q",                    # done
    ]

    runner = CliRunner()
    res = runner.invoke(
        app,
        ["init", "--config", str(cfg_file), "--secrets", str(secrets_file)],
        input="\n".join(inputs) + "\n",
        env={"COLUMNS": "200"},
    )
    assert res.exit_code == 0, res.stdout
    assert f"wrote {cfg_file}" in res.stdout
    # Verify saved configuration
    loaded = load_config(cfg_file)
    assert len(loaded.watches) == 1
    target = loaded.watches[0]
    assert target.guild_id == 987654
    assert target.channel_id == 0
    assert target.is_guild_watch is True
    assert target.ignored_channels == [500, 501]
    assert target.repo_paths == [str(repo1.resolve()), str(repo2.resolve())]
    assert target.bot_user_id == 12345
    assert target.post_hourly_cap == 5
