"""CLI tests for the OpenCode contract and retained controls."""

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import discord
import pytest
import typer
from typer.testing import CliRunner

from oi_agent import cli
from oi_agent.cli import app
from oi_agent.reflection.engine import ReflectionSummary
from oi_agent.store import Store

runner = CliRunner()


def _watch_config(tmp_path):
    """Create a watched git-like path and config through the CLI."""
    cfg_path = tmp_path / "config.toml"
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    result = runner.invoke(app, [
        "config", "watch-add", "--channel-id", "111", "--repo-path", str(repo),
        "--config", str(cfg_path),
    ])
    assert result.exit_code == 0, result.output
    return cfg_path, repo


def test_config_set_and_show_opencode_settings(tmp_path):
    """OpenCode model/steps are editable and visible without secrets."""
    cfg_path, _ = _watch_config(tmp_path)
    result = runner.invoke(app, [
        "config", "set", "opencode_model", "provider/model", "--config", str(cfg_path)
    ])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, [
        "config", "set", "opencode_steps", "40", "--config", str(cfg_path)
    ])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["config", "show", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "opencode_model = provider/model" in result.output
    assert "opencode_steps = 40" in result.output
    assert "api_key" not in result.output
    assert "base_url" not in result.output


def test_config_set_rejects_legacy_and_unknown_keys(tmp_path):
    """Legacy provider knobs have no compatibility path."""
    cfg_path, _ = _watch_config(tmp_path)
    for key in ("agent.model", "max_tool_iterations", "bogus"):
        result = runner.invoke(app, [
            "config", "set", key, "value", "--config", str(cfg_path)
        ])
        assert result.exit_code == 1


def test_watch_remove_preserves_previous_file_on_final_target(tmp_path):
    """The final watch cannot be removed into an unloadable config."""
    cfg_path, _ = _watch_config(tmp_path)
    before = cfg_path.read_bytes()
    result = runner.invoke(app, [
        "config", "watch-remove", "111", "--config", str(cfg_path)
    ])
    assert result.exit_code == 1
    assert cfg_path.read_bytes() == before


def test_run_help_has_no_legacy_transport_switches():
    """Runtime help exposes neither streaming nor answer-log controls."""
    result = runner.invoke(app, ["run", "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0
    assert "--stream" not in result.output
    assert "--log-answers" not in result.output


def test_daemon_dry_run_help_and_command(tmp_path):
    """Dry-run renders a hardened unit without requiring root mutation."""
    cfg_path, _ = _watch_config(tmp_path)
    result = runner.invoke(app, [
        "daemon", "install", "--dry-run", "--config", str(cfg_path),
    ])
    assert result.exit_code == 0, result.output
    assert "ProtectSystem=strict" in result.output
    assert "User=oi" in result.output
    assert "OPENCODE_DISABLE_PROJECT_CONFIG=1" in result.output


def test_missing_opencode_aborts_without_install_or_partial_state(
        tmp_path, monkeypatch):
    """Non-interactive init failure does not create OI/OpenCode state."""
    from oi_agent import cli
    from oi_agent.opencode.bootstrap import OpenCodeBootstrapError

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        cli,
        "check_version",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OpenCodeBootstrapError("missing")
        ),
    )
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: False))
    with pytest.raises(typer.Exit):
        cli._ensure_opencode(None, False)
    assert not (tmp_path / ".local/share/oi/opencode").exists()


def test_interactive_missing_opencode_requests_only_install_consent(
        tmp_path, monkeypatch):
    """Interactive init asks for consent before any other setup writes."""
    from oi_agent import cli
    from oi_agent.opencode.bootstrap import OpenCodeBootstrapError

    asked = []
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(
        cli,
        "check_version",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            OpenCodeBootstrapError("missing")
        ),
    )
    monkeypatch.setattr(cli.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(
        cli.typer,
        "confirm",
        lambda prompt, default=False: asked.append((prompt, default)) or False,
    )
    with pytest.raises(typer.Exit):
        cli._ensure_opencode(None, False)
    assert len(asked) == 1
    assert not (tmp_path / ".local/share/oi/opencode").exists()


def test_queue_status_command(tmp_path):
    """Verify queue status displays metrics table cleanly.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.

    Returns:
        None: Asserts exit code and output content.
    """
    cfg_path, repo = _watch_config(tmp_path)
    store_path = tmp_path / "state.db"
    runner.invoke(app, ["config", "set", "db_path", str(store_path), "--config", str(cfg_path)])
    store = Store(store_path)
    store.admit_mention_job(1001, 2001, 3001, repo)
    store.close()

    result = runner.invoke(app, ["queue", "status", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "Queue Status" in result.output
    assert "Pending Jobs" in result.output
    assert "1" in result.output


def test_queue_retry_command(tmp_path):
    """Verify queue retry works with --id, --all, and fails without either.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.

    Returns:
        None: Asserts exit codes and retry behavior.
    """
    cfg_path, repo = _watch_config(tmp_path)
    store_path = tmp_path / "state.db"
    runner.invoke(app, ["config", "set", "db_path", str(store_path), "--config", str(cfg_path)])
    store = Store(store_path)
    store.admit_mention_job(2001, 3001, 4001, repo)
    store.fail_jobs([2001], "permanent_error", is_permanent=True)
    store.close()

    # Missing flag
    err_result = runner.invoke(app, ["queue", "retry", "--config", str(cfg_path)])
    assert err_result.exit_code == 1
    assert "must specify --all or --id" in err_result.output

    # Retry specific ID
    id_result = runner.invoke(app, ["queue", "retry", "--config", str(cfg_path), "--id", "2001"])
    assert id_result.exit_code == 0, id_result.output
    assert "retried 1 job(s)" in id_result.output

    # Fail again and retry with --all
    store = Store(store_path)
    store.fail_jobs([2001], "permanent_error", is_permanent=True)
    store.close()
    all_result = runner.invoke(app, ["queue", "retry", "--config", str(cfg_path), "--all"])
    assert all_result.exit_code == 0, all_result.output
    assert "retried 1 job(s)" in all_result.output


def test_queue_purge_command(tmp_path):
    """Verify queue purge removes completed records and validates inputs.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.

    Returns:
        None: Asserts exit codes and output content.
    """
    cfg_path, repo = _watch_config(tmp_path)
    store_path = tmp_path / "state.db"
    runner.invoke(app, ["config", "set", "db_path", str(store_path), "--config", str(cfg_path)])
    store = Store(store_path)
    store.admit_mention_job(3001, 4001, 5001, repo)
    claimed = store.claim_next_conversation_jobs(lease_duration_seconds=60)
    assert claimed is not None
    store.commit_outbox("b1", 4001, 5001, repo, [3001], ["chunk1"], "s1")
    store.complete_delivery_batch("b1", repo, "s1")
    store._conn.execute("UPDATE mention_jobs SET updated_at = updated_at - 100 WHERE status = 'done'")
    store._conn.execute("UPDATE delivery_batches SET updated_at = updated_at - 100 WHERE status = 'done'")
    store._conn.commit()
    store.close()
    neg_res = runner.invoke(app, ["queue", "purge", "--config", str(cfg_path), "--older-than-days", "-1"])
    assert neg_res.exit_code == 1
    assert "must be non-negative" in neg_res.output

    # Purge with older-than-days 0
    purge_res = runner.invoke(app, ["queue", "purge", "--config", str(cfg_path), "--older-than-days", "0"])
    assert purge_res.exit_code == 0, purge_res.output
    assert "purged 1 completed record(s)" in purge_res.output


def test_doctor_command_detects_store_queue_failures(tmp_path, monkeypatch):
    """Verify doctor command reports failure when store has dead jobs.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.
        monkeypatch (pytest.MonkeyPatch): Pytest monkeypatch fixture.

    Returns:
        None: Asserts exit code and FAIL output.
    """
    cfg_path, repo = _watch_config(tmp_path)
    store_path = tmp_path / "state.db"
    runner.invoke(app, ["config", "set", "db_path", str(store_path), "--config", str(cfg_path)])
    version_mock = SimpleNamespace(binary="/bin/true", version=(1, 18, 25), raw="1.18.25")
    monkeypatch.setattr("oi_agent.daemon.check_version", lambda *args: version_mock)
    monkeypatch.setattr("oi_agent.daemon.authenticated", lambda *args: True)
    monkeypatch.setattr("oi_agent.daemon.list_models", lambda *args: ["openai/gpt-4o-mini"])
    monkeypatch.setenv("OI_DISCORD_TOKEN", "token")
    # Healthy store
    store = Store(store_path)
    store.close()
    ok_result = runner.invoke(app, ["doctor", "--config", str(cfg_path)])
    assert ok_result.exit_code == 0, ok_result.output
    assert "all checks passed" in ok_result.output

    # Add dead job to store
    store = Store(store_path)
    store.admit_mention_job(9999, 111, 222, repo)
    store.fail_jobs([9999], "permanent_error", is_permanent=True)
    store.close()

    fail_result = runner.invoke(app, ["doctor", "--config", str(cfg_path)])
    assert fail_result.exit_code == 1, fail_result.output
    assert "FAIL" in fail_result.output
    assert "dead job" in fail_result.output


def test_memory_cli_commands(tmp_path):
    """Verify oi memory status, show, prune, reset-member, and reset-scope commands."""
    cfg_path, repo = _watch_config(tmp_path)
    store_path = tmp_path / "state.db"
    runner.invoke(app, ["config", "set", "db_path", str(store_path), "--config", str(cfg_path)])

    store = Store(store_path)
    from oi_agent.dynamics import (
        CommunicationProfile,
        MemberProfile,
        TransientMemberState,
    )
    comm = CommunicationProfile(
        directness=0.7,
        detail_preference="balanced",
        challenge_preference="direct",
        humor_preference="medium",
        decision_style="recommendation",
    )
    prof = MemberProfile(
        schema_version=2,
        communication=comm,
        confidence={"directness": 0.8},
        evidence_count={"directness": 3},
        last_observed_at={"directness": 1000},
        recurring_topics=["architecture"],
        created_at=1000,
        updated_at=1000,
        last_interaction_at=1000,
    )
    store.upsert_member_profile("discord", "111", "u1", "user1", prof)
    store.upsert_transient_member_state(
        "discord", "111", "u1",
        TransientMemberState(energy="steady", current_focus=["testing"], observed_at=1000, expires_at=9999999999)
    )
    store.close()

    # 1. memory status
    res_status = runner.invoke(app, ["memory", "status", "--scope", "111", "--config", str(cfg_path)])
    assert res_status.exit_code == 0, res_status.output
    assert "Member Profiles" in res_status.output
    assert "1" in res_status.output

    # 2. memory show
    res_show = runner.invoke(app, ["memory", "show", "--scope", "111", "--member", "u1", "--config", str(cfg_path)])
    assert res_show.exit_code == 0, res_show.output
    assert "architecture" in res_show.output
    assert "steady" in res_show.output

    # 3. memory prune
    res_prune = runner.invoke(app, ["memory", "prune", "--scope", "111", "--config", str(cfg_path)])
    assert res_prune.exit_code == 0, res_prune.output
    assert "pruned" in res_prune.output

    # 4. memory reset-member requires --yes or fails
    res_reset = runner.invoke(
        app, ["memory", "reset-member", "--scope", "111", "--member", "u1", "--yes", "--config", str(cfg_path)]
    )
    assert res_reset.exit_code == 0, res_reset.output
    assert "deleted" in res_reset.output

    # 5. memory reset-scope
    res_reset_scope = runner.invoke(
        app, ["memory", "reset-scope", "--scope", "111", "--yes", "--config", str(cfg_path)]
    )
    assert res_reset_scope.exit_code == 0, res_reset_scope.output
    assert "wiped" in res_reset_scope.output


def test_bootstrap_cli_consent_and_dry_run(tmp_path):
    """Verify bootstrap requires configured scope, consent via --yes, and dry-run produces no model calls."""
    cfg_path, repo = _watch_config(tmp_path)

    # 1. Unconfigured scope fails
    res_unconf = runner.invoke(app, ["bootstrap", "--scope", "999", "--config", str(cfg_path)])
    assert res_unconf.exit_code == 1
    assert "not in configured watches" in res_unconf.output

    # 2. Dry run with configured scope succeeds without calling Discord or writing
    res_dry = runner.invoke(app, ["bootstrap", "--scope", "111", "--dry-run", "--config", str(cfg_path)])
    assert res_dry.exit_code == 0, res_dry.output
    assert "dry-run enabled" in res_dry.output

    # 3. Non-interactive without --yes fails closed
    res_no_yes = runner.invoke(app, ["bootstrap", "--scope", "111", "--config", str(cfg_path)])
    assert res_no_yes.exit_code == 1
    assert "--yes" in res_no_yes.output


# ---------------------------------------------------------------------------
# environment_mode through the CLI (C1)
# ---------------------------------------------------------------------------

def test_config_set_and_show_environment_mode(tmp_path):
    """environment_mode is settable, shown by config show, and strictly validated."""
    cfg_path, _ = _watch_config(tmp_path)
    result = runner.invoke(app, [
        "config", "set", "environment_mode", "server", "--config", str(cfg_path)
    ])
    assert result.exit_code == 0, result.output

    shown = runner.invoke(app, ["config", "show", "--config", str(cfg_path)])
    assert shown.exit_code == 0, shown.output
    assert "environment_mode = server" in shown.output

    bad = runner.invoke(app, [
        "config", "set", "environment_mode", "cluster", "--config", str(cfg_path)
    ])
    assert bad.exit_code == 1
    still = runner.invoke(app, ["config", "show", "--config", str(cfg_path)])
    assert "environment_mode = server" in still.output


# ---------------------------------------------------------------------------
# Bootstrap fakes (generic fixtures only)
# ---------------------------------------------------------------------------

_EPOCH = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


class _FakeMessage:
    """Generic Discord-like message with newest/oldest ordering by minute offset."""

    def __init__(self, mid, content, *, author_id="u1", author_name="user-a",
                 bot=False, mtype=None, minute_offset=0):
        self.id = mid
        self.content = content
        self.author = SimpleNamespace(
            id=author_id, bot=bot, display_name=author_name, name=author_name
        )
        self.type = mtype if mtype is not None else discord.MessageType.default
        self.created_at = _EPOCH + timedelta(minutes=minute_offset)


class _FakeChannel:
    """Serves messages newest-first like discord.py history."""

    def __init__(self, cid, messages, *, archived_threads=None, parent_id=None):
        self.id = cid
        self.parent_id = parent_id
        self._messages = messages
        self._archived_threads = archived_threads or []
        self.history_calls = 0
        self.archived_thread_calls = 0

    def history(self, limit=None):
        self.history_calls += 1

        async def _gen():
            messages = self._messages if limit is None else self._messages[:limit]
            for m in messages:
                yield m

        return _gen()

    def archived_threads(self, *, private=False, joined=False, limit=None):
        self.archived_thread_calls += 1

        async def _gen():
            for thread in self._archived_threads:
                yield thread

        return _gen()


class _FakeGuild:
    def __init__(self, gid, text_channels, active_threads=None):
        self.id = gid
        self.text_channels = text_channels
        self._active_threads = active_threads or []

    async def active_threads(self):
        return self._active_threads

def _make_fake_client(guilds, channels_by_id=None):
    """Return a discord.Client replacement wired to fake guild/channel objects."""

    class _FakeClient:
        def __init__(self, intents=None):
            self.guilds = guilds
            self.started = False
            self.closed = False
            self._on_ready = None

        def event(self, fn):
            self._on_ready = fn
            return fn

        def get_channel(self, channel_id):
            return (channels_by_id or {}).get(channel_id)

        async def start(self, token):
            self.started = True
            await self._on_ready()

        async def close(self):
            self.closed = True

    return _FakeClient


def _make_recording_store(record):
    class _RecordingStore:
        def __init__(self, path):
            record["store_constructed"] += 1
            self.path = path
            self.closed = False
            record["stores"].append(self)

        def close(self):
            self.closed = True

    return _RecordingStore


def _make_fake_engine(record, result=None, error=None):
    class _FakeReflectionEngine:
        """Mirror the REAL ReflectionEngine.analyze_history signature exactly.

        Kept signature-faithful on purpose: a drifting fake once masked a
        production TypeError between cli.py and reflection.engine (the suite
        stayed green while real bootstrap crashed). test_bootstrap_real_engine
        now also exercises the genuine engine as a belt-and-braces check.
        """

        def __init__(self, runner_obj, store_obj, **kwargs):
            record["analysis_timeout"] = kwargs.get("analysis_timeout")
            record["engine_constructed"] += 1

        async def analyze_history(self, platform, scope_id, messages, *,
                                  max_messages=120, max_total_chars=120_000):
            record["analyze_calls"] += 1
            record["platform"] = platform
            record["scope_id"] = scope_id
            record["messages"] = list(messages)
            record["max_messages"] = max_messages
            if error is not None:
                raise error
            return result

    return _FakeReflectionEngine


def _guild_watch_config(tmp_path, *, allowed="333,444", ignored=""):
    """Create a guild watch config through the CLI (scope 222)."""
    cfg_path = tmp_path / "config.toml"
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    args = [
        "config", "watch-add", "--guild-id", "222", "--repo-path", str(repo),
        "--config", str(cfg_path),
    ]
    if allowed:
        args += ["--allowed-channels", allowed]
    if ignored:
        args += ["--ignored-channels", ignored]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return cfg_path, repo


# ---------------------------------------------------------------------------
# Bootstrap contract (C3/C4/C6)
# ---------------------------------------------------------------------------

def test_bootstrap_dry_run_performs_no_reads_or_writes(tmp_path, monkeypatch):
    """Dry run must not construct a client, Store, or engine, nor fetch history."""
    cfg_path, _ = _watch_config(tmp_path)
    record = {"store_constructed": 0, "stores": [], "engine_constructed": 0,
              "analyze_calls": 0, "messages": [], "max_messages": None}

    def _no_client(intents=None):
        raise AssertionError("dry-run must not construct a Discord client")

    class _NoStore:
        def __init__(self, path):
            raise AssertionError("dry-run must not construct a Store")

    class _NoEngine:
        def __init__(self, runner_obj, store_obj):
            raise AssertionError("dry-run must not construct a ReflectionEngine")

    monkeypatch.setattr(discord, "Client", _no_client)
    monkeypatch.setattr(cli, "Store", _NoStore)
    monkeypatch.setattr("oi_agent.reflection.engine.ReflectionEngine", _NoEngine)
    monkeypatch.setenv("OI_DISCORD_TOKEN", "test-token")

    result = runner.invoke(app, [
        "bootstrap", "--scope", "111", "--dry-run", "--config", str(cfg_path)
    ])
    assert result.exit_code == 0, result.output
    assert "dry-run enabled" in result.output
    # Consent screen resolves and shows channel IDs + count before any history read.
    assert "Channels (1): 111" in result.output
    assert record["store_constructed"] == 0
    assert record["engine_constructed"] == 0
    assert record["analyze_calls"] == 0


def test_bootstrap_guild_not_found_reports_failure_and_fails_closed(tmp_path, monkeypatch):
    """A guild the bot cannot see is an explicit failure class, not success."""
    cfg_path, _ = _guild_watch_config(tmp_path)
    record = {"store_constructed": 0, "stores": [], "engine_constructed": 0,
              "analyze_calls": 0, "messages": [], "max_messages": None}
    monkeypatch.setattr(discord, "Client", _make_fake_client([]))
    monkeypatch.setattr(cli, "Store", _make_recording_store(record))
    monkeypatch.setattr(
        "oi_agent.reflection.engine.ReflectionEngine",
        _make_fake_engine(record, result=ReflectionSummary(True, False, False, 1)),
    )
    monkeypatch.setenv("OI_DISCORD_TOKEN", "test-token")

    result = runner.invoke(app, [
        "bootstrap", "--scope", "222", "--yes", "--config", str(cfg_path)
    ])
    assert result.exit_code == 1
    assert "guild not accessible to bot" in result.output
    assert record["analyze_calls"] == 0
    assert record["stores"] and record["stores"][0].closed


def test_bootstrap_yes_collects_chronological_history_with_exclusions(tmp_path, monkeypatch):
    """End-to-end: chronological per-channel batches, exclusion counts, no disallowed reads."""
    cfg_path, _ = _guild_watch_config(tmp_path, allowed="333,444")
    ch_disallowed = _FakeChannel(555, [_FakeMessage(1, "never-read message text")])
    ch_one = _FakeChannel(333, [
        # Newest-first, like discord.py: minute_offset decreases down the list.
        _FakeMessage(11, "newest message text", author_id="u1", minute_offset=50),
        _FakeMessage(12, "join announcement", author_id="bot1", bot=True, minute_offset=40),
        _FakeMessage(13, "pinned a message", author_id="u1",
                     mtype=discord.MessageType.pins_add, minute_offset=30),
        _FakeMessage(14, "   ", author_id="u2", minute_offset=20),
        _FakeMessage(15, "older message text", author_id="u1", minute_offset=10),
        _FakeMessage(16, "oldest message text", author_id="u2", minute_offset=5),
    ])
    ch_two = _FakeChannel(444, [
        _FakeMessage(21, "other channel note text", author_id="u3", minute_offset=15),
    ])
    payload = ReflectionSummary(True, True, True, 3, member_ids=("u1", "u2", "u3"))
    record = {"store_constructed": 0, "stores": [], "engine_constructed": 0,
              "analyze_calls": 0, "messages": [], "max_messages": None}
    monkeypatch.setattr(
        discord, "Client", _make_fake_client([_FakeGuild(222, [ch_disallowed, ch_one, ch_two])])
    )
    monkeypatch.setattr(cli, "Store", _make_recording_store(record))
    monkeypatch.setattr(
        "oi_agent.reflection.engine.ReflectionEngine",
        _make_fake_engine(record, result=payload),
    )
    monkeypatch.setenv("OI_DISCORD_TOKEN", "test-token")

    result = runner.invoke(app, [
        "bootstrap", "--scope", "222", "--yes", "--config", str(cfg_path)
    ])
    assert result.exit_code == 0, result.output

    # Disallowed channel is never read.
    assert ch_disallowed.history_calls == 0
    # Consent screen and post-connect resolution both show IDs + counts.
    assert "Channels (2): 333, 444" in result.output
    assert "Resolved channels to scan (2)" in result.output

    msgs = record["messages"]
    # Channel 333 batch reversed to chronological order, then channel 444's batch.
    assert [m.content for m in msgs] == [
        "oldest message text",
        "older message text",
        "newest message text",
        "other channel note text",
    ]
    assert [m.channel_id for m in msgs] == ["333", "333", "333", "444"]
    # Timestamps present and chronological within each channel block.
    assert all(m.timestamp_iso for m in msgs)
    stamps = [m.timestamp_iso for m in msgs]
    assert stamps[:3] == sorted(stamps[:3])
    assert stamps[3:] == sorted(stamps[3:])
    assert msgs[0].author_id == "u2"

    # Exclusion counts reported per failure class.
    assert "bot-excluded=1" in result.output
    assert "system-excluded=1" in result.output
    assert "empty-excluded=1" in result.output
    assert "permission-failed=0" in result.output
    assert "Initialized 3 member profile(s)" in result.output
    assert record["max_messages"] == 60  # bounded analysis batch, not collection cap
    assert record["stores"] and record["stores"][0].closed


def test_bootstrap_real_engine_wiring(tmp_path, monkeypatch):
    """Bootstrap must drive the REAL ReflectionEngine, not just a lookalike fake.

    Regression guard: the CLI once called analyze_history with a stale
    one-positional-arg signature; the test fake mirrored the bug, so the suite
    stayed green while real runs crashed with TypeError after the history
    fetch. This test stubs only the model transport and lets the genuine
    engine extract, validate, merge, and store.
    """
    cfg_path, _ = _guild_watch_config(tmp_path, allowed="333")
    ch_one = _FakeChannel(333, [
        _FakeMessage(31, "api deadline slipped again", author_id="u1", minute_offset=10),
        _FakeMessage(32, "design doc review note text", author_id="u2", minute_offset=5),
    ])
    payload = json.dumps({
        "members": [{
            "member_id": "u1",
            "handle": "u1",
            "directness": 0.8,
            "detail_preference": "brief",
            "confidence": 0.9,
            "topics": ["api governance"],
            "energy": "steady",
            "current_focus": ["api governance"],
        }],
        "team_pulse": {
            "focus_areas": ["api governance"],
            "friction_categories": ["scope_creep"],
            "momentum": "building",
            "confidence": 0.8,
        },
        "agent_calibration": {
            "preferred_verbosity": "concise",
            "preferred_directness": "direct",
            "formatting_avoidances": [],
            "confidence": 0.9,
        },
    })
    monkeypatch.setattr(discord, "Client", _make_fake_client([_FakeGuild(222, [ch_one])]))
    db_path = tmp_path / "wiring.db"
    monkeypatch.setattr(cli, "Store", lambda _path: Store(db_path))
    monkeypatch.setattr(
        "oi_agent.opencode.runner.OpenCodeRunner.run_raw_prompt",
        AsyncMock(return_value=f"```json\n{payload}\n```"),
    )
    monkeypatch.setenv("OI_DISCORD_TOKEN", "test-token")

    result = runner.invoke(app, [
        "bootstrap", "--scope", "222", "--yes", "--config", str(cfg_path)
    ])
    assert result.exit_code == 0, result.output
    assert "Initialized 1 member profile(s)" in result.output

    store = Store(db_path)
    try:
        profiles = store.list_member_profiles("discord", "222")
        assert len(profiles) == 1
        assert profiles[0]["member_id"] == "u1"
        # Scoped correctly: the same member under another scope is invisible.
        assert store.list_member_profiles("discord", "999") == []
    finally:
        store.close()


def test_bootstrap_reads_all_explicit_channels_without_collection_cap(tmp_path, monkeypatch):
    """Collection reads every explicitly selected channel; only analysis batches are bounded."""
    cfg_path, _ = _guild_watch_config(tmp_path, allowed=None)
    channels = [
        _FakeChannel(300 + i, [
            _FakeMessage(100 * (i + 1) + j, f"message {i}-{j} text",
                         author_id=f"u{j + 1}", minute_offset=10 - j)
            for j in range(4)
        ])
        for i in range(6)  # six channels; only the first five may be scanned
    ]
    payload = ReflectionSummary(True, False, False, 1, member_ids=("u1",))
    record = {"store_constructed": 0, "stores": [], "engine_constructed": 0,
              "analyze_calls": 0, "messages": [], "max_messages": None}
    monkeypatch.setattr(
        discord, "Client", _make_fake_client([_FakeGuild(222, channels)])
    )
    monkeypatch.setattr(cli, "Store", _make_recording_store(record))
    monkeypatch.setattr(
        "oi_agent.reflection.engine.ReflectionEngine",
        _make_fake_engine(record, result=payload),
    )
    monkeypatch.setenv("OI_DISCORD_TOKEN", "test-token")

    result = runner.invoke(app, [
        "bootstrap", "--scope", "222", "--yes", "--limit", "2",
        "--channels", "300,301,302,303,304,305", "--config", str(cfg_path),
    ])
    assert result.exit_code == 0, result.output
    assert len(record["messages"]) == 12  # 2 messages from all 6 channels
    assert record["max_messages"] == 60
    assert "channels=6, threads=0, collected=12" in result.output
    assert channels[5].history_calls == 1


def test_bootstrap_surfaces_engine_bound_error_as_failure(tmp_path, monkeypatch):
    """A ValueError from analyze_history is a bounded failure with non-zero exit."""
    cfg_path, _ = _watch_config(tmp_path)
    ch = _FakeChannel(111, [
        _FakeMessage(1, "first message text", author_id="u1", minute_offset=10),
        _FakeMessage(2, "second message text", author_id="u2", minute_offset=5),
    ])
    record = {"store_constructed": 0, "stores": [], "engine_constructed": 0,
              "analyze_calls": 0, "messages": [], "max_messages": None}
    monkeypatch.setattr(
        discord, "Client", _make_fake_client([], channels_by_id={111: ch})
    )
    monkeypatch.setattr(cli, "Store", _make_recording_store(record))
    monkeypatch.setattr(
        "oi_agent.reflection.engine.ReflectionEngine",
        _make_fake_engine(record, error=ValueError("bounds exceeded")),
    )
    monkeypatch.setenv("OI_DISCORD_TOKEN", "test-token")

    result = runner.invoke(app, [
        "bootstrap", "--scope", "111", "--yes", "--config", str(cfg_path)
    ])
    assert result.exit_code == 1
    assert "history analysis rejected input bounds" in result.output
    assert record["analyze_calls"] == 1
    assert record["stores"] and record["stores"][0].closed


# ---------------------------------------------------------------------------
# Gap 5: exact-known-channel consent for unrestricted guild watches
# ---------------------------------------------------------------------------

def _probe_no_construction(record):
    """Return patches that count, but forbid, client/Store/engine/secrets use."""

    def _no_client(intents=None):
        record["client_constructed"] += 1
        raise AssertionError("client must not be constructed on this path")

    class _NoStore:
        def __init__(self, path):
            record["store_constructed"] += 1
            raise AssertionError("Store must not be constructed on this path")

    class _NoEngine:
        def __init__(self, runner_obj, store_obj):
            record["engine_constructed"] += 1
            raise AssertionError("ReflectionEngine must not be constructed on this path")

    def _no_secrets(*args, **kwargs):
        record["secrets_loaded"] += 1
        raise AssertionError("secrets must not be loaded on this path")

    return _no_client, _NoStore, _NoEngine, _no_secrets


def test_bootstrap_dynamic_non_interactive_fails_closed_without_channels(tmp_path, monkeypatch):
    """Unrestricted guild watch + non-interactive: exit 1 with guidance, zero setup."""
    cfg_path, _ = _guild_watch_config(tmp_path, allowed=None)
    record = {"client_constructed": 0, "store_constructed": 0,
              "engine_constructed": 0, "secrets_loaded": 0}
    no_client, no_store, no_engine, no_secrets = _probe_no_construction(record)
    monkeypatch.setattr(discord, "Client", no_client)
    monkeypatch.setattr(cli, "Store", no_store)
    monkeypatch.setattr("oi_agent.reflection.engine.ReflectionEngine", no_engine)
    monkeypatch.setattr(cli, "load_secrets", no_secrets)
    monkeypatch.delenv("OI_DISCORD_TOKEN", raising=False)

    # With --yes (non-tty): still fail closed before any setup.
    result = runner.invoke(app, [
        "bootstrap", "--scope", "222", "--yes", "--config", str(cfg_path)
    ])
    assert result.exit_code == 1
    assert "cannot run non-interactively" in result.output
    # Actionable guidance names all three alternatives.
    assert "allowed-channels" in result.output
    assert "--channels" in result.output
    assert "run interactively" in result.output
    # Fail-closed before any construction, secrets load, or connection.
    assert record == {"client_constructed": 0, "store_constructed": 0,
                      "engine_constructed": 0, "secrets_loaded": 0}

    # Without --yes (non-tty): same fail-closed path, zero setup.
    result = runner.invoke(app, ["bootstrap", "--scope", "222", "--config", str(cfg_path)])
    assert result.exit_code == 1
    assert record == {"client_constructed": 0, "store_constructed": 0,
                      "engine_constructed": 0, "secrets_loaded": 0}


def test_bootstrap_channels_option_runs_exact_ids_non_interactively(tmp_path, monkeypatch):
    """--channels supplies the exact run set: only those IDs are read, ever."""
    cfg_path, _ = _guild_watch_config(tmp_path, allowed=None, ignored="555")
    ch_ignored = _FakeChannel(555, [_FakeMessage(1, "ignored channel message text")])
    ch_one = _FakeChannel(333, [_FakeMessage(11, "first channel message text", author_id="u1")])
    ch_two = _FakeChannel(444, [_FakeMessage(21, "second channel message text", author_id="u2")])
    ch_unrequested = _FakeChannel(666, [_FakeMessage(31, "never requested message text")])
    payload = ReflectionSummary(True, False, False, 1, member_ids=("u1", "u2"))
    record = {"store_constructed": 0, "stores": [], "engine_constructed": 0,
              "analyze_calls": 0, "messages": [], "max_messages": None}
    monkeypatch.setattr(
        discord, "Client",
        _make_fake_client([_FakeGuild(222, [ch_ignored, ch_one, ch_two, ch_unrequested])]),
    )
    monkeypatch.setattr(cli, "Store", _make_recording_store(record))
    monkeypatch.setattr(
        "oi_agent.reflection.engine.ReflectionEngine",
        _make_fake_engine(record, result=payload),
    )
    monkeypatch.setenv("OI_DISCORD_TOKEN", "test-token")

    result = runner.invoke(app, [
        "bootstrap", "--scope", "222", "--yes", "--channels", "333,444,444,555",
        "--config", str(cfg_path),
    ])
    assert result.exit_code == 0, result.output
    assert "Resolved channels to scan (2): 333, 444" in result.output
    # Ignored-by-watch IDs pass through filtering and are reported as skipped.
    assert "skipped 1 requested channel ID(s)" in result.output
    assert "555" in result.output
    # Ignored and unrequested channels are never read; duplicates read once.
    assert ch_ignored.history_calls == 0
    assert ch_unrequested.history_calls == 0
    assert ch_two.history_calls == 1
    assert [m.channel_id for m in record["messages"]] == ["333", "444"]
    assert record["stores"] and record["stores"][0].closed

def test_bootstrap_reads_active_and_archived_threads(tmp_path, monkeypatch):
    """Guild bootstrap includes active and archived thread history under selected channels."""
    cfg_path, _ = _guild_watch_config(tmp_path, allowed="333")
    archived = _FakeChannel(
        335,
        [_FakeMessage(35, "archived thread message", author_id="u3")],
        parent_id=333,
    )
    active = _FakeChannel(
        334,
        [_FakeMessage(34, "active thread message", author_id="u2")],
        parent_id=333,
    )
    parent = _FakeChannel(
        333,
        [_FakeMessage(33, "parent message", author_id="u1")],
        archived_threads=[archived],
    )
    payload = ReflectionSummary(True, False, False, 1, member_ids=("u1", "u2", "u3"))
    record = {"store_constructed": 0, "stores": [], "engine_constructed": 0,
              "analyze_calls": 0, "messages": [], "max_messages": None}
    monkeypatch.setattr(
        discord, "Client", _make_fake_client([_FakeGuild(222, [parent], [active])])
    )
    monkeypatch.setattr(cli, "Store", _make_recording_store(record))
    monkeypatch.setattr(
        "oi_agent.reflection.engine.ReflectionEngine",
        _make_fake_engine(record, result=payload),
    )
    monkeypatch.setenv("OI_DISCORD_TOKEN", "test-token")

    result = runner.invoke(app, [
        "bootstrap", "--scope", "222", "--yes", "--config", str(cfg_path)
    ])

    assert result.exit_code == 0, result.output
    assert "channels=1, threads=2, collected=3" in result.output
    assert parent.history_calls == 1
    assert active.history_calls == 1
    assert archived.history_calls == 1
    assert {message.channel_id for message in record["messages"]} == {"333", "334", "335"}



def test_bootstrap_channels_with_yes_never_prompts(tmp_path, monkeypatch):
    """--channels + --yes is fully pre-consented: exactly 0 prompts, history read directly."""
    cfg_path, _ = _guild_watch_config(tmp_path, allowed=None)
    ch_one = _FakeChannel(333, [_FakeMessage(11, "first message text", author_id="u1")])
    ch_two = _FakeChannel(444, [_FakeMessage(21, "second message text", author_id="u2")])
    payload = ReflectionSummary(True, False, False, 1, member_ids=("u1", "u2"))
    record = {"store_constructed": 0, "stores": [], "engine_constructed": 0,
              "analyze_calls": 0, "messages": [], "max_messages": None}
    prompts = []

    class _FakePrompt:
        @classmethod
        def ask(cls, label, **kwargs):
            prompts.append((label, kwargs))
            return "n"  # even a declining answer must never be consulted

    monkeypatch.setattr(cli, "Prompt", _FakePrompt)
    monkeypatch.setattr(cli, "_stdin_isatty", lambda: True)  # terminal context: still no prompt
    monkeypatch.setattr(
        discord, "Client", _make_fake_client([_FakeGuild(222, [ch_one, ch_two])])
    )
    monkeypatch.setattr(cli, "Store", _make_recording_store(record))
    monkeypatch.setattr(
        "oi_agent.reflection.engine.ReflectionEngine",
        _make_fake_engine(record, result=payload),
    )
    monkeypatch.setenv("OI_DISCORD_TOKEN", "test-token")

    result = runner.invoke(app, [
        "bootstrap", "--scope", "222", "--yes", "--channels", "333,444",
        "--config", str(cfg_path),
    ])
    assert result.exit_code == 0, result.output
    # The explicit exact-ID path is pre-consented: no stage-1 or stage-2 prompt at all.
    assert prompts == []
    # History was read directly from exactly the requested channels.
    assert ch_one.history_calls == 1
    assert ch_two.history_calls == 1
    assert [m.channel_id for m in record["messages"]] == ["333", "444"]
    assert record["analyze_calls"] == 1
    assert record["stores"] and record["stores"][0].closed


def test_bootstrap_dynamic_interactive_two_stage_consent(tmp_path, monkeypatch):
    """Interactive dynamic set: resolve, display exact IDs, confirm before any read."""
    cfg_path, _ = _guild_watch_config(tmp_path, allowed=None, ignored="999")
    ch_ignored = _FakeChannel(999, [_FakeMessage(1, "ignored channel message text")])
    ch_one = _FakeChannel(333, [_FakeMessage(11, "first message text", author_id="u1")])
    ch_two = _FakeChannel(444, [_FakeMessage(21, "second message text", author_id="u2")])
    payload = ReflectionSummary(True, False, False, 1, member_ids=("u1", "u2"))
    record = {"store_constructed": 0, "stores": [], "engine_constructed": 0,
              "analyze_calls": 0, "messages": [], "max_messages": None,
              "history_calls_at_prompt": None}
    prompts = []

    class _FakePrompt:
        @classmethod
        def ask(cls, label, **kwargs):
            prompts.append((label, kwargs))
            record["history_calls_at_prompt"] = (
                ch_ignored.history_calls + ch_one.history_calls + ch_two.history_calls
            )
            return "y"

    monkeypatch.setattr(cli, "Prompt", _FakePrompt)
    monkeypatch.setattr(cli, "_stdin_isatty", lambda: True)
    monkeypatch.setattr(
        discord, "Client", _make_fake_client([_FakeGuild(222, [ch_ignored, ch_one, ch_two])])
    )
    monkeypatch.setattr(cli, "Store", _make_recording_store(record))
    monkeypatch.setattr(
        "oi_agent.reflection.engine.ReflectionEngine",
        _make_fake_engine(record, result=payload),
    )
    monkeypatch.setenv("OI_DISCORD_TOKEN", "test-token")

    result = runner.invoke(app, ["bootstrap", "--scope", "222", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    # Exact resolved IDs displayed post-connect.
    assert "Resolved channels to scan (2): 333, 444" in result.output
    assert "Exact scope for consent" in result.output
    # Two stages: pre-connect consent, then post-resolve exact-channel confirmation.
    assert [label for label, _ in prompts] == [
        "Proceed with reading channel/thread history and synthesizing team memory?",
        "Read all history from these 2 channels/threads now?",
    ]
    assert prompts[1][1].get("default") == "n"  # second stage defaults to No
    # The exact-channel confirmation happened before any history fetch.
    assert record["history_calls_at_prompt"] == 0
    assert ch_ignored.history_calls == 0
    assert record["analyze_calls"] == 1


def test_bootstrap_dynamic_interactive_decline_aborts_cleanly(tmp_path, monkeypatch):
    """Declining the second-stage prompt reads nothing, writes nothing, exits 0."""
    cfg_path, _ = _guild_watch_config(tmp_path, allowed=None)
    ch_one = _FakeChannel(333, [_FakeMessage(11, "first message text", author_id="u1")])
    ch_two = _FakeChannel(444, [_FakeMessage(21, "second message text", author_id="u2")])
    record = {"store_constructed": 0, "stores": [], "engine_constructed": 0,
              "analyze_calls": 0, "messages": [], "max_messages": None}
    answers = iter(["y", "n"])

    class _FakePrompt:
        @classmethod
        def ask(cls, label, **kwargs):
            return next(answers)

    monkeypatch.setattr(cli, "Prompt", _FakePrompt)
    monkeypatch.setattr(cli, "_stdin_isatty", lambda: True)
    monkeypatch.setattr(
        discord, "Client", _make_fake_client([_FakeGuild(222, [ch_one, ch_two])])
    )
    monkeypatch.setattr(cli, "Store", _make_recording_store(record))
    monkeypatch.setattr(
        "oi_agent.reflection.engine.ReflectionEngine",
        _make_fake_engine(record, result=ReflectionSummary(True, False, False, 1)),
    )
    monkeypatch.setenv("OI_DISCORD_TOKEN", "test-token")

    result = runner.invoke(app, ["bootstrap", "--scope", "222", "--config", str(cfg_path)])
    assert result.exit_code == 0, result.output
    assert "aborted" in result.output
    assert "no history read" in result.output
    assert ch_one.history_calls == 0
    assert ch_two.history_calls == 0
    assert record["analyze_calls"] == 0
    assert record["messages"] == []
    assert record["stores"] and record["stores"][0].closed


def test_bootstrap_dynamic_yes_fails_closed_immediately(tmp_path, monkeypatch):
    """--yes cannot consent to an unresolved set: immediate exit 1, zero setup, zero prompts."""
    cfg_path, _ = _guild_watch_config(tmp_path, allowed=None)
    ch_one = _FakeChannel(333, [_FakeMessage(11, "first message text", author_id="u1")])
    ch_two = _FakeChannel(444, [_FakeMessage(21, "second message text", author_id="u2")])
    record = {"client_constructed": 0, "store_constructed": 0,
              "engine_constructed": 0, "secrets_loaded": 0}
    no_client, no_store, no_engine, no_secrets = _probe_no_construction(record)
    prompts = []

    class _FakePrompt:
        @classmethod
        def ask(cls, label, **kwargs):
            prompts.append((label, kwargs))
            return "n"

    monkeypatch.setattr(cli, "Prompt", _FakePrompt)
    monkeypatch.setattr(cli, "_stdin_isatty", lambda: True)  # even on a tty, --yes must fail
    monkeypatch.setattr(discord, "Client", no_client)
    monkeypatch.setattr(cli, "Store", no_store)
    monkeypatch.setattr("oi_agent.reflection.engine.ReflectionEngine", no_engine)
    monkeypatch.setattr(cli, "load_secrets", no_secrets)
    monkeypatch.setenv("OI_DISCORD_TOKEN", "test-token")

    result = runner.invoke(app, ["bootstrap", "--scope", "222", "--yes", "--config", str(cfg_path)])
    assert result.exit_code == 1, result.output
    # Actionable guidance names all three alternatives.
    assert "--yes" in result.output
    assert "allowed-channels" in result.output
    assert "--channels" in result.output
    assert "interactively" in result.output
    assert "WITHOUT --yes" in result.output
    # Fail-closed before any connection, secrets load, or construction.
    assert record == {"client_constructed": 0, "store_constructed": 0,
                      "engine_constructed": 0, "secrets_loaded": 0}
    # No Prompt.ask is reachable with --yes, regardless of tty state.
    assert prompts == []
    assert ch_one.history_calls == 0
    assert ch_two.history_calls == 0


def test_bootstrap_dry_run_unrestricted_guild_states_dynamic_set(tmp_path, monkeypatch):
    """Dry-run on an unrestricted guild watch names the dynamic set and connects to nothing."""
    cfg_path, _ = _guild_watch_config(tmp_path, allowed=None)
    record = {"client_constructed": 0, "store_constructed": 0,
              "engine_constructed": 0, "secrets_loaded": 0}
    no_client, no_store, no_engine, no_secrets = _probe_no_construction(record)
    monkeypatch.setattr(discord, "Client", no_client)
    monkeypatch.setattr(cli, "Store", no_store)
    monkeypatch.setattr("oi_agent.reflection.engine.ReflectionEngine", no_engine)
    monkeypatch.setattr(cli, "load_secrets", no_secrets)

    result = runner.invoke(app, [
        "bootstrap", "--scope", "222", "--dry-run", "--config", str(cfg_path)
    ])
    assert result.exit_code == 0, result.output
    assert "dynamic" in result.output
    assert "Channels (" not in result.output
    assert record == {"client_constructed": 0, "store_constructed": 0,
                      "engine_constructed": 0, "secrets_loaded": 0}


def test_bootstrap_channels_option_validation_without_cap(tmp_path, monkeypatch):
    """--channels validates numeric input and deduplicates without a five-channel cap."""
    cfg_path, _ = _guild_watch_config(tmp_path, allowed=None)

    class _NoStore:
        def __init__(self, path):
            raise AssertionError("no Store may be constructed for invalid --channels")

    monkeypatch.setattr(cli, "Store", _NoStore)

    # --help documents the option and the exact-consent behavior.
    help_res = runner.invoke(app, ["bootstrap", "--help"], env={"COLUMNS": "200"})
    assert help_res.exit_code == 0, help_res.output
    assert "all available history" in help_res.output

    # Non-numeric ID: clean validation error, no Store construction.
    bad = runner.invoke(app, [
        "bootstrap", "--scope", "222", "--yes", "--channels", "333,abc",
        "--config", str(cfg_path),
    ])
    assert bad.exit_code == 1
    assert "comma-separated numeric channel IDs" in bad.output

    # Empty list: rejected.
    empty = runner.invoke(app, [
        "bootstrap", "--scope", "222", "--yes", "--channels", ",",
        "--config", str(cfg_path),
    ])
    assert empty.exit_code == 1
    assert "at least one numeric channel ID" in empty.output

    # Rejected where the exact channel set is already known (channel watch).
    known_cfg, _ = _watch_config(tmp_path / "known")
    wrong = runner.invoke(app, [
        "bootstrap", "--scope", "111", "--yes", "--channels", "333",
        "--config", str(known_cfg),
    ])
    assert wrong.exit_code == 1
    assert "unrestricted guild watches" in wrong.output

    # More than 5 IDs are all accepted and read.
    channels = [
        _FakeChannel(cid, [_FakeMessage(cid, f"message {cid} text", author_id=f"u{cid}")])
        for cid in (333, 444, 555, 666, 777, 888)
    ]
    by_id = {c.id: c for c in channels}
    payload = ReflectionSummary(True, False, False, 1, member_ids=("u333",))
    record = {"store_constructed": 0, "stores": [], "engine_constructed": 0,
              "analyze_calls": 0, "messages": [], "max_messages": None}
    monkeypatch.setattr(
        discord, "Client", _make_fake_client([_FakeGuild(222, channels)])
    )
    monkeypatch.setattr(cli, "Store", _make_recording_store(record))
    monkeypatch.setattr(
        "oi_agent.reflection.engine.ReflectionEngine",
        _make_fake_engine(record, result=payload),
    )
    monkeypatch.setenv("OI_DISCORD_TOKEN", "test-token")

    cap_res = runner.invoke(app, [
        "bootstrap", "--scope", "222", "--yes",
        "--channels", "333,444,555,666,777,888", "--config", str(cfg_path),
    ])
    assert cap_res.exit_code == 0, cap_res.output
    assert "Resolved channels to scan (6)" in cap_res.output
    assert "channels=6, threads=0, collected=6" in cap_res.output
    assert all(by_id[cid].history_calls == 1 for cid in (333, 444, 555, 666, 777, 888))


# ---------------------------------------------------------------------------
# Memory controls (C2, reset confirmation, scope isolation, PIN-B)
# ---------------------------------------------------------------------------

def test_memory_prune_rejects_retention_days_below_one(tmp_path, monkeypatch):
    """--retention-days < 1 fails closed before any Store is constructed."""
    cfg_path, _ = _watch_config(tmp_path)

    class _NoStore:
        def __init__(self, path):
            raise AssertionError("Store must not be constructed for invalid retention")

    monkeypatch.setattr(cli, "Store", _NoStore)
    for bad in ("0", "-3"):
        result = runner.invoke(app, [
            "memory", "prune", "--retention-days", bad, "--config", str(cfg_path)
        ])
        assert result.exit_code == 1
        assert "at least 1" in result.output

    record = {"prune_calls": []}

    class _PruneStore:
        def __init__(self, path):
            pass

        def prune_memory(self, scope_id=None, stable_retention_seconds=90 * 86400):
            record["prune_calls"].append((scope_id, stable_retention_seconds))
            return {"member_profiles": 1}

        def close(self):
            pass

    monkeypatch.setattr(cli, "Store", _PruneStore)
    ok = runner.invoke(app, [
        "memory", "prune", "--retention-days", "7", "--config", str(cfg_path)
    ])
    assert ok.exit_code == 0, ok.output
    assert record["prune_calls"] == [(None, 7 * 86400)]


def test_memory_reset_commands_fail_closed_without_yes(tmp_path, monkeypatch):
    """Non-interactive reset-member/reset-scope without --yes never touch the Store."""

    class _NoResetStore:
        def __init__(self, path):
            pass

        def reset_member_memory(self, *args, **kwargs):
            raise AssertionError("reset-member must not run without confirmation")

        def reset_scope_memory(self, *args, **kwargs):
            raise AssertionError("reset-scope must not run without confirmation")

        def close(self):
            pass

    cfg_path, _ = _watch_config(tmp_path)
    monkeypatch.setattr(cli, "Store", _NoResetStore)

    res_member = runner.invoke(app, [
        "memory", "reset-member", "--scope", "111", "--member", "u1",
        "--config", str(cfg_path),
    ])
    assert res_member.exit_code == 1
    assert "--yes" in res_member.output

    res_scope = runner.invoke(app, [
        "memory", "reset-scope", "--scope", "111", "--config", str(cfg_path)
    ])
    assert res_scope.exit_code == 1
    assert "--yes" in res_scope.output


def test_memory_show_and_status_do_not_cross_scopes(tmp_path):
    """Scope A's operator view never exposes Scope B's memory for the same member."""
    cfg_path, _ = _watch_config(tmp_path)
    store_path = tmp_path / "state.db"
    runner.invoke(app, [
        "config", "set", "db_path", str(store_path), "--config", str(cfg_path)
    ])

    from oi_agent.dynamics import CommunicationProfile, MemberProfile

    def _profile(topic):
        return MemberProfile(
            schema_version=2,
            communication=CommunicationProfile(
                directness=0.7,
                detail_preference="balanced",
                challenge_preference="direct",
                humor_preference="medium",
                decision_style="recommendation",
            ),
            confidence={"directness": 0.8},
            evidence_count={"directness": 3},
            last_observed_at={"directness": 1000},
            recurring_topics=[topic],
            created_at=1000,
            updated_at=1000,
            last_interaction_at=1000,
        )

    store = Store(store_path)
    store.upsert_member_profile("discord", "111", "u1", "user-a", _profile("topic-alpha"))
    store.upsert_member_profile("discord", "222", "u1", "user-b", _profile("topic-beta"))
    store.close()

    res_a = runner.invoke(app, [
        "memory", "show", "--scope", "111", "--member", "u1", "--config", str(cfg_path)
    ])
    assert res_a.exit_code == 0, res_a.output
    assert "topic-alpha" in res_a.output
    assert "topic-beta" not in res_a.output

    res_b = runner.invoke(app, [
        "memory", "show", "--scope", "222", "--member", "u1", "--config", str(cfg_path)
    ])
    assert res_b.exit_code == 0, res_b.output
    assert "topic-beta" in res_b.output
    assert "topic-alpha" not in res_b.output

    res_none = runner.invoke(app, [
        "memory", "show", "--scope", "333", "--member", "u1", "--config", str(cfg_path)
    ])
    assert res_none.exit_code == 0
    assert "no profile found" in res_none.output

    st_a = runner.invoke(app, ["memory", "status", "--scope", "111", "--config", str(cfg_path)])
    assert st_a.exit_code == 0, st_a.output
    assert "Member Profiles" in st_a.output


def test_migrate_legacy_corrupt_rows_fail_closed_with_guidance(tmp_path, monkeypatch):
    """PIN-B: corrupt legacy rows abort the whole migration with purge guidance."""
    from oi_agent.store import LegacyMigrationError

    cfg_path, _ = _watch_config(tmp_path)

    class _FailingLegacyStore:
        def __init__(self, path):
            pass

        def preview_legacy_migration(self):
            return [
                {"handle": "user-a", "member_id": "u1"},
                {"handle": "user-b", "member_id": "u2"},
            ]

        def migrate_legacy_profiles(self, scope):
            raise LegacyMigrationError(migrated=0, skipped=2)

        def close(self):
            pass

    monkeypatch.setattr(cli, "Store", _FailingLegacyStore)
    result = runner.invoke(app, [
        "memory", "migrate-legacy", "--scope", "111", "--yes", "--config", str(cfg_path)
    ])
    assert result.exit_code == 1
    assert "skipped 2 corrupt record(s)" in result.output
    assert "nothing migrated" in result.output
    assert "purge-legacy" in result.output


def test_migrate_legacy_success_mentions_table_drop(tmp_path, monkeypatch):
    """A successful migration reports counts and that legacy tables were dropped."""
    cfg_path, _ = _watch_config(tmp_path)

    class _OkLegacyStore:
        def __init__(self, path):
            pass

        def preview_legacy_migration(self):
            return [{"handle": "user-a", "member_id": "u1"}]

        def migrate_legacy_profiles(self, scope):
            return 3

        def close(self):
            pass

    monkeypatch.setattr(cli, "Store", _OkLegacyStore)
    result = runner.invoke(app, [
        "memory", "migrate-legacy", "--scope", "111", "--yes", "--config", str(cfg_path)
    ])
    assert result.exit_code == 0, result.output
    assert "migrated 3 legacy profile(s)" in result.output
    assert "legacy tables were dropped" in result.output
