"""Unit and integration tests for durable at-least-once Discord delivery.

Covers:
- Immediate SQLite admission before processing
- Downtime message recovery during on_ready
- Active and archived thread recovery
- Burst quiet-period coalescing without dropping source message IDs
- Resuming pending outbox batches without re-running OpenCode
- Stale lease expiry recovery after simulated worker crash
"""

from __future__ import annotations

import asyncio
import json
import re
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import discord
import pytest

from oi_agent.agent.reply import Reply
from oi_agent.config import Config, WatchTarget
from oi_agent.opencode.paths import resolve_opencode_paths
from oi_agent.opencode.prompt import FINAL_OUTPUT_MARKER
from oi_agent.opencode.provenance import worktree_fingerprint
from oi_agent.opencode.review_target import provider_ref_template
from oi_agent.poster import DeliveryResult
from oi_agent.store import Store
from oi_agent.watch import discord_client
from oi_agent.watch.discord_client import OIWatcher


class FakeAuthor:
    """Minimal Discord author object."""

    def __init__(self, user_id: int, name: str = "alice", bot: bool = False) -> None:
        """Initialize author attributes.

        Args:
            user_id (int): Snowflake user ID.
            name (str): Display name.
            bot (bool): Whether author is a bot.
        """
        self.id = user_id
        self.display_name = name
        self.bot = bot


class FakeResponse:
    """Mock response for discord exception initialization."""

    def __init__(self, status: int = 404, reason: str = "Not Found") -> None:
        self.status = status
        self.reason = reason


class FakeMention:
    """Minimal Discord mention object."""

    def __init__(self, user_id: int, display_name: str = "ultron") -> None:
        self.id = user_id
        self.display_name = display_name

class FakeChannel:
    """Fake Discord channel or thread with message history."""

    def __init__(
        self,
        channel_id: int,
        history: list[FakeMessage] | None = None,
        parent_id: int | None = None,
        archived_threads: list[FakeChannel] | None = None,
        guild_id: int | None = None,
    ) -> None:
        """Initialize channel attributes.

        Args:
            channel_id (int): Channel snowflake ID.
            history (list[FakeMessage] | None): Initial message history.
            parent_id (int | None): Parent channel ID if thread.
            archived_threads (list[FakeChannel] | None): Mock archived threads.
            guild_id (int | None): Owning guild ID for guild-wide watch resolution.
        """
        self.id = channel_id
        self.parent_id = parent_id
        self.guild_id = guild_id
        self._history = list(history or [])
        self._archived_threads = list(archived_threads or [])

    async def history(
        self, limit: int = 50, after: discord.Object | None = None, oldest_first: bool = False
    ):
        """Yield fake messages newest-first or oldest-first with after filtering.

        Args:
            limit (int): Max messages to yield.
            after (discord.Object | None): Filter messages strictly after ID.
            oldest_first (bool): Sort direction.

        Yields:
            FakeMessage: Each message in sequence.
        """
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
        """Fetch message by snowflake ID.

        Args:
            message_id (int): Message snowflake ID.

        Returns:
            FakeMessage: Found message.
        """
        for m in self._history:
            if m.id == message_id:
                return m
        raise discord.NotFound(FakeResponse(404), "message not found")

    async def archived_threads(self, limit: int = 50) -> list[FakeChannel]:
        """Return fake archived threads.

        Args:
            limit (int): Limit count.

        Returns:
            list[FakeChannel]: Archived threads list.
        """
        return self._archived_threads[:limit]


class FakeGuild:
    """Fake Discord guild with threads."""

    def __init__(self, threads: list[FakeChannel] | None = None) -> None:
        """Initialize guild threads.

        Args:
            threads (list[FakeChannel] | None): Active threads in guild.
        """
        self.threads = list(threads or [])

    def active_threads(self) -> list[FakeChannel]:
        """Return active threads.

        Returns:
            list[FakeChannel]: Active threads list.
        """
        return self.threads


class FakeMessage:
    """Minimal incoming Discord message."""

    def __init__(
        self,
        message_id: int,
        channel: FakeChannel,
        author: FakeAuthor,
        content: str,
        mentions: list[FakeMention] | None = None,
    ) -> None:
        """Initialize message attributes.

        Args:
            message_id (int): Message snowflake ID.
            channel (FakeChannel): Origin channel.
            author (FakeAuthor): Message sender.
            content (str): Text content.
            mentions (list[FakeMention] | None): Mentioned users.
        """
        self.id = message_id
        self.channel = channel
        self.author = author
        self.content = content
        self.mentions = mentions or []
        self.created_at = datetime.now(UTC)
        self.attachments = []
        self.embeds = []


def _make_watcher(tmp_path: Path, channel_id: int = 111, bot_id: int = 99) -> OIWatcher:
    """Construct an isolated OIWatcher for tests.

    Args:
        tmp_path (Path): Temporary test directory.
        channel_id (int): Watch channel snowflake ID.
        bot_id (int): Bot user snowflake ID.

    Returns:
        OIWatcher: Configured watcher instance.
    """
    cfg = Config()
    cfg.watches = [WatchTarget(channel_id, str(tmp_path / "repo"), bot_id, 10)]
    store = Store(tmp_path / "test_state.db")
    return OIWatcher(cfg, store)


@pytest.mark.asyncio
async def test_admission_writes_to_sqlite_before_processing(tmp_path: Path, monkeypatch) -> None:
    """Verify that on_message admits the job to SQLite immediately before processing."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0)
    watcher = _make_watcher(tmp_path)
    channel = FakeChannel(111)
    message = FakeMessage(1001, channel, FakeAuthor(7), "<@99> run audit", [FakeMention(99)])
    channel._history = [message]

    # Block OpenCode runner so we can inspect DB state right after admission
    runner_entered = asyncio.Event()
    runner_release = asyncio.Event()

    async def fake_run(*args, **kwargs):
        runner_entered.set()
        await runner_release.wait()
        return Reply("audit result", "sha123", session_id="sess-1")

    async def fake_deliver(self, *args, **kwargs):
        return DeliveryResult(ok=True, discord_message_id=555)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)

    await watcher.on_message(message)

    # Verify that the job is already written in SQLite before worker finishes
    row = watcher._store._conn.execute(
        "SELECT source_message_id, conversation_id, owner_channel_id, status "
        "FROM mention_jobs WHERE source_message_id = 1001"
    ).fetchone()
    assert row is not None
    assert row[0] == 1001
    assert row[1] == 111
    assert row[2] == 111
    assert row[3] in ("pending", "leased")
    assert watcher._store.get_scope_cursor("111") == 1001

    runner_release.set()
    await asyncio.sleep(0.05)
    await watcher.close()

    # After completion, job should be marked 'done'
    final_status = watcher._store._conn.execute(
        "SELECT status FROM mention_jobs WHERE source_message_id = 1001"
    ).fetchone()[0]
    assert final_status == "done"


@pytest.mark.asyncio
async def test_downtime_message_recovery_on_on_ready(tmp_path: Path, monkeypatch) -> None:
    """Verify that on_ready catches up on unread messages after cursor."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0)
    watcher = _make_watcher(tmp_path)
    channel = FakeChannel(111)

    # Set prior cursor to 100
    watcher._store.update_scope_cursor("111", 111, 100)

    # Messages 101, 102, 103 occurred while watcher was offline
    m101 = FakeMessage(101, channel, FakeAuthor(7), "regular chat 1")
    m102 = FakeMessage(102, channel, FakeAuthor(7), "<@99> missed request", [FakeMention(99)])
    m103 = FakeMessage(103, channel, FakeAuthor(7), "regular chat 2")
    channel._history = [m101, m102, m103]

    deliveries = []

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None, bot_name="", bot_id=None):
        return Reply("recovered reply", "sha123", session_id="sess-recovered")

    async def fake_deliver(self, channel_id, target_channel_id, chunk, nonce, reply_to_message_id=None, client=None):
        deliveries.append((channel_id, chunk, reply_to_message_id))
        return DeliveryResult(ok=True, discord_message_id=900)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    watcher._channel_cache[111] = channel

    await watcher.on_ready()
    await asyncio.sleep(0.05)
    await watcher.close()

    assert watcher._store.get_scope_cursor("111") == 103
    assert len(deliveries) == 1
    assert deliveries[0][0] == 111
    assert deliveries[0][1] == "recovered reply"
    assert deliveries[0][2] == 102
    assert watcher._store.get_opencode_session(111, tmp_path / "repo") == "sess-recovered"


@pytest.mark.asyncio
async def test_active_and_archived_thread_recovery(tmp_path: Path, monkeypatch) -> None:
    """Verify on_ready discovers active guild threads and archived channel threads."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0)
    watcher = _make_watcher(tmp_path, channel_id=111)

    active_thread = FakeChannel(201, parent_id=111)
    m_active = FakeMessage(2001, active_thread, FakeAuthor(7), "<@99> active thread question", [FakeMention(99)])
    active_thread._history = [m_active]

    archived_thread = FakeChannel(301, parent_id=111)
    m_archived = FakeMessage(3001, archived_thread, FakeAuthor(7), "<@99> archived thread question", [FakeMention(99)])
    archived_thread._history = [m_archived]

    parent_channel = FakeChannel(111, archived_threads=[archived_thread])
    watcher._channel_cache[111] = parent_channel

    guild = FakeGuild(threads=[active_thread])
    watcher._test_guilds = [guild]
    runs = []

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None, bot_name="", bot_id=None):
        runs.append((question, session_id))
        return Reply(f"ans for {question}", "sha", session_id="thread-sess")

    async def fake_deliver(self, channel_id, target_channel_id, chunk, nonce, reply_to_message_id=None, client=None):
        return DeliveryResult(ok=True, discord_message_id=777)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)

    await watcher.on_ready()
    await asyncio.sleep(0.05)
    await watcher.close()

    assert watcher._store.get_scope_cursor("201") == 2001
    assert watcher._store.get_scope_cursor("301") == 3001
    assert len(runs) == 2


@pytest.mark.asyncio
async def test_burst_coalescing_preserves_all_source_message_ids(tmp_path: Path, monkeypatch) -> None:
    """Verify that multiple messages in a burst are linked to the same batch without dropping IDs."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0.05)
    watcher = _make_watcher(tmp_path)
    channel = FakeChannel(111)

    m1 = FakeMessage(501, channel, FakeAuthor(7), "<@99> part 1", [FakeMention(99)])
    m2 = FakeMessage(502, channel, FakeAuthor(7), "<@99> part 2 final", [FakeMention(99)])
    channel._history = [m1, m2]

    questions_seen = []

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None, bot_name="", bot_id=None):
        questions_seen.append(question)
        return Reply("combined answer", "sha123", session_id="burst-session")

    async def fake_deliver(self, channel_id, target_channel_id, chunk, nonce, reply_to_message_id=None, client=None):
        return DeliveryResult(ok=True, discord_message_id=888)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)

    # Send both messages rapidly
    await watcher.on_message(m1)
    await asyncio.sleep(0.01)
    await watcher.on_message(m2)

    # Wait for burst quiet period to elapse and worker to process
    await asyncio.sleep(0.12)
    await watcher.close()

    assert questions_seen == ["@ultron (user id 99) part 2 final"]

    # Verify BOTH message 501 and 502 are marked done with the same batch_id
    jobs = watcher._store._conn.execute(
        "SELECT source_message_id, batch_id, status FROM mention_jobs WHERE source_message_id IN (501, 502)"
    ).fetchall()
    assert len(jobs) == 2
    assert jobs[0][2] == "done"
    assert jobs[1][2] == "done"
    assert jobs[0][1] is not None
    assert jobs[0][1] == jobs[1][1]


@pytest.mark.asyncio
async def test_resume_pending_outbox_without_rerunning_opencode(tmp_path: Path, monkeypatch) -> None:
    """Verify that an interrupted delivery batch is resumed and sent without running model."""
    watcher = _make_watcher(tmp_path)

    # Pre-commit an outbox batch with 2 chunks
    batch_id = "test-resume-batch-001"
    repo_path = tmp_path / "repo"
    watcher._store.admit_mention_job(701, 111, 111, repo_path)
    watcher._store.commit_outbox(
        batch_id=batch_id,
        conversation_id=111,
        owner_channel_id=111,
        repo_path=repo_path,
        source_message_ids=[701],
        chunks=["chunk one", "chunk two"],
        candidate_session_id="resumed-session-id",
    )

    delivered_chunks = []

    async def fake_run(*args, **kwargs):
        pytest.fail("OpenCode runner should NEVER be invoked for existing outbox batches")

    async def fake_deliver(self, channel_id, target_channel_id, chunk, nonce, reply_to_message_id=None, client=None):
        delivered_chunks.append((chunk, nonce))
        return DeliveryResult(ok=True, discord_message_id=1000 + len(delivered_chunks))

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)

    # Start worker loop directly
    _task = asyncio.create_task(watcher._durable_worker_loop())
    await asyncio.sleep(0.05)
    await watcher.close()

    assert len(delivered_chunks) == 2
    assert delivered_chunks[0][0] == "chunk one"
    assert delivered_chunks[1][0] == "chunk two"

    # Verify batch completed and chunk bodies scrubbed
    batch_status = watcher._store._conn.execute(
        "SELECT status FROM delivery_batches WHERE batch_id = ?", (batch_id,)
    ).fetchone()[0]
    assert batch_status == "done"

    bodies = watcher._store._conn.execute(
        "SELECT body FROM outbound_chunks WHERE batch_id = ?", (batch_id,)
    ).fetchall()
    assert all(b[0] is None for b in bodies)

    assert watcher._store.get_opencode_session(111, repo_path) == "resumed-session-id"


@pytest.mark.asyncio
async def test_lease_expiry_recovery_after_simulated_crash(tmp_path: Path, monkeypatch) -> None:
    """Verify that a job with an expired lease from a crashed worker is claimed and completed."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0)
    watcher = _make_watcher(tmp_path)
    channel = FakeChannel(111)
    msg = FakeMessage(801, channel, FakeAuthor(7), "<@99> recover me", [FakeMention(99)])
    channel._history = [msg]
    watcher._channel_cache[111] = channel

    now = int(time.time())
    repo_path = tmp_path / "repo"

    # Insert a leased job with expired lease (simulating worker killed 2 minutes ago)
    watcher._store._conn.execute(
        "INSERT INTO mention_jobs ("
        "source_message_id, conversation_id, owner_channel_id, repo_path, "
        "status, batch_id, attempt_count, available_at, lease_until, created_at, updated_at"
        ") VALUES (?, 111, 111, ?, 'leased', NULL, 1, ?, ?, ?, ?)",
        (801, str(repo_path.resolve()), now - 120, now - 60, now - 120, now - 120),
    )

    runs = []

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None, bot_name="", bot_id=None):
        runs.append(question)
        return Reply("recovered from crash", "sha999", session_id="crash-sess")

    async def fake_deliver(self, *args, **kwargs):
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)

    _task = asyncio.create_task(watcher._durable_worker_loop())
    await asyncio.sleep(0.05)
    await watcher.close()

    assert runs == ["@ultron (user id 99) recover me"]
    job_status = watcher._store._conn.execute(
        "SELECT status FROM mention_jobs WHERE source_message_id = 801"
    ).fetchone()[0]
    assert job_status == "done"

@pytest.mark.asyncio
async def test_partial_outbox_chunk_resume_deduplication(tmp_path: Path, monkeypatch) -> None:
    """Verify that a batch with partially delivered chunks only sends the remaining chunks."""
    watcher = _make_watcher(tmp_path)
    repo_path = tmp_path / "repo"
    batch_id = "partial-batch-456"

    watcher._store.admit_mention_job(901, 111, 111, repo_path)
    watcher._store.commit_outbox(
        batch_id=batch_id,
        conversation_id=111,
        owner_channel_id=111,
        repo_path=repo_path,
        source_message_ids=[901],
        chunks=["part 1 sent", "part 2 pending"],
        candidate_session_id="partial-sess",
    )
    # Mark chunk 0 as already delivered
    watcher._store.record_chunk_delivery(batch_id, 0, discord_message_id=7777)

    delivered_chunks = []

    async def fake_deliver(self, channel_id, target_channel_id, chunk, nonce, reply_to_message_id=None, client=None):
        delivered_chunks.append((chunk, nonce))
        return DeliveryResult(ok=True, discord_message_id=8888)

    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)

    _task = asyncio.create_task(watcher._durable_worker_loop())
    await asyncio.sleep(0.05)
    await watcher.close()

    # Only chunk 1 should have been delivered over network
    assert len(delivered_chunks) == 1
    assert delivered_chunks[0][0] == "part 2 pending"

    # Batch is now done
    batch_status = watcher._store._conn.execute(
        "SELECT status FROM delivery_batches WHERE batch_id = ?", (batch_id,)
    ).fetchone()[0]
    assert batch_status == "done"


@pytest.mark.asyncio
async def test_permanent_error_dead_letters_job(tmp_path: Path, monkeypatch) -> None:
    """Verify that permanent errors (e.g. trigger message deleted 404) dead-letter the job."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0)
    watcher = _make_watcher(tmp_path)
    channel = FakeChannel(111)
    # History is empty, so fetch_message will raise discord.NotFound
    channel._history = []
    watcher._channel_cache[111] = channel

    repo_path = tmp_path / "repo"
    watcher._store.admit_mention_job(999, 111, 111, repo_path)

    _task = asyncio.create_task(watcher._durable_worker_loop())
    await asyncio.sleep(0.05)
    await watcher.close()

    job = watcher._store._conn.execute(
        "SELECT status, last_error_class FROM mention_jobs WHERE source_message_id = 999"
    ).fetchone()
    assert job is not None
    assert job[0] == "dead"
    assert job[1] == "message_not_found"


@pytest.mark.asyncio
async def test_pause_defers_jobs_without_dropping(tmp_path: Path, monkeypatch) -> None:
    """Verify that jobs arriving while paused are deferred into future available_at."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0)
    watcher = _make_watcher(tmp_path)
    channel = FakeChannel(111)
    msg = FakeMessage(1101, channel, FakeAuthor(7), "<@99> paused request", [FakeMention(99)])
    channel._history = [msg]
    watcher._channel_cache[111] = channel

    watcher._store.set_paused(True)
    watcher._store.admit_mention_job(1101, 111, 111, tmp_path / "repo")

    _task = asyncio.create_task(watcher._durable_worker_loop())
    await asyncio.sleep(0.05)
    await watcher.close()

    job = watcher._store._conn.execute(
        "SELECT status, available_at, last_error_class, attempt_count FROM mention_jobs WHERE source_message_id = 1101"
    ).fetchone()
    assert job is not None
    assert job[0] == "pending"
    assert job[1] > int(time.time())
    assert job[2] == "paused"
    assert job[3] == 0

@pytest.mark.asyncio
async def test_hourly_cap_exceeded_defers_jobs(tmp_path: Path, monkeypatch) -> None:
    """Verify that jobs exceeding the hourly cap are deferred without dropping."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0)
    watcher = _make_watcher(tmp_path)
    channel = FakeChannel(111)
    msg = FakeMessage(1201, channel, FakeAuthor(7), "<@99> capped request", [FakeMention(99)])
    channel._history = [msg]
    watcher._channel_cache[111] = channel

    # Spend the cap (cap is 10 in _make_watcher)
    now = int(time.time())
    for i in range(10):
        watcher._store._conn.execute(
            "INSERT INTO posts(ts, channel_id) VALUES(?, ?)", (now, 111)
        )

    watcher._store.admit_mention_job(1201, 111, 111, tmp_path / "repo")

    _task = asyncio.create_task(watcher._durable_worker_loop())
    await asyncio.sleep(0.05)
    await watcher.close()

    job = watcher._store._conn.execute(
        "SELECT status, available_at, last_error_class, attempt_count FROM mention_jobs WHERE source_message_id = 1201"
    ).fetchone()
    assert job is not None
    assert job[0] == "pending"
    assert job[1] > int(time.time())
    assert job[2] == "hourly_cap_exceeded"
    assert job[3] == 0


@pytest.mark.asyncio
async def test_concurrent_conversations_burst_parallelism(
    tmp_path: Path, monkeypatch
) -> None:
    """Multiple distinct conversations process their quiet periods and runs concurrently without blocking each other."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0.05)
    cfg = Config()
    cfg.watches = [
        WatchTarget(111, str(tmp_path / "repo1"), 99, 10),
        WatchTarget(222, str(tmp_path / "repo2"), 99, 10),
    ]
    store = Store(tmp_path / "test_state.db")
    watcher = OIWatcher(cfg, store)

    ch1 = FakeChannel(111)
    ch2 = FakeChannel(222)
    msg1 = FakeMessage(1301, ch1, FakeAuthor(7), "<@99> conv 1", [FakeMention(99)])
    msg2 = FakeMessage(1302, ch2, FakeAuthor(8), "<@99> conv 2", [FakeMention(99)])
    ch1._history = [msg1]
    ch2._history = [msg2]
    watcher._channel_cache[111] = ch1
    watcher._channel_cache[222] = ch2
    started = []
    release = asyncio.Event()

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None, bot_name="", bot_id=None):
        started.append(question)
        if len(started) == 2:
            release.set()
        await release.wait()
        return Reply("answer", "sha", session_id="test-session")

    monkeypatch.setattr(watcher._runner, "run", fake_run)

    delivered = []

    async def fake_deliver(self, *args, **kwargs):
        delivered.append(kwargs.get("chunk") or args[2] if len(args) > 2 else "chunk")
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)

    # Admit mention jobs in both conversations
    watcher._store.admit_mention_job(1301, 111, 111, tmp_path / "repo1")
    watcher._store.admit_mention_job(1302, 222, 222, tmp_path / "repo2")

    _task = asyncio.create_task(watcher._durable_worker_loop())

    # Both conversations should start concurrently and reach fake_run
    await asyncio.wait_for(release.wait(), timeout=2.0)
    assert len(started) == 2
    assert set(started) == {
        "@ultron (user id 99) conv 1",
        "@ultron (user id 99) conv 2",
    }

    await asyncio.sleep(0.05)
    await watcher.close()
    assert len(delivered) == 2


@pytest.mark.asyncio
async def test_mentions_prior_to_installed_at_are_ignored(tmp_path: Path) -> None:
    """Mentions created before installed_at cutoff are ignored both on_message and on reconciliation."""
    import datetime

    cfg = Config()
    cfg.watches = [WatchTarget(channel_id=111, repo_path=str(tmp_path / "repo"), bot_user_id=99)]
    store = Store(tmp_path / "test_installed.db")
    cutoff = 1_700_000_000
    store.set_installed_at(cutoff)
    watcher = OIWatcher(cfg, store)

    channel = FakeChannel(111)
    m_old = FakeMessage(501, channel, FakeAuthor(7), "<@99> old question from the past", [FakeMention(99)])
    m_old.created_at = datetime.datetime.fromtimestamp(cutoff - 100, tz=datetime.timezone.utc)

    m_new = FakeMessage(502, channel, FakeAuthor(7), "<@99> new question after install", [FakeMention(99)])
    m_new.created_at = datetime.datetime.fromtimestamp(cutoff + 100, tz=datetime.timezone.utc)

    # 1. on_message test
    await watcher.on_message(m_old)
    stats = store.get_queue_stats()
    assert stats.get("pending_jobs", 0) == 0

    await watcher.on_message(m_new)
    stats = store.get_queue_stats()
    assert stats.get("pending_jobs", 0) == 1

    # 2. Scope reconciliation test: old messages are skipped
    store2 = Store(tmp_path / "test_reconcile.db")
    store2.set_installed_at(cutoff)
    watcher2 = OIWatcher(cfg, store2)
    channel2 = FakeChannel(111)
    channel2._history = [m_old]
    watcher2._channel_cache[111] = channel2

    await watcher2._reconcile_scope(channel2, cfg.watches[0], 111)
    stats2 = store2.get_queue_stats()
    assert stats2.get("pending_jobs", 0) == 0
    # Cursor advanced past old message
    assert store2.get_scope_cursor("111") == 501

    await watcher.close()
    await watcher2.close()


@pytest.mark.asyncio
async def test_bounded_reflection_lifecycle_and_overload(tmp_path: Path) -> None:
    """Reflection queue buffers deliveries up to capacity, drops oldest on overload, and isolates errors."""
    db = tmp_path / "test_ref.db"
    store = Store(db)
    cfg = Config()
    cfg.watches = [WatchTarget(channel_id=111, repo_path="/tmp/repo")]
    watcher = OIWatcher(cfg, store)

    # Enqueue 35 items into maxsize=32 queue
    for i in range(35):
        watcher._enqueue_reflection(
            batch_id=f"b_{i}",
            platform="discord",
            scope_id="111",
            author_id=f"u_{i}",
            author_name=f"user_{i}",
            question="q",
            reply_text="r",
            thread_excerpt="",
        )

    assert watcher._reflection_dropped_count == 3
    assert watcher._reflection_queue.qsize() == 32

    # Idempotence: re-enqueuing same batch_id is ignored
    watcher._enqueue_reflection(
        batch_id="b_34",
        platform="discord",
        scope_id="111",
        author_id="u_34",
        author_name="user_34",
        question="q",
        reply_text="r",
        thread_excerpt="",
    )
    assert watcher._reflection_queue.qsize() == 32
    await watcher.close()


# --- Reflection lifecycle contract (plan Phase 4; findings B1/B7/B8) ----------------


@pytest.mark.asyncio
async def test_failed_delivery_queues_no_reflection(tmp_path: Path, monkeypatch) -> None:
    """A delivery that fails must not enqueue any reflection observation."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0)
    watcher = _make_watcher(tmp_path)
    channel = FakeChannel(111)
    msg = FakeMessage(1401, channel, FakeAuthor(7), "<@99> deliver me", [FakeMention(99)])
    channel._history = [msg]
    watcher._channel_cache[111] = channel

    async def fake_run(*args, **kwargs):
        return Reply("answer", "sha", session_id="sess-fail")

    async def failing_deliver(self, *args, **kwargs):
        return DeliveryResult(ok=False, error_class="forbidden", status_code=403)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", failing_deliver)

    _task = asyncio.create_task(watcher._durable_worker_loop())
    await asyncio.sleep(0.1)
    await watcher.close()

    assert watcher._reflection_queue.qsize() == 0
    assert len(watcher._processed_reflection_deliveries) == 0


@pytest.mark.asyncio
async def test_recovered_outbox_delivery_queues_no_reflection(tmp_path: Path, monkeypatch) -> None:
    """Recovered outbox deliveries never fabricate reflection (plan Phase 4.6)."""
    watcher = _make_watcher(tmp_path)
    repo_path = tmp_path / "repo"
    watcher._store.admit_mention_job(1501, 111, 111, repo_path)
    watcher._store.commit_outbox(
        batch_id="recovered-batch-777",
        conversation_id=111,
        owner_channel_id=111,
        repo_path=repo_path,
        source_message_ids=[1501],
        chunks=["recovered chunk"],
        candidate_session_id=None,
    )

    async def fake_deliver(self, *args, **kwargs):
        return DeliveryResult(ok=True, discord_message_id=31337)

    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)

    _task = asyncio.create_task(watcher._durable_worker_loop())
    await asyncio.sleep(0.1)
    await watcher.close()

    assert watcher._reflection_queue.empty()
    assert len(watcher._processed_reflection_deliveries) == 0


@pytest.mark.asyncio
async def test_reflection_worker_consumes_exceptions_and_continues(tmp_path: Path) -> None:
    """Engine exceptions are consumed: no crash, and the next item still processes."""
    cfg = Config()
    cfg.watches = [WatchTarget(channel_id=111, repo_path="/tmp/repo")]
    watcher = OIWatcher(cfg, Store(tmp_path / "state.db"))

    effects = [RuntimeError("reflection exploded"), None]
    watcher._reflection_engine.reflect_after_delivery = AsyncMock(side_effect=effects)

    for i in range(2):
        watcher._enqueue_reflection(
            batch_id=f"exc_{i}",
            platform="discord",
            scope_id="111",
            author_id=f"u_{i}",
            author_name=f"user_{i}",
            question="q",
            reply_text="r",
            thread_excerpt="",
        )

    worker = asyncio.create_task(watcher._reflection_worker_loop())
    await asyncio.wait_for(watcher._reflection_queue.join(), timeout=5.0)
    await asyncio.sleep(0.05)

    assert watcher._reflection_engine.reflect_after_delivery.await_count == 2
    assert not worker.done()  # worker survived the exception (no unhandled task)
    await watcher.close()
    assert watcher._reflection_queue.empty()


@pytest.mark.asyncio
async def test_shutdown_drains_reflection_queue_within_bound(tmp_path: Path) -> None:
    """B7: close() keeps the worker draining until EMPTY within the fixed bound."""
    cfg = Config()
    cfg.watches = [WatchTarget(channel_id=111, repo_path="/tmp/repo")]
    watcher = OIWatcher(cfg, Store(tmp_path / "state.db"))

    drained: list[str] = []

    async def slow_reflect(**kwargs):
        await asyncio.sleep(0.05)
        drained.append(kwargs["author_id"])

    watcher._reflection_engine.reflect_after_delivery = slow_reflect

    total = 8
    for i in range(total):
        watcher._enqueue_reflection(
            batch_id=f"drain_{i}",
            platform="discord",
            scope_id="111",
            author_id=f"u_{i}",
            author_name=f"user_{i}",
            question="q",
            reply_text="r",
            thread_excerpt="",
        )
    assert watcher._reflection_queue.qsize() == total

    watcher._reflection_worker_task = asyncio.create_task(watcher._reflection_worker_loop())
    await asyncio.sleep(0.08)  # let the worker pick up the first item
    started = time.monotonic()
    await watcher.close()
    elapsed = time.monotonic() - started

    # Drained to EMPTY (not abandoned after one item) within the fixed bound.
    assert watcher._reflection_queue.empty()
    assert len(drained) == total
    assert elapsed < 2.0
    # Intake stays stopped after close.
    watcher._enqueue_reflection(
        batch_id="post_close",
        platform="discord",
        scope_id="111",
        author_id="u_x",
        author_name="user_x",
        question="q",
        reply_text="r",
        thread_excerpt="",
    )
    assert watcher._reflection_queue.empty()
    assert "post_close" not in watcher._processed_reflection_deliveries
    assert watcher._reflection_worker_task.done()


@pytest.mark.asyncio
async def test_idempotence_fifo_evicts_oldest_at_bound(tmp_path: Path) -> None:
    """B8: tracked deliveries evict the OLDEST entry deterministically at the bound."""
    from oi_agent.watch.discord_client import MAX_TRACKED_REFLECTION_DELIVERIES

    cfg = Config()
    cfg.watches = [WatchTarget(channel_id=111, repo_path="/tmp/repo")]
    watcher = OIWatcher(cfg, Store(tmp_path / "state.db"))

    for i in range(MAX_TRACKED_REFLECTION_DELIVERIES + 50):
        watcher._enqueue_reflection(
            batch_id=f"fifo_{i}",
            platform="discord",
            scope_id="111",
            author_id="u",
            author_name="user",
            question="q",
            reply_text="r",
            thread_excerpt="",
        )

    tracked = watcher._processed_reflection_deliveries
    assert len(tracked) == MAX_TRACKED_REFLECTION_DELIVERIES
    # Oldest 50 were evicted deterministically; recent ids are still tracked.
    assert "fifo_0" not in tracked
    assert "fifo_49" not in tracked
    assert "fifo_50" in tracked
    assert next(iter(tracked)) == "fifo_50"  # FIFO order: oldest first

    # Idempotence still holds for recent batches after overflow.
    before = watcher._reflection_queue.qsize()
    watcher._enqueue_reflection(
        batch_id="fifo_149",
        platform="discord",
        scope_id="111",
        author_id="u",
        author_name="user",
        question="q",
        reply_text="r",
        thread_excerpt="",
    )
    assert watcher._reflection_queue.qsize() == before
    await watcher.close()


@pytest.mark.asyncio
async def test_guild_watch_reflection_stores_under_guild_scope(tmp_path: Path, monkeypatch) -> None:
    """B1 regression: guild-wide watch enqueues reflection under the GUILD scope."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0)
    cfg = Config()
    cfg.watches = [
        WatchTarget(channel_id=0, guild_id=555, repo_path=str(tmp_path / "repo"), bot_user_id=99)
    ]
    store = Store(tmp_path / "state.db")
    watcher = OIWatcher(cfg, store)

    channel = FakeChannel(777, guild_id=555)
    msg = FakeMessage(1601, channel, FakeAuthor(2002, "guildie"), "<@99> guild audit", [FakeMention(99)])
    channel._history = [msg]
    watcher._channel_cache[777] = channel

    async def fake_run(*args, **kwargs):
        return Reply("guild answer", "sha-g", session_id="sess-g")

    async def fake_deliver(self, *args, **kwargs):
        return DeliveryResult(ok=True, discord_message_id=606)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)

    observation = json.dumps({
        "member": {"detail_preference": "brief", "confidence": 0.9, "topics": ["deploy"]}
    })
    watcher._runner.run_raw_prompt = AsyncMock(return_value=f"```json\n{observation}\n```")

    watcher._worker_task = asyncio.create_task(watcher._durable_worker_loop())
    watcher._reflection_worker_task = asyncio.create_task(watcher._reflection_worker_loop())

    await watcher.on_message(msg)
    for _ in range(200):
        if store.get_member_profile("discord", "555", "2002") is not None:
            break
        await asyncio.sleep(0.05)
    await watcher.close()

    # Stored under the GUILD scope (the runner's canonical rule), not channel 777.
    assert store.get_member_profile("discord", "555", "2002") is not None
    assert store.get_member_profile("discord", "777", "2002") is None
    snapshot = store.memory_snapshot("discord", "555")
    assert any(p.handle == "guildie" for p in snapshot.profiles)
    assert store.memory_snapshot("discord", "777").is_empty
    store.close()


# --- Final-only output at the Discord boundary (plan Phase 6 regression) ---------
#
# A real Discord review showed OpenCode's intermediate narration ("Let me
# check...", tool commentary) concatenated with the answer and posted. These
# tests pin the full path: fake OpenCode binary -> OpenCodeRunner -> outbox ->
# Poster, using the exact pinned synthetic stream structure (several assistant
# narration messages plus one completed final message).


def _committed_repo(tmp_path: Path) -> Path:
    """Create a minimal committed git fixture for the watched checkout."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "marker.txt").write_text("UNIQUE_MARKER\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "marker.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    return repo


def _spam_opencode_script(
    tmp_path: Path, narration_lines: list[str], final_parts: list[str]
) -> Path:
    """Fake OpenCode emitting narration messages plus one multi-part final answer."""
    session = "sess-spam"
    stream: list[dict[str, object]] = []
    part_index = 0
    for step, narration in enumerate(narration_lines):
        message = f"msg_narration_{step}"
        part = {"id": f"p{part_index}", "messageID": message,
                "sessionID": session, "type": "step-start"}
        stream.append({"type": "step_start", "timestamp": 0, "sessionID": session, "part": part})
        part_index += 1
        part = {"id": f"p{part_index}", "messageID": message,
                "sessionID": session, "type": "text", "text": narration}
        stream.append({"type": "text", "timestamp": 0, "sessionID": session, "part": part})
        part_index += 1
        part = {"id": f"p{part_index}", "messageID": message, "sessionID": session,
                "type": "tool", "tool": "read", "callID": f"call-{step}",
                "state": {"status": "completed", "input": {}, "output": ""}}
        stream.append({"type": "tool_use", "timestamp": 0, "sessionID": session, "part": part})
        part_index += 1
        part = {"id": f"p{part_index}", "messageID": message,
                "sessionID": session, "type": "step-finish", "reason": "tool-calls"}
        stream.append({"type": "step_finish", "timestamp": 0, "sessionID": session, "part": part})
        part_index += 1
    message = "msg_final"
    part = {"id": f"p{part_index}", "messageID": message,
            "sessionID": session, "type": "step-start"}
    stream.append({"type": "step_start", "timestamp": 0, "sessionID": session, "part": part})
    part_index += 1
    for final_index, final_text in enumerate(final_parts):
        if final_index == 0:
            final_text = f"{FINAL_OUTPUT_MARKER}\n{final_text}"
        part = {"id": f"p{part_index}", "messageID": message,
                "sessionID": session, "type": "text", "text": final_text}
        stream.append({"type": "text", "timestamp": 0, "sessionID": session, "part": part})
        part_index += 1
    part = {"id": f"p{part_index}", "messageID": message,
            "sessionID": session, "type": "step-finish", "reason": "stop"}
    stream.append({"type": "step_finish", "timestamp": 0, "sessionID": session, "part": part})

    body = f"""#!{sys.executable}
import json, sys
for event in {stream!r}:
    print(json.dumps(event))
"""
    path = tmp_path / "fake-opencode-spam"
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _bare_remote_with_mr(
    tmp_path: Path, *, provider: str = "gitlab", number: int = 188
) -> tuple[Path, Path, str, str]:
    """Create a bare remote and watched clone with synthetic provider MR refs."""
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", str(remote)], check=True)
    clone = _committed_repo(tmp_path)
    subprocess.run(["git", "-C", str(clone), "remote", "add", "origin", str(remote)], check=True)
    base_sha = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", "-b", "feature"], check=True)
    (clone / "marker.txt").write_text("HEAD_MARKER\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(clone), "commit", "-qm", "head"], check=True)
    head_sha = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "-C", str(clone), "checkout", "-q", "main"], check=True)
    subprocess.run(
        ["git", "-C", str(clone), "merge", "-q", "--no-ff", "-m", "merge result", "feature"],
        check=True,
    )
    merge_sha = subprocess.run(
        ["git", "-C", str(clone), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    subprocess.run(["git", "-C", str(clone), "push", "-q", "origin", "main", "feature"], check=True)
    subprocess.run(
        ["git", "-C", str(remote), "update-ref",
         provider_ref_template(provider, number, "head"), head_sha],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(remote), "update-ref",
         provider_ref_template(provider, number, "merge"), merge_sha],
        check=True,
    )
    return remote, clone, base_sha, head_sha


def _reviews_glob() -> list[Path]:
    reviews = resolve_opencode_paths().state_home / "reviews"
    if not reviews.exists():
        return []
    return list(reviews.glob("oi-review-*"))


def _collapse(text: str) -> str:
    """Whitespace-normalized form for chunk-boundary-insensitive comparison."""
    return re.sub(r"\s+", " ", text).strip()


def _outbox_rows(store: Store) -> list[tuple[str, int, str | None]]:
    return store._conn.execute(
        "SELECT batch_id, chunk_index, body FROM outbound_chunks ORDER BY batch_id, chunk_index"
    ).fetchall()


@pytest.mark.asyncio
async def test_final_only_reply_reaches_outbox_and_discord(tmp_path: Path, monkeypatch) -> None:
    """Narration from a multi-step stream never reaches the outbox or Discord."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0)
    repo = _committed_repo(tmp_path)
    narration = ["Let me check the repository now.", "Narrowing the answer down."]
    final_body = "VERIFIED_FINAL_BEGIN " + ("A verified conclusion sentence. " * 90) + "FINAL_ANSWER_END"
    binary = _spam_opencode_script(tmp_path, narration, [final_body])
    expected_text = final_body

    cfg = Config()
    cfg.watches = [WatchTarget(111, str(repo), 99, 10)]
    cfg.opencode_binary = str(binary)
    cfg.max_reply_chars = 5000  # force Discord transport chunking of one logical reply
    store = Store(tmp_path / "state.db")
    watcher = OIWatcher(cfg, store)
    watcher._reflection_engine.reflect_after_delivery = AsyncMock(return_value=None)

    channel = FakeChannel(111)
    message = FakeMessage(1701, channel, FakeAuthor(7), "<@99> audit this", [FakeMention(99)])
    channel._history = [message]

    delivered: list[str] = []
    bodies_seen_at_delivery: list[list[str | None]] = []

    async def fake_deliver(self, channel_id, target_channel_id, chunk, nonce,
                           reply_to_message_id=None, client=None):
        # Before any delivery confirmation the outbox already holds the full
        # logical reply and nothing else.
        bodies_seen_at_delivery.append([body for _b, _i, body in _outbox_rows(store)])
        delivered.append(chunk)
        return DeliveryResult(ok=True, discord_message_id=5000 + len(delivered))

    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)

    await watcher.on_message(message)
    for _ in range(200):
        if store.get_queue_stats().get("done_jobs") == 1:
            break
        await asyncio.sleep(0.05)
    await watcher.close()

    assert store.get_opencode_session(111, repo) is not None  # the audit itself succeeded
    # Only the final answer was delivered, in order, split purely for transport.
    assert len(delivered) == 2
    for chunk in delivered:
        assert len(chunk) <= 2000
    assert _collapse(" ".join(delivered)) == _collapse(expected_text)
    for line in narration:
        for chunk in delivered:
            assert line not in chunk
    # Pre-confirmation snapshot: all chunk bodies present, exactly the answer.
    assert len(bodies_seen_at_delivery) == 2
    pre_confirmation = bodies_seen_at_delivery[0]
    assert all(body is not None for body in pre_confirmation)
    assert _collapse(" ".join(pre_confirmation)) == _collapse(expected_text)
    for body in pre_confirmation:
        for line in narration:
            assert line not in body
    # After delivery confirmation the bodies are scrubbed...
    rows = _outbox_rows(store)
    assert len(rows) == 2
    assert all(body is None for _b, _i, body in rows)
    # ...and exactly one completed batch exists.
    batches = store._conn.execute("SELECT status FROM delivery_batches").fetchall()
    assert batches == [("done",)]


@pytest.mark.asyncio
async def test_mr_review_final_only_through_outbox_and_discord(tmp_path: Path, monkeypatch) -> None:
    """An MR review delivers only the final review text."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0)
    monkeypatch.setenv("HOME", str(tmp_path / "operator"))
    _remote, clone, base_sha, head_sha = _bare_remote_with_mr(tmp_path)
    fingerprint = worktree_fingerprint(clone)
    narration = ["Let me check the merge request now.", "Comparing both snapshots."]
    final_body = "REVIEW_VERDICT: the proposed change is acceptable."
    binary = _spam_opencode_script(tmp_path, narration, [final_body])
    expected_text = final_body

    cfg = Config()
    cfg.watches = [WatchTarget(111, str(clone), 99, 10)]
    cfg.opencode_binary = str(binary)
    store = Store(tmp_path / "state.db")
    watcher = OIWatcher(cfg, store)
    watcher._reflection_engine.reflect_after_delivery = AsyncMock(return_value=None)

    channel = FakeChannel(111)
    message = FakeMessage(1801, channel, FakeAuthor(7), "<@99> review MR 188", [FakeMention(99)])
    channel._history = [message]

    delivered: list[str] = []
    bodies_seen_at_delivery: list[list[str | None]] = []

    async def fake_deliver(self, channel_id, target_channel_id, chunk, nonce,
                           reply_to_message_id=None, client=None):
        bodies_seen_at_delivery.append([body for _b, _i, body in _outbox_rows(store)])
        delivered.append(chunk)
        return DeliveryResult(ok=True, discord_message_id=6000 + len(delivered))

    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)

    await watcher.on_message(message)
    for _ in range(200):
        if store.get_queue_stats().get("done_jobs") == 1:
            break
        await asyncio.sleep(0.05)
    await watcher.close()

    # Exactly one chunk: the bounded final review text.
    assert delivered == [expected_text]
    assert bodies_seen_at_delivery == [[expected_text]]
    # Review sessions are never stored, so nothing can be resumed.
    assert store.get_opencode_session(111, clone) is None
    # Outbox scrubbed after confirmation; snapshots gone; worktree untouched.
    assert all(body is None for _b, _i, body in _outbox_rows(store))
    assert _reviews_glob() == []
    assert worktree_fingerprint(clone) == fingerprint
