"""Mocked end-to-end tests for the Discord admission and OpenCode path."""

import asyncio
from datetime import UTC, datetime

import discord
import pytest

from oi_agent.agent.reply import Reply
from oi_agent.config import Config, WatchTarget
from oi_agent.poster import DeliveryResult
from oi_agent.store import Store
from oi_agent.watch import discord_client
from oi_agent.watch.discord_client import OIWatcher


class FakeAuthor:
    """Minimal Discord author object."""

    def __init__(self, user_id, name="user1", bot=False):
        """Store author fields consumed by the watcher."""
        self.id = user_id
        self.display_name = name
        self.bot = bot


class FakeMention:
    """Minimal Discord mention object."""

    def __init__(self, user_id):
        """Store the mentioned user ID."""
        self.id = user_id

class FakeResponse:
    """Mock response for discord exception initialization."""

    def __init__(self, status=404, reason="Not Found"):
        self.status = status
        self.reason = reason


class FakeChannel:

    def __init__(self, channel_id, history=None, parent_id=None):
        """Create a fake channel."""
        self.id = channel_id
        self.parent_id = parent_id
        self._history = list(history or [])

    async def history(self, limit=50, after=None, oldest_first=False):
        """Yield fake messages with optional after filter."""
        msgs = list(self._history)
        if after is not None:
            after_id = after.id if hasattr(after, "id") else int(after)
            msgs = [m for m in msgs if m.id > after_id]
        if limit is not None:
            msgs = msgs[:limit]
        for message in msgs:
            yield message

    async def fetch_message(self, message_id):
        """Fetch a fake message by snowflake ID."""
        for m in self._history:
            if m.id == message_id:
                return m
        raise discord.NotFound(FakeResponse(404), "message not found")


class FakeMessage:
    """Minimal incoming Discord message."""

    def __init__(self, message_id, channel, author, content, mentions):
        """Create a fake event message."""
        self.id = message_id
        self.channel = channel
        self.author = author
        self.content = content
        self.mentions = mentions
        self.created_at = datetime.now(UTC)
        self.attachments = []
        self.embeds = []


def _watcher(tmp_path, channel_id=111):
    """Build a watcher with one configured target and isolated state."""
    cfg = Config()
    cfg.watches = [WatchTarget(channel_id, str(tmp_path / "repo"), 99, 3)]
    return OIWatcher(cfg, Store(tmp_path / "state.db"))


@pytest.fixture(autouse=True)
def immediate_burst_flush(monkeypatch):
    """Remove production debounce delay from event-path tests."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0)


@pytest.mark.asyncio
async def test_direct_mention_reaches_runner_and_poster_once(tmp_path, monkeypatch):
    """An explicit mention runs OpenCode, posts once, then stores its session."""
    watcher = _watcher(tmp_path)
    channel = FakeChannel(111)
    message = FakeMessage(1, channel, FakeAuthor(7), "<@99> inspect", [FakeMention(99)])
    channel._history = [message]
    calls = []

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None):
        """Return a deterministic successful OpenCode reply."""
        calls.append((author, question, session_id))
        return Reply("answer", "abc123", session_id="new-session")

    async def fake_deliver(self, channel_id, target_channel_id, chunk, nonce, reply_to_message_id=None, client=None):
        """Capture the Poster boundary."""
        calls.append((channel_id, target_channel_id, chunk))
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await watcher.on_message(message)
    await asyncio.sleep(0.05)

    assert calls[0] == ("user1", "<@99> inspect", None)
    assert calls[-1] == (111, 111, "answer")
    assert watcher._store.get_opencode_session(111, tmp_path / "repo") == "new-session"


@pytest.mark.asyncio
async def test_poster_failure_does_not_persist_session(tmp_path, monkeypatch):
    """A reply blocked or failed at the Poster boundary does not persist session state."""
    watcher = _watcher(tmp_path)
    channel = FakeChannel(111)
    message = FakeMessage(1, channel, FakeAuthor(7), "<@99> inspect", [FakeMention(99)])
    channel._history = [message]

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None):
        return Reply("answer", "abc123", session_id="unsent-session")

    async def fake_deliver_fail(
        self, channel_id, target_channel_id, chunk, nonce, reply_to_message_id=None, client=None
    ):
        return DeliveryResult(ok=False, error_class="network_error")

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver_fail)
    await watcher.on_message(message)
    await asyncio.sleep(0.05)

    assert watcher._store.get_opencode_session(111, tmp_path / "repo") is None


@pytest.mark.asyncio
async def test_burst_coalescing_keeps_latest_question(tmp_path, monkeypatch):
    """A quiet-period burst creates one run using the latest explicit question."""
    watcher = _watcher(tmp_path)
    channel = FakeChannel(111)
    first = FakeMessage(1, channel, FakeAuthor(7), "<@99> old", [FakeMention(99)])
    second = FakeMessage(2, channel, FakeAuthor(7), "<@99> latest", [FakeMention(99)])
    channel._history = [first, second]
    questions = []

    async def fake_run(*args, **kwargs):
        """Capture the single coalesced request."""
        questions.append(args[2])
        return Reply("answer", "sha", session_id="session")

    async def fake_deliver(self, *args, **kwargs):
        """Accept the post without network access."""
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await watcher.on_message(first)
    await watcher.on_message(second)
    await asyncio.sleep(0.05)

    assert questions == ["<@99> latest"]


@pytest.mark.asyncio
async def test_trigger_during_run_gets_one_followup(tmp_path, monkeypatch):
    """A newer trigger waits for the current run and does not race it."""
    watcher = _watcher(tmp_path)
    channel = FakeChannel(111)
    first = FakeMessage(1, channel, FakeAuthor(7), "<@99> first", [FakeMention(99)])
    second = FakeMessage(2, channel, FakeAuthor(7), "<@99> follow-up", [FakeMention(99)])
    channel._history = [first, second]
    started = asyncio.Event()
    release = asyncio.Event()
    questions = []

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None):
        """Hold the first run open while the second trigger arrives."""
        questions.append(question)
        if question == first.content:
            started.set()
            await release.wait()
        return Reply("answer", "sha", session_id=f"session-{len(questions)}")

    async def fake_deliver(self, *args, **kwargs):
        """Accept each post."""
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await watcher.on_message(first)
    await asyncio.sleep(0.02)
    assert started.is_set()
    await watcher.on_message(second)
    await asyncio.sleep(0)
    assert questions == [first.content]
    release.set()
    await asyncio.sleep(0.05)
    assert questions == [first.content, second.content]


@pytest.mark.parametrize(
    "failure_class",
    ["unknown_session", "protocol_missing_final_marker", "protocol_empty_final_answer"],
)
@pytest.mark.asyncio
async def test_stale_session_is_cleared_and_retried_once(
    tmp_path, monkeypatch, failure_class,
):
    """A stale or unframed stored session triggers one fresh retry."""
    watcher = _watcher(tmp_path)
    target = watcher._cfg.watches[0]
    watcher._store.set_opencode_session(111, target.repo_path, "stale")
    channel = FakeChannel(111)
    message = FakeMessage(1, channel, FakeAuthor(7), "<@99> inspect", [FakeMention(99)])
    channel._history = [message]
    session_args = []

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None):
        """Fail once as stale, then succeed fresh."""
        session_args.append(session_id)
        if len(session_args) == 1:
            return Reply("failure", "sha", ok=False, error_class=failure_class)
        return Reply("answer", "sha", session_id="fresh")

    async def fake_deliver(self, *args, **kwargs):
        """Accept the final post."""
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await watcher.on_message(message)
    await asyncio.sleep(0.05)

    assert session_args == ["stale", None]
    assert watcher._store.get_opencode_session(111, target.repo_path) == "fresh"


@pytest.mark.asyncio
async def test_thread_sessions_are_separated_from_parent_channel(tmp_path, monkeypatch):
    """Thread conversation stores its OpenCode session under thread ID."""
    watcher = _watcher(tmp_path, channel_id=111)
    thread = FakeChannel(222, parent_id=111)
    message = FakeMessage(1, thread, FakeAuthor(7), "<@99> thread question", [FakeMention(99)])
    thread._history = [message]

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None):
        return Reply("thread answer", "abc123", session_id="thread-session")

    async def fake_deliver(self, channel_id, target_channel_id, chunk, nonce, reply_to_message_id=None, client=None):
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await watcher.on_message(message)
    await asyncio.sleep(0.05)

    assert watcher._store.get_opencode_session(222, tmp_path / "repo") == "thread-session"
    assert watcher._store.get_opencode_session(111, tmp_path / "repo") is None
