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

    def __init__(self, user_id, display_name="ultron"):
        """Store the mentioned user ID and its display name."""
        self.id = user_id
        self.display_name = display_name

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

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None, bot_name="", bot_id=None):
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

    assert calls[0] == ("user1", "@ultron (user id 99) inspect", None)
    assert calls[-1] == (111, 111, "answer")
    assert watcher._store.get_opencode_session(111, tmp_path / "repo") == "new-session"


@pytest.mark.asyncio
async def test_poster_failure_does_not_persist_session(tmp_path, monkeypatch):
    """A reply blocked or failed at the Poster boundary does not persist session state."""
    watcher = _watcher(tmp_path)
    channel = FakeChannel(111)
    message = FakeMessage(1, channel, FakeAuthor(7), "<@99> inspect", [FakeMention(99)])
    channel._history = [message]

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None, bot_name="", bot_id=None):
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

    assert questions == ["@ultron (user id 99) latest"]


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

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None, bot_name="", bot_id=None):
        """Hold the first run open while the second trigger arrives."""
        questions.append(question)
        if question == "@ultron (user id 99) first":
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
    assert questions == ["@ultron (user id 99) first"]
    release.set()
    await asyncio.sleep(0.05)
    assert questions == [
        "@ultron (user id 99) first",
        "@ultron (user id 99) follow-up",
    ]


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

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None, bot_name="", bot_id=None):
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
async def test_protocol_failure_without_session_is_retried_once(tmp_path, monkeypatch):
    """A protocol failure without stored session still gets one fresh retry."""
    watcher = _watcher(tmp_path)
    target = watcher._cfg.watches[0]
    channel = FakeChannel(111)
    message = FakeMessage(1, channel, FakeAuthor(7), "<@99> explain this", [FakeMention(99)])
    channel._history = [message]
    session_args = []

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None, bot_name="", bot_id=None):
        """Fail once without a stored session, then succeed."""
        session_args.append(session_id)
        if len(session_args) == 1:
            return Reply("failure", "sha", ok=False, error_class="protocol_empty_final_answer")
        return Reply("answer", "sha", session_id="fresh")

    async def fake_deliver(self, *args, **kwargs):
        """Accept the recovered post."""
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await watcher.on_message(message)
    await asyncio.sleep(0.05)

    assert session_args == [None, None]
    assert watcher._store.get_opencode_session(111, target.repo_path) == "fresh"


@pytest.mark.asyncio
async def test_thread_sessions_are_separated_from_parent_channel(tmp_path, monkeypatch):
    """Thread conversation stores its OpenCode session under thread ID."""
    watcher = _watcher(tmp_path, channel_id=111)
    thread = FakeChannel(222, parent_id=111)
    message = FakeMessage(1, thread, FakeAuthor(7), "<@99> thread question", [FakeMention(99)])
    thread._history = [message]

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None, bot_name="", bot_id=None):
        return Reply("thread answer", "abc123", session_id="thread-session")

    async def fake_deliver(self, channel_id, target_channel_id, chunk, nonce, reply_to_message_id=None, client=None):
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await watcher.on_message(message)
    await asyncio.sleep(0.05)

    assert watcher._store.get_opencode_session(222, tmp_path / "repo") == "thread-session"
    assert watcher._store.get_opencode_session(111, tmp_path / "repo") is None


def test_resolve_user_mentions_resolves_known_and_unknown_ids():
    """Known mentions render as name + id; unknown ids degrade to an explicit marker."""
    from oi_agent.watch.discord_client import _resolve_user_mentions

    resolved = _resolve_user_mentions(
        "hey <@99> and <@!42> ping <@123456789>",
        [FakeMention(99, "ultron"), FakeAuthor(42, "sowa")],
    )
    assert resolved == (
        "hey @ultron (user id 99) and @sowa (user id 42) ping @unknown (user id 123456789)"
    )
    # No mention markup: content passes through untouched.
    assert _resolve_user_mentions("plain text", []) == "plain text"
    assert _resolve_user_mentions("", [FakeMention(99)]) == ""


@pytest.mark.asyncio
async def test_question_and_excerpt_reach_runner_with_resolved_mentions(tmp_path, monkeypatch):
    """Mention markup is resolved in both the question and every excerpt line."""
    watcher = _watcher(tmp_path)
    channel = FakeChannel(111)
    older = FakeMessage(1, channel, FakeAuthor(7), "hello <@99>", [FakeMention(99, "ultron")])
    latest = FakeMessage(2, channel, FakeAuthor(7), "<@99> status?", [FakeMention(99, "ultron")])
    channel._history = [older, latest]
    captured = {}

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None, bot_name="", bot_id=None):
        captured["question"] = question
        captured["excerpt"] = excerpt
        return Reply("answer", "sha", session_id="session")

    async def fake_deliver(self, *args, **kwargs):
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await watcher.on_message(latest)
    await asyncio.sleep(0.05)

    assert captured["question"] == "@ultron (user id 99) status?"
    assert "@ultron (user id 99)" in captured["excerpt"]
    assert "<@99>" not in captured["excerpt"]


@pytest.mark.asyncio
async def test_runner_receives_bot_identity(tmp_path, monkeypatch):
    """The watcher passes its Discord display name and id to the runner."""
    watcher = _watcher(tmp_path)
    watcher._connection.user = FakeAuthor(99, "UltronBot")
    channel = FakeChannel(111)
    message = FakeMessage(1, channel, FakeAuthor(7), "<@99> inspect", [FakeMention(99)])
    channel._history = [message]
    captured = {}

    async def fake_run(target, author, question, excerpt, session_id=None, member_id=None, bot_name="", bot_id=None):
        captured["bot_name"] = bot_name
        captured["bot_id"] = bot_id
        return Reply("answer", "sha", session_id="session")

    async def fake_deliver(self, *args, **kwargs):
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await watcher.on_message(message)
    await asyncio.sleep(0.05)

    assert captured == {"bot_name": "UltronBot", "bot_id": 99}


async def _process_messages(watcher, channel, *messages):
    """Run admitted messages through the durable processor without worker timing.

    Args:
        watcher (OIWatcher): Watcher backed by an isolated real store.
        channel (FakeChannel): Message source channel or thread.
        messages (FakeMessage): Messages admitted to this single batch.

    Returns:
        None: Processing and immediate delivery have completed.
    """
    watcher._channel_cache[channel.id] = channel
    target, owner = watcher._resolve_target(channel.id, channel)
    for message in messages:
        watcher._store.admit_mention_job(
            source_message_id=message.id,
            conversation_id=channel.id,
            owner_channel_id=owner,
            repo_path=target.primary_repo,
        )
    conversation_id, jobs = watcher._store.claim_next_conversation_jobs()
    await watcher._process_conversation_jobs(conversation_id, jobs)


@pytest.mark.parametrize(
    "content",
    ["<@99> !fresh inspect", "<@!99> !new inspect", "!fresh <@99> inspect"],
)
@pytest.mark.asyncio
async def test_fresh_command_clears_before_run_and_resumes_delivered_replacement(
    tmp_path, monkeypatch, content,
):
    """Reset tokens disappear from context; only a delivered replacement resumes."""
    watcher = _watcher(tmp_path)
    repo = watcher._cfg.watches[0].repo_path
    watcher._store.set_opencode_session(111, repo, "old")
    watcher._store.set_opencode_session(222, repo, "other-conversation")
    channel = FakeChannel(111)
    first = FakeMessage(1, channel, FakeAuthor(7), content, [FakeMention(99)])
    channel._history = [first]
    calls, posted = [], []

    async def fake_run(target, author, question, excerpt, session_id=None, **kwargs):
        """Capture clean prompt context and the durable mapping at execution time."""
        calls.append((session_id, watcher._store.get_opencode_session(111, repo)))
        assert "!fresh" not in question + excerpt
        assert "!new" not in question + excerpt
        assert "inspect" in question
        return Reply("answer", "sha", session_id="replacement")

    async def fake_deliver(self, **kwargs):
        """Observe that replacement state is not committed before delivery."""
        if not posted:
            assert watcher._store.get_opencode_session(111, repo) is None
        posted.append(kwargs["chunk"])
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await _process_messages(watcher, channel, first)
    assert first.content == content
    assert watcher._store.get_opencode_session_info(111, repo)["turns"] == 1
    second = FakeMessage(2, channel, FakeAuthor(7), "<@99> inspect again", [FakeMention(99)])
    channel._history.append(second)
    await _process_messages(watcher, channel, second)
    assert calls == [(None, None), ("replacement", "replacement")]
    assert posted == ["answer", "answer"]
    assert watcher._store.get_opencode_session_info(111, repo)["turns"] == 2
    assert watcher._store.get_opencode_session(222, repo) == "other-conversation"


@pytest.mark.parametrize(
    "content",
    ["<@99> explain !fresh", "<@99> !freshness inspect", "<@99> !new-session", "<@7> !fresh <@99> inspect"],
)
@pytest.mark.asyncio
async def test_noncommand_text_keeps_current_session(tmp_path, monkeypatch, content):
    """Later words, longer tokens, and another user's mention cannot request reset."""
    watcher = _watcher(tmp_path)
    repo = watcher._cfg.watches[0].repo_path
    watcher._store.set_opencode_session(111, repo, "current")
    channel = FakeChannel(111)
    message = FakeMessage(1, channel, FakeAuthor(7), content, [FakeMention(99), FakeMention(7)])
    channel._history = [message]
    calls = []

    async def fake_run(target, author, question, excerpt, session_id=None, **kwargs):
        """Preserve literal user text when it is not a leading command token."""
        calls.append(session_id)
        assert "!" in question
        return Reply("answer", "sha", session_id="current")

    async def fake_deliver(self, **kwargs):
        """Require no rotation notice on an ordinary resumed turn."""
        assert kwargs["chunk"] == "answer"
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await _process_messages(watcher, channel, message)
    assert calls == ["current"]


@pytest.mark.asyncio
async def test_earlier_burst_reset_applies_to_latest_question(tmp_path, monkeypatch):
    """An earlier admitted reset survives coalescing but leaves the latest question."""
    watcher = _watcher(tmp_path)
    repo = watcher._cfg.watches[0].repo_path
    watcher._store.set_opencode_session(111, repo, "old")
    channel = FakeChannel(111)
    first = FakeMessage(1, channel, FakeAuthor(7), "<@99> !new old", [FakeMention(99)])
    second = FakeMessage(2, channel, FakeAuthor(7), "<@99> latest", [FakeMention(99)])
    channel._history = [first, second]
    calls = []

    async def fake_run(target, author, question, excerpt, session_id=None, **kwargs):
        """Capture the coalesced fresh run with sanitized context."""
        calls.append((question, session_id))
        assert "!new" not in excerpt
        return Reply("answer", "sha", session_id="new")

    async def fake_deliver(self, **kwargs):
        """Accept one coalesced batch."""
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await _process_messages(watcher, channel, first, second)
    assert calls == [("@ultron (user id 99) latest", None)]
    assert watcher._store.get_opencode_session_info(111, repo)["turns"] == 1


@pytest.mark.parametrize(
    "turns,age,ttl,rotates",
    [(30, 0, 7, True), (29, 7 * 86400, 7, True), (29, 7 * 86400 - 1, 7, False), (29, 9 * 86400, 0, False)],
)
@pytest.mark.asyncio
async def test_session_rotation_boundaries(tmp_path, monkeypatch, turns, age, ttl, rotates):
    """Rotate at cap or exact TTL; disabled TTL and below-boundary turns resume."""
    watcher = _watcher(tmp_path)
    watcher._cfg.session_ttl_days = ttl
    watcher._cfg.max_reply_chars = 50
    repo = watcher._cfg.watches[0].repo_path
    watcher._store.set_opencode_session(111, repo, "old")
    now = int(discord_client.time.time())
    watcher._store._conn.execute(
        "UPDATE opencode_sessions SET turns=?, updated_at=? WHERE conversation_id=111",
        (turns, now - age),
    )
    watcher._store._conn.commit()
    monkeypatch.setattr(discord_client.time, "time", lambda: now)
    channel = FakeChannel(111)
    message = FakeMessage(1, channel, FakeAuthor(7), "<@99> inspect", [FakeMention(99)])
    channel._history = [message]
    calls, posted = [], []
    answer = "Verified repository answer. " * 3

    async def fake_run(target, author, question, excerpt, session_id=None, **kwargs):
        """Inspect pre-run mapping and return the session actually executed."""
        calls.append(session_id)
        assert watcher._store.get_opencode_session(111, repo) == (None if rotates else "old")
        return Reply(answer, "sha", session_id="replacement" if rotates else "old")

    async def fake_deliver(self, **kwargs):
        """Capture every chunk to ensure exactly one notice survives splitting."""
        posted.append(kwargs["chunk"])
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await _process_messages(watcher, channel, message)
    assert calls == [None if rotates else "old"]
    assert " ".join(posted).count("Started a fresh session") == int(rotates)
    info = watcher._store.get_opencode_session_info(111, repo)
    assert info["session_id"] == ("replacement" if rotates else "old")
    assert info["turns"] == (1 if rotates else turns + 1)


@pytest.mark.parametrize("fresh", [False, True])
@pytest.mark.asyncio
async def test_reset_failure_stays_fresh_without_rotation_notice(tmp_path, monkeypatch, fresh):
    """A failed rotated/explicitly reset run cannot restore old state or claim success."""
    watcher = _watcher(tmp_path)
    repo = watcher._cfg.watches[0].repo_path
    watcher._cfg.max_session_turns = 1
    watcher._store.set_opencode_session(111, repo, "old")
    watcher._store._conn.execute("UPDATE opencode_sessions SET turns=1")
    watcher._store._conn.commit()
    channel = FakeChannel(111)
    content = "<@99> !fresh inspect" if fresh else "<@99> inspect"
    message = FakeMessage(1, channel, FakeAuthor(7), content, [FakeMention(99)])
    channel._history = [message]
    calls, posted = [], []

    async def fake_run(*args, session_id=None, **kwargs):
        """Return a protocol failure twice, exercising the existing fresh retry."""
        calls.append(session_id)
        assert watcher._store.get_opencode_session(111, repo) is None
        return Reply("failure", "sha", ok=False, error_class="protocol_empty_final_answer")

    async def fake_deliver(self, **kwargs):
        """Observe the final failure without an automatic-rotation success notice."""
        posted.append(kwargs["chunk"])
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(watcher._runner, "run", fake_run)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await _process_messages(watcher, channel, message)
    assert calls == [None, None]
    assert posted == ["failure"]
    assert watcher._store.get_opencode_session(111, repo) is None


@pytest.mark.asyncio
async def test_fresh_review_stays_isolated_from_conversation_sessions(tmp_path, monkeypatch):
    """A reset review follows real review dispatch and leaves no resumable mapping."""
    from oi_agent.opencode import runner

    watcher = _watcher(tmp_path)
    repo = watcher._cfg.watches[0].repo_path
    watcher._store.set_opencode_session(111, repo, "old")
    channel = FakeChannel(111)
    message = FakeMessage(1, channel, FakeAuthor(7), "<@99> !fresh review PR #188", [FakeMention(99)])
    channel._history = [message]
    review_target = object()
    posted = []

    def resolve_review(question, repo_paths):
        """Recognize the review only after the control token is stripped."""
        assert "!fresh" not in question
        assert "review PR #188" in question
        return review_target

    async def review_only(target, author, question, excerpt, *, bot_name, bot_id):
        """Model the isolated review result without accepting any resume argument."""
        assert watcher._store.get_opencode_session(111, repo) is None
        assert "!fresh" not in excerpt
        return Reply("review answer", "sha", session_id=None)

    async def fake_deliver(self, **kwargs):
        """Deliver the review without creating branch-audit state."""
        posted.append(kwargs["chunk"])
        return DeliveryResult(ok=True, discord_message_id=999)

    monkeypatch.setattr(runner, "resolve_review_target", resolve_review)
    monkeypatch.setattr(watcher._runner, "_run_review_locked", review_only)
    monkeypatch.setattr(discord_client.Poster, "deliver_chunk", fake_deliver)
    await _process_messages(watcher, channel, message)
    assert posted == ["review answer"]
    assert watcher._store.get_opencode_session(111, repo) is None
