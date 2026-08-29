"""Mocked end-to-end tests for one Discord message event."""

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
import pytest

from oi_agent.agent import responder
from oi_agent.config import Config, LLMConfig, WatchTarget
from oi_agent.store import Store
from oi_agent.watch import discord_client
from oi_agent.watch.discord_client import OIWatcher


class FakeAuthor:
    """Minimal Discord author object used by the event simulation."""

    def __init__(self, user_id: int, name: str = "alice", bot: bool = False):
        """Store the fields consumed by the watcher.

        Args:
            user_id: Discord user id.
            name: Display name.
            bot: Whether this author is a bot account.
        """
        self.id = user_id
        self.display_name = name
        self.bot = bot


class FakeMention:
    """Minimal Discord mention object."""

    def __init__(self, user_id: int):
        """Store a mentioned user id.

        Args:
            user_id: Discord user id.
        """
        self.id = user_id


class FakeChannel:
    """Messageable fake with async history and optional thread parent."""

    def __init__(self, channel_id: int, history=None, parent_id=None):
        """Create a fake channel.

        Args:
            channel_id: Channel or thread id.
            history: Messages yielded newest-first by Discord history.
            parent_id: Parent channel id for a thread.
        """
        self.id = channel_id
        self.parent_id = parent_id
        self._history = list(history or [])

    async def history(self, limit: int):
        """Yield at most ``limit`` fake messages newest-first."""
        for message in self._history[:limit]:
            yield message


class FakeMessage:
    """Minimal Discord message event object."""

    def __init__(self, message_id: int, channel, author, content, mentions):
        """Create a fake incoming message.

        Args:
            message_id: Discord message id.
            channel: Fake channel containing the message.
            author: Fake author.
            content: Message body.
            mentions: Mentioned fake users.
        """
        self.id = message_id
        self.channel = channel
        self.author = author
        self.content = content
        self.mentions = mentions
        self.created_at = datetime.now(UTC)


def _watcher(tmp_path, channel_id=111):
    """Build a watcher with one configured target and isolated state."""
    cfg = Config()
    cfg.watches = [WatchTarget(
        channel_id=channel_id,
        repo_path=str(tmp_path / "repo"),
        bot_user_id=99,
        post_hourly_cap=3,
    )]
    return OIWatcher(cfg, Store(tmp_path / "state.db"))


@pytest.fixture(autouse=True)
def immediate_burst_flush(monkeypatch):
    """Remove production debounce delay from isolated event-path tests."""
    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0)


@pytest.mark.asyncio
async def test_direct_mention_runs_full_pipeline(tmp_path, monkeypatch):
    """A bot mention reaches audit, gets stamped, and passes to the poster."""
    watcher = _watcher(tmp_path)
    channel = FakeChannel(111)
    message = FakeMessage(
        1, channel, FakeAuthor(7), "<@99> where is the bug?", [FakeMention(99)]
    )
    channel._history = [message]
    calls = []

    monkeypatch.setattr(responder, "git_pull", lambda _: True)
    monkeypatch.setattr(responder, "refresh_repo_manifest", lambda _: None)
    monkeypatch.setattr(responder.tools, "audit_sha", lambda _: "abc123")
    monkeypatch.setattr(responder.tools, "manifest_summary", lambda _: "# Code Map")

    async def fake_chat_agentic(configs, system, user, *, tools, execute,
                                max_iterations, max_tokens, temperature):
        """Return a deterministic model answer and record the prompt."""
        calls.append(user)
        return "The answer is in the code.", "mock/model"

    async def fake_send(self, channel_id, target_channel_id, content):
        """Capture the poster boundary without making an HTTP request."""
        calls.append((channel_id, target_channel_id, content))
        return True

    monkeypatch.setattr(responder, "chat_agentic", fake_chat_agentic)
    monkeypatch.setattr(discord_client.Poster, "send", fake_send)

    await watcher.on_message(message)
    await asyncio.sleep(0.05)

    assert any(isinstance(call, str) and "Question:" in call for call in calls)
    posted = calls[-1]
    assert posted[:2] == (111, 111)
    assert "audited at abc123" in posted[2]
    # Confirmed send must fold the exchange into rolling memory (5a), and
    # nothing else: the message archive was removed as unbounded dead weight.
    _, exchanges = watcher._store.get_memory(111)
    assert len(exchanges) == 1



@pytest.mark.asyncio
async def test_trigger_burst_runs_one_latest_response(tmp_path, monkeypatch):
    """A burst in one conversation produces one response for its latest event."""
    watcher = _watcher(tmp_path)
    channel = FakeChannel(111)
    first = FakeMessage(
        1, channel, FakeAuthor(7), "<@99> inspect the old thing",
        [FakeMention(99)],
    )
    second = FakeMessage(
        2, channel, FakeAuthor(7), "<@99> inspect the latest thing",
        [FakeMention(99)],
    )
    channel._history = [second, first]
    questions = []
    posted = []

    async def fake_responder(*args, **kwargs):
        """Capture one coalesced responder invocation."""
        questions.append(args[4])
        return responder.Reply("coalesced answer", "abc123")

    async def fake_send(self, channel_id, target_channel_id, content):
        """Capture the single post at the poster boundary."""
        posted.append((channel_id, target_channel_id, content))
        return True

    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0.01)
    monkeypatch.setattr(discord_client.responder, "respond", fake_responder)
    monkeypatch.setattr(discord_client.Poster, "send", fake_send)

    await watcher.on_message(first)
    await watcher.on_message(second)
    await asyncio.sleep(0.03)

    assert questions == ["<@99> inspect the latest thing"]
    assert posted == [(111, 111, "coalesced answer")]


@pytest.mark.asyncio
async def test_unmarked_followup_extends_pending_burst(tmp_path, monkeypatch):
    """Unmarked voice-to-text fragments extend a pending explicit request."""
    watcher = _watcher(tmp_path)
    channel = FakeChannel(111)
    trigger = FakeMessage(
        1, channel, FakeAuthor(7), "<@99> inspect this issue",
        [FakeMention(99)],
    )
    followup = FakeMessage(
        2, channel, FakeAuthor(7), "the screenshot is attached here", []
    )
    channel._history = [followup, trigger]
    questions = []

    async def fake_responder(*args, **kwargs):
        """Capture the original explicit request."""
        questions.append(args[4])
        return responder.Reply("answer", "abc123")

    async def fake_send(self, channel_id, target_channel_id, content):
        """Accept the coalesced reply without network access."""
        return True

    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0.01)
    monkeypatch.setattr(discord_client.responder, "respond", fake_responder)
    monkeypatch.setattr(discord_client.Poster, "send", fake_send)

    await watcher.on_message(trigger)
    await asyncio.sleep(0.005)
    await watcher.on_message(followup)
    await asyncio.sleep(0.03)
    assert questions == [trigger.content]



@pytest.mark.asyncio
async def test_trigger_during_audit_gets_one_followup(tmp_path, monkeypatch):
    """A newer trigger waits for the current audit instead of racing it."""
    watcher = _watcher(tmp_path)
    channel = FakeChannel(111)
    first = FakeMessage(
        1, channel, FakeAuthor(7), "<@99> inspect the first thing",
        [FakeMention(99)],
    )
    second = FakeMessage(
        2, channel, FakeAuthor(7), "<@99> inspect the follow-up",
        [FakeMention(99)],
    )
    channel._history = [second, first]
    started = asyncio.Event()
    release = asyncio.Event()
    questions = []
    posted = []

    async def fake_responder(*args, **kwargs):
        """Hold the first audit open while a newer event arrives."""
        question = args[4]
        questions.append(question)
        if question == first.content:
            started.set()
            await release.wait()
        return responder.Reply("answer", "abc123")

    async def fake_send(self, channel_id, target_channel_id, content):
        """Capture each completed coalesced response."""
        posted.append(content)
        return True

    monkeypatch.setattr(discord_client, "BURST_QUIET_SECONDS", 0)
    monkeypatch.setattr(discord_client.responder, "respond", fake_responder)
    monkeypatch.setattr(discord_client.Poster, "send", fake_send)

    await watcher.on_message(first)
    await asyncio.sleep(0.02)
    assert started.is_set()

    await watcher.on_message(second)
    await asyncio.sleep(0)
    assert questions == [first.content]

    release.set()
    await asyncio.sleep(0.05)

    assert questions == [first.content, second.content]
    assert posted == ["answer", "answer"]



@pytest.mark.asyncio
async def test_thread_memory_is_keyed_per_thread_not_parent(
        tmp_path, monkeypatch):
    """A thread message stores recall under the thread id, not the parent.

    The hourly cap still keys on the parent channel, but memory must be
    per-conversation so sibling threads never cross-contaminate each other's
    rolling context.
    """
    watcher = _watcher(tmp_path, channel_id=111)
    # Message arrives in thread 222 whose parent is the watched channel 111.
    thread = FakeChannel(222, parent_id=111)
    message = FakeMessage(
        1, thread, FakeAuthor(7), "<@99> what changed here?", [FakeMention(99)]
    )
    thread._history = [message]

    monkeypatch.setattr(responder, "git_pull", lambda _: True)
    monkeypatch.setattr(responder, "refresh_repo_manifest", lambda _: None)
    monkeypatch.setattr(responder.tools, "audit_sha", lambda _: "abc123")
    monkeypatch.setattr(responder.tools, "manifest_summary", lambda _: "# Map")

    async def fake_chat_agentic(configs, system, user, *, tools, execute,
                                max_iterations, max_tokens, temperature):
        """Return a deterministic answer."""
        return "Thread-scoped answer.", "mock/model"

    async def fake_send(self, channel_id, target_channel_id, content):
        """Accept the post without network access."""
        return True

    monkeypatch.setattr(responder, "chat_agentic", fake_chat_agentic)
    monkeypatch.setattr(discord_client.Poster, "send", fake_send)

    await watcher.on_message(message)
    await asyncio.sleep(0.05)

    # Recall lives under the thread id...
    _, thread_exchanges = watcher._store.get_memory(222)
    assert len(thread_exchanges) == 1
    # ...and NOT under the parent channel id.
    _, parent_exchanges = watcher._store.get_memory(111)
    assert parent_exchanges == []


@pytest.mark.asyncio
async def test_conversational_message_skips_the_audit(tmp_path, monkeypatch):
    """A chat message is answered by triage without pull/manifest/tools."""
    target = WatchTarget(channel_id=111, repo_path=str(tmp_path / "repo"))
    audited = []

    async def fake_chat(configs, system, user, *, max_tokens, temperature):
        """Triage answers conversationally (no audit sentinel)."""
        return "haha good one, no notes.", "mock/model"

    async def boom_context(*a, **k):
        """Fail loudly if the audit pipeline is entered for chat."""
        audited.append("context")
        raise AssertionError("audit must not run for a chat message")

    async def boom_agentic(*a, **k):
        audited.append("agentic")
        raise AssertionError("tool loop must not run for a chat message")

    monkeypatch.setattr(responder, "chat", fake_chat)
    monkeypatch.setattr(responder, "build_context", boom_context)
    monkeypatch.setattr(responder, "chat_agentic", boom_agentic)

    reply = await responder.respond(
        LLMConfig("https://big.example/v1", "big-model"),
        LLMConfig("https://small.example/v1", "small-model"),
        target, "alice", "crack a joke",
        "alice: lol\nbob: nice", memory_key=111,
    )

    assert reply.text == "haha good one, no notes."
    assert reply.sha == "none"                 # audited no code
    assert "audited at" not in reply.text      # no provenance footer
    assert reply.ok is True                    # real reply, folds into memory
    assert audited == []                       # pipeline never entered


@pytest.mark.asyncio
async def test_new_code_question_triggers_audit(tmp_path, monkeypatch):
    """The audit sentinel from triage drops into the full audit pipeline."""
    target = WatchTarget(channel_id=111, repo_path=str(tmp_path / "repo"))
    ran = []

    async def fake_chat(configs, system, user, *, max_tokens, temperature):
        """Triage decides this needs fresh code reading."""
        return responder.AUDIT_SENTINEL, "mock/model"

    async def fake_context(*a, **k):
        ran.append("context")
        return responder._SeedContext("AUDITED SHA: abc123", "abc123", "fp")

    async def fake_agentic(*a, **k):
        ran.append("agentic")
        return "Here is what the code does, per file X.", "mock/model"

    monkeypatch.setattr(responder, "chat", fake_chat)
    monkeypatch.setattr(responder, "build_context", fake_context)
    monkeypatch.setattr(responder, "chat_agentic", fake_agentic)
    monkeypatch.setattr(responder, "_mutation_warning", lambda *_a: "")

    reply = await responder.respond(
        LLMConfig("https://big.example/v1", "big-model"),
        LLMConfig("https://small.example/v1", "small-model"),
        target, "alice", "did the new settlement query get implemented?",
        "", memory_key=111,
    )

    assert ran == ["context", "agentic"]       # full audit ran
    assert reply.sha == "abc123"
    assert "abc123" in reply.text              # stamped provenance


@pytest.mark.asyncio
async def test_triage_failure_falls_through_to_audit(tmp_path, monkeypatch):
    """A triage provider failure audits rather than dropping the message."""
    target = WatchTarget(channel_id=111, repo_path=str(tmp_path / "repo"))
    ran = []

    async def boom_chat(*a, **k):
        raise RuntimeError("triage provider down")

    async def fake_context(*a, **k):
        ran.append("context")
        return responder._SeedContext("AUDITED SHA: abc123", "abc123", "fp")

    async def fake_agentic(*a, **k):
        ran.append("agentic")
        return "answer from the audit", "mock/model"

    monkeypatch.setattr(responder, "chat", boom_chat)
    monkeypatch.setattr(responder, "build_context", fake_context)
    monkeypatch.setattr(responder, "chat_agentic", fake_agentic)
    monkeypatch.setattr(responder, "_mutation_warning", lambda *_a: "")

    reply = await responder.respond(
        LLMConfig("https://big.example/v1", "big-model"),
        LLMConfig("https://small.example/v1", "small-model"),
        target, "alice", "anything", "", memory_key=111,
    )

    assert ran == ["context", "agentic"]       # fell through to audit
    assert reply.sha == "abc123"


@pytest.mark.asyncio
async def test_unmarked_founder_message_is_ignored(tmp_path, monkeypatch):
    """Founder identity and claim words do not bypass the explicit gate."""
    watcher = _watcher(tmp_path)
    message = FakeMessage(
        2, FakeChannel(111), FakeAuthor(7), "the bug report is resolved", []
    )
    calls = []

    async def fake_responder(*args, **kwargs):
        """Fail the test if ignored chatter reaches the responder."""
        calls.append((args, kwargs))
        return responder.Reply("short answer", "abc123")

    monkeypatch.setattr(discord_client.responder, "respond", fake_responder)

    await watcher.on_message(message)

    assert calls == []


@pytest.mark.asyncio
async def test_context_failure_becomes_honest_reply(tmp_path, monkeypatch):
    """Repository failures become a stamped error instead of escaping the event."""
    watcher = _watcher(tmp_path)
    channel = FakeChannel(111)
    message = FakeMessage(
        3, channel, FakeAuthor(7), "<@99> inspect this", [FakeMention(99)]
    )
    posted = []

    def fail_refresh(_):
        """Simulate the manifest failure from the reported path."""
        raise RuntimeError("manifest unavailable")

    async def fake_send(self, channel_id, target_channel_id, content):
        """Capture the controlled failure response."""
        posted.append(content)
        return True

    monkeypatch.setattr(responder, "git_pull", lambda _: True)
    monkeypatch.setattr(responder, "refresh_repo_manifest", fail_refresh)
    monkeypatch.setattr(responder.tools, "audit_sha", lambda _: "none")
    monkeypatch.setattr(discord_client.Poster, "send", fake_send)

    await watcher.on_message(message)
    await asyncio.sleep(0.05)

    assert len(posted) == 1
    assert "Couldn't inspect the repository" in posted[0]
    assert "audited at none" in posted[0]
    # The honest-failure reply posts but must NOT be folded into memory.
    _, exchanges = watcher._store.get_memory(111)
    assert exchanges == []


@pytest.mark.asyncio
async def test_non_text_model_output_becomes_honest_reply(tmp_path, monkeypatch):
    """A successful HTTP response with null content cannot crash drafting."""
    target = WatchTarget(channel_id=111, repo_path=str(tmp_path / "repo"))

    monkeypatch.setattr(responder, "_mutation_warning", lambda *_a: "")

    async def fake_context(*args, **kwargs):
        """Return deterministic context without touching a repository."""
        return responder._SeedContext("AUDITED SHA: abc123", "abc123", "fp")

    async def fake_chat_agentic(*args, **kwargs):
        """Simulate the malformed provider response from the log."""
        return None, "mock/provider"
    monkeypatch.setattr(responder, "build_context", fake_context)
    monkeypatch.setattr(responder, "chat_agentic", fake_chat_agentic)

    reply = await responder.respond(
        LLMConfig("https://big.example/v1", "big-model"),
        LLMConfig("https://small.example/v1", "small-model"),
        target,
        "alice",
        "inspect this",
        "",
    )

    assert "Couldn't complete the audit" in reply.text
    assert "audited at abc123" in reply.text


@pytest.mark.asyncio
async def test_model_loop_greps_and_reads_through_jailed_executor(
        tmp_path, monkeypatch):
    """Retrieval is model-driven: the loop's executor greps and reads."""
    repo = tmp_path / "repo"
    (repo / "backend").mkdir(parents=True)
    (repo / "backend" / "contracts.py").write_text(
        "COST_CENTER_DEFAULTS = 'monthly'\n"
        "default_cost_relevance = operating_cost\n")
    target = WatchTarget(channel_id=111, repo_path=str(repo))
    searched = []
    read_paths = []

    monkeypatch.setattr(responder, "git_pull", lambda _: True)
    monkeypatch.setattr(responder, "refresh_repo_manifest", lambda _: None)
    monkeypatch.setattr(responder.tools, "audit_sha", lambda _: "abc123")
    monkeypatch.setattr(responder.tools, "manifest_summary", lambda _: "# Code Map")

    # Spy-wrappers keep the REAL jailed tools in the path while recording.
    real_grep = responder.tools.grep_multi

    def spy_grep(repo_root, patterns, max_results=30):
        """Record requested patterns, then run the genuine search."""
        searched.extend(patterns)
        return real_grep(repo_root, patterns, max_results=max_results)

    monkeypatch.setattr(responder.tools, "grep_multi", spy_grep)

    real_read = responder.tools.read_file

    def spy_read(repo_root, rel_path, max_bytes=200_000, truncate=False):
        """Record requested paths, then run the genuine contained read."""
        read_paths.append(rel_path)
        return real_read(repo_root, rel_path, max_bytes=max_bytes,
                         truncate=truncate)

    monkeypatch.setattr(responder.tools, "read_file", spy_read)

    async def fake_chat_agentic(configs, system, user, *, tools, execute,
                                max_iterations, max_tokens, temperature):
        """Model behavior: grep the thread's terms, then read the hit."""
        hits = execute("grep_repo",
                       {"patterns": ["cost_center", "defaults"]})
        source = execute("read_file", {"path": "backend/contracts.py"})
        return f"{hits}\n{source}", "mock/model"

    monkeypatch.setattr(responder, "chat_agentic", fake_chat_agentic)

    reply = await responder.respond(
        LLMConfig("https://big.example/v1", "big-model"),
        LLMConfig("https://small.example/v1", "small-model"),
        target,
        "alice",
        "check plz",
        "bob: cost center defaults ledger scratch",
    )

    assert "cost_center" in searched
    assert "defaults" in searched
    assert read_paths == ["backend/contracts.py"]
    assert "default_cost_relevance = operating_cost" in reply.text


@pytest.mark.asyncio
async def test_thread_excerpt_preserves_attachment_metadata():
    """Attachment names and URLs remain visible to the evidence builder."""
    author = FakeAuthor(7)
    message = SimpleNamespace(
        author=author,
        content="see screenshot",
        attachments=[SimpleNamespace(
            filename="classification.png",
            url="https://cdn.discordapp.com/classification.png",
            content_type="image/png",
        )],
        embeds=[],
    )
    excerpt = await discord_client._thread_excerpt(
        FakeChannel(111, history=[message])
    )

    assert "classification.png" in excerpt
    assert "https://cdn.discordapp.com/classification.png" in excerpt


def test_uncached_thread_resolves_from_event_channel(tmp_path):
    """Thread parent resolution uses the channel attached to the event."""
    watcher = _watcher(tmp_path, channel_id=111)
    thread = FakeChannel(222, parent_id=111)

    target, owner = watcher._resolve_target(222, thread)

    assert target is watcher._cfg.watches[0]
    assert owner == 111


@pytest.mark.asyncio
async def test_poster_network_failure_is_blocked(tmp_path, monkeypatch):
    """A Discord transport exception returns false instead of escaping."""
    watcher = _watcher(tmp_path)

    class BrokenClient:
        """Async client whose request fails at the network boundary."""

        def __init__(self, *args, **kwargs):
            """Accept the httpx client constructor arguments."""

        async def __aenter__(self):
            """Enter the fake HTTP client."""
            return self

        async def __aexit__(self, *args):
            """Leave the fake HTTP client."""
            return False

        async def post(self, *args, **kwargs):
            """Raise the simulated transport failure."""
            raise OSError("network down")

    monkeypatch.setattr("oi_agent.poster.httpx.AsyncClient", BrokenClient)
    poster = discord_client.Poster(watcher._cfg, watcher._store, "token")

    assert await poster.send(111, 111, "answer") is False
