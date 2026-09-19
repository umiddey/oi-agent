"""Tests for deterministic reminder parsing, gateway interception, and durable state."""

from types import SimpleNamespace

import pytest

from oi_agent.config import Config, WatchTarget
from oi_agent.reminders import COMPLETION_EMOJIS, parse_reminder_declaration, render_reminder
from oi_agent.store import Store
from oi_agent.watch.discord_client import OIWatcher


def test_parse_hourly_reminder_and_targets():
    """Parse a bot-tagged ten-hour reminder and exclude the bot from targets."""
    declaration = parse_reminder_declaration(
        "<@99> remind <@7> every 10 hours: register the tax ID",
        [99, 7],
        99,
    )

    assert declaration is not None
    assert declaration.interval_seconds == 36_000
    assert declaration.target_member_ids == (7,)
    assert declaration.task_text == "register the tax ID"


def test_parser_rejects_ambiguous_or_missing_schedule():
    """Do not schedule untagged, zero, or unscheduled messages."""
    assert parse_reminder_declaration("every 10 hours: task", [7], 99) is None
    assert parse_reminder_declaration("<@99> every 0 hours: task", [99], 99) is None
    assert parse_reminder_declaration("<@99> remind me: task", [99], 99) is None


def test_store_reminder_is_idempotent_and_repeats(tmp_path):
    """Persist one declaration once, claim it, advance it, and complete it."""
    store = Store(tmp_path / "state.db")
    assert store.create_reminder(100, 200, 7, [8], 3_600, now=1_000)
    assert not store.create_reminder(100, 200, 7, [8], 3_600, now=1_000)

    assert store.claim_due_reminders(now=4_600)[0]["source_message_id"] == 100
    assert store.advance_reminder(100, sent_at=4_600)
    reminder = store.get_reminder(100)
    assert reminder["next_due_at"] == 8_200
    assert reminder["status"] == "active"

    assert store.set_reminder_completed(100, True)
    assert store.claim_due_reminders(now=100_000) == []
    assert store.set_reminder_completed(100, False)
    assert store.get_reminder(100)["status"] == "active"
    store.close()


def test_render_contains_source_and_targets():
    """Render a bounded Discord reminder with target mention and source link."""
    body = render_reminder("submit the form", 100, 200, [7], 1_000)
    assert "<@7>" in body
    assert "submit the form" in body
    assert "/200/100" in body
    assert "☑️" in COMPLETION_EMOJIS


@pytest.mark.asyncio
async def test_gateway_intercepts_reminder_without_open_code(tmp_path, monkeypatch):
    """A valid reminder declaration is stored and never enters audit admission."""
    cfg = Config(reminder_channel_id=200)
    cfg.watches = [WatchTarget(channel_id=200, repo_path=str(tmp_path), bot_user_id=99)]
    watcher = OIWatcher(cfg, Store(tmp_path / "state.db"))
    async def fake_deliver(*args, **kwargs):
        """Accept the acknowledgement without contacting Discord."""
        return SimpleNamespace(ok=True, discord_message_id=101)

    monkeypatch.setattr("oi_agent.watch.discord_client.Poster.deliver_chunk", fake_deliver)
    message = SimpleNamespace(
        id=100,
        channel=SimpleNamespace(id=200),
        author=SimpleNamespace(id=7),
        content="<@99> remind <@8> every 10 hours: submit tax ID",
        mentions=[SimpleNamespace(id=99), SimpleNamespace(id=8)],
    )

    await watcher.on_message(message)

    reminder = watcher._store.get_reminder(100)
    assert reminder is not None
    assert reminder["target_member_ids"] == (8,)
    assert reminder["interval_seconds"] == 36_000
    watcher._store.close()


@pytest.mark.asyncio
async def test_tick_reaction_completes_reminder(tmp_path):
    """A human completion tick closes the reminder and prevents due claims."""
    cfg = Config(reminder_channel_id=200)
    cfg.watches = [WatchTarget(channel_id=200, repo_path=str(tmp_path), bot_user_id=99)]
    store = Store(tmp_path / "state.db")
    store.create_reminder(100, 200, 7, [], 60, now=1)
    watcher = OIWatcher(cfg, store)
    payload = SimpleNamespace(message_id=100, user_id=7, emoji=SimpleNamespace(name="☑️"))

    await watcher.on_raw_reaction_add(payload)

    assert store.get_reminder(100)["status"] == "completed"
    assert store.claim_due_reminders(now=10_000) == []
    store.close()
