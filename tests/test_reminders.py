"""Tests for deterministic reminder parsing, gateway interception, and durable state."""

from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from oi_agent.config import Config, WatchTarget
from oi_agent.reminders import (
    COMPLETION_EMOJIS,
    format_schedule,
    next_daily_occurrence,
    parse_reminder_declaration,
    render_reminder,
    tomorrow_at,
)
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


def test_parse_daily_at_time():
    """Daily-at-time repeats every day at the Berlin wall-clock time."""
    declaration = parse_reminder_declaration(
        "<@99> remind me daily at 11am: call DeutschePost",
        [99],
        99,
    )

    assert declaration is not None
    assert declaration.kind == "daily"
    assert (declaration.hour, declaration.minute) == (11, 0)
    assert declaration.interval_seconds == 86_400
    assert declaration.task_text == "call DeutschePost"
    assert declaration.target_member_ids == ()


def test_parse_every_day_at_time_variants():
    """Every-day wording with 24h, pm, and targeted forms."""
    first = parse_reminder_declaration(
        "<@99> every day at 10am - check the employee portal",
        [99],
        99,
    )
    assert first is not None
    assert (first.kind, first.hour, first.minute) == ("daily", 10, 0)
    assert first.task_text == "check the employee portal"

    second = parse_reminder_declaration(
        "<@99> remind <@7> every day at 6pm: standup notes",
        [99, 7],
        99,
    )
    assert second is not None
    assert (second.kind, second.hour, second.minute) == ("daily", 18, 0)
    assert second.target_member_ids == (7,)

    third = parse_reminder_declaration(
        "<@99> remind me daily at 14:30: review inbox",
        [99],
        99,
    )
    assert third is not None
    assert (third.hour, third.minute) == (14, 30)

    assert parse_reminder_declaration("<@99> remind me daily at 25am: nope", [99], 99) is None
    assert parse_reminder_declaration("<@99> remind me daily at 11:99: nope", [99], 99) is None


def test_parse_timeless_daily_keeps_legacy_task_text():
    """Timeless 'every day' keeps the exact legacy task extraction."""
    declaration = parse_reminder_declaration(
        "<@99> Remind us to pray to God once every day",
        [99],
        99,
    )

    assert declaration is not None
    assert declaration.kind == "daily"
    assert declaration.hour is None
    assert declaration.task_text == "us to pray to God once"


def test_parse_one_shot_tomorrow():
    """One-shot 'tomorrow at time' fires once without guessing."""
    declaration = parse_reminder_declaration(
        "<@99> Please remind me to call DeutschePost at 11am tomorrow",
        [99],
        99,
    )

    assert declaration is not None
    assert declaration.kind == "once"
    assert (declaration.hour, declaration.minute) == (11, 0)
    assert declaration.task_text == "call DeutschePost"

    reversed_order = parse_reminder_declaration(
        "<@99> remind me tomorrow at 10am: book the dentist",
        [99],
        99,
    )
    assert reversed_order is not None
    assert (reversed_order.kind, reversed_order.hour, reversed_order.task_text) == (
        "once",
        10,
        "book the dentist",
    )

    assert parse_reminder_declaration("<@99> remind me tomorrow: no time", [99], 99) is None
    assert parse_reminder_declaration("<@99> remind me daily", [99], 99) is None


def test_wall_time_helpers():
    """Berlin occurrence math picks today/future and tomorrow correctly."""
    berlin = ZoneInfo("Europe/Berlin")
    evening = int(datetime(2026, 9, 21, 22, 57, tzinfo=berlin).timestamp())

    morning_due = next_daily_occurrence(11, 0, evening)
    assert datetime.fromtimestamp(morning_due, tz=berlin).strftime("%Y-%m-%d %H:%M") == "2026-09-22 11:00"

    same_evening = next_daily_occurrence(23, 30, evening)
    assert datetime.fromtimestamp(same_evening, tz=berlin).strftime("%Y-%m-%d %H:%M") == "2026-09-21 23:30"

    one_shot_due = tomorrow_at(11, 0, evening)
    assert datetime.fromtimestamp(one_shot_due, tz=berlin).strftime("%Y-%m-%d %H:%M") == "2026-09-22 11:00"


def test_daily_advance_survives_dst_fall_back():
    """CEST->CET transition keeps the 11:00 Berlin wall time (25h later in UTC)."""
    berlin = ZoneInfo("Europe/Berlin")
    sent = int(datetime(2026, 10, 24, 11, 0, tzinfo=berlin).timestamp())
    following = next_daily_occurrence(11, 0, sent)
    assert datetime.fromtimestamp(following, tz=berlin).strftime("%Y-%m-%d %H:%M") == "2026-10-25 11:00"
    assert following - sent == 90_000


def test_format_schedule_labels():
    """Acknowledgement labels stay deterministic per schedule kind."""
    daily = parse_reminder_declaration("<@99> remind me daily at 11am: call X", [99], 99)
    assert daily is not None
    assert format_schedule(daily, 1790067600) == "every day at 11:00 Europe/Berlin"

    once = parse_reminder_declaration("<@99> remind me tomorrow at 10am: call Y", [99], 99)
    assert once is not None
    assert format_schedule(once, 1790064000) == "once on 2026-09-22 10:00 Europe/Berlin"

    timeless = parse_reminder_declaration("<@99> Remind us to pray once every day", [99], 99)
    assert timeless is not None
    assert format_schedule(timeless, None) == "every day"


def test_store_daily_and_one_shot_lifecycle(tmp_path):
    """Daily-at-time advances on the wall clock; one-shot completes on fire."""
    store = Store(tmp_path / "state.db")
    assert store.create_reminder(
        101, 200, 7, [], 86_400, now=1_000,
        at_hour=11, at_minute=0, first_due_at=5_000,
    )
    reminder = store.get_reminder(101)
    assert reminder is not None
    assert reminder["next_due_at"] == 5_000
    assert reminder["at_hour"] == 11 and reminder["at_minute"] == 0
    assert reminder["one_shot"] is False

    assert store.advance_daily_reminder(101, sent_at=5_000)
    assert store.get_reminder(101)["next_due_at"] == 36_000

    assert store.create_reminder(
        102, 200, 7, [], 86_400, now=1_000,
        one_shot=True, at_hour=11, at_minute=0, first_due_at=1_000_000,
    )
    assert store.get_reminder(102)["one_shot"] is True
    due_ids = {item["source_message_id"] for item in store.claim_due_reminders(now=1_000_000)}
    assert due_ids == {101, 102}
    assert store.set_reminder_completed(102, True)
    assert [item["source_message_id"] for item in store.claim_due_reminders(now=10_000_000)] == [101]
    store.close()


def test_render_one_shot_omits_next_check():
    """One-shot posts read as single-fire, not repeating."""
    body = render_reminder("call DeutschePost", 100, 200, [], 1_000_000, one_shot=True)
    assert "One-time reminder" in body
    assert "Next check" not in body
    assert "/200/100" in body


@pytest.mark.asyncio
async def test_gateway_intercepts_daily_at_time(tmp_path, monkeypatch):
    """A daily-at-time declaration is stored with schedule fields, no audit."""
    cfg = Config(reminder_channel_id=200)
    cfg.watches = [WatchTarget(channel_id=200, repo_path=str(tmp_path), bot_user_id=99)]
    watcher = OIWatcher(cfg, Store(tmp_path / "state.db"))
    sent_chunks: list[str] = []

    async def fake_deliver(*args, **kwargs):
        """Capture the acknowledgement without contacting Discord."""
        sent_chunks.append(kwargs.get("chunk", ""))
        return SimpleNamespace(ok=True, discord_message_id=101)

    monkeypatch.setattr("oi_agent.watch.discord_client.Poster.deliver_chunk", fake_deliver)
    message = SimpleNamespace(
        id=103,
        channel=SimpleNamespace(id=200),
        author=SimpleNamespace(id=7),
        content="<@99> remind me daily at 11am: call DeutschePost",
        mentions=[SimpleNamespace(id=99)],
    )

    await watcher.on_message(message)

    reminder = watcher._store.get_reminder(103)
    assert reminder is not None
    assert reminder["one_shot"] is False
    assert (reminder["at_hour"], reminder["at_minute"]) == (11, 0)
    assert reminder["next_due_at"] > 0
    assert sent_chunks and "every day at 11:00 Europe/Berlin" in sent_chunks[0]
    watcher._store.close()
