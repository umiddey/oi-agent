"""Deterministic parsing and rendering for Discord reminder declarations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from datetime import time as dtime
from typing import Iterable
from zoneinfo import ZoneInfo

#: Wall-time zone for all daily/one-shot fire times (user decision).
REMINDER_TZ_NAME = "Europe/Berlin"

COMPLETION_EMOJIS = frozenset({"☑️", "✅", "✔️", "☑", "✔"})
_INTERVAL_RE = re.compile(
    r"\bevery\s+(?:(?P<count>\d+)\s*)?(?P<unit>minute|minutes|hour|hours|day|days)\b",
    re.IGNORECASE,
)
_DAILY_RE = re.compile(r"\b(?:every\s+day|everyday|daily)\b", re.IGNORECASE)
# Time with an explicit `at` (used for daily-at-time and one-shot).
_TIME_RE = re.compile(
    r"\bat\s+(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ampm>am|pm|a\.m\.|p\.m\.)?",
    re.IGNORECASE,
)
# Bare clock (requires am/pm so random numbers are never treated as times).
_BARE_CLOCK_RE = re.compile(
    r"\b(?P<h>\d{1,2})(?::(?P<m>\d{2}))?\s*(?P<ampm>am|pm|a\.m\.|p\.m\.)",
    re.IGNORECASE,
)
_TOMORROW_RE = re.compile(r"\btomorrow\b", re.IGNORECASE)
_BOT_MENTION_RE = re.compile(r"<@!?\d+>")
_SEPARATOR_RE = re.compile(r"^\s*(?::|-|–|—)\s*")
# Extended leading-verb strip for the new syntaxes (old interval path unchanged).
_VERB_RE = re.compile(r"^\s*(?:please\s+)?remind(?:\s+me|\s+us)?\b", re.IGNORECASE)
_LEGACY_VERB_RE = re.compile(r"^\s*remind(?:\s+me)?\b", re.IGNORECASE)
_LEADING_TO_RE = re.compile(r"^\s*to\b", re.IGNORECASE)


@dataclass(frozen=True)
class ReminderDeclaration:
    """A validated reminder request, with task text kept transient only."""

    interval_seconds: int
    target_member_ids: tuple[int, ...]
    task_text: str
    #: 'interval' (every N...), 'daily' (daily/every day, optional at-time), or
    #: 'once' (tomorrow at time, single fire).
    kind: str = "interval"
    #: Berlin wall-clock fire time for daily/once kinds; None when untimed.
    hour: int | None = None
    minute: int | None = None


def _parse_clock(hour_text: str, minute_text: str | None, ampm_text: str | None) -> tuple[int, int] | None:
    """Validate a clock reading without guessing ambiguous values."""
    try:
        hour_raw, minute = int(hour_text), int(minute_text or "0")
    except ValueError:
        return None
    if not 0 <= minute <= 59:
        return None
    if ampm_text:
        marker = ampm_text.casefold().replace(".", "")
        if marker not in ("am", "pm") or not 1 <= hour_raw <= 12:
            return None
        return (hour_raw % 12 + (12 if marker == "pm" else 0)), minute
    if not 0 <= hour_raw <= 23:
        return None
    return hour_raw, minute


def _strip_new_task(text: str) -> str:
    """Strip the extended leading verb plus one leading 'to' for new syntaxes."""
    text = _VERB_RE.sub("", text).strip()
    # "remind me to call X" -> "call X" (new syntaxes only; legacy path untouched).
    text = _LEADING_TO_RE.sub("", text).strip()
    return _SEPARATOR_RE.sub("", text)


def parse_reminder_declaration(
    content: str,
    mention_ids: Iterable[int],
    bot_user_id: int,
) -> ReminderDeclaration | None:
    """Parse an explicit OI reminder command without guessing ambiguous schedules.

    Accepted schedules (bot mention required):
    - ``every N minutes|hours|days`` (legacy, unchanged behavior);
    - ``daily`` / ``every day`` / ``everyday``, with optional ``at H[:MM] [am|pm]``;
    - one-shot ``tomorrow`` with an explicit ``at H[:MM] [am|pm]`` (either order).

    Args:
        content: Raw Discord message content.
        mention_ids: Discord-resolved user IDs mentioned in the message.
        bot_user_id: OI's Discord user ID.

    Returns:
        A validated declaration, or ``None`` when the message is not a reminder command.
    """
    if not isinstance(content, str) or not content.strip() or bot_user_id <= 0:
        return None
    ids = tuple(dict.fromkeys(int(value) for value in mention_ids if int(value) > 0))
    if bot_user_id not in ids:
        return None
    targets = tuple(member_id for member_id in ids if member_id != bot_user_id)

    body = _BOT_MENTION_RE.sub("", content).strip()

    daily_match = _DAILY_RE.search(body)
    if daily_match is not None:
        time_match = _TIME_RE.search(body)
        if time_match is None:
            # Timeless daily keeps the exact legacy task-text behavior.
            task = _BOT_MENTION_RE.sub("", content).strip()
            task = _LEGACY_VERB_RE.sub("", task).strip()
            task = _DAILY_RE.sub("", task, count=1).strip()
            task = _SEPARATOR_RE.sub("", task)
            if not task or len(task) > 1_500:
                return None
            return ReminderDeclaration(86_400, targets, task, "daily", None, None)
        clock = _parse_clock(time_match.group("h"), time_match.group("m"), time_match.group("ampm"))
        if clock is None:
            return None
        hour, minute = clock
        task = _DAILY_RE.sub("", body, count=1).strip()
        task = _TIME_RE.sub("", task, count=1).strip()
        task = _strip_new_task(task)
        if not task or len(task) > 1_500:
            return None
        return ReminderDeclaration(86_400, targets, task, "daily", hour, minute)

    tomorrow_match = _TOMORROW_RE.search(body)
    if tomorrow_match is not None:
        time_match = _TIME_RE.search(body) or _BARE_CLOCK_RE.search(body)
        if time_match is None:
            return None
        clock = _parse_clock(time_match.group("h"), time_match.group("m"), time_match.group("ampm"))
        if clock is None:
            return None
        hour, minute = clock
        task = _TOMORROW_RE.sub("", body, count=1).strip()
        if _TIME_RE.search(task):
            task = _TIME_RE.sub("", task, count=1).strip()
        else:
            task = _BARE_CLOCK_RE.sub("", task, count=1).strip()
        task = _strip_new_task(task)
        if not task or len(task) > 1_500:
            return None
        return ReminderDeclaration(86_400, targets, task, "once", hour, minute)

    match = _INTERVAL_RE.search(content)
    if match is None:
        return None
    count = int(match.group("count") or "1")
    if count <= 0:
        return None
    unit = match.group("unit").casefold()
    multiplier = 60 if unit.startswith("minute") else 3_600 if unit.startswith("hour") else 86_400
    interval_seconds = count * multiplier
    if interval_seconds > 365 * 86_400:
        return None

    task = _BOT_MENTION_RE.sub("", content).strip()
    task = _LEGACY_VERB_RE.sub("", task).strip()
    task = _INTERVAL_RE.sub("", task, count=1).strip()
    task = _SEPARATOR_RE.sub("", task)
    if not task or len(task) > 1_500:
        return None

    return ReminderDeclaration(interval_seconds, targets, task)


def _berlin_month_day(after_ts: int) -> datetime:
    """Return the Berlin-local now for a unix timestamp."""
    return datetime.fromtimestamp(int(after_ts), tz=ZoneInfo(REMINDER_TZ_NAME))


def next_daily_occurrence(hour: int, minute: int, after_ts: int) -> int:
    """Return the next Berlin HH:MM strictly after ``after_ts`` (DST-safe).

    On DST transition days a nonexistent wall time resolves per ``zoneinfo``
    convention instead of raising; documented best-effort edge.
    """
    if not 0 <= int(hour) <= 23 or not 0 <= int(minute) <= 59:
        raise ValueError("hour/minute out of range")
    base = _berlin_month_day(after_ts)
    candidate = base.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
    if int(candidate.timestamp()) <= int(after_ts):
        candidate += timedelta(days=1)
    return int(candidate.timestamp())


def tomorrow_at(hour: int, minute: int, after_ts: int) -> int:
    """Return next-calendar-date Berlin HH:MM for a one-shot reminder."""
    if not 0 <= int(hour) <= 23 or not 0 <= int(minute) <= 59:
        raise ValueError("hour/minute out of range")
    base = _berlin_month_day(after_ts)
    day = (base.date() + timedelta(days=1))
    candidate = datetime.combine(day, dtime(int(hour), int(minute)), tzinfo=ZoneInfo(REMINDER_TZ_NAME))
    return int(candidate.timestamp())


def format_schedule(declaration: ReminderDeclaration, first_due_at: int | None = None) -> str:
    """Format a validated schedule for an acknowledgement message."""
    if declaration.kind == "once" and declaration.hour is not None:
        due = _berlin_month_day(first_due_at) if first_due_at else None
        stamp = due.strftime("%Y-%m-%d %H:%M") if due else f"{declaration.hour:02d}:{declaration.minute or 0:02d}"
        return f"once on {stamp} {REMINDER_TZ_NAME}"
    if declaration.kind == "daily" and declaration.hour is not None:
        return f"every day at {declaration.hour:02d}:{declaration.minute or 0:02d} {REMINDER_TZ_NAME}"
    if declaration.kind == "daily":
        return "every day"
    return f"every {format_interval(declaration.interval_seconds)}"


def format_due_time(timestamp: int) -> str:
    """Format a persisted due timestamp for a concise Discord message."""
    return datetime.fromtimestamp(int(timestamp), tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def format_interval(interval_seconds: int) -> str:
    """Format a validated interval for an acknowledgement message."""
    if interval_seconds % 86_400 == 0:
        count, unit = interval_seconds // 86_400, "day"
    elif interval_seconds % 3_600 == 0:
        count, unit = interval_seconds // 3_600, "hour"
    else:
        count, unit = interval_seconds // 60, "minute"
    return f"{count} {unit}{'' if count == 1 else 's'}"


def render_reminder(
    task_text: str,
    source_message_id: int,
    channel_id: int,
    target_member_ids: Iterable[int],
    next_due_at: int,
    *,
    one_shot: bool = False,
) -> str:
    """Render a bounded reminder reply with a source jump link and due time."""
    mentions = " ".join(f"<@{int(member_id)}>" for member_id in target_member_ids)
    prefix = f"{mentions} " if mentions else ""
    jump_url = f"https://discord.com/channels/@me/{channel_id}/{source_message_id}"
    if one_shot:
        return (
            f"{prefix}One-time reminder: {task_text[:1_500].strip()}\n"
            f"Source: {jump_url}"
        )
    return (
        f"{prefix}Reminder: {task_text[:1_500].strip()}\n"
        f"Next check: {format_due_time(next_due_at)}\n"
        f"Source: {jump_url}"
    )
