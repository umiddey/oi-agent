"""Deterministic parsing and rendering for Discord reminder declarations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Iterable

COMPLETION_EMOJIS = frozenset({"☑️", "✅", "✔️", "☑", "✔"})
_INTERVAL_RE = re.compile(
    r"\bevery\s+(?:(?P<count>\d+)\s*)?(?P<unit>minute|minutes|hour|hours|day|days)\b",
    re.IGNORECASE,
)
_BOT_MENTION_RE = re.compile(r"<@!?\d+>")
_SEPARATOR_RE = re.compile(r"^\s*(?::|-|–|—)\s*")


@dataclass(frozen=True)
class ReminderDeclaration:
    """A validated reminder request, with task text kept transient only."""

    interval_seconds: int
    target_member_ids: tuple[int, ...]
    task_text: str


def parse_reminder_declaration(
    content: str,
    mention_ids: Iterable[int],
    bot_user_id: int,
) -> ReminderDeclaration | None:
    """Parse an explicit OI reminder command without guessing ambiguous schedules.

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
    task = re.sub(r"^\s*remind(?:\s+me)?\b", "", task, flags=re.IGNORECASE).strip()
    task = _INTERVAL_RE.sub("", task, count=1).strip()
    task = _SEPARATOR_RE.sub("", task)
    if not task or len(task) > 1_500:
        return None

    targets = tuple(member_id for member_id in ids if member_id != bot_user_id)
    return ReminderDeclaration(interval_seconds, targets, task)


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
) -> str:
    """Render a bounded reminder reply with a source jump link and due time."""
    mentions = " ".join(f"<@{int(member_id)}>" for member_id in target_member_ids)
    prefix = f"{mentions} " if mentions else ""
    jump_url = f"https://discord.com/channels/@me/{channel_id}/{source_message_id}"
    return (
        f"{prefix}Reminder: {task_text[:1_500].strip()}\n"
        f"Next check: {format_due_time(next_due_at)}\n"
        f"Source: {jump_url}"
    )
