"""Deterministic explicit-trigger gate — decides what deserves attention.

Rules (in order):
1. Bot mentioned -> RESPOND directly.
2. ``oi:`` command-style prefix -> RESPOND.
3. ``#oi``, ``#audit`` or ``#bugreport`` marker -> RESPOND.
4. Everything else -> IGNORE.

Author identity and ordinary claim vocabulary are context, not permission to
interrupt a conversation with an autonomous reply.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class Action(str, Enum):
    """What the pipeline should do with a message."""

    IGNORE = "ignore"
    RESPOND = "respond"


@dataclass
class GateDecision:
    """Result of gating one incoming message.

    Attributes:
        action: Pipeline action to take.
        reason: Short human-readable explanation for logs.
    """

    action: Action
    reason: str


_COMMAND_RE = re.compile(r"^\s*oi\s*:", re.IGNORECASE)
_MARKER_RE = re.compile(
    r"(?<!\w)#(?:oi|audit|bugreport)\b", re.IGNORECASE
)


def evaluate(
    content: str,
    mention_ids: list[int],
    bot_user_id: int,
) -> GateDecision:
    """Admit only messages that explicitly request OI's attention.

    Args:
        content: Message text to inspect for command/hashtag markers.
        mention_ids: Ids of users mentioned in the message.
        bot_user_id: Our bot application user id (mention trigger).

    Returns:
        GateDecision with the action and its justification.
    """
    if bot_user_id and bot_user_id in mention_ids:
        return GateDecision(Action.RESPOND, "bot mentioned")
    if _COMMAND_RE.search(content):
        return GateDecision(Action.RESPOND, "oi command")
    marker = _MARKER_RE.search(content)
    if marker:
        return GateDecision(
            Action.RESPOND, f"explicit marker '{marker.group(0).lower()}'"
        )
    return GateDecision(Action.IGNORE, "no explicit trigger")
