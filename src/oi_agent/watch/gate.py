"""Deterministic message gate — decides what deserves attention, no LLM.

Rules (in order):
1. Bot mentioned -> RESPOND directly.
2. Author is a founder -> RESPOND.
3. Content contains a claim keyword -> RESPOND.
4. Everything else -> IGNORE.
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


def evaluate(
    content: str,
    mention_ids: list[int],
    author_id: int,
    bot_user_id: int,
    founder_ids: list[int],
    claim_keywords: list[str],
) -> GateDecision:
    """Apply the deterministic gate rules to one message.

    Args:
        content: Message text (lowercased for keyword matching).
        mention_ids: Ids of users mentioned in the message.
        author_id: Message author id.
        bot_user_id: Our bot application user id (mention trigger).
        founder_ids: Author ids that always pass the gate.
        claim_keywords: Lowercase substrings forcing a response.

    Returns:
        GateDecision with the action and its justification.
    """
    if bot_user_id and bot_user_id in mention_ids:
        return GateDecision(Action.RESPOND, "bot mentioned")
    if author_id in founder_ids:
        return GateDecision(Action.RESPOND, "founder message")
    low = content.lower()
    # Word-boundary match: a plain substring test would fire "bug" on
    # "debugging" or "model" on "modeling" and trigger unwanted full audits.
    hit = next(
        (k for k in claim_keywords
         if re.search(rf"\b{re.escape(k)}\b", low)),
        None,
    )
    if hit:
        return GateDecision(Action.RESPOND, f"claim keyword '{hit}'")
    return GateDecision(Action.IGNORE, "no trigger")
